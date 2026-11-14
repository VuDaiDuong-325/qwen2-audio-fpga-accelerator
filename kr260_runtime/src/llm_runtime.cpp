// =============================================================================
// llm_runtime.cpp - Runtime C++ cho phan LLM (prefill + decode) cua Qwen2-Audio INT8 tren KR260.
//   Doc goi `qwen2audio-int8-pkg/1` (pkg_v1) bang mmap, cung ngu nghia voi ref_runtime.py (cau hinh P1.5).
//   PL (NPU INT8 16x16) lam MOI phep GEMM; PS lam phan con lai (RMSNorm, RoPE, attention INT8, SiLU, requant, argmax).
//   Hang ngoai lai (fallback) KHONG tinh tren PS: x duoc tach 3 "chu so" int8 (--digits), chay 1 luot PL INT32 voi M=3 (gan nhu bang gia M=1,
//   xem npu_mbench) roi hau xu ly float64 tren PS  ==  ref_runtime.py --fb-digits 3.
//
// Backend:   --backend soft   : GEMM INT8 chinh xac tren CPU (de kiem tra tren may bat ky, KHONG can NPU)
//            --backend npu    : NPU that (can -DNPU_DMAHEAP, root, NPU_PL_OK=1). Trong so nam san trong cac dma-buf (CMA), 'dc cvac' 1 LAN.
//            (bien dich voi -DRT_FAKE_NPU: duong NPU chay nhung 'phan cung' la ham C++ doc dung dia chi/stride da cau hinh -> kiem tra layout)
//
// Build (board):  g++ -O3 -mcpu=cortex-a53 -ffp-contract=off -DNPU_DMAHEAP -pthread -o llm_runtime llm_runtime.cpp -lm   (tu kiem tra: ./llm_runtime --selftest)
// Chay  (board):  sudo env NPU_PL_OK=1 ./llm_runtime --pkg ../pkg_v1 --ids ids.i32 --audio rows.f32 --backend npu [--max-new 80]
//   ids.i32  : int32 [P] token prompt (da co <|AUDIO|> lap lai);  rows.f32 : float32 [n_audio, hidden] hang audio (dau ra projector)
//   Tao 2 file nay bang:  python3 make_cpp_inputs.py --pkg ../pkg_v1 --clip 0 --out in_clip00   (Colab / may co numpy)
// Tuy chon: --teacher gen.i32 (teacher-forced: nap token cho truoc, in top-5 moi vi tri)  --dump-top5 f.txt  --dump-logits f.f32
//           (env RT_NOPREFETCH=1 tat luong nap truoc trong so encoder)  --digits N (mac dinh 3)  --wav file.wav --template ids.i32 (am thanh -> encoder C++ -> chu; thay cho --ids/--audio)  --dump-rows f --dump-mel f  --detok tokens.txt --vocab DIR (doi token -> chu)  --text-out f.txt  --threads N (mac dinh 1; 2..4 chia attention/requant/silu cho cac nhan A53, chi dung khi board khong ai khac dung)  --max-new N  --out tokens.txt  --scratch-mb 16  --verbose
// =============================================================================
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <atomic>
#include <functional>
#include <thread>

#if defined(NPU_DMAHEAP) && !defined(RT_FAKE_NPU)
#include "kr260_npu_driver.h"
#if !defined(NPU_HEAP_DEV) || !defined(KR260_NPU_DRIVER_H)
#error "kr260_npu_driver.h trong thu muc nay KHONG phai ban KR260 (thieu NPU_HEAP_DEV). Chep lai kr260_npu_driver.h moi nhat roi bien dich lai."
#endif
#define RT_HAVE_NPU 1
#elif defined(RT_FAKE_NPU)
#define RT_HAVE_NPU 1
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#define HAVE_NEON 1
#elif defined(USE_N2S)
#include "NEON_2_SSE.h"
#define HAVE_NEON 1
#define vcvtnq_s32_f32(a) _mm_cvtps_epi32(a)          /* chi de TEST tren x86 */
#else
#define HAVE_NEON 0
#endif

static double now_s() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
[[noreturn]] static void die(const char *fmt, const char *a = "", const char *b = "") { fprintf(stderr, fmt, a, b); fprintf(stderr, "\n"); exit(2); }

// ----------------------------------------------------------------------------- nhom luong (tuy chon --threads N; mac dinh 1 = khong tao luong)
struct Pool {
    int n = 1; std::vector<std::thread> th; std::atomic<unsigned> gen{0}; std::atomic<int> done{0}; std::atomic<bool> stop{false}; const std::function<void(int)> *job = nullptr;
    static inline void relax() {
#if defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#else
        asm volatile("" ::: "memory");
#endif
    }
    void start(int nt) {
        n = std::max(1, std::min(nt, 8));
        for (int id = 1; id < n; id++) th.emplace_back([this, id]() {
            // spin thuan (ngu 40us moi lan lam vung song song sau moi luot PL mat ~150-300us de thuc day -> cham hon ca 1 luong); chi ngu khi nhan roi > 100 ms
            unsigned seen = 0; unsigned spins = 0; double idle0 = 0;
            for (;;) {
                unsigned g = gen.load(std::memory_order_acquire);
                if (g == seen) {
                    if (stop.load(std::memory_order_relaxed)) return; relax();
                    if ((++spins & 1023) == 0) { double t = now_s(); if (idle0 == 0) idle0 = t; else if (t - idle0 > 0.1) { struct timespec ts = {0, 50000}; nanosleep(&ts, nullptr); } }
                    continue;
                }
                seen = g; spins = 0; idle0 = 0; (*job)(id); done.fetch_add(1, std::memory_order_release);
            }
        });
    }
    ~Pool() { stop = true; for (auto &t : th) t.join(); }
    void run(const std::function<void(int)> &f) {
        if (n == 1) { f(0); return; }
        job = &f; done.store(0, std::memory_order_relaxed); gen.fetch_add(1, std::memory_order_release);
        f(0); while (done.load(std::memory_order_acquire) < n - 1) relax();
    }
    // chia [0,items) thanh n doan lien tiep (ranh gioi la boi cua unit), goi f(i0,i1,tid); n==1 -> goi thang f(0,items,0)
    void split(int items, int unit, const std::function<void(int, int, int)> &f) {
        if (n == 1 || items <= 0) { f(0, items, 0); return; }
        const int nu = (items + unit - 1) / unit;
        std::function<void(int)> job2 = [&](int tid) { int u0 = (int)((long)nu * tid / n), u1 = (int)((long)nu * (tid + 1) / n); int i0 = u0 * unit, i1 = std::min(items, u1 * unit); if (i0 < i1) f(i0, i1, tid); };
        run(job2);
    }
};
static Pool POOL;

// ----------------------------------------------------------------------------- JSON toi thieu
struct J {
    enum T { Nul, Bool, Str, Num, Arr, Obj } t = Nul;
    double num = 0; bool b = false; std::string s; std::vector<J> a; std::vector<std::pair<std::string, J>> o;
    const J &operator[](const char *k) const { for (auto &p : o) if (p.first == k) return p.second; die("JSON: thieu khoa '%s'", k); }
    const J &operator[](size_t i) const { return a.at(i); }
    bool has(const char *k) const { for (auto &p : o) if (p.first == k) return true; return false; }
    int i() const { return (int)num; }
};
struct JP {
    const char *p, *e;
    void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++; }
    std::string str() {
        std::string r; p++;
        while (p < e && *p != '"') {
            if (*p == '\\') { p++; char c = *p++; switch (c) { case 'n': r += '\n'; break; case 't': r += '\t'; break; case 'r': r += '\r'; break; case 'b': r += '\b'; break; case 'f': r += '\f'; break;
                case 'u': { unsigned v = (unsigned)strtoul(std::string(p, 4).c_str(), nullptr, 16); p += 4; if (v < 0x80) r += (char)v; else if (v < 0x800) { r += (char)(0xC0 | (v >> 6)); r += (char)(0x80 | (v & 63)); } else { r += (char)(0xE0 | (v >> 12)); r += (char)(0x80 | ((v >> 6) & 63)); r += (char)(0x80 | (v & 63)); } break; }
                default: r += c; } }
            else r += *p++;
        }
        p++; return r;
    }
    J val() {
        ws(); J j;
        if (*p == '{') { j.t = J::Obj; p++; ws(); if (*p == '}') { p++; return j; }
            for (;;) { ws(); std::string k = str(); ws(); p++; j.o.emplace_back(k, val()); ws(); if (*p == ',') { p++; continue; } p++; break; } }
        else if (*p == '[') { j.t = J::Arr; p++; ws(); if (*p == ']') { p++; return j; }
            for (;;) { j.a.push_back(val()); ws(); if (*p == ',') { p++; continue; } p++; break; } }
        else if (*p == '"') { j.t = J::Str; j.s = str(); }
        else if (!strncmp(p, "true", 4)) { j.t = J::Bool; j.b = true; p += 4; }
        else if (!strncmp(p, "false", 5)) { j.t = J::Bool; j.b = false; p += 5; }
        else if (!strncmp(p, "null", 4)) { p += 4; }
        else if (!strncmp(p, "NaN", 3)) { j.t = J::Num; j.num = NAN; p += 3; }
        else if (!strncmp(p, "Infinity", 8)) { j.t = J::Num; j.num = INFINITY; p += 8; }
        else if (!strncmp(p, "-Infinity", 9)) { j.t = J::Num; j.num = -INFINITY; p += 9; }
        else { char *q; j.t = J::Num; j.num = strtod(p, &q); if (q == p) die("JSON: loi cu phap"); p = q; }
        return j;
    }
};

// ----------------------------------------------------------------------------- tep mmap
struct File { const uint8_t *p = nullptr; size_t n = 0; };
static File map_file(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY); if (fd < 0) die("khong mo duoc %s", path.c_str());
    struct stat st; fstat(fd, &st);
    void *m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (m == MAP_FAILED) die("mmap that bai: %s", path.c_str());
    File f; f.p = (const uint8_t *)m; f.n = st.st_size; return f;
}
static std::vector<uint8_t> read_all(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb"); if (!f) die("khong mo duoc %s", path.c_str());
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n); if (n && fread(v.data(), 1, n, f) != (size_t)n) die("doc loi %s", path.c_str()); fclose(f); return v;
}

// ----------------------------------------------------------------------------- goi
struct Member { int c0, c1; };
struct Gemm {
    std::string name; int K = 0, N = 0; double sin = 0; float sin_f = 0, thr_in = 0; bool int32 = false, rowfb = false;
    std::vector<Member> mem;
    const int8_t *B = nullptr; const float *sw = nullptr, *bias = nullptr, *sout = nullptr;
    std::vector<int32_t> bi, mult, shift, zp; std::vector<double> rawk, sinsw;
    std::vector<int8_t> nsh8; std::vector<int16_t> zp16; std::vector<int32_t> tmax; bool neon_rq = false, tiled = false; int rq_why = 0;   // requant NEON: -shift (int8, them 8 phan tu dem), zp (int16); tmax[n] = |acc+bi| lon nhat khong bao hoa (chinh xac theo double)
    // vi tri trong NPU (cac manh theo cot N)
    struct Piece { int n0, np; int chunk; size_t off; };
    std::vector<Piece> pieces;
};
struct Attn { float sq, sk, fac, scaling; };
// tinh san mang phu cho requant NEON + tmax (|acc+bias| lon nhat khong bao hoa, tinh chinh xac theo double nhu ref)
static void finalize_requant(Gemm &g) {
    g.nsh8.assign(g.N + 8, 0); g.zp16.assign(g.N + 8, 0); g.tmax.resize(g.N); bool ok = true; g.rq_why = 0;
            for (int n = 0; n < g.N; n++) {
                if (g.shift[n] < 0 || g.shift[n] > 62 || g.zp[n] < -32768 || g.zp[n] > 32767) { ok = false; g.rq_why |= 1; } else { g.nsh8[n] = (int8_t)(-g.shift[n]); g.zp16[n] = (int16_t)g.zp[n]; }
                if (llabs((long long)g.bi[n]) + (long long)g.K * 128 * 128 >= (1LL << 31)) { ok = false; g.rq_why |= 2; }       // acc+bias phai nam trong int32
                double rk = g.rawk[n]; int64_t c;
                if (!(rk > 0)) c = INT32_MAX; else { c = (int64_t)floor(127.5 / rk); if (c > INT32_MAX) c = INT32_MAX; if (c < 0) c = 0;
                    while (c < INT32_MAX && fabs((double)(c + 1) * rk) <= 127.5) c++;
                    while (c > 0 && fabs((double)c * rk) > 127.5) c--; }
                g.tmax[n] = (int32_t)c;
            }
            g.neon_rq = ok && HAVE_NEON; { int e = 0; g.tiled = !g.mem.empty(); for (auto &mb : g.mem) { if (mb.c0 != e) g.tiled = false; e = mb.c1; } if (e != g.N) g.tiled = false; }
}
struct Pkg {
    std::string root; J man; std::map<std::string, File> files;
    int layers, hid, H, Hkv, d, inter, vocab; double theta, rms_eps;
    int eos, audio_id, max_new; float penalty;
    std::vector<Gemm> g;                                    // [3*? ] llm.j.{qkv,o_proj,gate_up,down_proj} theo lop; cuoi cung lm_head
    std::vector<const float *> in_ln, post_ln, down_inv; const float *final_norm = nullptr;
    std::vector<Attn> att;
    const uint16_t *emb = nullptr;
    // ---- encoder audio (chi nap khi dung --wav): conv1, conv2, [enc.i.{qkv,out_proj,fc1,fc2}]*, proj
    int e_layers = 0, e_d = 0, e_heads = 0, e_ffn = 0, n_fft = 400, hop = 160, n_samples = 480000, n_mels = 128, n_freq = 201; double e_ln_eps = 1e-5; float e_qscale = 0.25f;
    std::vector<Gemm> eg; std::vector<Attn> eatt; std::vector<const float *> e_ln1w, e_ln1b, e_ln2w, e_ln2b;
    const float *e_lnw = nullptr, *e_lnb = nullptr, *e_pos = nullptr, *mel_filt = nullptr, *hann = nullptr;
    Gemm &EG(int layer, int k) { return eg[2 + layer * 4 + k]; }   // k: 0 qkv, 1 out_proj, 2 fc1, 3 fc2
    Gemm &CONV1() { return eg[0]; } Gemm &CONV2() { return eg[1]; } Gemm &PROJ() { return eg.back(); }
    void load_encoder() {
        const J &E = man["dims"]["enc"], &A = man["dims"]["audio"];
        e_layers = E["layers"].i(); e_d = E["d_model"].i(); e_heads = E["heads"].i(); e_ffn = E["ffn"].i(); e_ln_eps = E["ln_eps"].num; e_qscale = (float)E["q_scaling"].num;
        n_fft = A["n_fft"].i(); hop = A["hop"].i(); n_samples = A["n_samples"].i(); n_mels = A["feature_size"].i();
        std::map<std::string, const J *> ge; for (auto &e : man["gemms"].a) ge[e["name"].s] = &e;
        const char *nm[4] = {"qkv", "out_proj", "fc1", "fc2"};
        eg.resize(2 + e_layers * 4 + 1);
        load_gemm(eg[0], *ge.at("conv1")); load_gemm(eg[1], *ge.at("conv2"));
        for (int j = 0; j < e_layers; j++) for (int k = 0; k < 4; k++) load_gemm(eg[2 + j * 4 + k], *ge.at("enc." + std::to_string(j) + "." + nm[k]));
        load_gemm(eg.back(), *ge.at("proj"));
        auto T = [&](const std::string &k) { return (const float *)blob(man["tensors"][k.c_str()]); };
        for (int j = 0; j < e_layers; j++) {
            std::string pre = "audio_tower.layers." + std::to_string(j);
            e_ln1w.push_back(T(pre + ".self_attn_layer_norm.weight")); e_ln1b.push_back(T(pre + ".self_attn_layer_norm.bias"));
            e_ln2w.push_back(T(pre + ".final_layer_norm.weight")); e_ln2b.push_back(T(pre + ".final_layer_norm.bias"));
            const J &a = man["attention"][("enc." + std::to_string(j)).c_str()]; eatt.push_back({(float)a["sq"].num, (float)a["sk"].num, (float)a["fac"].num, (float)a["scaling"].num});
        }
        e_lnw = T("audio_tower.layer_norm.weight"); e_lnb = T("audio_tower.layer_norm.bias"); e_pos = T("audio_tower.embed_positions.weight"); mel_filt = T("mel_filters"); hann = T("hann_window");
        n_freq = man["tensors"]["mel_filters"]["shape"][(size_t)0].i();
        if (eg.back().N != hid) die("proj N != hidden LLM");
    }
    const uint8_t *blob(const J &e) {
        const std::string &f = e["file"].s;
        auto it = files.find(f); if (it == files.end()) it = files.emplace(f, map_file(root + "/" + f)).first;
        return it->second.p + (size_t)e["offset"].num;
    }
    Gemm &G(int layer, int k) { return g[layer * 4 + k]; }   // k: 0 qkv, 1 o_proj, 2 gate_up, 3 down_proj
    Gemm &LM() { return g.back(); }
    void load(const std::string &r) {
        root = r; auto mb = read_all(r + "/MANIFEST.json"); JP jp{(const char *)mb.data(), (const char *)mb.data() + mb.size()}; man = jp.val();
        if (man["format"].s != "qwen2audio-int8-pkg/1") die("format khong dung");
        const J &L = man["dims"]["llm"]; layers = L["layers"].i(); hid = L["hidden"].i(); H = L["heads"].i(); Hkv = L["kv_heads"].i(); d = L["head_dim"].i();
        inter = L["inter"].i(); vocab = L["vocab"].i(); theta = L["rope_theta"].num; rms_eps = L["rms_eps"].num;
        const J &D = man["decode"]; eos = D["eos_id"].i(); audio_id = D["audio_token_id"].i(); max_new = D["max_new"].i(); penalty = (float)D["repetition_penalty"].num;
        std::map<std::string, const J *> ge; for (auto &e : man["gemms"].a) ge[e["name"].s] = &e;
        const char *nm[4] = {"qkv", "o_proj", "gate_up", "down_proj"};
        g.resize(layers * 4 + 1);
        for (int j = 0; j < layers; j++) for (int k = 0; k < 4; k++) load_gemm(g[j * 4 + k], *ge.at("llm." + std::to_string(j) + "." + nm[k]));
        load_gemm(g.back(), *ge.at("lm_head"));
        for (int j = 0; j < layers; j++) {
            std::string pre = "language_model.model.layers." + std::to_string(j);
            in_ln.push_back((const float *)blob(man["tensors"][(pre + ".input_layernorm.weight").c_str()]));
            post_ln.push_back((const float *)blob(man["tensors"][(pre + ".post_attention_layernorm.weight").c_str()]));
            down_inv.push_back((const float *)blob(man["tensors"][("llm." + std::to_string(j) + ".down_inv").c_str()]));
            const J &a = man["attention"][("llm." + std::to_string(j)).c_str()];
            att.push_back({(float)a["sq"].num, (float)a["sk"].num, (float)a["fac"].num, (float)a["scaling"].num});
        }
        final_norm = (const float *)blob(man["tensors"]["language_model.model.norm.weight"]);
        emb = (const uint16_t *)blob(man["embedding"]);
    }
    void load_gemm(Gemm &g, const J &e) {
        g.name = e["name"].s; g.K = e["K"].i(); g.N = e["N"].i(); g.sin = e["sin"].num; g.int32 = e["out"].s == "int32"; g.rowfb = e["rowfb"].b;
        for (auto &m : e["members"].a) g.mem.push_back({m["cols"][(size_t)0].i(), m["cols"][(size_t)1].i()});
        g.B = (const int8_t *)blob(e["B"]); g.sw = (const float *)blob(e["sw"]); g.bias = (const float *)blob(e["bias_f32"]);
        g.sin_f = (float)g.sin; g.thr_in = (float)(127.5 * g.sin);
        if (g.int32) { const double *swd = (const double *)blob(e["sw_f64"]); g.sinsw.resize(g.N); for (int n = 0; n < g.N; n++) g.sinsw[n] = g.sin * swd[n]; }
        else {
            const int32_t *qp = (const int32_t *)blob(e["qparams"]); g.sout = (const float *)blob(e["sout"]);
            g.bi.resize(g.N); g.mult.resize(g.N); g.shift.resize(g.N); g.zp.resize(g.N); g.rawk.resize(g.N);
            for (int n = 0; n < g.N; n++) { g.bi[n] = qp[4 * n]; g.mult[n] = qp[4 * n + 1]; g.shift[n] = qp[4 * n + 2]; g.zp[n] = qp[4 * n + 3]; g.rawk[n] = g.sin * (double)g.sw[n] / (double)g.sout[n]; }
            finalize_requant(g);
        }
    }
};
static float half2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, f;
    if (e == 0) { if (m == 0) f = s << 31; else { int sh = 0; while (!(m & 1024)) { m <<= 1; sh++; } m &= 1023; f = (s << 31) | ((uint32_t)(113 - sh) << 23) | (m << 13); } }
    else if (e == 31) f = (s << 31) | 0x7f800000u | (m << 13);
    else f = (s << 31) | ((e + 112) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r;
}

// ----------------------------------------------------------------------------- backend GEMM INT8 -> INT32
struct Stats { double t_dyn = 0, pl_wall = 0, pl_hw = 0, ps_rq = 0, ps_i32 = 0, ps_dig = 0, ps_pre = 0; long calls = 0, rows_dig = 0, rows_sat = 0, passes = 0; };
static Stats ST;

struct Backend {
    virtual ~Backend() {}
    virtual void prepare(Pkg &) {}
    // C [M][N] int32 = A [M][K] int8 x B[K][N]
    virtual void gemm32(Gemm &g, const int8_t *A, int M, int32_t *C) = 0;
};
struct SoftBackend : Backend {
    void gemm32(Gemm &g, const int8_t *A, int M, int32_t *C) override {
        const int K = g.K, N = g.N;
        for (int m = 0; m < M; m++) {
            int32_t *c = C + (size_t)m * N; memset(c, 0, sizeof(int32_t) * N);
            for (int k = 0; k < K; k++) { int a = A[(size_t)m * K + k]; if (!a) continue; const int8_t *b = g.B + (size_t)k * N; for (int n = 0; n < N; n++) c[n] += a * b[n]; }
        }
    }
};

#ifdef RT_HAVE_NPU
struct Chunk { uint8_t *va; uint64_t pa; size_t size, used; int fd; };
struct NpuBackend : Backend {
    std::vector<Chunk> chunks; size_t chunk_bytes = 32u << 20;
    volatile uint32_t *regs = nullptr; int memfd = -1;
    // vung scratch (A, C8 gia, C32, params)
    Chunk scr{}; size_t off_a = 0, off_c8 = 0, off_c32 = 0, off_p = 0; int max_m = 32, max_k = 0, max_np = 0;
    int piece_cols = 16384;
    Chunk dyn{}; const Gemm *dyn_owner = nullptr; size_t dyn_bytes = 0;      // GEMM encoder: trong so nap 1 GEMM/lan vao 1 chunk nho (dung chung)

    static int cma_alloc(Chunk &c, size_t len) {
#ifdef RT_FAKE_NPU
        c.va = (uint8_t *)aligned_alloc(4096, len); if (!c.va) return -1; memset(c.va, 0, len); c.pa = (uint64_t)(uintptr_t)c.va; c.size = len; c.used = 0; c.fd = -1; return 0;
#else
        int hfd = open(NPU_HEAP_DEV, O_RDWR | O_CLOEXEC); if (hfd < 0) { fprintf(stderr, "[npu] open %s: %s\n", NPU_HEAP_DEV, strerror(errno)); return -1; }
        int tries = 40; { const char *er = getenv("NPU_ALLOC_RETRY"); if (er && atoi(er) >= 0) tries = atoi(er) + 1; }
        struct npu_heap_alloc a; int rc = -1;
        for (int att = 0; att < tries; att++) {
            memset(&a, 0, sizeof a); a.len = len; a.fd_flags = O_RDWR | O_CLOEXEC; rc = ioctl(hfd, NPU_HEAP_IOCTL_ALLOC, &a);
            if (rc >= 0) break; if (errno != ENOMEM && errno != EBUSY && errno != EAGAIN) break;
            if (att == 1) sync(); usleep(att < 10 ? 100000 : 300000);
        }
        close(hfd);
        if (rc < 0) { fprintf(stderr, "[npu] cap %zu MB tu CMA that bai: %s\n", len >> 20, strerror(errno)); return -1; }
        c.fd = (int)a.fd; c.size = len; c.used = 0;
        c.va = (uint8_t *)mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, c.fd, 0);
        if (c.va == MAP_FAILED) { perror("[npu] mmap dma-buf"); close(c.fd); return -1; }
        memset(c.va, 0, len);
        int pm = open("/proc/self/pagemap", O_RDONLY); if (pm < 0) { perror("[npu] pagemap"); return -1; }
        size_t np = len >> 12; std::vector<uint64_t> ent(np);
        ssize_t got = pread(pm, ent.data(), np * 8, (off_t)(((uintptr_t)c.va >> 12) * 8)); close(pm);
        if (got != (ssize_t)(np * 8)) { fprintf(stderr, "[npu] doc pagemap that bai\n"); return -1; }
        uint64_t pfn0 = ent[0] & ((1ULL << 55) - 1); bool good = (ent[0] >> 63) && pfn0;
        for (size_t i = 1; good && i < np; i++) { uint64_t pfn = ent[i] & ((1ULL << 55) - 1); if (!(ent[i] >> 63) || pfn != pfn0 + i) good = false; }
        if (!good) { fprintf(stderr, "[npu] buffer khong lien tuc / pagemap khong cho PFN (can root)\n"); return -1; }
        c.pa = pfn0 << 12;
        if (c.pa + len > (1ULL << 32)) { fprintf(stderr, "[npu] buffer vuot 4 GiB\n"); return -1; }
        npu_cache_flush_inv(c.va, len);
        return 0;
#endif
    }
    int open_regs() {
#ifdef RT_FAKE_NPU
        return 0;
#else
        const char *ok = getenv("NPU_PL_OK");
        if (!ok || strcmp(ok, "1") != 0) { fprintf(stderr, "[npu] DUNG LAI: chua xac nhan bitstream NPU da nap. Chay: sudo env NPU_PL_OK=1 ...\n"); return -1; }
        memfd = open("/dev/mem", O_RDWR | O_SYNC); if (memfd < 0) { perror("[npu] open /dev/mem (can sudo)"); return -1; }
        regs = (volatile uint32_t *)mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, NPU_CTRL_BASE);
        if (regs == (volatile uint32_t *)MAP_FAILED) { perror("[npu] mmap regs"); return -1; }
        return 0;
#endif
    }
    static size_t up16(size_t x) { return (x + 15) & ~(size_t)15; }
    static size_t up4k(size_t x) { return (x + 4095) & ~(size_t)4095; }

    void prepare(Pkg &P) override {
        if (open_regs() < 0) exit(2);
        // 1) chia cac GEMM thanh manh (lm_head chia theo cot), tinh scratch can dung
        for (auto &g : P.g) {
            g.pieces.clear();
            int step = (g.N > 2 * piece_cols) ? piece_cols : g.N;
            for (int n0 = 0; n0 < g.N; n0 += step) { int np = std::min(step, g.N - n0); g.pieces.push_back({n0, np, -1, 0}); max_np = std::max(max_np, np); }
            max_k = std::max(max_k, g.K);
            if (g.N % 16) die("N=%s khong chia het 16 (can pad)", g.name.c_str());
            if (g.K % 16) die("K=%s khong chia het 16 (can pad)", g.name.c_str());
        }
        for (auto &g : P.eg) {                                                                // encoder: khong nap san, chi tinh kich thuoc
            g.pieces.clear(); if (g.N > 2 * piece_cols) die("encoder GEMM %s qua rong", g.name.c_str());
            g.pieces.push_back({0, g.N, -1, 0}); max_np = std::max(max_np, g.N); max_k = std::max(max_k, g.K);
            if (g.N % 16 || g.K % 16) die("encoder GEMM %s: N,K phai chia het 16", g.name.c_str());
            dyn_bytes = std::max(dyn_bytes, up4k((size_t)g.K * g.N));
        }
        off_a = 0; off_c8 = up4k(off_a + (size_t)max_m * up16(max_k)); off_c32 = up4k(off_c8 + (size_t)max_m * up16(max_np));
        off_p = up4k(off_c32 + (size_t)max_m * 4 * up16(max_np)); size_t scr_bytes = up4k(off_p + (size_t)max_np * 16);
        scr_bytes = std::max<size_t>(scr_bytes, 1u << 20);
        if (cma_alloc(scr, scr_bytes) < 0) exit(2);
        fprintf(stderr, "[npu] scratch %.1f MB tai PA 0x%llx\n", scr_bytes / 1048576.0, (unsigned long long)scr.pa);
        // 2) nap trong so vao cac chunk CMA (1 lan), 'dc cvac' 1 lan
        double t0 = now_s(); size_t total = 0; int lastpct = -1; size_t need = 0;
        for (auto &g : P.g) need += (size_t)g.K * g.N;
        for (auto &g : P.g) for (auto &pc : g.pieces) {
            size_t bytes = (size_t)g.K * pc.np; size_t al = up4k(bytes);
            int ci = -1; for (size_t i = 0; i < chunks.size(); i++) if (chunks[i].size - chunks[i].used >= al) { ci = (int)i; break; }
            if (ci < 0) { Chunk c{}; size_t len = std::max(chunk_bytes, al); if (cma_alloc(c, len) < 0) { fprintf(stderr, "[npu] het CMA sau %.0f MB / can ~%.0f MB trong so\n", total / 1048576.0, need / 1048576.0); exit(2); } chunks.push_back(c); ci = (int)chunks.size() - 1; }
            Chunk &c = chunks[ci]; pc.chunk = ci; pc.off = c.used; c.used += al;
            for (int k = 0; k < g.K; k++) memcpy(c.va + pc.off + (size_t)k * pc.np, g.B + (size_t)k * g.N + pc.n0, pc.np);
#ifndef RT_FAKE_NPU
            npu_cache_clean(c.va + pc.off, bytes);
#endif
            total += bytes; int pct = (int)(100.0 * total / need); if (pct / 10 != lastpct / 10) { fprintf(stderr, "[npu] nap trong so %d%% (%.0f MB, %zu chunk)\n", pct, total / 1048576.0, chunks.size()); lastpct = pct; }
        }
        fprintf(stderr, "[npu] da nap %.0f MB trong so vao %zu chunk CMA trong %.1f s\n", total / 1048576.0, chunks.size(), now_s() - t0);
        if (dyn_bytes) { if (cma_alloc(dyn, dyn_bytes) < 0) exit(2); fprintf(stderr, "[npu] chunk nap dong cho encoder: %.1f MB\n", dyn_bytes / 1048576.0); }
    }
    int run(uint64_t pa_a, uint64_t pa_b, uint64_t lda, uint64_t ldb, int M, int K, int N, uint8_t *c32_va, size_t c32_bytes, uint64_t pa_c32, uint64_t ldc32, uint64_t ldc) {
#ifdef RT_FAKE_NPU
        (void)ldc; (void)K;
        const int8_t *A = (const int8_t *)(uintptr_t)pa_a, *B = (const int8_t *)(uintptr_t)pa_b; int32_t *C = (int32_t *)(uintptr_t)pa_c32;
        for (int m = 0; m < M; m++) { int32_t *c = (int32_t *)((uint8_t *)C + (size_t)m * ldc32); for (int n = 0; n < N; n++) c[n] = 0;
            for (int k = 0; k < K; k++) { int a = A[(size_t)m * lda + k]; const int8_t *b = B + (size_t)k * ldb; for (int n = 0; n < N; n++) c[n] += a * b[n]; } }
        (void)c32_va; (void)c32_bytes; ST.pl_hw += 0; return 0;
#else
        regs[NPU_REG_M] = M; regs[NPU_REG_N] = N; regs[NPU_REG_K] = K; regs[NPU_REG_KDIM] = 128;
        regs[NPU_REG_BASE_A] = (uint32_t)pa_a; regs[NPU_REG_BASE_B] = (uint32_t)pa_b; regs[NPU_REG_BASE_C] = (uint32_t)(scr.pa + off_c8);
        regs[NPU_REG_BASE_C32] = (uint32_t)pa_c32; regs[NPU_REG_BASE_P] = (uint32_t)(scr.pa + off_p);
        regs[NPU_REG_LDA] = (uint32_t)lda; regs[NPU_REG_LDB] = (uint32_t)ldb; regs[NPU_REG_LDC] = (uint32_t)ldc; regs[NPU_REG_LDC32] = (uint32_t)ldc32;
        regs[NPU_REG_OUT_MODE] = NPU_OUT_INT32;
        npu_cache_flush_inv(c32_va, c32_bytes);
        __sync_synchronize();
        double t0 = now_s(); regs[NPU_REG_CTRL] = NPU_CTRL_START;
        uint32_t st; do { st = regs[NPU_REG_CTRL]; if (now_s() - t0 > 10.0) { fprintf(stderr, "[npu] TIMEOUT st=0x%x\n", st); return -2; } } while (!(st & (NPU_ST_DONE | NPU_ST_ERR)));
        __sync_synchronize(); ST.pl_hw += (now_s() - t0);
        npu_cache_flush_inv(c32_va, c32_bytes);
        if (st & NPU_ST_ERR) { fprintf(stderr, "[npu] ERR (cau hinh khong hop le)\n"); return -1; }
        return 0;
#endif
    }
    void gemm32(Gemm &g, const int8_t *A, int M, int32_t *C) override {
        const int K = g.K; const size_t lda = K;                 // K chia het 16
        for (int m0 = 0; m0 < M; m0 += max_m) {
            int mm = std::min(max_m, M - m0);
            uint8_t *sa = scr.va + off_a; for (int m = 0; m < mm; m++) memcpy(sa + (size_t)m * lda, A + (size_t)(m0 + m) * K, K);
#ifndef RT_FAKE_NPU
            npu_cache_clean(sa, (size_t)mm * lda);
#endif
            for (auto &pc : g.pieces) {
                Chunk &c = pc.chunk >= 0 ? chunks[pc.chunk] : dyn; uint64_t ldc32 = 4ull * pc.np, ldc = pc.np;
                if (pc.chunk < 0 && dyn_owner != &g) {                                              // GEMM encoder: nap trong so (1 lan/GEMM, dung lai cho cac luot chu so)
                    double td = now_s(); memcpy(dyn.va, g.B, (size_t)g.K * g.N);
#ifndef RT_FAKE_NPU
                    npu_cache_clean(dyn.va, (size_t)g.K * g.N);
#endif
                    dyn_owner = &g; ST.t_dyn += now_s() - td;
                }
                uint8_t *cv = scr.va + off_c32;
                if (run(scr.pa + off_a, c.pa + pc.off, lda, pc.np, mm, K, pc.np, cv, (size_t)mm * ldc32, scr.pa + off_c32, ldc32, ldc) < 0) die("npu_run that bai (%s)", g.name.c_str());
                ST.passes++;
                for (int m = 0; m < mm; m++) memcpy(C + (size_t)(m0 + m) * g.N + pc.n0, cv + (size_t)m * ldc32, sizeof(int32_t) * pc.np);
            }
        }
    }
};
#endif


// ----------------------------------------------------------------------------- kernel PS nhanh (NEON; ban vo huong du phong). Dung cho head_dim = 64.
static inline float exp_poly(float x) {
    x = x > 88.f ? 88.f : (x < -87.f ? -87.f : x);
    float t = x * 1.44269504f; float nf = (t + 12582912.0f) - 12582912.0f;
    float r = x - nf * 0.693145751953125f - nf * 1.42860677e-6f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f; p = p * r + 8.3334519073e-3f; p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f; p = p * r + 5.0000001201e-1f; p = p * r * r + r + 1.0f;
    int32_t e = ((int32_t)nf + 127) << 23; float sc; memcpy(&sc, &e, 4); return p * sc;
}
#if HAVE_NEON
static inline float32x4_t exp4(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.f)), vdupq_n_f32(88.f));
    int32x4_t n = vcvtnq_s32_f32(vmulq_f32(x, vdupq_n_f32(1.44269504f)));
    float32x4_t nf = vcvtq_f32_s32(n);
    float32x4_t r = vmlsq_f32(x, nf, vdupq_n_f32(0.693145751953125f)); r = vmlsq_f32(r, nf, vdupq_n_f32(1.42860677e-6f));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vmlaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r); p = vmlaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vmlaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r); p = vmlaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vmlaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    p = vmlaq_f32(vaddq_f32(r, vdupq_n_f32(1.0f)), p, vmulq_f32(r, r));
    return vmulq_f32(p, vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23)));
}
static inline float32x4_t recip4(float32x4_t d) { float32x4_t r = vrecpeq_f32(d); r = vmulq_f32(r, vrecpsq_f32(d, r)); return vmulq_f32(r, vrecpsq_f32(d, r)); }
#endif
// out[i] = silu(g[i]) * u[i] * dinv[i]   (cung thu tu nhan voi ref: (silu*u)*down_inv)
static void silu_mul_dinv(const float *g, const float *u, const float *dinv, float *out, int n) {
    int i = 0;
#if HAVE_NEON
    float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 8 <= n; i += 8) {
        float32x4_t g0 = vld1q_f32(g + i), g1 = vld1q_f32(g + i + 4);
        float32x4_t r0 = recip4(vaddq_f32(exp4(vnegq_f32(g0)), one)), r1 = recip4(vaddq_f32(exp4(vnegq_f32(g1)), one));
        vst1q_f32(out + i,     vmulq_f32(vmulq_f32(vmulq_f32(g0, r0), vld1q_f32(u + i)),     vld1q_f32(dinv + i)));
        vst1q_f32(out + i + 4, vmulq_f32(vmulq_f32(vmulq_f32(g1, r1), vld1q_f32(u + i + 4)), vld1q_f32(dinv + i + 4)));
    }
#endif
    for (; i < n; i++) out[i] = (g[i] / (1.0f + exp_poly(-g[i])) * u[i]) * dinv[i];
}
static inline int32_t dot64_s8(const int8_t *a, const int8_t *b) {
#if HAVE_NEON
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < 64; i += 16) {
        int8x16_t x = vld1q_s8(a + i), y = vld1q_s8(b + i);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x), vget_low_s8(y))); acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(x), vget_high_s8(y)));
    }
    int32_t t[4]; vst1q_s32(t, acc); return t[0] + t[1] + t[2] + t[3];
#else
    int32_t s = 0; for (int i = 0; i < 64; i++) s += a[i] * b[i]; return s;
#endif
}
// 1 dau attention INT8 (hang q KHONG ngoai lai): score = (q8.k8)*fac ; softmax ; P8 = rint(127*e/emax_u) ; o = (P8.V8)*sp*vscale
//   Cac cot khoa bi danh dau (fpos[0..nf), tang dan, < S): score tinh lai bang FP (qf.kf*scaling), p8 = 0, dong gop V cong bang FP (p=e/sum);
//   sp = emax_u/(127*sum) voi emax_u = max e tren cac cot KHONG bi danh dau (giong ref: sp = pmax/127). nf == 0 -> emax_u = 1.
// K,V: [S][64] int8 lien tiep; sc: float[S+8]; p8: int8[S+8]
static float maxf_n(const float *a, int n) {
    float m = -1e30f; int i = 0;
#if HAVE_NEON
    float32x4_t vm = vdupq_n_f32(-1e30f); for (; i + 4 <= n; i += 4) vm = vmaxq_f32(vm, vld1q_f32(a + i));
    float t[4]; vst1q_f32(t, vm); m = fmaxf(fmaxf(t[0], t[1]), fmaxf(t[2], t[3]));
#endif
    for (; i < n; i++) m = a[i] > m ? a[i] : m; return m;
}
static void attn_head_fast(const int8_t *q, const int8_t *K, const int8_t *V, int S, float fac, const float *vscale, float *o, float *sc, int8_t *p8,
                           const int *fpos, int nf, const float *qf, const float *kf, const float *vf, float scaling) {
    float mx = -1e30f; int s = 0;
#if HAVE_NEON
    int8x16_t q0 = vld1q_s8(q), q1 = vld1q_s8(q + 16), q2 = vld1q_s8(q + 32), q3 = vld1q_s8(q + 48);
    float32x4_t vfac = vdupq_n_f32(fac), vmx = vdupq_n_f32(-1e30f);
    for (; s + 4 <= S; s += 4) {
        int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
        const int8_t *k0 = K + (size_t)s * 64, *k1 = k0 + 64, *k2 = k1 + 64, *k3 = k2 + 64;
#define QKSTEP(A, KP, J, QQ) { int8x16_t kk = vld1q_s8(KP + 16 * J); A = vpadalq_s16(A, vmull_s8(vget_low_s8(QQ), vget_low_s8(kk))); A = vpadalq_s16(A, vmull_s8(vget_high_s8(QQ), vget_high_s8(kk))); }
        QKSTEP(a0, k0, 0, q0) QKSTEP(a1, k1, 0, q0) QKSTEP(a2, k2, 0, q0) QKSTEP(a3, k3, 0, q0)
        QKSTEP(a0, k0, 1, q1) QKSTEP(a1, k1, 1, q1) QKSTEP(a2, k2, 1, q1) QKSTEP(a3, k3, 1, q1)
        QKSTEP(a0, k0, 2, q2) QKSTEP(a1, k1, 2, q2) QKSTEP(a2, k2, 2, q2) QKSTEP(a3, k3, 2, q2)
        QKSTEP(a0, k0, 3, q3) QKSTEP(a1, k1, 3, q3) QKSTEP(a2, k2, 3, q3) QKSTEP(a3, k3, 3, q3)
#undef QKSTEP
        int32x2_t d0 = vadd_s32(vget_low_s32(a0), vget_high_s32(a0)), d1 = vadd_s32(vget_low_s32(a1), vget_high_s32(a1));
        int32x2_t d2 = vadd_s32(vget_low_s32(a2), vget_high_s32(a2)), d3 = vadd_s32(vget_low_s32(a3), vget_high_s32(a3));
        float32x4_t v = vmulq_f32(vcvtq_f32_s32(vcombine_s32(vpadd_s32(d0, d1), vpadd_s32(d2, d3))), vfac);
        vst1q_f32(sc + s, v); vmx = vmaxq_f32(vmx, v);
    }
    { float t[4]; vst1q_f32(t, vmx); mx = fmaxf(fmaxf(t[0], t[1]), fmaxf(t[2], t[3])); }
#endif
    for (; s < S; s++) { float v = (float)dot64_s8(q, K + (size_t)s * 64) * fac; sc[s] = v; mx = v > mx ? v : mx; }
    if (nf) {                                                   // cot bi danh dau: score FP, roi tinh lai max
        for (int i = 0; i < nf; i++) { const float *kv = kf + (size_t)fpos[i] * 64; float acc = 0.f; for (int d = 0; d < 64; d++) acc += qf[d] * kv[d]; sc[fpos[i]] = acc * scaling; }
        mx = maxf_n(sc, S);
    }
    float sum = 0.f; s = 0;
#if HAVE_NEON
    {
        float32x4_t vsum = vdupq_n_f32(0.f), vmxx = vdupq_n_f32(mx), v127 = vdupq_n_f32(127.0f);
        for (; s + 8 <= S; s += 8) {
            float32x4_t e0 = exp4(vsubq_f32(vld1q_f32(sc + s), vmxx)), e1 = exp4(vsubq_f32(vld1q_f32(sc + s + 4), vmxx));
            vsum = vaddq_f32(vsum, vaddq_f32(e0, e1)); vst1q_f32(sc + s, e0); vst1q_f32(sc + s + 4, e1);       // sc <- e
            int16x8_t pp = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(e0, v127))), vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(e1, v127))));
            vst1_s8(p8 + s, vqmovn_s16(pp));
        }
        float t[4]; vst1q_f32(t, vsum); sum = t[0] + t[1] + t[2] + t[3];
    }
#endif
    for (; s < S; s++) { float e = exp_poly(sc[s] - mx); sum += e; sc[s] = e; float v = e * 127.0f; v = (v + 12582912.0f) - 12582912.0f; p8[s] = (int8_t)(int)v; }
    float emax_u = 1.0f, ef[64];
    if (nf) {                                                   // loai cot danh dau khoi P8 (INT8) va chuan hoa theo max cua cac cot con lai
        for (int i = 0; i < nf; i++) { ef[i] = sc[fpos[i]]; sc[fpos[i]] = 0.f; }
        emax_u = maxf_n(sc, S); if (!(emax_u > 1e-30f)) emax_u = 1e-30f;
        const float k127 = 127.0f / emax_u; int t = 0;
#if HAVE_NEON
        float32x4_t vk = vdupq_n_f32(k127);
        for (; t + 8 <= S; t += 8) {
            int16x8_t pp = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(sc + t), vk))), vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(sc + t + 4), vk))));
            vst1_s8(p8 + t, vqmovn_s16(pp));
        }
#endif
        for (; t < S; t++) { float v = sc[t] * k127; v = (v + 12582912.0f) - 12582912.0f; p8[t] = (int8_t)(int)v; }
    }
    if (S & 1) p8[S] = 0;
    const float spv = emax_u / (127.0f * sum);
#if HAVE_NEON
    int32x4_t acc[16]; for (int j = 0; j < 16; j++) acc[j] = vdupq_n_s32(0);
    for (s = 0; s + 1 < S; s += 2) {
        int8x8_t pa = vdup_n_s8(p8[s]), pb = vdup_n_s8(p8[s + 1]); const int8_t *va = V + (size_t)s * 64, *vb = va + 64;
        for (int j = 0; j < 8; j++) {                           // |p|<=127, |v|<=128 -> tong 2 tich < 32767
            int16x8_t t = vmull_s8(vld1_s8(va + 8 * j), pa); t = vmlal_s8(t, vld1_s8(vb + 8 * j), pb);
            acc[2 * j] = vaddw_s16(acc[2 * j], vget_low_s16(t)); acc[2 * j + 1] = vaddw_s16(acc[2 * j + 1], vget_high_s16(t));
        }
    }
    if (s < S) { int8x8_t pa = vdup_n_s8(p8[s]); const int8_t *va = V + (size_t)s * 64;
        for (int j = 0; j < 8; j++) { int16x8_t t = vmull_s8(vld1_s8(va + 8 * j), pa); acc[2 * j] = vaddw_s16(acc[2 * j], vget_low_s16(t)); acc[2 * j + 1] = vaddw_s16(acc[2 * j + 1], vget_high_s16(t)); } }
    { float32x4_t vsp = vdupq_n_f32(spv); for (int j = 0; j < 16; j++) vst1q_f32(o + 4 * j, vmulq_f32(vmulq_f32(vcvtq_f32_s32(acc[j]), vsp), vld1q_f32(vscale + 4 * j))); }
#else
    int32_t tmp[64]; memset(tmp, 0, sizeof tmp);
    for (int s2 = 0; s2 < S; s2++) { const int8_t *v = V + (size_t)s2 * 64; int p = p8[s2]; for (int d = 0; d < 64; d++) tmp[d] += p * v[d]; }
    for (int d = 0; d < 64; d++) o[d] = (float)tmp[d] * spv * vscale[d];
#endif
    for (int i = 0; i < nf; i++) { const float w = ef[i] / sum; const float *vv = vf + (size_t)fpos[i] * 64; for (int d = 0; d < 64; d++) o[d] += w * vv[d]; }
}

// ----------------------------------------------------------------------------- do thoi gian PS theo nhom
enum { P_RMS, P_ROPE, P_ATT, P_SILU, P_RES, P_OUT, P_N };
static const char *P_NAME[P_N] = {"rmsnorm", "rope+ghi cache", "attention", "silu*up", "cong residual", "logits+penalty+argmax"};
static double PRF[P_N]; static long ATT_T[8][2][5], ATT_ENC[8][2][5];   // [decode?][tat ca, NEON khong cot FP, NEON co cot FP, cham do hang q ngoai lai, cham do >64 cot FP]
struct Tic { int k; double t; explicit Tic(int k_) : k(k_), t(now_s()) {} ~Tic() { PRF[k] += now_s() - t; } };

// ----------------------------------------------------------------------------- phep toan float32 (cung quy uoc voi ref_runtime.py, vector hoa NEON bit-exact)
#if HAVE_NEON && defined(__aarch64__)
#define RT_DIV4(a, b) vdivq_f32(a, b)
#elif HAVE_NEON
#define RT_DIV4(a, b) _mm_div_ps(a, b)                  /* NEON_2_SSE: float32x4_t = __m128 (chi de TEST tren x86) */
#endif

static void rms_norm(const float *x, const float *w, float *out, int n, float eps) {
    float ss = 0.f; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float var = ss / (float)n; float r = 1.0f / sqrtf(var + eps);
    int i = 0;
#if HAVE_NEON
    float32x4_t vr = vdupq_n_f32(r);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(out + i,     vmulq_f32(vld1q_f32(w + i),     vmulq_f32(vld1q_f32(x + i),     vr)));
        vst1q_f32(out + i + 4, vmulq_f32(vld1q_f32(w + i + 4), vmulq_f32(vld1q_f32(x + i + 4), vr)));
    }
#endif
    for (; i < n; i++) out[i] = w[i] * (x[i] * r);
}
static inline void add_inplace_range(float *dst, const float *src, int i0, int i1) {
    int i = i0;
#if HAVE_NEON
    for (; i + 8 <= i1; i += 8) {
        vst1q_f32(dst + i,     vaddq_f32(vld1q_f32(dst + i),     vld1q_f32(src + i)));
        vst1q_f32(dst + i + 4, vaddq_f32(vld1q_f32(dst + i + 4), vld1q_f32(src + i + 4)));
    }
#endif
    for (; i < i1; i++) dst[i] = dst[i] + src[i];
}
static inline void add_inplace_f32(float *dst, const float *src, int n) {
    if (POOL.n > 1 && n >= 16384) POOL.split(n, 1024, [&](int i0, int i1, int) { add_inplace_range(dst, src, i0, i1); });
    else add_inplace_range(dst, src, 0, n);
}
static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }

// ----------------------------------------------------------------------------- vong PS cua 1 GEMM (NEON, cung ket qua voi vong vo huong)
static float maxabs_f32(const float *x, int K) {
    float mx = 0.f; int k = 0;
#if HAVE_NEON
    float32x4_t vm = vdupq_n_f32(0.f); for (; k + 4 <= K; k += 4) vm = vmaxq_f32(vm, vabsq_f32(vld1q_f32(x + k)));
    float t[4]; vst1q_f32(t, vm); mx = fmaxf(fmaxf(t[0], t[1]), fmaxf(t[2], t[3]));
#endif
    for (; k < K; k++) mx = fmaxf(mx, fabsf(x[k])); return mx;
}
// a[k] = clip(rint(x[k] / s), -128, 127)  (chia that, khong nhan nghich dao -> cung ket qua voi nearbyintf(x/s))
static void quant_row_s8(const float *x, int8_t *a, int K, float s) {
    int k = 0;
#if HAVE_NEON
    float32x4_t vs = vdupq_n_f32(s);
    for (; k + 8 <= K; k += 8) {
        int32x4_t r0 = vcvtnq_s32_f32(RT_DIV4(vld1q_f32(x + k), vs)), r1 = vcvtnq_s32_f32(RT_DIV4(vld1q_f32(x + k + 4), vs));
        vst1_s8(a + k, vqmovn_s16(vcombine_s16(vqmovn_s32(r0), vqmovn_s32(r1))));
    }
#endif
    for (; k < K; k++) { float r = nearbyintf(x[k] / s); r = r > 127.f ? 127.f : (r < -128.f ? -128.f : r); a[k] = (int8_t)r; }
}
// yr[n] = clip(((ac+bi)*mult + 2^(sh-1)) >> sh + zp, -128, 127) * sout cho n in [n0,n1); tra true neu co n voi |ac+bi| > tmax[n] (bao hoa)
static bool requant_range(const Gemm &g, const int32_t *ac, float *yr, int n0, int n1) {
    int n = n0; bool sat = false;
#if HAVE_NEON
    if (g.neon_rq) {
        uint32x4_t vs = vdupq_n_u32(0); const int32x4_t lo = vdupq_n_s32(-128), hi = vdupq_n_s32(127);
        for (; n + 4 <= n1; n += 4) {
            if ((n & 15) == 0) { __builtin_prefetch(g.bi.data() + n + 128); __builtin_prefetch(g.mult.data() + n + 128); __builtin_prefetch(g.tmax.data() + n + 128); __builtin_prefetch(g.sout + n + 128); __builtin_prefetch(g.nsh8.data() + n + 128); __builtin_prefetch(g.zp16.data() + n + 128); __builtin_prefetch(ac + n + 128); }
            int32x4_t t = vaddq_s32(vld1q_s32(ac + n), vld1q_s32(g.bi.data() + n)), mu = vld1q_s32(g.mult.data() + n);
            int64x2_t p0 = vmull_s32(vget_low_s32(t), vget_low_s32(mu)), p1 = vmull_s32(vget_high_s32(t), vget_high_s32(mu));
            int32x4_t sh32 = vmovl_s16(vget_low_s16(vmovl_s8(vld1_s8(g.nsh8.data() + n)))), zp32 = vmovl_s16(vld1_s16(g.zp16.data() + n));      // -shift, zp -> int32
            p0 = vaddq_s64(vrshlq_s64(p0, vmovl_s32(vget_low_s32(sh32))), vmovl_s32(vget_low_s32(zp32)));                                         // rounding shift right
            p1 = vaddq_s64(vrshlq_s64(p1, vmovl_s32(vget_high_s32(sh32))), vmovl_s32(vget_high_s32(zp32)));
            int32x4_t v = vminq_s32(vmaxq_s32(vcombine_s32(vqmovn_s64(p0), vqmovn_s64(p1)), lo), hi);
            vst1q_f32(yr + n, vmulq_f32(vcvtq_f32_s32(v), vld1q_f32(g.sout + n)));
            vs = vorrq_u32(vs, vcgtq_s32(vabsq_s32(t), vld1q_s32(g.tmax.data() + n)));
        }
        uint32_t tt[4]; vst1q_u32(tt, vs); sat = (tt[0] | tt[1] | tt[2] | tt[3]) != 0;
    }
#endif
    for (; n < n1; n++) {
        int64_t t = (int64_t)ac[n] + g.bi[n]; int64_t rnd = g.shift[n] > 0 ? ((int64_t)1 << (g.shift[n] - 1)) : 0;
        int64_t v = ((t * g.mult[n] + rnd) >> g.shift[n]) + g.zp[n]; v = v < -128 ? -128 : (v > 127 ? 127 : v);
        yr[n] = (float)v * g.sout[n]; if ((t < 0 ? -t : t) > (int64_t)g.tmax[n]) sat = true;
    }
    return sat;
}

// ----------------------------------------------------------------------------- 1 GEMM co fallback (hang ngoai lai) bang chu so int8 tren PL
struct Runner {
    Pkg &P; Backend &be; int digits = 3;
    Runner(Pkg &p, Backend &b) : P(p), be(b) {}
    std::vector<int8_t> A; std::vector<int32_t> C; std::vector<float> tmp;
    std::vector<uint8_t> local_flags; std::vector<char> infl, satrow; std::vector<int> nrm, dig, drows; std::vector<double> sc_dig;

    // x [M][K] float -> y [M][N] float ; flags [n_members][M] (uint8) neu != nullptr
    void gemm(Gemm &g, const float *x, int M, float *y, std::vector<uint8_t> *flags) {
        const int K = g.K, N = g.N; const int nm = (int)g.mem.size(); if (nm > 8) die("qua nhieu thanh vien");
        if (!flags) flags = &local_flags;
        flags->assign((size_t)nm * M, 0);
        infl.assign(M, 0); nrm.clear(); dig.clear();
        double t_pre = now_s();
        for (int m = 0; m < M; m++) {
            if (g.rowfb) infl[m] = maxabs_f32(x + (size_t)m * K, K) > g.thr_in;
            (infl[m] ? dig : nrm).push_back(m);
        }
        satrow.assign(M, 0);
        double t0;
        // luot A: cac hang thuong
        if (!nrm.empty()) {
            const int nn = (int)nrm.size(); A.resize((size_t)nn * K); C.resize((size_t)nn * N);
            auto qrows = [&](int i0, int i1, int) { for (int i = i0; i < i1; i++) quant_row_s8(x + (size_t)nrm[i] * K, A.data() + (size_t)i * K, K, g.sin_f); };
            if (POOL.n > 1 && nn >= 4 * POOL.n) POOL.split(nn, 1, qrows); else qrows(0, nn, 0);
            ST.ps_pre += now_s() - t_pre; t0 = now_s();
            be.gemm32(g, A.data(), nn, C.data());
            ST.pl_wall += now_s() - t0; t0 = now_s();
            auto post = [&](int i, int n0, int n1, bool *sat) {                       // 1 hang, cot [n0,n1)
                const int m = nrm[i]; const int32_t *ac = C.data() + (size_t)i * N; float *yr = y + (size_t)m * N;
                if (g.int32) {
                    int n = n0;
#if defined(__aarch64__) && HAVE_NEON
                    for (; n + 4 <= n1; n += 4) {
                        int32x4_t vi = vld1q_s32(ac + n);
                        float64x2_t d0 = vmulq_f64(vcvtq_f64_s64(vmovl_s32(vget_low_s32(vi))), vld1q_f64(g.sinsw.data() + n));
                        float64x2_t d1 = vmulq_f64(vcvtq_f64_s64(vmovl_s32(vget_high_s32(vi))), vld1q_f64(g.sinsw.data() + n + 2));
                        vst1q_f32(yr + n, vcombine_f32(vcvt_f32_f64(d0), vcvt_f32_f64(d1)));
                    }
#endif
                    for (; n < n1; n++) yr[n] = (float)((double)ac[n] * g.sinsw[n]);
                    return;
                }
                if (!g.rowfb) { requant_range(g, ac, yr, n0, n1); return; }
                for (int mi = 0; mi < nm; mi++) { int c0 = std::max(n0, g.mem[mi].c0), c1 = std::min(n1, g.mem[mi].c1); if (c0 < c1 && requant_range(g, ac, yr, c0, c1)) sat[mi] = true; }
            };
            auto apply = [&](int i, const bool *sat) { const int m = nrm[i]; for (int mi = 0; mi < nm; mi++) if (sat[mi]) { (*flags)[(size_t)mi * M + m] = 1; satrow[m] = 1; } };
            if (!g.int32 && g.rowfb && !g.tiled) {                                    // thanh vien khong lat kin cot (hiem): requant ca hang roi kiem tra bao hoa
                for (int i = 0; i < nn; i++) { const int m = nrm[i]; const int32_t *ac = C.data() + (size_t)i * N; requant_range(g, ac, y + (size_t)m * N, 0, N);
                    for (int mi = 0; mi < nm; mi++) { bool sat = false; for (int n = g.mem[mi].c0; n < g.mem[mi].c1 && !sat; n++) { int64_t t = (int64_t)ac[n] + g.bi[n]; sat = (t < 0 ? -t : t) > (int64_t)g.tmax[n]; }
                        if (sat) { (*flags)[(size_t)mi * M + m] = 1; satrow[m] = 1; } } }
            } else if (POOL.n > 1 && nn >= 2 * POOL.n) {                              // nhieu hang (prefill): chia theo hang
                POOL.split(nn, 1, [&](int i0, int i1, int) { for (int i = i0; i < i1; i++) { bool sat[8] = {}; post(i, 0, N, sat); apply(i, sat); } });
            } else if (POOL.n > 1 && N >= 4096) {                                      // it hang, N lon (decode): chia theo cot
                for (int i = 0; i < nn; i++) { bool satT[8][8] = {}; POOL.split(N, 64, [&](int n0, int n1, int tid) { post(i, n0, n1, satT[tid]); });
                    bool sat[8] = {}; for (int tt = 0; tt < POOL.n; tt++) for (int mi = 0; mi < nm; mi++) sat[mi] = sat[mi] || satT[tt][mi]; apply(i, sat); }
            } else { for (int i = 0; i < nn; i++) { bool sat[8] = {}; post(i, 0, N, sat); apply(i, sat); } }
            (g.int32 ? ST.ps_i32 : ST.ps_rq) += now_s() - t0;
        }
        // luot B: hang ngoai lai (infl: ca hang; sat: theo thanh vien) bang `digits` chu so int8 -> INT32
        drows = dig; for (int m = 0; m < M; m++) if (satrow[m]) drows.push_back(m);
        if (!drows.empty()) {
            double td = now_s(); int nd = (int)drows.size(); const int D = digits;
            A.resize((size_t)nd * D * K); C.resize((size_t)nd * D * N);
            sc_dig.resize((size_t)nd * D);
            auto prep_dig = [&](int i0, int i1, int) {
                std::vector<double> r(K);
                for (int i = i0; i < i1; i++) {
                    const float *xr = x + (size_t)drows[i] * K; double amax = 0; for (int k = 0; k < K; k++) amax = fmax(amax, fabs((double)xr[k]));
                    double s = amax > 0 ? amax / 127.0 : 1.0; for (int k = 0; k < K; k++) r[k] = (double)xr[k];
                    for (int dd = 0; dd < D; dd++) {
                        int8_t *ar = A.data() + ((size_t)i * D + dd) * K; sc_dig[(size_t)i * D + dd] = s;
                        for (int k = 0; k < K; k++) { double q = nearbyint(r[k] / s); q = q > 127 ? 127 : (q < -127 ? -127 : q); ar[k] = (int8_t)q; r[k] -= q * s; }
                        s /= 254.0;
                    }
                }
            };
            if (POOL.n > 1 && nd >= 2 * POOL.n) POOL.split(nd, 1, prep_dig); else prep_dig(0, nd, 0);
            t0 = now_s(); ST.ps_dig += t0 - td;
            be.gemm32(g, A.data(), nd * D, C.data());
            ST.pl_wall += now_s() - t0; td = now_s();
            for (int i = 0; i < nd; i++) {
                int m = drows[i]; bool full = infl[m]; float *yr = y + (size_t)m * N;
                auto col_range = [&](int c0, int c1) {
                    int n = c0;
#if defined(__aarch64__) && HAVE_NEON
                    if (D == 3) {
                        const int32_t *c_0 = &C[((size_t)i * 3 + 0) * N], *c_1 = &C[((size_t)i * 3 + 1) * N], *c_2 = &C[((size_t)i * 3 + 2) * N];
                        float64x2_t s0 = vdupq_n_f64(sc_dig[(size_t)i * 3 + 0]), s1 = vdupq_n_f64(sc_dig[(size_t)i * 3 + 1]), s2 = vdupq_n_f64(sc_dig[(size_t)i * 3 + 2]);
                        for (; n + 2 <= c1; n += 2) {
                            float64x2_t acc = vmulq_f64(vcvtq_f64_s64(vmovl_s32(vld1_s32(c_0 + n))), s0);
                            acc = vaddq_f64(acc, vmulq_f64(vcvtq_f64_s64(vmovl_s32(vld1_s32(c_1 + n))), s1));
                            acc = vaddq_f64(acc, vmulq_f64(vcvtq_f64_s64(vmovl_s32(vld1_s32(c_2 + n))), s2));
                            float64x2_t swd = vcvt_f64_f32(vld1_f32(g.sw + n)), bid = vcvt_f64_f32(vld1_f32(g.bias + n));
                            vst1_f32(yr + n, vcvt_f32_f64(vaddq_f64(vmulq_f64(acc, swd), bid)));
                        }
                    }
#endif
                    for (; n < c1; n++) {
                        double acc = 0; for (int dd = 0; dd < D; dd++) acc += (double)C[((size_t)i * D + dd) * N + n] * sc_dig[(size_t)i * D + dd];
                        yr[n] = (float)(acc * (double)g.sw[n] + (double)g.bias[n]);
                    }
                };
                if (full) { col_range(0, N); ST.rows_dig++; for (int mi = 0; mi < nm; mi++) (*flags)[(size_t)mi * M + m] = 1; }
                else { for (int mi = 0; mi < nm; mi++) if ((*flags)[(size_t)mi * M + m]) col_range(g.mem[mi].c0, g.mem[mi].c1); ST.rows_sat++; }
            }
            ST.ps_dig += now_s() - td;
        }
        ST.calls++;
    }
};

// 1 dau attention INT8 cho 1 hang q (dung chung cho LLM: nhan qua + cache, va encoder: khong nhan qua). Tat ca con tro K8/V8/KF/VF/kcol tro vao dau (head) tuong ung, khoa 0..lim.
//   fl: cac vi tri khoa bi danh dau cua dau do (tang dan). sc, p: bo dem cho nhanh cham (>= lim+1 phan tu); q8: d int8.
static void attend_head(const float *qv, int d, const Attn &a, float scaling, bool pf_q, int lim, const int8_t *K8, const int8_t *V8, const float *KF, const float *VF, const uint8_t *kcol,
                        const std::vector<int> &fl, const float *vs, float *out, bool fast, float *scb, int8_t *p8b, std::vector<float> &sc, std::vector<float> &p, std::vector<int8_t> &q8,
                        long (*AT)[5], int dec) {
    float mxq = 0.f;
    for (int i = 0; i < d; i++) { mxq = fmaxf(mxq, fabsf(qv[i])); float r = nearbyintf(qv[i] / a.sq); q8[i] = (int8_t)(r > 127.f ? 127.f : (r < -128.f ? -128.f : r)); }
    static const bool dbg_noq = getenv("RT_DBG_NOQROW") != nullptr;                      // chi de TEST: bo co hang q ngoai lai de so duong NEON voi duong vo huong tren cung du lieu
    const bool qrow = !dbg_noq && ((mxq > (float)(127.5 * a.sq)) || pf_q);
    AT[dec][0]++;
    const int nfl = (int)(std::upper_bound(fl.begin(), fl.end(), lim) - fl.begin());
    if (fast && !qrow && nfl <= 64 && nfl < lim + 1) { AT[dec][nfl ? 2 : 1]++;                // duong NEON (co the kem vai cot FP)
        attn_head_fast(q8.data(), K8, V8, lim + 1, a.fac, vs, out, scb, p8b, fl.data(), nfl, qv, KF, VF, scaling);
        return;
    }
    if (fast) AT[dec][qrow ? 3 : 4]++;
    float m = -INFINITY;
    for (int s = 0; s <= lim; s++) {
        const bool fp = qrow || kcol[s];
        float v;
        if (!fp) { int acc = 0; const int8_t *k8 = &K8[(size_t)s * d]; for (int i = 0; i < d; i++) acc += (int)q8[i] * k8[i]; v = (float)acc * a.fac; }
        else { float acc = 0.f; const float *kv = &KF[(size_t)s * d]; for (int i = 0; i < d; i++) acc += qv[i] * kv[i]; v = acc * scaling; }
        sc[s] = v; m = fmaxf(m, v);
    }
    float sum = 0.f; for (int s = 0; s <= lim; s++) { p[s] = expf(sc[s] - m); sum += p[s]; }
    float pmax = 0.f; for (int s = 0; s <= lim; s++) { p[s] = p[s] / sum; if (!kcol[s]) pmax = fmaxf(pmax, p[s]); }
    float sp = fmaxf(pmax / 127.0f, 1e-30f);
    std::vector<double> accd(d, 0.0); std::vector<float> fpacc(d, 0.f); bool anyfp = false;
    for (int s = 0; s <= lim; s++) {
        if (kcol[s]) { anyfp = true; const float *vv = &VF[(size_t)s * d]; for (int i = 0; i < d; i++) fpacc[i] += p[s] * vv[i]; continue; }
        float p8 = nearbyintf(p[s] / sp); const int8_t *v8 = &V8[(size_t)s * d];
        for (int i = 0; i < d; i++) accd[i] += (double)(p8 * (float)v8[i]);
    }
    for (int i = 0; i < d; i++) { double r = accd[i] * (double)sp * (double)vs[i]; if (anyfp) r += (double)fpacc[i]; out[i] = (float)r; }
}

// ----------------------------------------------------------------------------- mo hinh LLM
struct Model {
    Pkg &P; Runner &R; int maxlen = 0, n = 0;
    std::vector<std::vector<float>> kc, vc;            // [layer][Hkv][maxlen][d] (K sau RoPE)
    std::vector<std::vector<int8_t>> k8c, v8c;
    std::vector<std::vector<uint8_t>> kcol;            // [layer][Hkv][maxlen]
    std::vector<std::vector<uint8_t>> pfc;             // [layer][maxlen] hang ngoai lai (bat ky thanh vien)
    std::vector<std::vector<std::vector<int>>> flist;  // [layer][Hkv] cac vi tri khoa bi danh dau (kcol), tang dan
    std::vector<std::vector<float>> sc_bufs; std::vector<std::vector<int8_t>> p8_bufs; bool fast = true;
    // bo dem tai su dung giua cac buoc (loai bo malloc/free moi buoc decode/prefill)
    std::vector<float> h, qkv, o, gu, act, tmp, qh, cs, sn, inv, h_lg;
    std::vector<uint8_t> fl, pf;
    FILE *trace_fp = nullptr;                          // RT_TRACE=file: ghi x sau moi lop (debug, chi so voi ref_runtime)
    Model(Pkg &p, Runner &r) : P(p), R(r) {
        const char *t = getenv("RT_TRACE"); if (t) trace_fp = fopen(t, "wb");
        const int h2 = P.d / 2; inv.resize(h2);
        for (int i = 0; i < h2; i++) inv[i] = 1.0f / powf((float)P.theta, (float)(2 * i) / (float)P.d);
        h_lg.resize(P.hid);
    }
    void new_cache(int ml) {
        maxlen = ml; n = 0; int L = P.layers;
        kc.assign(L, std::vector<float>((size_t)P.Hkv * ml * P.d)); vc = kc;
        k8c.assign(L, std::vector<int8_t>((size_t)P.Hkv * ml * P.d)); v8c = k8c;
        kcol.assign(L, std::vector<uint8_t>((size_t)P.Hkv * ml)); pfc.assign(L, std::vector<uint8_t>(ml));
        flist.assign(L, std::vector<std::vector<int>>(P.Hkv)); sc_bufs.assign(POOL.n, std::vector<float>(ml + 16, 0.f)); p8_bufs.assign(POOL.n, std::vector<int8_t>(ml + 16, 0)); fast = (P.d == 64) && !getenv("RT_NOFAST");
    }
    // attention INT8 cho T hang moi (vi tri s0..s0+T-1) cua lop j; q [T][H][d] da RoPE; tra o [T][H*d]
    void attention(int j, const float *q, int T, int s0, const std::vector<uint8_t> &pfq, const float *vs, float *o) {
        const int H = P.H, Hkv = P.Hkv, d = P.d, rep = H / Hkv, S = s0 + T; const Attn &a = P.att[j];
        auto work = [&](int it0, int it1, int tid) {
            std::vector<float> sc(S), p(S); std::vector<int8_t> q8(d); float *scb = sc_bufs[tid].data(); int8_t *p8b = p8_bufs[tid].data(); long (*AT)[5] = ATT_T[tid];
            const int dec = (T == 1) ? 1 : 0;
            for (int it = it0; it < it1; it++) {      // duyet theo dau (cung kv-head lien tiep) de K/V con trong cache
                const int h = it / T, t = it % T, kvh = h / rep; const size_t hb = (size_t)kvh * maxlen * d;
                const int lim = (T > 1) ? (s0 + t) : (S - 1);            // khoa cuoi duoc phep (mat na nhan qua)
                attend_head(q + ((size_t)t * H + h) * d, d, a, a.scaling, pfq[t] != 0, lim, &k8c[j][hb], &v8c[j][hb], &kc[j][hb], &vc[j][hb], &kcol[j][(size_t)kvh * maxlen], flist[j][kvh],
                            vs + kvh * d, o + ((size_t)t * H + h) * d, fast, scb, p8b, sc, p, q8, AT, dec);
            }
        };
        if (POOL.n > 1 && T * H >= 8) POOL.split(T * H, 1, work); else work(0, T * H, 0);
    }
    // T hang moi noi tiep cache; x [T][hid] (cap nhat tai cho)
    void rows(float *x, int T) {
        const int hid = P.hid, H = P.H, Hkv = P.Hkv, d = P.d, inter = P.inter, s0 = n, h2 = d / 2;
        const int qn = H * d, kn = Hkv * d, tot = qn + 2 * kn;
        h.resize((size_t)T * hid); qkv.resize((size_t)T * tot); o.resize((size_t)T * hid);
        gu.resize((size_t)T * 2 * inter); act.resize((size_t)T * inter); tmp.resize((size_t)T * hid);
        qh.resize((size_t)T * qn); cs.resize((size_t)T * d); sn.resize((size_t)T * d); pf.resize(T);
        for (int t = 0; t < T; t++) for (int i = 0; i < h2; i++) {
            float fr = (float)(s0 + t) * inv[i];
            cs[(size_t)t * d + i] = cs[(size_t)t * d + i + h2] = cosf(fr);
            sn[(size_t)t * d + i] = sn[(size_t)t * d + i + h2] = sinf(fr);
        }
        for (int j = 0; j < P.layers; j++) {
            { Tic tc(P_RMS);
              if (POOL.n > 1 && T >= 4) POOL.split(T, 1, [&](int t0, int t1, int) { for (int t = t0; t < t1; t++) rms_norm(x + (size_t)t * hid, P.in_ln[j], h.data() + (size_t)t * hid, hid, (float)P.rms_eps); });
              else for (int t = 0; t < T; t++) rms_norm(x + (size_t)t * hid, P.in_ln[j], h.data() + (size_t)t * hid, hid, (float)P.rms_eps); }
            R.gemm(P.G(j, 0), h.data(), T, qkv.data(), &fl);
            for (int t = 0; t < T; t++) pf[t] = fl[t] | fl[(size_t)T + t] | fl[(size_t)2 * T + t];
            const float *vs = P.G(j, 0).sout + qn + kn;
            double t_rope = now_s();
            auto rope_row = [&](int t) {
                float *row = qkv.data() + (size_t)t * tot; const float *c = &cs[(size_t)t * d], *s = &sn[(size_t)t * d];
                auto apply = [&](float *v) {
                    int i = 0;
#if HAVE_NEON
                    for (; i + 4 <= h2; i += 4) {
                        const float32x4_t v0 = vld1q_f32(v + i), v1 = vld1q_f32(v + i + h2);
                        const float32x4_t c0 = vld1q_f32(c + i), c1 = vld1q_f32(c + i + h2);
                        const float32x4_t s0_v = vld1q_f32(s + i), s1_v = vld1q_f32(s + i + h2);
                        const float32x4_t r0 = vnegq_f32(v1), r1 = v0;
                        vst1q_f32(v + i, vaddq_f32(vmulq_f32(v0, c0), vmulq_f32(r0, s0_v)));
                        vst1q_f32(v + i + h2, vaddq_f32(vmulq_f32(v1, c1), vmulq_f32(r1, s1_v)));
                    }
#endif
                    if (i < h2) {
                        float r[256];
                        for (int k = i; k < h2; k++) { r[k] = -v[k + h2]; r[k + h2] = v[k]; }
                        for (int k = i; k < h2; k++) { v[k] = v[k] * c[k] + r[k] * s[k]; v[k + h2] = v[k + h2] * c[k + h2] + r[k + h2] * s[k + h2]; }
                    }
                };
                for (int hh = 0; hh < H; hh++) apply(row + (size_t)hh * d);
                for (int hh = 0; hh < Hkv; hh++) apply(row + qn + (size_t)hh * d);
            };
            if (POOL.n > 1 && T >= 4) POOL.split(T, 1, [&](int t0, int t1, int) { for (int t = t0; t < t1; t++) rope_row(t); });
            else for (int t = 0; t < T; t++) rope_row(t);
            // ghi cache (k sau RoPE, v), luong tu k/v va co ngoai lai
            const Attn &a = P.att[j]; const float thr_k = (float)(127.5 * a.sk);
            auto kv_quant_row = [&](int t) {
                for (int kv = 0; kv < Hkv; kv++) {
                    const float *kr = qkv.data() + (size_t)t * tot + qn + (size_t)kv * d, *vr = qkv.data() + (size_t)t * tot + qn + kn + (size_t)kv * d;
                    size_t base = ((size_t)kv * maxlen + s0 + t) * d;
                    bool col = false; float mk = 0.f;
                    for (int i = 0; i < d; i++) {
                        mk = fmaxf(mk, fabsf(kr[i])); if (fabsf(vr[i]) > 127.5f * vs[kv * d + i]) col = true;
                        float r = nearbyintf(kr[i] / a.sk); k8c[j][base + i] = (int8_t)(r > 127.f ? 127.f : (r < -128.f ? -128.f : r));
                        float r2 = nearbyintf(vr[i] / vs[kv * d + i]); v8c[j][base + i] = (int8_t)(r2 > 127.f ? 127.f : (r2 < -128.f ? -128.f : r2));
                        kc[j][base + i] = kr[i]; vc[j][base + i] = vr[i];
                    }
                    if (mk > thr_k) col = true; if (pf[t]) col = true;
                    kcol[j][(size_t)kv * maxlen + s0 + t] = col;
                }
            };
            if (POOL.n > 1 && T >= 4) POOL.split(T, 1, [&](int t0, int t1, int) { for (int t = t0; t < t1; t++) kv_quant_row(t); });
            else for (int t = 0; t < T; t++) kv_quant_row(t);
            for (int t = 0; t < T; t++) {
                pfc[j][s0 + t] = pf[t];
                for (int kv = 0; kv < Hkv; kv++) if (kcol[j][(size_t)kv * maxlen + s0 + t]) flist[j][kv].push_back(s0 + t);
            }
            PRF[P_ROPE] += now_s() - t_rope;
            for (int t = 0; t < T; t++) memcpy(&qh[(size_t)t * qn], qkv.data() + (size_t)t * tot, sizeof(float) * qn);
            { Tic tc(P_ATT); attention(j, qh.data(), T, s0, pf, vs, o.data()); }
            R.gemm(P.G(j, 1), o.data(), T, tmp.data(), nullptr);
            { Tic tc(P_RES); add_inplace_f32(x, tmp.data(), (size_t)T * hid); }
            { Tic tc(P_RMS);
              if (POOL.n > 1 && T >= 4) POOL.split(T, 1, [&](int t0, int t1, int) { for (int t = t0; t < t1; t++) rms_norm(x + (size_t)t * hid, P.post_ln[j], h.data() + (size_t)t * hid, hid, (float)P.rms_eps); });
              else for (int t = 0; t < T; t++) rms_norm(x + (size_t)t * hid, P.post_ln[j], h.data() + (size_t)t * hid, hid, (float)P.rms_eps); }
            R.gemm(P.G(j, 2), h.data(), T, gu.data(), nullptr);
            { Tic tc(P_SILU);
              auto sf = [&](int t, int c0, int c1) { silu_mul_dinv(&gu[(size_t)t * 2 * inter + c0], &gu[(size_t)t * 2 * inter + inter + c0], P.down_inv[j] + c0, &act[(size_t)t * inter + c0], c1 - c0); };
              if (POOL.n > 1 && T == 1) POOL.split(inter, 64, [&](int c0, int c1, int) { sf(0, c0, c1); });
              else if (POOL.n > 1 && T >= 2 * POOL.n) POOL.split(T, 1, [&](int t0_, int t1_, int) { for (int t = t0_; t < t1_; t++) sf(t, 0, inter); });
              else for (int t = 0; t < T; t++) sf(t, 0, inter); }
            R.gemm(P.G(j, 3), act.data(), T, tmp.data(), nullptr);
            { Tic tc(P_RES); add_inplace_f32(x, tmp.data(), (size_t)T * hid); }
            if (trace_fp) fwrite(x, 4, (size_t)T * hid, trace_fp);
        }
        n = s0 + T;
    }
    // logits cua 1 hang x[hid] (dau ra lop cuoi): norm cuoi + lm_head INT32
    void logits(const float *x, float *lg) {
        rms_norm(x, P.final_norm, h_lg.data(), P.hid, (float)P.rms_eps);
        R.gemm(P.LM(), h_lg.data(), 1, lg, nullptr);
    }
};


// ----------------------------------------------------------------------------- WAV -> log-mel -> encoder -> projector (dung chung Runner: GEMM tren PL, phan con lai tren PS)
static std::vector<float> read_wav16(const std::string &path) {
    auto b = read_all(path); if (b.size() < 44 || memcmp(b.data(), "RIFF", 4) || memcmp(b.data() + 8, "WAVE", 4)) die("%s khong phai file WAV (RIFF/WAVE)", path.c_str());
    auto u16 = [&](size_t o) { return (int)(b[o] | (b[o + 1] << 8)); };
    auto u32 = [&](size_t o) { return (uint32_t)(b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | ((uint32_t)b[o + 3] << 24)); };
    int fmt = 0, ch = 1, bits = 0; uint32_t sr = 16000; size_t dpos = 0, dlen = 0;
    for (size_t o = 12; o + 8 <= b.size();) {
        uint32_t len = u32(o + 4);
        if (!memcmp(&b[o], "fmt ", 4)) { fmt = u16(o + 8); ch = u16(o + 10); sr = u32(o + 12); bits = u16(o + 22); }
        else if (!memcmp(&b[o], "data", 4)) { dpos = o + 8; dlen = std::min<size_t>(len, b.size() - dpos); break; }
        o += 8 + (size_t)len + (len & 1);
    }
    if (!dpos || ch < 1) die("WAV %s khong co chunk fmt/data hop le", path.c_str());
    const bool pcm16 = (fmt == 1 || fmt == 0xFFFE) && bits == 16, f32 = fmt == 3 && bits == 32;
    if (!pcm16 && !f32) die("WAV %s: chi ho tro PCM 16-bit hoac float32 (fmt=%s)", path.c_str(), (std::to_string(fmt) + "/" + std::to_string(bits) + "bit").c_str());
    const size_t bps = bits / 8, nfr = dlen / (bps * ch); std::vector<float> w(nfr);
    for (size_t i = 0; i < nfr; i++) { float acc = 0.f; for (int c = 0; c < ch; c++) { size_t o = dpos + (i * ch + c) * bps; float v; if (pcm16) v = (float)(int16_t)(b[o] | (b[o + 1] << 8)) / 32768.0f; else { uint32_t u = u32(o); memcpy(&v, &u, 4); } acc += v; } w[i] = ch > 1 ? acc / (float)ch : acc; }
    if (sr != 16000 && nfr > 1) {                                                         // noi suy tuyen tinh nhu ref_runtime.read_wav (np.interp)
        const size_t n = (size_t)llround((double)nfr * 16000.0 / (double)sr); std::vector<float> r(n);
        for (size_t i = 0; i < n; i++) { double x = n > 1 ? (double)i * (double)(nfr - 1) / (double)(n - 1) : 0.0; size_t i0 = (size_t)x; if (i0 >= nfr - 1) i0 = nfr - 2; double f = x - (double)i0; r[i] = (float)((double)w[i0] * (1.0 - f) + (double)w[i0 + 1] * f); }
        fprintf(stderr, "[wav] %u Hz -> 16000 Hz (noi suy tuyen tinh)\n", sr); return r;
    }
    return w;
}
// log-mel nhu ref_runtime.log_mel / HF WhisperFeatureExtractor: chi tinh cac khung can dung (L+5 khung dau), cac khung sau vung hop le khong anh huong max va khong duoc encoder dung.
// feats: [n_mels][ncols] (hang = kenh mel); L = ceil(n/hop) (so khung hop le)
static void log_mel(const Pkg &P, const std::vector<float> &wave, std::vector<float> &feats, int &ncols, int &Lout) {
    const int n_fft = P.n_fft, hop = P.hop, ns = P.n_samples, nm = P.n_mels, nf = P.n_freq, pad = n_fft / 2;
    const int n = (int)std::min<size_t>(wave.size(), (size_t)ns); const int L = (n + hop - 1) / hop;
    const int nfr_out = (1 + (ns + 2 * pad - n_fft) / hop) - 1; const int need = std::min(nfr_out, L + 5);
    auto samp = [&](long i) -> double { long j = i - pad; if (j < 0) j = -j; else if (j >= ns) j = 2L * (ns - 1) - j; return (j >= 0 && j < n) ? (double)wave[j] : 0.0; };
    std::vector<double> cs(n_fft), sn(n_fft), win(n_fft); for (int m = 0; m < n_fft; m++) { cs[m] = cos(2.0 * M_PI * m / n_fft); sn[m] = sin(2.0 * M_PI * m / n_fft); win[m] = (double)P.hann[m]; }
    // Bang cos/sin lien tuc [nf][n_fft]: loai bo 55.7 trieu phep nhanh if(m>=n_fft) trong STFT, bit-exact 100%
    std::vector<double> cs_tbl((size_t)nf * n_fft), sn_tbl((size_t)nf * n_fft);
    for (int k = 0; k < nf; k++) {
        int m = 0; double *ck = &cs_tbl[(size_t)k * n_fft], *sk = &sn_tbl[(size_t)k * n_fft];
        for (int i = 0; i < n_fft; i++) { ck[i] = cs[m]; sk[i] = sn[m]; m += k; if (m >= n_fft) m -= n_fft; }
    }
    std::vector<double> mag((size_t)need * nf);
    POOL.split(need, 4, [&](int f0, int f1, int) {
        std::vector<double> x(n_fft);
        for (int f = f0; f < f1; f++) {
            for (int i = 0; i < n_fft; i++) x[i] = samp((long)f * hop + i) * win[i];
            for (int k = 0; k < nf; k++) {
                const double *ck = &cs_tbl[(size_t)k * n_fft], *sk = &sn_tbl[(size_t)k * n_fft];
                double re = 0, im = 0;
                for (int i = 0; i < n_fft; i++) { re += x[i] * ck[i]; im -= x[i] * sk[i]; }
                mag[(size_t)f * nf + k] = re * re + im * im;
            }
        }
    });
    // Chuyen vi mel_filt sang [nm][nf] lien tuc va chia luong qua nm (bit-exact 100%)
    std::vector<double> mf_t((size_t)nm * nf);
    for (int m = 0; m < nm; m++) for (int k = 0; k < nf; k++) mf_t[(size_t)m * nf + k] = (double)P.mel_filt[(size_t)k * nm + m];
    std::vector<double> ls((size_t)nm * need);
    std::vector<double> tmax(POOL.n, -1e300);
    POOL.split(nm, 2, [&](int m0, int m1, int tid) {
        double lmax = -1e300;
        for (int m = m0; m < m1; m++) {
            const double *mf = &mf_t[(size_t)m * nf];
            for (int f = 0; f < need; f++) {
                const double *mg = &mag[(size_t)f * nf];
                double a = 0; for (int k = 0; k < nf; k++) a += mf[k] * mg[k];
                double v = log10(a > 1e-10 ? a : 1e-10);
                ls[(size_t)m * need + f] = v; if (v > lmax) lmax = v;
            }
        }
        tmax[tid] = lmax;
    });
    double gmax = -1e300; for (int t = 0; t < POOL.n; t++) if (tmax[t] > gmax) gmax = tmax[t];
    feats.resize((size_t)nm * need); for (size_t i = 0; i < feats.size(); i++) { double v = ls[i] < gmax - 8.0 ? gmax - 8.0 : ls[i]; feats[i] = (float)((v + 4.0) / 4.0); }
    ncols = need; Lout = L;
}
static void layer_norm_row(const float *x, const float *w, const float *b, float *out, int n, float eps) {
    float s = 0.f; for (int i = 0; i < n; i++) s += x[i]; const float m = s / (float)n;
    float v = 0.f; for (int i = 0; i < n; i++) { float c = x[i] - m; v += c * c; } v /= (float)n;
    const float sd = sqrtf(v + eps);
    int i = 0;
#if HAVE_NEON
    const float32x4_t vm = vdupq_n_f32(m), vsd = vdupq_n_f32(sd);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t vx = vld1q_f32(x + i), vw = vld1q_f32(w + i), vb = vld1q_f32(b + i);
        const float32x4_t norm = RT_DIV4(vsubq_f32(vx, vm), vsd);
        vst1q_f32(out + i, vaddq_f32(vmulq_f32(norm, vw), vb));
    }
#endif
    for (; i < n; i++) out[i] = ((x[i] - m) / sd) * w[i] + b[i];
}
// erf float: x*P(x^2)/Q(x^2) (xap xi hu ty le, sai so tuyet doi < 2e-7 tren [-3.83, 3.83], ngoai khoang kep ve +-1) - NEON 4 phan tu
#if HAVE_NEON
static inline float32x4_t erf4(float32x4_t x) {
    static const float A[7] = {-2.72614225801306e-10f, 2.77068142495902e-08f, -2.10102402082508e-06f, -5.69250639462346e-05f, -7.34990630326855e-04f, -2.95459980854025e-03f, -1.60960333262415e-02f};
    static const float B[5] = {-1.45660718464996e-05f, -2.13374055278905e-04f, -1.68282697438203e-03f, -7.37332916720468e-03f, -1.42647390514189e-02f};
    x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(3.832506856900711f)), vdupq_n_f32(-3.832506856900711f));
    const float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t p = vdupq_n_f32(A[0]); for (int i = 1; i < 7; i++) p = vmlaq_f32(vdupq_n_f32(A[i]), p, x2);
    float32x4_t q = vdupq_n_f32(B[0]); for (int i = 1; i < 5; i++) q = vmlaq_f32(vdupq_n_f32(B[i]), q, x2);
    return RT_DIV4(vmulq_f32(x, p), q);
}
#endif
static bool GELU_EXACT = false;                     // RT_GELU_EXACT=1: dung erff cua libm (cham hon ~10 lan) de doi chieu
static void gelu_range(float *a, int i0, int i1) {
    int i = i0;
#if HAVE_NEON
    if (!GELU_EXACT) {
        const float32x4_t kh = vdupq_n_f32(0.5f), k1 = vdupq_n_f32(1.0f), ks = vdupq_n_f32(0.70710678118654752440f);
        for (; i + 4 <= i1; i += 4) { const float32x4_t x = vld1q_f32(a + i); const float32x4_t y = vmulq_f32(vmulq_f32(x, kh), vaddq_f32(k1, erf4(vmulq_f32(x, ks)))); vst1q_f32(a + i, vbslq_f32(vcltq_f32(x, vdupq_n_f32(-5.5f)), vdupq_n_f32(0.f), y)); }   // x<-5.5: gelu < 2e-7 ~ 0
    }
#endif
    for (; i < i1; i++) { float x = a[i]; a[i] = x * 0.5f * (1.0f + erff(x * 0.70710678118654752440f)); }
}
static void gelu_arr(float *a, size_t n) {          // F.gelu(approximate='none'): (x*0.5)*(1+erf(x*0.70710678))
    POOL.split((int)n, 4096, [&](int i0, int i1, int) { gelu_range(a, i0, i1); });
}
// Luong nen nap truoc trong so encoder (file duoc mmap): cham tung trang theo thu tu dung de doc dia chay song song voi tinh toan.
// Chi doc bo nho chi-doc nen khong co tranh chap du lieu; lan chay sau (cache nong) thi gan nhu khong ton gi. Tat bang RT_NOPREFETCH=1.
struct Prefetch {
    std::thread th; std::atomic<bool> stop{false}; std::atomic<size_t> done{0}; size_t total = 0; bool on = false; volatile uint8_t sink = 0;
    void start(const Pkg &P) {
        if (getenv("RT_NOPREFETCH")) return;
        std::vector<std::pair<const uint8_t *, size_t>> rg; for (const Gemm &g : P.eg) { size_t n = (size_t)g.K * g.N; if (g.B && n) { rg.push_back({(const uint8_t *)g.B, n}); total += n; } }
        if (rg.empty()) return; on = true;
        th = std::thread([this, rg]() {
            const size_t PG = 4096; uint8_t acc = 0;
            auto hint = [&](size_t i) { if (i >= rg.size()) return; uintptr_t a = (uintptr_t)rg[i].first & ~(uintptr_t)(PG - 1), e = (uintptr_t)rg[i].first + rg[i].second; madvise((void *)a, e - a, MADV_WILLNEED); };
            hint(0); hint(1);
            for (size_t i = 0; i < rg.size() && !stop.load(std::memory_order_relaxed); i++) {
                hint(i + 2);
                const volatile uint8_t *p = rg[i].first; const size_t n = rg[i].second;
                for (size_t o = 0; o < n; o += PG) { acc ^= p[o]; if (stop.load(std::memory_order_relaxed)) break; }
                done.fetch_add(n, std::memory_order_relaxed);
            }
            sink = acc;
        });
    }
    double finish() { if (!on) return -1; double f = total ? 100.0 * (double)done.load() / (double)total : 100.0; stop = true; th.join(); on = false; return f; }
    ~Prefetch() { if (on) { stop = true; th.join(); } }
};
struct Lap { double t = now_s(); void lap(double &acc) { double n = now_s(); acc += n - t; t = n; } };
// tra [n_rows][hid] hang audio (da qua projector)
static std::vector<float> encode_audio(Pkg &P, Runner &R, const std::vector<float> &wave, int &n_rows, const char *dump_mel = nullptr) {
    Prefetch pf_th; pf_th.start(P);
    double t0 = now_s(); std::vector<float> feats; int ncols = 0, L = 0; log_mel(P, wave, feats, ncols, L); const double t_mel = now_s() - t0; t0 = now_s();
    double ph_ln = 0, ph_kv = 0, ph_att = 0, ph_gemm = 0, ph_gelu = 0, ph_res = 0, ph_misc = 0; Lap lp;
    if (dump_mel) { FILE *f = fopen(dump_mel, "wb"); if (f) { fwrite(feats.data(), 4, feats.size(), f); fclose(f); } }
    const int d = P.e_d, H = P.e_heads, D = d / H, ffn = P.e_ffn;
    const int L2 = (L - 1) / 2 + 1, Lp = (L2 - 2) / 2 + 1, R1 = std::min(3000, 2 * L2), ncx = std::min(R1 + 1, 3000);
    if (L2 < 4 || Lp < 1 || ncx > ncols) die("audio qua ngan hoac khong nhat quan (L=%s)", std::to_string(L).c_str());
    Stats e0 = ST;
    std::vector<float> X1((size_t)R1 * 3 * P.n_mels), Y1((size_t)R1 * d);                                          // conv1 (k=3, pad=1) -> GELU
    for (int r = 0; r < R1; r++) for (int k = 0; k < 3; k++) { const int j = r + k - 1; for (int c = 0; c < P.n_mels; c++) X1[(size_t)r * 3 * P.n_mels + k * P.n_mels + c] = (j >= 0 && j < ncx) ? feats[(size_t)c * ncols + j] : 0.f; }
    lp.lap(ph_misc);
    R.gemm(P.CONV1(), X1.data(), R1, Y1.data(), nullptr); lp.lap(ph_gemm); gelu_arr(Y1.data(), Y1.size()); lp.lap(ph_gelu);
    std::vector<float> X2((size_t)L2 * 3 * d), h((size_t)L2 * d);                                                    // conv2 (k=3, stride 2, pad=1) -> GELU + vi tri
    for (int i = 0; i < L2; i++) for (int k = 0; k < 3; k++) { const int j = k + 2 * i - 1; float *dst = &X2[(size_t)i * 3 * d + (size_t)k * d]; if (j >= 0 && j < R1) memcpy(dst, &Y1[(size_t)j * d], sizeof(float) * d); else memset(dst, 0, sizeof(float) * d); }
    lp.lap(ph_misc);
    R.gemm(P.CONV2(), X2.data(), L2, h.data(), nullptr); lp.lap(ph_gemm); gelu_arr(h.data(), h.size()); lp.lap(ph_gelu);
    add_inplace_f32(h.data(), P.e_pos, h.size());
    lp.lap(ph_res);
    std::vector<float> a_in((size_t)L2 * d), qkv((size_t)L2 * 3 * d), qs((size_t)L2 * d), o((size_t)L2 * d), tmp((size_t)L2 * d), f1((size_t)L2 * ffn);
    std::vector<int8_t> K8((size_t)H * L2 * D), V8((size_t)H * L2 * D); std::vector<float> KF((size_t)H * L2 * D), VF((size_t)H * L2 * D); std::vector<uint8_t> kcol((size_t)H * L2); std::vector<std::vector<int>> flist(H);
    std::vector<uint8_t> fl, pf(L2);
    const bool fast = (D == 64) && !getenv("RT_NOFAST");
    for (int li = 0; li < P.e_layers; li++) {
        const Attn &at = P.eatt[li];
        POOL.split(L2, 8, [&](int ra, int rb, int) { for (int t = ra; t < rb; t++) layer_norm_row(&h[(size_t)t * d], P.e_ln1w[li], P.e_ln1b[li], &a_in[(size_t)t * d], d, (float)P.e_ln_eps); });
        lp.lap(ph_ln);
        Gemm &gq = P.EG(li, 0); R.gemm(gq, a_in.data(), L2, qkv.data(), &fl); lp.lap(ph_gemm);
        for (int t = 0; t < L2; t++) pf[t] = fl[t] | fl[(size_t)L2 + t] | fl[(size_t)2 * L2 + t];
        const float *vscale = gq.sout + 2 * d; const float thr_k = (float)(127.5 * at.sk);
        POOL.split(L2, 8, [&](int t0_q, int t1_q, int) {
            for (int t = t0_q; t < t1_q; t++) {
                const float *src = &qkv[(size_t)t * 3 * d]; float *dst = &qs[(size_t)t * d];
                int c = 0;
#if HAVE_NEON
                const float32x4_t vqs = vdupq_n_f32(P.e_qscale);
                for (; c + 4 <= d; c += 4) vst1q_f32(dst + c, vmulq_f32(vld1q_f32(src + c), vqs));
#endif
                for (; c < d; c++) dst[c] = src[c] * P.e_qscale;
            }
        });
        POOL.split(H, 1, [&](int h0, int h1, int) { for (int hh = h0; hh < h1; hh++) { flist[hh].clear(); for (int s = 0; s < L2; s++) {
            const float *kr = &qkv[(size_t)s * 3 * d + d + hh * D], *vr = &qkv[(size_t)s * 3 * d + 2 * d + hh * D]; const size_t base = ((size_t)hh * L2 + s) * D; bool col = pf[s] != 0; float mk = 0.f;
            for (int c = 0; c < D; c++) { mk = fmaxf(mk, fabsf(kr[c])); const float vsc = vscale[hh * D + c]; if (fabsf(vr[c]) > 127.5f * vsc) col = true;
                float r1 = nearbyintf(kr[c] / at.sk); K8[base + c] = (int8_t)(r1 > 127.f ? 127.f : (r1 < -128.f ? -128.f : r1));
                float r2 = nearbyintf(vr[c] / vsc); V8[base + c] = (int8_t)(r2 > 127.f ? 127.f : (r2 < -128.f ? -128.f : r2)); KF[base + c] = kr[c]; VF[base + c] = vr[c]; }
            if (mk > thr_k) col = true; kcol[(size_t)hh * L2 + s] = col; if (col) flist[hh].push_back(s); } } });
        lp.lap(ph_kv);
        POOL.split(L2 * H, 1, [&](int i0, int i1, int tid) {
            std::vector<float> sc(L2), p(L2); std::vector<int8_t> q8(D), p8b(L2 + 16); std::vector<float> scb(L2 + 16);
            for (int it = i0; it < i1; it++) { const int hh = it / L2, t = it % L2; const size_t hb = (size_t)hh * L2 * D;      // theo dau: K/V cua 1 dau (~44 KB) duoc dung lai cho L2 truy van lien tiep
                attend_head(&qs[(size_t)t * d + hh * D], D, at, 1.0f, pf[t] != 0, L2 - 1, &K8[hb], &V8[hb], &KF[hb], &VF[hb], &kcol[(size_t)hh * L2], flist[hh], vscale + hh * D, &o[(size_t)t * d + hh * D], fast, scb.data(), p8b.data(), sc, p, q8, ATT_ENC[tid], 0); }
        });
        lp.lap(ph_att);
        R.gemm(P.EG(li, 1), o.data(), L2, tmp.data(), nullptr); lp.lap(ph_gemm);
        add_inplace_f32(h.data(), tmp.data(), h.size());
        lp.lap(ph_res);
        POOL.split(L2, 8, [&](int ra, int rb, int) { for (int t = ra; t < rb; t++) layer_norm_row(&h[(size_t)t * d], P.e_ln2w[li], P.e_ln2b[li], &a_in[(size_t)t * d], d, (float)P.e_ln_eps); });
        lp.lap(ph_ln);
        R.gemm(P.EG(li, 2), a_in.data(), L2, f1.data(), nullptr); lp.lap(ph_gemm); gelu_arr(f1.data(), f1.size()); lp.lap(ph_gelu);
        R.gemm(P.EG(li, 3), f1.data(), L2, tmp.data(), nullptr); lp.lap(ph_gemm);
        add_inplace_f32(h.data(), tmp.data(), h.size());
        lp.lap(ph_res);
    }
    std::vector<float> hp((size_t)Lp * d), out((size_t)Lp * P.hid);                                                  // avg-pool k=2,s=2 -> LayerNorm -> projector
    POOL.split(Lp, 4, [&](int i0, int i1, int) {
        for (int i = i0; i < i1; i++) {
            float *row = &hp[(size_t)i * d]; const float *h0 = &h[(size_t)2 * i * d], *h1 = &h[(size_t)(2 * i + 1) * d];
            for (int c = 0; c < d; c++) row[c] = 0.5f * (h0[c] + h1[c]);
            layer_norm_row(row, P.e_lnw, P.e_lnb, row, d, (float)P.e_ln_eps);
        }
    });
    lp.lap(ph_ln);
    R.gemm(P.PROJ(), hp.data(), Lp, out.data(), nullptr); lp.lap(ph_gemm);
    const double t_enc = now_s() - t0, pfp = pf_th.finish();
    fprintf(stderr, "[encoder] mel %d -> %d -> %d hang | log-mel %.2f s | encoder %.2f s (PL wall %.2f s, nap trong so dong %.2f s)\n", L, L2, Lp, t_mel, t_enc, ST.pl_wall - e0.pl_wall, ST.t_dyn - e0.t_dyn);
    {   // chi tiet: GEMM (gom PL + phan PS trong GEMM) va phan PS ngoai GEMM
        const double pre = ST.ps_pre - e0.ps_pre, rq = ST.ps_rq - e0.ps_rq, i32 = ST.ps_i32 - e0.ps_i32, dg = ST.ps_dig - e0.ps_dig, pl = ST.pl_wall - e0.pl_wall, dyn = ST.t_dyn - e0.t_dyn, hw = ST.pl_hw - e0.pl_hw;
        fprintf(stderr, "[encoder] GEMM %.2f s = PL %.2f (nap dong %.2f, PL chay thuc %.2f, chep A/C %.2f) + tien xu ly/luong tu hoa %.2f + requant %.2f + int32->f32 %.2f + chu so du phong %.2f + con lai %.2f | %ld luot PL, %ld hang fallback day du, %ld hang bao hoa\n",
                ph_gemm, pl, dyn, hw, pl - dyn - hw, pre, rq, i32, dg, ph_gemm - pl - pre - rq - i32 - dg, ST.passes - e0.passes, ST.rows_dig - e0.rows_dig, ST.rows_sat - e0.rows_sat);
        fprintf(stderr, "[encoder] PS ngoai GEMM: attention %.2f s | dung K/V int8 %.2f | LayerNorm %.2f | GELU %.2f | cong tat (residual/pos) %.2f | khac %.2f%s\n", ph_att, ph_kv, ph_ln, ph_gelu, ph_res, ph_misc,
                pfp >= 0 ? "" : " | (prefetch tat)");
        if (pfp >= 0) fprintf(stderr, "[encoder] luong nap truoc: da cham %.0f%% trong so khi encoder xong\n", pfp);
    }
    n_rows = Lp; return out;
}
// prompt = tien to + n_rows x <|AUDIO|> + hau to (lay tu 1 ids.i32 mau, vi du in_clip00/ids.i32: template chat khong doi giua cac clip)
static std::vector<int> build_ids(const std::vector<int> &tmpl, int audio_id, int n_rows) {
    size_t a = tmpl.size(), b = 0; for (size_t i = 0; i < tmpl.size(); i++) if (tmpl[i] == audio_id) { if (a == tmpl.size()) a = i; b = i; }
    if (a == tmpl.size()) die("template ids khong co token <|AUDIO|>");
    std::vector<int> ids(tmpl.begin(), tmpl.begin() + a); ids.insert(ids.end(), (size_t)n_rows, audio_id); ids.insert(ids.end(), tmpl.begin() + b + 1, tmpl.end()); return ids;
}

// ----------------------------------------------------------------------------- giai ma token -> chu (byte-level BPE cua Qwen2): chi can vocab.json (hoac tokenizer.json) trong hf_processor
struct Vocab { std::vector<std::string> tok; size_t n_found = 0; };
static void append_utf8(std::string &s, uint32_t cp) {
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 63)); s += (char)(0x80 | (cp & 63)); }
    else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 63)); s += (char)(0x80 | ((cp >> 6) & 63)); s += (char)(0x80 | (cp & 63)); }
}
static int g_cp2byte[324]; static bool g_cp2byte_init = false;
static void init_cp2byte() {                       // nghich dao cua bytes_to_unicode() cua GPT-2
    if (g_cp2byte_init) return; g_cp2byte_init = true; for (int i = 0; i < 324; i++) g_cp2byte[i] = -1;
    int n = 0; for (int b = 0; b < 256; b++) { bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174; if (keep) g_cp2byte[b] = b; else g_cp2byte[256 + n++] = b; }
}
static bool load_vocab(const std::string &path_in, Vocab &v) {
    init_cp2byte(); std::string path = path_in; struct stat st;
    if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) { std::string a = path + "/vocab.json"; if (stat(a.c_str(), &st) == 0) path = a; else path += "/tokenizer.json"; }
    FILE *f = fopen(path.c_str(), "rb"); if (!f) return false;
    std::string s; { char buf[65536]; size_t k; while ((k = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, k); } fclose(f);
    size_t i = 0; if (path.size() >= 14 && path.compare(path.size() - 14, 14, "tokenizer.json") == 0) { i = s.find("\"vocab\""); if (i == std::string::npos) return false; }
    i = s.find('{', i); if (i == std::string::npos) return false; i++;
    auto hex4 = [&](size_t at) { uint32_t x = 0; for (int k = 0; k < 4; k++) { char c = s[at + k]; x = x * 16 + (c <= '9' ? c - '0' : (c | 32) - 'a' + 10); } return x; };
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t' || s[i] == ',')) i++;
        if (i >= s.size() || s[i] == '}') break; if (s[i] != '"') return false; i++;
        std::string key;                                                            // chuoi JSON -> UTF-8
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') { i++; char c = s[i++];
                if (c == 'u') { uint32_t cp = hex4(i); i += 4; if (cp >= 0xD800 && cp < 0xDC00 && s[i] == '\\' && s[i + 1] == 'u') { uint32_t lo = hex4(i + 2); i += 6; cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); } append_utf8(key, cp); }
                else key += c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c == 'b' ? '\b' : c == 'f' ? '\f' : c;
            } else key += s[i++];
        }
        i++; while (i < s.size() && (s[i] == ' ' || s[i] == ':')) i++;
        long id = 0; bool neg = false; if (s[i] == '-') { neg = true; i++; } while (i < s.size() && s[i] >= '0' && s[i] <= '9') id = id * 10 + (s[i++] - '0'); if (neg) id = -id;
        std::string bytes; size_t j = 0;                                             // ky tu byte-level -> byte that
        while (j < key.size()) {
            uint32_t cp; unsigned char c = key[j];
            if (c < 0x80) { cp = c; j += 1; } else if ((c >> 5) == 6) { cp = ((c & 31) << 6) | (key[j + 1] & 63); j += 2; }
            else if ((c >> 4) == 14) { cp = ((c & 15) << 12) | ((key[j + 1] & 63) << 6) | (key[j + 2] & 63); j += 3; } else { cp = ((c & 7) << 18) | ((key[j + 1] & 63) << 12) | ((key[j + 2] & 63) << 6) | (key[j + 3] & 63); j += 4; }
            if (cp < 324 && g_cp2byte[cp] >= 0) bytes += (char)g_cp2byte[cp]; else append_utf8(bytes, cp);
        }
        if (id >= 0) { if ((size_t)id >= v.tok.size()) v.tok.resize((size_t)id + 1); if (v.tok[id].empty()) v.n_found++; v.tok[id] = bytes; }
    }
    return v.n_found > 0;
}
// token khong co trong vocab (token dac biet: <|im_end|>, <|endoftext|>, <|AUDIO|>...) bi bo, giong skip_special_tokens=True
static std::string detok(const Vocab &v, const std::vector<int> &ids) {
    std::string out; for (int t : ids) if (t >= 0 && (size_t)t < v.tok.size()) out += v.tok[t];
    size_t a = 0, b = out.size(); while (a < b && (out[a] == ' ' || out[a] == '\n' || out[a] == '\t' || out[a] == '\r')) a++; while (b > a && (out[b - 1] == ' ' || out[b - 1] == '\n' || out[b - 1] == '\t' || out[b - 1] == '\r')) b--;
    return out.substr(a, b - a);
}
static std::vector<int> read_token_file(const std::string &path) {
    std::vector<int> t; FILE *f = fopen(path.c_str(), "r"); if (!f) die("khong mo duoc %s", path.c_str()); int x; while (fscanf(f, "%d", &x) == 1) t.push_back(x); fclose(f); return t;
}

// ----------------------------------------------------------------------------- --selftest: so NEON voi vong vo huong (chay duoc tren board, vai giay, khong can NPU/goi)
static int selftest() {
    uint64_t rs = 88172645463325252ull; auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    auto urand = [&](double lo, double hi) { return lo + (hi - lo) * (double)(rnd() >> 11) / 9007199254740992.0; };
    int bad = 0;
    // 1) requant + co bao hoa
    {
        Gemm g; g.K = 896; g.N = 1027; for (int n = 0; n < g.N; n++) {
            g.bi.push_back((int32_t)urand(-2e6, 2e6)); g.mult.push_back((int32_t)urand(1 << 28, 2147483000.0)); g.shift.push_back((int32_t)urand(18, 42)); g.zp.push_back((int32_t)urand(-30, 30));
            g.rawk.push_back(urand(2e-6, 4e-5)); }
        std::vector<float> so(g.N); for (auto &v : so) v = (float)urand(0.01, 0.3); g.sout = so.data(); g.mem = {{0, 256}, {256, 640}, {640, 1027}}; finalize_requant(g);
        std::vector<int32_t> ac(g.N); std::vector<float> y1(g.N), y2(g.N); long diff = 0, sdiff = 0, nsat = 0, tests = 0;
        for (int it = 0; it < 300; it++) {
            double amp = it < 100 ? 3e5 : (it < 200 ? 4e6 : 2e7); for (auto &v : ac) v = (int32_t)urand(-amp, amp);
            for (auto &mb : g.mem) {
                bool sn = requant_range(g, ac.data(), y1.data(), mb.c0, mb.c1), sr = false;
                for (int n = mb.c0; n < mb.c1; n++) {
                    int64_t t = (int64_t)ac[n] + g.bi[n]; int64_t rn = g.shift[n] > 0 ? ((int64_t)1 << (g.shift[n] - 1)) : 0; int64_t v = ((t * g.mult[n] + rn) >> g.shift[n]) + g.zp[n]; v = v < -128 ? -128 : (v > 127 ? 127 : v);
                    y2[n] = (float)v * g.sout[n]; if (fabs((double)t * g.rawk[n]) > 127.5) sr = true;
                    if (memcmp(&y1[n], &y2[n], 4)) diff++;
                }
                if (sn != sr) sdiff++; nsat += sr; tests++;
            }
        }
        printf("[selftest] requant: %ld khac gia tri, %ld khac co bao hoa / %ld thanh vien (%ld co bao hoa), neon_rq=%d  %s\n", diff, sdiff, tests, nsat, (int)g.neon_rq, (diff || sdiff) ? "LOI" : "OK"); bad += (diff || sdiff);
    }
    // 1b) nhom luong: requant chia theo cot (cac khuc 64) == requant ca hang; lap nhieu lan de do loi dong bo
    {
        const int saved = POOL.n; if (POOL.n == 1) POOL.start(4);
        Gemm g; g.K = 4864; g.N = 9728; for (int n = 0; n < g.N; n++) {
            g.bi.push_back((int32_t)urand(-2e6, 2e6)); g.mult.push_back((int32_t)urand(1 << 28, 2147483000.0)); g.shift.push_back((int32_t)urand(18, 42)); g.zp.push_back((int32_t)urand(-30, 30));
            g.rawk.push_back(urand(2e-6, 4e-5)); }
        std::vector<float> so(g.N); for (auto &v : so) v = (float)urand(0.01, 0.3); g.sout = so.data(); g.mem = {{0, 4864}, {4864, 9728}}; finalize_requant(g);
        std::vector<int32_t> ac(g.N); std::vector<float> y1(g.N), y2(g.N); long bad_it = 0;
        for (int it = 0; it < 300; it++) {
            for (auto &v : ac) v = (int32_t)urand(-3e7, 3e7);
            bool sref[2]; for (int mi = 0; mi < 2; mi++) sref[mi] = requant_range(g, ac.data(), y1.data(), g.mem[mi].c0, g.mem[mi].c1);
            bool satT[8][8] = {}; POOL.split(g.N, 64, [&](int n0, int n1, int tid) { for (int mi = 0; mi < 2; mi++) { int c0 = std::max(n0, g.mem[mi].c0), c1 = std::min(n1, g.mem[mi].c1); if (c0 < c1 && requant_range(g, ac.data(), y2.data(), c0, c1)) satT[tid][mi] = true; } });
            bool sp[2] = {false, false}; for (int tt = 0; tt < 8; tt++) for (int mi = 0; mi < 2; mi++) sp[mi] = sp[mi] || satT[tt][mi];
            if (memcmp(y1.data(), y2.data(), 4 * (size_t)g.N) || sp[0] != sref[0] || sp[1] != sref[1]) bad_it++;
        }
        printf("[selftest] nhom luong (%d luong), requant chia cot: %ld/300 lan khac  %s\n", POOL.n, bad_it, bad_it ? "LOI" : "OK"); bad += (bad_it != 0); (void)saved;
    }
    // 2) luong tu hoa hang + max|x|
    {
        long diff = 0, mdiff = 0; const int K = 4867; std::vector<float> x(K); std::vector<int8_t> a1(K), a2(K);
        for (int it = 0; it < 200; it++) {
            float sc = (float)urand(0.005, 0.2); for (int k = 0; k < K; k++) { double r = urand(0, 1); x[k] = r < 0.2 ? ((float)(int)urand(-120, 120) + 0.5f) * sc : (float)urand(-130 * sc, 130 * sc); }
            quant_row_s8(x.data(), a1.data(), K, sc);
            for (int k = 0; k < K; k++) { float r = nearbyintf(x[k] / sc); r = r > 127.f ? 127.f : (r < -128.f ? -128.f : r); a2[k] = (int8_t)r; if (a1[k] != a2[k]) diff++; }
            float m1 = maxabs_f32(x.data(), K), m2 = 0; for (int k = 0; k < K; k++) m2 = fmaxf(m2, fabsf(x[k])); if (m1 != m2) mdiff++;
        }
        printf("[selftest] luong tu hoa: %ld phan tu khac, max|x| khac %ld  %s\n", diff, mdiff, (diff || mdiff) ? "LOI" : "OK"); bad += (diff || mdiff);
    }
    // 3) attention 1 dau (head_dim 64) co cot FP vs vong vo huong (double, giong nhanh cham cua Model::attention)
    {
        double worst = 0; long flips = 0, total = 0; const int maxS = 600;
        std::vector<int8_t> K8((size_t)maxS * 64), V8((size_t)maxS * 64), p8(maxS + 16), q8(64); std::vector<float> Kf((size_t)maxS * 64), Vf((size_t)maxS * 64), qf(64), vs(64), sc(maxS + 16), o1(64), o2(64);
        for (int it = 0; it < 400; it++) {
            const int S = 1 + (int)urand(0, maxS - 1); const float sq = 0.05f, sk = 0.04f, fac = sq * sk * 0.125f * (float)urand(0.5, 4.0), scaling = 0.125f;
            for (int i = 0; i < 64; i++) { qf[i] = (float)urand(-6, 6); q8[i] = (int8_t)std::max(-128.f, std::min(127.f, nearbyintf(qf[i] / sq))); vs[i] = (float)urand(0.01, 0.05); }
            for (size_t i = 0; i < (size_t)S * 64; i++) { K8[i] = (int8_t)urand(-128, 127.99); V8[i] = (int8_t)urand(-128, 127.99); Kf[i] = K8[i] * sk; Vf[i] = V8[i] * vs[i % 64]; }
            std::vector<uint8_t> flg(S, 0); std::vector<int> fp; int nf = it % 4 == 0 ? 0 : (int)urand(1, std::min(64, S) - 0.001); if (nf >= S) nf = S - 1;
            for (int i = 0; i < nf; i++) { int pos = (int)urand(0, S - 0.001); if (!flg[pos]) { flg[pos] = 1; } }
            for (int i = 0; i < S; i++) if (flg[i]) { fp.push_back(i); for (int d = 0; d < 64; d++) { Kf[(size_t)i * 64 + d] *= (it % 3 == 0 ? 8.f : 1.f); Vf[(size_t)i * 64 + d] *= 5.f; } }
            attn_head_fast(q8.data(), K8.data(), V8.data(), S, fac, vs.data(), o1.data(), sc.data(), p8.data(), fp.data(), (int)fp.size(), qf.data(), Kf.data(), Vf.data(), scaling);
            // vo huong
            std::vector<float> scr(S), p(S); float m = -INFINITY;
            for (int s = 0; s < S; s++) { if (flg[s]) { float acc = 0; for (int d = 0; d < 64; d++) acc += qf[d] * Kf[(size_t)s * 64 + d]; scr[s] = acc * scaling; } else { int acc = 0; for (int d = 0; d < 64; d++) acc += (int)q8[d] * K8[(size_t)s * 64 + d]; scr[s] = (float)acc * fac; } m = fmaxf(m, scr[s]); }
            float sum = 0; for (int s = 0; s < S; s++) { p[s] = expf(scr[s] - m); sum += p[s]; }
            float pmax = 0; for (int s = 0; s < S; s++) { p[s] /= sum; if (!flg[s]) pmax = fmaxf(pmax, p[s]); }
            float sp = fmaxf(pmax / 127.0f, 1e-30f); std::vector<double> accd(64, 0.0), fpa(64, 0.0);
            for (int s = 0; s < S; s++) { if (flg[s]) { for (int d = 0; d < 64; d++) fpa[d] += (double)p[s] * Vf[(size_t)s * 64 + d]; continue; } float pq = nearbyintf(p[s] / sp); for (int d = 0; d < 64; d++) accd[d] += (double)(pq * (float)V8[(size_t)s * 64 + d]); }
            double mxo = 1e-9; for (int d = 0; d < 64; d++) { o2[d] = (float)(accd[d] * (double)sp * (double)vs[d] + fpa[d]); mxo = std::max(mxo, (double)fabsf(o2[d])); }
            double df = 0; for (int d = 0; d < 64; d++) df = std::max(df, (double)fabsf(o1[d] - o2[d])); df /= mxo; worst = std::max(worst, df); total++; if (df > 2e-3) flips++;
        }
        printf("[selftest] attention 1 dau (co cot FP): sai so tuong doi lon nhat %.2e, %ld/%ld ca > 2e-3 (do lam tron P8 o bien)  %s\n", worst, flips, total, worst > 5e-2 ? "LOI" : "OK"); bad += (worst > 5e-2);
    }
    {   // GELU NEON (erf huu ty) so voi erf double: x trong [-12, 12], + chia luong
        const int n = 240001; std::vector<float> g(n); for (int i = 0; i < n; i++) g[i] = -12.f + 24.f * (float)i / (float)(n - 1);
        std::vector<float> x0 = g; GELU_EXACT = false; gelu_arr(g.data(), g.size());
        double mabs = 0, mrel = 0, mabs_neg = 0; for (int i = 0; i < n; i++) { const double x = x0[i], r = x * 0.5 * (1.0 + erf(x * 0.70710678118654752440)), e = fabs((double)g[i] - r);
            mabs = std::max(mabs, e); if (fabs(r) > 1e-2) mrel = std::max(mrel, e / fabs(r)); if (x < -3.0) mabs_neg = std::max(mabs_neg, e); }
        const bool ok = mabs < 2e-6 && mrel < 1e-4; bad += !ok;
        printf("[selftest] GELU NEON: sai so tuyet doi lon nhat %.2e (x<-3: %.2e), tuong doi lon nhat (|gelu|>1e-2) %.2e  %s\n", mabs, mabs_neg, mrel, ok ? "OK" : "LOI");
    }
    // 5) rms_norm, layer_norm_row, add_inplace_f32 (NEON bit-exact so voi vo huong)
    {
        long diff_rms = 0, diff_ln = 0, diff_add = 0; const int N = 1283;
        std::vector<float> x(N), w(N), b(N), y1(N), y2(N);
        for (int it = 0; it < 100; it++) {
            for (int i = 0; i < N; i++) { x[i] = (float)urand(-5, 5); w[i] = (float)urand(0.1, 2.0); b[i] = (float)urand(-1, 1); }
            rms_norm(x.data(), w.data(), y1.data(), N, 1e-6f);
            float ss = 0.f; for (int i = 0; i < N; i++) ss += x[i] * x[i];
            float r = 1.0f / sqrtf(ss / (float)N + 1e-6f);
            for (int i = 0; i < N; i++) { y2[i] = w[i] * (x[i] * r); if (memcmp(&y1[i], &y2[i], 4)) diff_rms++; }
            layer_norm_row(x.data(), w.data(), b.data(), y1.data(), N, 1e-5f);
            float sm = 0; for (int i = 0; i < N; i++) sm += x[i]; float m = sm / (float)N;
            float sv = 0; for (int i = 0; i < N; i++) { float c = x[i] - m; sv += c * c; } sv /= (float)N;
            float sd = sqrtf(sv + 1e-5f);
            for (int i = 0; i < N; i++) { y2[i] = ((x[i] - m) / sd) * w[i] + b[i]; if (memcmp(&y1[i], &y2[i], 4)) diff_ln++; }
            y1 = x; y2 = x; add_inplace_f32(y1.data(), b.data(), N);
            for (int i = 0; i < N; i++) { y2[i] += b[i]; if (memcmp(&y1[i], &y2[i], 4)) diff_add++; }
        }
        const bool ok = (diff_rms == 0 && diff_ln == 0 && diff_add == 0); bad += !ok;
        printf("[selftest] NEON bit-exact (rms_norm=%ld, layer_norm=%ld, add_inplace=%ld khac)  %s\n", diff_rms, diff_ln, diff_add, ok ? "OK" : "LOI");
    }
    printf("[selftest] %s\n", bad ? "CO LOI" : "TAT CA OK"); return bad ? 1 : 0;
}

// ----------------------------------------------------------------------------- main
static void apply_penalty(float *lg, int V, const std::vector<int> &hist, float pen) {
    static std::vector<uint8_t> seen;
    if ((int)seen.size() < V) seen.assign(V, 0);
    for (int t : hist) if (t >= 0 && t < V && !seen[t]) { seen[t] = 1; float v = lg[t]; lg[t] = v > 0 ? v / pen : v * pen; }
    for (int t : hist) if (t >= 0 && t < V) seen[t] = 0;
}
static int argmax(const float *v, int n) { int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }

int main(int argc, char **argv) {
    std::string pkg, ids_f, audio_f, backend = "soft", teacher_f, top5_f, logits_f, out_f; int max_new = -1, digits = 3, scratch_mb = 16, threads = 1; std::string vocab_f, detok_f, text_out, wav_f, tmpl_f, rows_out, mel_out; bool verbose = false; (void)scratch_mb;
    if (getenv("RT_GELU_EXACT")) GELU_EXACT = true;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--selftest")) return selftest();
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i]; auto nx = [&]() -> std::string { if (i + 1 >= argc) die("thieu gia tri cho %s", a.c_str()); return argv[++i]; };
        if (a == "--pkg") pkg = nx(); else if (a == "--ids") ids_f = nx(); else if (a == "--audio") audio_f = nx(); else if (a == "--backend") backend = nx();
        else if (a == "--teacher") teacher_f = nx(); else if (a == "--dump-top5") top5_f = nx(); else if (a == "--dump-logits") logits_f = nx(); else if (a == "--out") out_f = nx();
        else if (a == "--max-new") max_new = atoi(nx().c_str()); else if (a == "--digits") digits = atoi(nx().c_str()); else if (a == "--scratch-mb") scratch_mb = atoi(nx().c_str());
        else if (a == "--threads") threads = atoi(nx().c_str()); else if (a == "--wav") wav_f = nx(); else if (a == "--template") tmpl_f = nx(); else if (a == "--dump-rows") rows_out = nx(); else if (a == "--dump-mel") mel_out = nx(); else if (a == "--vocab") vocab_f = nx(); else if (a == "--detok") detok_f = nx(); else if (a == "--text-out") text_out = nx(); else if (a == "--verbose") verbose = true; else die("tuy chon la: %s", a.c_str());
    }
    if (!detok_f.empty()) {                                                          // chi doi token -> chu (khong can NPU)
        Vocab V; std::string vp = vocab_f.empty() ? pkg + "/hf_processor" : vocab_f; if (!load_vocab(vp, V)) die("khong doc duoc vocab.json/tokenizer.json o %s (dung --vocab FILE|DIR hoac --pkg DIR)", vp.c_str());
        std::string txt = detok(V, read_token_file(detok_f)); printf("%s\n", txt.c_str()); return 0;
    }
    if (pkg.empty() || (ids_f.empty() && wav_f.empty() && !(!audio_f.empty() && !tmpl_f.empty()))) die("dung: llm_runtime --pkg DIR (--ids ids.i32 --audio rows.f32 | --audio rows.f32 --template ids.i32 | --wav file.wav --template ids.i32) [--backend soft|npu] ...");
    setvbuf(stdout, nullptr, _IOLBF, 0);
    POOL.start(threads); if (POOL.n > 1) fprintf(stderr, "[pool] %d luong (attention theo dau, requant theo cot, silu)\n", POOL.n);
    auto rq_cov = [&](const std::vector<Gemm> &v, const char *nm) {
        int tot = 0, neon = 0, i32 = 0; std::string bad; for (const Gemm &g : v) { if (g.int32) { i32++; continue; } tot++; if (g.neon_rq) neon++; else if (bad.size() < 160) bad += g.name + "(" + (g.rq_why & 1 ? "shift/zp" : "") + (g.rq_why & 2 ? " bias" : "") + ") "; }
        fprintf(stderr, "[pkg] requant NEON %s: %d/%d GEMM (+%d int32)%s%s\n", nm, neon, tot, i32, bad.empty() ? "" : " | KHONG NEON: ", bad.c_str()); };
    Pkg P; double t0 = now_s(); P.load(pkg); fprintf(stderr, "[pkg] %d lop, hidden %d, vocab %d, %zu GEMM | doc manifest %.2f s\n", P.layers, P.hid, P.vocab, P.g.size(), now_s() - t0); rq_cov(P.g, "LLM");
    Backend *be = nullptr;
    if (backend == "soft") be = new SoftBackend();
#ifdef RT_HAVE_NPU
    else if (backend == "npu") be = new NpuBackend();
#endif
    else die("backend '%s' khong co trong ban build nay (npu can -DNPU_DMAHEAP)", backend.c_str());
    if (!wav_f.empty()) { t0 = now_s(); P.load_encoder(); fprintf(stderr, "[pkg] encoder: %d lop, d_model %d, %d dau | nap manifest %.2f s\n", P.e_layers, P.e_d, P.e_heads, now_s() - t0); rq_cov(P.eg, "encoder"); }
    t0 = now_s(); be->prepare(P); if (backend != "soft") fprintf(stderr, "[npu] prepare %.1f s\n", now_s() - t0);
    Runner R(P, *be); R.digits = digits; Model M(P, R);

    std::vector<int> ids; std::vector<float> rows_buf; const float *arow = nullptr; size_t na = 0, used = 0; std::vector<uint8_t> ab;
    if (!wav_f.empty()) {                                                                  // WAV -> log-mel -> encoder -> projector -> prompt
        if (tmpl_f.empty()) die("--wav can --template ids.i32 (vi du in_clip00/ids.i32): template chat cua prompt");
        std::vector<float> wave = read_wav16(wav_f); fprintf(stderr, "[wav] %zu mau (%.2f s)\n", wave.size(), wave.size() / 16000.0);
        int nrow = 0; rows_buf = encode_audio(P, R, wave, nrow, mel_out.empty() ? nullptr : mel_out.c_str()); arow = rows_buf.data(); na = (size_t)nrow;
        auto tb = read_all(tmpl_f); std::vector<int> tm(tb.size() / 4); { const int32_t *p = (const int32_t *)tb.data(); for (size_t i = 0; i < tm.size(); i++) tm[i] = p[i]; }
        ids = build_ids(tm, P.audio_id, nrow);
        if (!rows_out.empty()) { FILE *f = fopen(rows_out.c_str(), "wb"); if (f) { fwrite(rows_buf.data(), 4, rows_buf.size(), f); fclose(f); } }
    } else if (ids_f.empty() && !tmpl_f.empty() && !audio_f.empty()) {                      // hang audio co san (vd tu encoder FP32) + template: tu dung prompt theo so hang
        ab = read_all(audio_f); arow = (const float *)ab.data(); na = ab.size() / 4 / P.hid;
        if (na * 4 * (size_t)P.hid != ab.size()) die("--audio: kich thuoc file khong chia het cho hidden %s", std::to_string(P.hid).c_str());
        auto tb = read_all(tmpl_f); std::vector<int> tm(tb.size() / 4); { const int32_t *p = (const int32_t *)tb.data(); for (size_t i = 0; i < tm.size(); i++) tm[i] = p[i]; }
        ids = build_ids(tm, P.audio_id, (int)na); fprintf(stderr, "[audio] %zu hang tu %s (template %s)\n", na, audio_f.c_str(), tmpl_f.c_str());
    } else {
        auto ib = read_all(ids_f); ids.resize(ib.size() / 4); { const int32_t *p = (const int32_t *)ib.data(); for (size_t i = 0; i < ids.size(); i++) ids[i] = p[i]; }
        if (!audio_f.empty()) ab = read_all(audio_f);
        arow = (const float *)ab.data(); na = ab.size() / 4 / P.hid;
    }
    std::vector<int> teacher; if (!teacher_f.empty()) { auto tb = read_all(teacher_f); const int32_t *p = (const int32_t *)tb.data(); for (size_t i = 0; i < tb.size() / 4; i++) teacher.push_back(p[i]); }
    if (max_new < 0) max_new = P.max_new;
    const int Pn = (int)ids.size(), hid = P.hid; M.new_cache(Pn + (teacher.empty() ? max_new : (int)teacher.size()) + 2);
    std::vector<float> x((size_t)Pn * hid);
    for (int t = 0; t < Pn; t++) {
        if (ids[t] == P.audio_id) { if (used >= na) die("so <|AUDIO|> > so hang audio"); memcpy(&x[(size_t)t * hid], arow + used * hid, sizeof(float) * hid); used++; }
        else for (int i = 0; i < hid; i++) x[(size_t)t * hid + i] = half2f(P.emb[(size_t)ids[t] * hid + i]);
    }
    if (used != na) die("so <|AUDIO|> (%s) != so hang audio", std::to_string(used).c_str());
    // prefill
    Stats s0 = ST; double PRP[P_N]; memcpy(PRP, PRF, sizeof PRF); t0 = now_s(); M.rows(x.data(), Pn); double t_pre = now_s() - t0; Stats sp = ST;
    fprintf(stderr, "[prefill] %d hang: %.2f s (PL wall %.2f s, fallback %ld hang)\n", Pn, t_pre, sp.pl_wall - s0.pl_wall, sp.rows_dig - s0.rows_dig);
    {   const double pre = sp.ps_pre - s0.ps_pre, rq = sp.ps_rq - s0.ps_rq, i32 = sp.ps_i32 - s0.ps_i32, dg = sp.ps_dig - s0.ps_dig; double sum = pre + rq + i32 + dg, pl = sp.pl_wall - s0.pl_wall;
        fprintf(stderr, "[prefill] PS %.2f s: luong tu hoa x+kiem tra %.2f | requant %.2f | int32->f32 %.2f | chu so du phong %.2f", t_pre - pl, pre, rq, i32, dg);
        for (int i = 0; i < P_N; i++) { const double v = PRF[i] - PRP[i]; fprintf(stderr, " | %s %.2f", P_NAME[i], v); sum += v; }
        fprintf(stderr, " | con lai %.2f\n", t_pre - pl - sum); }
    std::vector<float> lg(P.vocab), last(x.end() - hid, x.end());
    std::vector<int> hist(ids), outt; FILE *f5 = top5_f.empty() ? nullptr : fopen(top5_f.c_str(), "w"), *fl = logits_f.empty() ? nullptr : fopen(logits_f.c_str(), "wb");
    const int steps = teacher.empty() ? max_new : (int)teacher.size();
    double t_dec = 0; std::vector<double> tok_ms; Stats d0 = ST; double PR0[P_N]; memcpy(PR0, PRF, sizeof PRF);
    for (int step = 0; step < steps; step++) {
        double ts = now_s();
        M.logits(last.data(), lg.data());
        double to = now_s();
        if (fl) fwrite(lg.data(), 4, P.vocab, fl);
        if (f5) { int ix[5] = {-1, -1, -1, -1, -1};                  // top-5 mot luot (hoa theo chi so nho truoc khi bang nhau)
            for (int i = 0; i < P.vocab; i++) { int p = 5; while (p > 0 && (ix[p - 1] < 0 || lg[i] > lg[ix[p - 1]])) p--; if (p >= 5) continue; for (int q = 4; q > p; q--) ix[q] = ix[q - 1]; ix[p] = i; }
            fprintf(f5, "%d", step); for (int i = 0; i < 5; i++) fprintf(f5, " %d %.6f", ix[i], lg[ix[i]]); fprintf(f5, "\n"); }
        apply_penalty(lg.data(), P.vocab, hist, P.penalty);
        int tok = argmax(lg.data(), P.vocab);
        PRF[P_OUT] += now_s() - to;
        int feed = teacher.empty() ? tok : teacher[step];
        outt.push_back(tok);
        if (teacher.empty() && tok == P.eos) { tok_ms.push_back((now_s() - ts) * 1e3); t_dec += now_s() - ts; break; }
        if (step + 1 == steps) { tok_ms.push_back((now_s() - ts) * 1e3); t_dec += now_s() - ts; break; }
        hist.push_back(feed);
        for (int i = 0; i < hid; i++) last[i] = half2f(P.emb[(size_t)feed * hid + i]);
        M.rows(last.data(), 1);
        double dt = now_s() - ts; t_dec += dt; tok_ms.push_back(dt * 1e3);
        if (verbose) fprintf(stderr, "  token %d -> %d (%.1f ms)\n", step, tok, dt * 1e3);
    }
    if (f5) fclose(f5); if (fl) fclose(fl);
    printf("TOKENS:"); for (int t : outt) printf(" %d", t); printf("\n");
    { Vocab V; std::string vp = vocab_f.empty() ? pkg + "/hf_processor" : vocab_f;
      if (load_vocab(vp, V)) { std::string txt = detok(V, outt); printf("\n===== VAN BAN SINH RA =====\n%s\n===========================\n\n", txt.c_str());
          if (!text_out.empty()) { FILE *ft = fopen(text_out.c_str(), "w"); if (ft) { fprintf(ft, "%s\n", txt.c_str()); fclose(ft); } } }
      else fprintf(stderr, "[text] khong tim thay vocab.json/tokenizer.json o %s -> khong giai ma chu (dung --vocab; hoac giai ma tren Colab)\n", vp.c_str()); }
    if (!out_f.empty()) { FILE *fo = fopen(out_f.c_str(), "w"); for (int t : outt) fprintf(fo, "%d\n", t); fclose(fo); }
    const int nt = (int)tok_ms.size(); Stats d1 = ST;
    std::vector<double> sorted = tok_ms; std::sort(sorted.begin(), sorted.end());
    printf("DECODE: %d token trong %.2f s = %.2f token/s | ms/token: tb %.1f p50 %.1f p90 %.1f max %.1f\n", nt, t_dec, nt / std::max(t_dec, 1e-9), 1e3 * t_dec / std::max(nt, 1),
           nt ? sorted[nt / 2] : 0.0, nt ? sorted[(size_t)(nt * 0.9)] : 0.0, nt ? sorted.back() : 0.0);
    printf("  PL wall %.1f ms/token (phan cung %.1f ms/token, %ld luot) | PS (con lai) %.1f ms/token | fallback: %ld hang chu so, %ld hang bao hoa / %d token\n",
           1e3 * (d1.pl_wall - d0.pl_wall) / std::max(nt, 1), 1e3 * (d1.pl_hw - d0.pl_hw) / std::max(nt, 1), d1.passes - d0.passes,
           1e3 * (t_dec - (d1.pl_wall - d0.pl_wall)) / std::max(nt, 1), d1.rows_dig - d0.rows_dig, d1.rows_sat - d0.rows_sat, nt);
    {
        double tot = 1e3 * t_dec / std::max(nt, 1), sum = 0; printf("  PS chi tiet (ms/token): ");
        const double nt1 = std::max(nt, 1); double pre = 1e3 * (d1.ps_pre - d0.ps_pre) / nt1, rq = 1e3 * (d1.ps_rq - d0.ps_rq) / nt1, i32 = 1e3 * (d1.ps_i32 - d0.ps_i32) / nt1, dg = 1e3 * (d1.ps_dig - d0.ps_dig) / nt1;
        printf("luong tu hoa x+kiem tra ngoai lai %.1f | requant int8+bao hoa %.1f | lm_head int32->float %.1f | chu so (chuan bi+hau xu ly) %.1f", pre, rq, i32, dg); sum += pre + rq + i32 + dg;
        for (int i = 0; i < P_N; i++) { double v = 1e3 * (PRF[i] - PR0[i]) / std::max(nt, 1); printf(" | %s %.1f", P_NAME[i], v); sum += v; }
        double pl = 1e3 * (d1.pl_wall - d0.pl_wall) / std::max(nt, 1);
        printf(" | con lai %.1f   (tong %.1f = PL %.1f + PS %.1f)\n", tot - pl - sum, tot, pl, tot - pl);
        { long n[5] = {0, 0, 0, 0, 0}; for (int tt = 0; tt < 8; tt++) for (int k = 0; k < 5; k++) n[k] += ATT_ENC[tt][0][k];
          if (n[0]) printf("  attention encoder: %ld lan (dau x lop) | NEON %.1f%% (trong do %.1f%% co cot FP) | cham: %ld do hang q ngoai lai, %ld do >64 cot FP\n", n[0], 100.0 * (n[1] + n[2]) / n[0], 100.0 * n[2] / n[0], n[3], n[4]); }
        for (int dc = 0; dc < 2; dc++) { long n[5] = {0, 0, 0, 0, 0}; for (int tt = 0; tt < 8; tt++) for (int k = 0; k < 5; k++) n[k] += ATT_T[tt][dc][k]; if (!n[0]) continue;
            printf("  attention %s: %ld lan (dau x lop) | NEON %.1f%% (trong do %.1f%% co cot FP) | cham: %ld do hang q ngoai lai, %ld do >64 cot FP\n", dc ? "decode " : "prefill", n[0],
                   100.0 * (n[1] + n[2]) / n[0], 100.0 * n[2] / n[0], n[3], n[4]); }
    }
    return 0;
}