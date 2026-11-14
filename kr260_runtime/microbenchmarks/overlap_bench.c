// =============================================================================
// overlap_bench.c - do thoi gian DECODE 1 token (24 lop + lm_head) khi CHONG LAP nap trong so voi tinh toan PL.
//
//   Hien tai moi phep GEMM chay noi tiep: nap A,B,params -> PL tinh -> doc C. Voi decode (M=1) PL chi ~205 ms/token nhung nap
//   trong so ~450-550 ms. O day: trong luc PL tinh phep i, 1 luong CPU nap trong so + params cua phep i+1 vao "slot" kia
//   (vung DDR chia 2 slot). Khong can doi cau hinh board, khong can RTL moi, khong dung CMA.
//
// Dung dung hinh dang that cua 1 token Qwen2 (M=1): moi lop qkv 896x1152, o 896x896, gate_up 896x9728, down 4864x896;
// cuoi cung lm_head 896x156032 chia 10 manh cot (16384). Trong so/dau vao ngau nhien (chi do thoi gian; dung dung duoc
// kiem bang so sanh checksum dau ra cua che do noi tiep va che do chong lap - cung dau vao phai ra cung ket qua).
// KHONG mo phong phan CPU giua cac phep (softmax, RMSNorm...) -> day la can duoi thoi gian chep du lieu; xem --cpu-us.
//
//   gcc -O3 -fopenmp -mcpu=native -Wall -o overlap_bench overlap_bench.c
//   sudo ./overlap_bench [--layers 24] [--tokens 3] [--cpu-us 0] [--kdim auto]
//   Bien moi truong cua driver (NPU_BACKEND, NPU_DDR_PA ...) dung binh thuong.
// Kiem thu khong can phan cung:  gcc -O2 -fopenmp -DNPU_SOFTSIM -o overlap_sim overlap_bench.c && ./overlap_sim --layers 1 --tokens 1
// =============================================================================
#ifdef NPU_SOFTSIM
#define NPU_DDR_SIZE 0x4000000UL
#define npu_open  npu_open_hw_unused
#define npu_run   npu_run_hw_unused
#define npu_start npu_start_hw_unused
#define npu_wait  npu_wait_hw_unused
#define npu_close npu_close_hw_unused
#endif
#include "npu_driver.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef NPU_SOFTSIM
#undef npu_open
#undef npu_run
#undef npu_start
#undef npu_wait
#undef npu_close
static inline int npu_open(npu_t *d) { memset(d, 0, sizeof(*d)); d->ddr = (uint8_t *)calloc(1, NPU_DDR_SIZE); d->ddr_pa = NPU_DDR_PA; d->ddr_size = NPU_DDR_SIZE; d->read_threads = 1; return d->ddr ? 0 : -1; }
static inline void npu_close(npu_t *d) { free(d->ddr); }
static int g_sim_rc;
static inline int npu_start(npu_t *d, const npu_layout_t *l, uint32_t M, uint32_t K, uint32_t N, uint32_t kdim, uint32_t mode) {
    (void)kdim; (void)mode;
    const npu_qparam_t *qp = (const npu_qparam_t *)(d->ddr + l->off_p);
    for (uint32_t m = 0; m < M; m++) {
        const int8_t *a = (const int8_t *)(d->ddr + l->off_a + (uint64_t)m * l->lda);
        for (uint32_t n = 0; n < N; n++) {
            uint32_t acc = 0;
            for (uint32_t k = 0; k < K; k++) acc += (uint32_t)(int32_t)((int32_t)a[k] * (int32_t)(int8_t)d->ddr[l->off_b + (uint64_t)k * l->ldb + n]);
            *(int8_t *)(d->ddr + l->off_c + (uint64_t)m * l->ldc + n) = npu_requant_ref((int32_t)acc, qp[n]);
        }
    }
    d->last_hw_ms = 0.0; g_sim_rc = 0; return 0;
}
static inline int npu_wait(npu_t *d, double t) { (void)d; (void)t; return g_sim_rc; }
static inline int npu_run(npu_t *d, const npu_layout_t *l, uint32_t M, uint32_t K, uint32_t N, uint32_t kdim, uint32_t mode, double t) {
    (void)t; return npu_start(d, l, M, K, N, kdim, mode) ? -1 : npu_wait(d, t);
}
#endif

typedef struct { const char *name; uint32_t K, N; int8_t *B; npu_qparam_t *qp; int8_t *A; } wt_t;
typedef struct { wt_t *w; npu_layout_t l; uint32_t M, K, N; } op_t;

static uint64_t rs = 0x9E3779B97F4A7C15ULL;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static wt_t make_wt(const char *name, uint32_t K, uint32_t N) {
    wt_t w; w.name = name; w.K = K; w.N = N;
    w.B = (int8_t *)malloc((size_t)K * N); w.qp = (npu_qparam_t *)malloc((size_t)N * sizeof(npu_qparam_t)); w.A = (int8_t *)malloc(K);
    for (size_t i = 0; i < (size_t)K * N; i += 8) { uint64_t v = rnd(); size_t n = ((size_t)K * N - i) < 8 ? ((size_t)K * N - i) : 8; memcpy(w.B + i, &v, n); }
    for (uint32_t i = 0; i < K; i++) w.A[i] = (int8_t)(rnd() & 0xFF);
    for (uint32_t n = 0; n < N; n++) { w.qp[n].bias = (int32_t)(rnd() % 2001) - 1000; w.qp[n].mult = 1; w.qp[n].shift = 12; w.qp[n].zp = 0; }
    return w;
}

static uint64_t fnv(uint64_t h, const uint8_t *p, size_t n) { for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; } return h; }
static void spin_us(double us) { if (us <= 0) return; double t = npu_now(); while ((npu_now() - t) * 1e6 < us) { } }

int main(int argc, char **argv) {
    int layers = 24, tokens = 3, lt = 1; double cpu_us = 0; uint32_t kdim = NPU_KDIM_AUTO;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--load-threads") && i + 1 < argc) lt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu-us") && i + 1 < argc) cpu_us = atof(argv[++i]);
        else if (!strcmp(argv[i], "--kdim") && i + 1 < argc) { ++i; kdim = strcmp(argv[i], "auto") ? (uint32_t)atoi(argv[i]) : NPU_KDIM_AUTO; }
        else { fprintf(stderr, "dung: %s [--layers 24] [--tokens 3] [--cpu-us 0] [--kdim auto] [--load-threads 1]\n", argv[0]); return 2; }
    }
    if (lt < 1) lt = 1;
    if (lt > 3) lt = 3;
#ifdef _OPENMP
    omp_set_dynamic(0);
    if (lt > omp_get_max_threads() - 1) lt = omp_get_max_threads() - 1;   // can 1 luong cho PL + lt luong nap
    if (lt < 1) { fprintf(stderr, "can it nhat 2 luong OpenMP (OMP_NUM_THREADS)\n"); return 2; }
#endif
    npu_t dev; if (npu_open(&dev)) return 2;
    dev.load_threads = lt;
    uint64_t slot = (dev.ddr_size / 2) & ~4095ULL;

    // cac loai trong so (tai su dung cho moi lop: nguon nam trong RAM, > cache nen luon doc tu DDR nhu trong suy luan that)
    wt_t qkv = make_wt("qkv", 896, 1152), op = make_wt("o_proj", 896, 896), gu = make_wt("gate_up", 896, 9728),
         dn = make_wt("down", 4864, 896), lmA = make_wt("lm_head[16384]", 896, 16384), lmB = make_wt("lm_head[8576]", 896, 156032 - 9 * 16384);
    wt_t *all[] = {&qkv, &op, &gu, &dn};
    int nops = layers * 4 + 10, k = 0;
    op_t *ops = (op_t *)calloc((size_t)nops, sizeof(op_t));
    for (int L = 0; L < layers; L++) for (int j = 0; j < 4; j++) { ops[k].w = all[j]; k++; }
    for (int c = 0; c < 10; c++) { ops[k].w = (c < 9) ? &lmA : &lmB; k++; }
    double wbytes = 0;
    for (int i = 0; i < nops; i++) {
        op_t *o = &ops[i]; o->M = 1; o->K = o->w->K; o->N = o->w->N;
        o->l = npu_layout(1, o->K, o->N);
        if (o->l.total > slot) { fprintf(stderr, "phep %d can %llu byte > slot %llu\n", i, (unsigned long long)o->l.total, (unsigned long long)slot); return 2; }
        wbytes += (double)o->K * o->N;
    }
    printf("== overlap_bench (nap B bang %d luong): %d phep/token (%d lop + lm_head 10 manh), trong so %.0f MB/token, vung DDR %llu MB (2 slot x %llu MB) ==\n",
           lt, nops, layers, wbytes / 1048576.0, (unsigned long long)(dev.ddr_size >> 20), (unsigned long long)(slot >> 20));

    int8_t *C = (int8_t *)malloc(16384 + 64);
    uint64_t sum_serial = 0, sum_over = 0; double best_s = 1e30, best_o = 1e30, ld = 0, pl = 0, rd = 0, pl_o = 0;

    for (int t = 0; t < tokens; t++) {
        // ---- noi tiep: nap tat ca -> PL -> doc (nhu npu_run_case hien tai) ----
        uint64_t h = 1469598103934665603ULL; double tld = 0, tpl = 0, trd = 0, t0 = npu_now();
        for (int i = 0; i < nops; i++) {
            op_t *o = &ops[i]; double a = npu_now();
            npu_load_A(&dev, &o->l, o->w->A, 1, o->K); npu_load_B(&dev, &o->l, o->w->B, o->K, o->N); npu_load_params(&dev, &o->l, o->w->qp, o->N);
            double b = npu_now();
            if (npu_run(&dev, &o->l, 1, o->K, o->N, kdim, NPU_OUT_INT8, 30.0)) return 2;
            double c = npu_now(); npu_read_C8(&dev, &o->l, C, 1, o->N); h = fnv(h, (uint8_t *)C, o->N); spin_us(cpu_us);
            double e = npu_now(); tld += b - a; tpl += c - b; trd += e - c;
        }
        double ts = npu_now() - t0; sum_serial = h;
        if (ts < best_s) { best_s = ts; ld = tld; pl = tpl; rd = trd; }

        // ---- chong lap: trong luc PL tinh phep i, 1 luong nap trong so phep i+1 vao slot kia ----
        h = 1469598103934665603ULL; t0 = npu_now();
        npu_layout_t l0 = ops[0].l;
        npu_load_B(&dev, &l0, ops[0].w->B, ops[0].K, ops[0].N); npu_load_params(&dev, &l0, ops[0].w->qp, ops[0].N);
        int err = 0; double plsum = 0;
        for (int i = 0; i < nops; i++) {
            op_t *o = &ops[i]; npu_layout_t l = npu_layout_shift(o->l, (i & 1) ? slot : 0);
            npu_load_A(&dev, &l, o->w->A, 1, o->K);                       // dau vao phu thuoc phep truoc -> nap ngay truoc khi chay (rat nho)
            op_t *nx = (i + 1 < nops) ? &ops[i + 1] : NULL;
            npu_layout_t ln; if (nx) ln = npu_layout_shift(nx->l, ((i + 1) & 1) ? slot : 0);
            #pragma omp parallel num_threads(1 + lt)
            {
                int tid = omp_get_thread_num();
                if (tid == 0) { if (npu_start(&dev, &l, 1, o->K, o->N, kdim, NPU_OUT_INT8) || npu_wait(&dev, 30.0)) err = 1; }
                else if (nx) {                                   // luong 1..lt: nap song song cac khoang hang cua B phep ke tiep
                    int j = tid - 1;
                    npu_load_B_rows(&dev, &ln, nx->w->B, nx->N, (uint32_t)((uint64_t)nx->K * j / lt), (uint32_t)((uint64_t)nx->K * (j + 1) / lt));
                    if (j == 0) npu_load_params(&dev, &ln, nx->w->qp, nx->N);
                }
            }
            if (err) return 2;
            plsum += dev.last_hw_ms;
            npu_read_C8(&dev, &l, C, 1, o->N); h = fnv(h, (uint8_t *)C, o->N); spin_us(cpu_us);
        }
        double to = npu_now() - t0; sum_over = h;
        if (to < best_o) { best_o = to; pl_o = plsum * 1e-3; }
        printf("token %d: noi tiep %.1f ms | chong lap %.1f ms | checksum %s\n", t, ts * 1e3, to * 1e3, sum_serial == sum_over ? "KHOP" : "LECH !!!");
        if (sum_serial != sum_over) return 1;
    }
    printf("\nTot nhat: noi tiep %.1f ms/token (nap %.1f | PL %.1f | doc+CPU %.1f)  ->  chong lap %.1f ms/token  (nhanh %.2fx, %.2f -> %.2f token/s)\n",
           best_s * 1e3, ld * 1e3, pl * 1e3, rd * 1e3, best_o * 1e3, best_s / best_o, 1.0 / best_s, 1.0 / best_o);
    printf("Che do chong lap: tong PL %.1f ms (noi tiep %.1f ms) -> phan con lai %.1f ms la dong bo/nap A/doc C/chen CPU (PL cham di neu > noi tiep: tranh chap DDR)\n", pl_o * 1e3, pl * 1e3, (best_o - pl_o) * 1e3);
    printf("Can duoi ly thuyet (PL thuan / nap thuan, cai lon hon): %.1f ms\n", (pl > ld ? pl : ld) * 1e3);
    npu_close(&dev);
    return 0;
}
