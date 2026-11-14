/* =============================================================================
 * ps_bench.c - DO CHI PHI PS (CPU A53) cho 1 token decode cua Qwen2-0.5B-INT8 tren KR260.
 *
 * CHI TINH TREN CPU: KHONG mo /dev/mem, KHONG cham PL, KHONG xin CMA, KHONG can sudo, KHONG can bitstream.
 * An toan tren board dung chung: 1 luong, ~20-30 s, vai MB bo nho.
 *
 *   gcc -O3 -mcpu=cortex-a53 -fopenmp-simd -o ps_bench ps_bench.c -lm
 *   ./ps_bench            # day du (kiem tra dung/sai truoc, roi do)
 *   ./ps_bench -q         # nhanh (~8 s)
 *   ./ps_bench -t         # chi tu kiem tra dung/sai cac kernel (khong do)
 *
 * Do: tung kernel PS cua 1 token (RMSNorm, luong tu hoa, dequant, kiem tra hang ngoai lai, RoPE, attention INT8 theo
 * do dai ngu canh S, SiLU*up, lm_head hau xu ly + argmax) + bang thong luong ma tran-vector fallback FP (int8 -> float).
 * Ket qua: ms PS / token theo S, token/s uoc tinh khi cong voi PL (204.3 ms/token da do).
 * ========================================================================== */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#define HAVE_NEON 1
#elif defined(USE_N2S)
#include "NEON_2_SSE.h"
#define HAVE_NEON 1
#define vcvtnq_s32_f32(a) _mm_cvtps_epi32(a)          /* RNE (MXCSR mac dinh) - chi dung de TEST tren x86 */
#else
#define HAVE_NEON 0
#endif

/* ---------------- kich thuoc model (Qwen2-0.5B ben LLM cua Qwen2-Audio) ---------------- */
enum { HID = 896, NH = 14, NKV = 2, HD = 64, INTER = 4864, NL = 24, VOCAB = 156032, REP = NH / NKV, QKVN = (NH + 2 * NKV) * HD };

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static volatile float g_sink_f; static volatile int g_sink_i;
static uint32_t rs = 12345u;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float frand(float lo, float hi) { return lo + (hi - lo) * (rnd() & 0xFFFFFF) / 16777216.0f; }
static void fill_f(float *p, size_t n, float lo, float hi) { for (size_t i = 0; i < n; i++) p[i] = frand(lo, hi); }
static void fill_i8(int8_t *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = (int8_t)(rnd() & 0xFF); }
static void *amalloc(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, (n + 63) & ~(size_t)63)) { perror("alloc"); exit(1); } return p; }

/* ============================ KERNELS ============================ */
/* RMSNorm (Qwen2): out = w * x * rsqrt(mean(x^2)+eps) */
static void rmsnorm(const float *restrict x, const float *restrict w, float *restrict out, int n, float eps) {
    float ss = 0.f;
#pragma omp simd reduction(+:ss)
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

/* luong tu hoa: clip(rint(x*inv), -128, 127). Ban vo huong lam tham chieu; ban NEON: fcvtns (lam tron ve chan) + sqxtn (bao hoa) */
static void quant_i8_ref(const float *x, float inv, int8_t *out, int n) {
    for (int i = 0; i < n; i++) { float r = nearbyintf(x[i] * inv); r = r > 127.f ? 127.f : (r < -128.f ? -128.f : r); out[i] = (int8_t)r; }
}
static void quant_i8(const float *restrict x, float inv, int8_t *restrict out, int n) {
    int i = 0;
#if HAVE_NEON
    float32x4_t vi = vdupq_n_f32(inv);
    for (; i + 16 <= n; i += 16) {
        int32x4_t a = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i),      vi));
        int32x4_t b = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 4),  vi));
        int32x4_t c = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 8),  vi));
        int32x4_t d = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 12), vi));
        vst1q_s8(out + i, vcombine_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b))), vqmovn_s16(vcombine_s16(vqmovn_s32(c), vqmovn_s32(d)))));
    }
#endif
    for (; i < n; i++) {
        float v = x[i] * inv; v = v > 128.f ? 128.f : (v < -128.f ? -128.f : v); v = (v + 12582912.0f) - 12582912.0f;
        int q = (int)v; out[i] = (int8_t)(q > 127 ? 127 : q);
    }
}
/* dequant: y = q * scale[n] (scale rieng tung kenh dau ra) */
static void dequant_i8(const int8_t *restrict q, const float *restrict sc, float *restrict y, int n) {
    for (int i = 0; i < n; i++) y[i] = (float)q[i] * sc[i];
}
/* kiem tra hang ngoai lai tu INT32: any |(acc+bias)*k| > 127.5, k = sin*sw/sout (tung kenh) -> tra max |.| */
static float flag_maxabs(const int32_t *restrict acc, const int32_t *restrict bias, const float *restrict k, int n) {
    float m = 0.f; int i = 0;
#if HAVE_NEON
    float32x4_t m0 = vdupq_n_f32(0.f), m1 = m0, m2 = m0, m3 = m0;
    for (; i + 16 <= n; i += 16) {
        m0 = vmaxq_f32(m0, vabsq_f32(vmulq_f32(vcvtq_f32_s32(vaddq_s32(vld1q_s32(acc + i),      vld1q_s32(bias + i))),      vld1q_f32(k + i))));
        m1 = vmaxq_f32(m1, vabsq_f32(vmulq_f32(vcvtq_f32_s32(vaddq_s32(vld1q_s32(acc + i + 4),  vld1q_s32(bias + i + 4))),  vld1q_f32(k + i + 4))));
        m2 = vmaxq_f32(m2, vabsq_f32(vmulq_f32(vcvtq_f32_s32(vaddq_s32(vld1q_s32(acc + i + 8),  vld1q_s32(bias + i + 8))),  vld1q_f32(k + i + 8))));
        m3 = vmaxq_f32(m3, vabsq_f32(vmulq_f32(vcvtq_f32_s32(vaddq_s32(vld1q_s32(acc + i + 12), vld1q_s32(bias + i + 12))), vld1q_f32(k + i + 12))));
    }
    float t[4]; vst1q_f32(t, vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3))); m = fmaxf(fmaxf(t[0], t[1]), fmaxf(t[2], t[3]));
#endif
    for (; i < n; i++) { float v = fabsf((float)(acc[i] + bias[i]) * k[i]); m = v > m ? v : m; }
    return m;
}
/* RoPE kieu rotate_half: x[i], x[i+32] */
static void rope(float *restrict x, int heads, const float *restrict c, const float *restrict s) {
    for (int h = 0; h < heads; h++) {
        float *p = x + h * HD;
        for (int i = 0; i < HD / 2; i++) { float a = p[i], b = p[i + HD / 2]; p[i] = a * c[i] - b * s[i]; p[i + HD / 2] = b * c[i] + a * s[i]; }
    }
}
/* exp da thuc (sai so tuong doi ~1e-7), tu vector hoa duoc */
static inline float exp_poly(float x) {
    x = x > 88.f ? 88.f : (x < -87.f ? -87.f : x);
    float t = x * 1.44269504f;
    float nf = (t + 12582912.0f) - 12582912.0f;
    float r = x - nf * 0.693145751953125f - nf * 1.42860677e-6f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f; p = p * r + 8.3334519073e-3f; p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f; p = p * r + 5.0000001201e-1f; p = p * r * r + r + 1.0f;
    int32_t e = ((int32_t)nf + 127) << 23; float sc; __builtin_memcpy(&sc, &e, 4);
    return p * sc;
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
static inline float32x4_t recip4(float32x4_t d) {            /* 1/d: uoc luong + 2 buoc Newton (~24 bit) */
    float32x4_t r = vrecpeq_f32(d); r = vmulq_f32(r, vrecpsq_f32(d, r)); return vmulq_f32(r, vrecpsq_f32(d, r));
}
#endif
static void silu_mul_poly(const float *restrict g, const float *restrict u, float *restrict out, int n) {
    int i = 0;
#if HAVE_NEON
    float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 8 <= n; i += 8) {
        float32x4_t g0 = vld1q_f32(g + i), g1 = vld1q_f32(g + i + 4);
        float32x4_t e0 = exp4(vnegq_f32(g0)), e1 = exp4(vnegq_f32(g1));
        float32x4_t r0 = recip4(vaddq_f32(e0, one)), r1 = recip4(vaddq_f32(e1, one));
        vst1q_f32(out + i,     vmulq_f32(vmulq_f32(g0, r0), vld1q_f32(u + i)));
        vst1q_f32(out + i + 4, vmulq_f32(vmulq_f32(g1, r1), vld1q_f32(u + i + 4)));
    }
#endif
    for (; i < n; i++) out[i] = g[i] / (1.0f + exp_poly(-g[i])) * u[i];
}
static void silu_mul_libm(const float *restrict g, const float *restrict u, float *restrict out, int n) {
    for (int i = 0; i < n; i++) out[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}
static void add_inplace(float *restrict a, const float *restrict b, int n) { for (int i = 0; i < n; i++) a[i] += b[i]; }

/* ---- tich vo huong int8 64 phan tu ---- */
static inline int32_t dot64_s8_ref(const int8_t *a, const int8_t *b) { int32_t s = 0; for (int i = 0; i < 64; i++) s += a[i] * b[i]; return s; }
#if HAVE_NEON
static inline int32_t hsum_s32(int32x4_t v) { int32_t t[4]; vst1q_s32(t, v); return t[0] + t[1] + t[2] + t[3]; }
static inline int32_t dot64_s8(const int8_t *a, const int8_t *b) {
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < 64; i += 16) {
        int8x16_t x = vld1q_s8(a + i), y = vld1q_s8(b + i);
        int16x8_t lo = vmull_s8(vget_low_s8(x), vget_low_s8(y));
        int16x8_t hi = vmull_s8(vget_high_s8(x), vget_high_s8(y));
        acc = vpadalq_s16(acc, lo); acc = vpadalq_s16(acc, hi);
    }
    return hsum_s32(acc);
}
#else
#define dot64_s8 dot64_s8_ref
#endif

/* ---- attention INT8 (decode, M=1): q8[NH][64]; K8,V8 [NKV][S][64]; fac = he so nhan diem; vscale[NKV*64] (kenh) ----
 *  score = (q8 . k8) * fac ; softmax ; P8 = rint(127*e) (vi sp = max/127 = 1/(127*sum)) ; O = (P8 . V8) * sp * vscale
 *  A53 chay TRONG THU TU (in-order): moi vong xu ly 4 vi tri K cung luc (4 chuoi cong doc lap) de che do tre.                      */
static void attn_i8(const int8_t *restrict q8, const int8_t *restrict K8, const int8_t *restrict V8, int S, float fac,
                    const float *restrict vscale, float *restrict O, float *restrict sc /* [S+8] */, int8_t *restrict p8 /* [S+8] */) {
    for (int kv = 0; kv < NKV; kv++) {
        const int8_t *K = K8 + (size_t)kv * S * HD, *V = V8 + (size_t)kv * S * HD;
        for (int hh = 0; hh < REP; hh++) {
            int h = kv * REP + hh; const int8_t *q = q8 + h * HD;
            float mx = -1e30f; int s = 0;
#if HAVE_NEON
            int8x16_t q0 = vld1q_s8(q), q1 = vld1q_s8(q + 16), q2 = vld1q_s8(q + 32), q3 = vld1q_s8(q + 48);
            float32x4_t vfac = vdupq_n_f32(fac), vmx = vdupq_n_f32(-1e30f);
            for (; s + 4 <= S; s += 4) {
                int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
                const int8_t *k0 = K + (size_t)s * HD, *k1 = k0 + HD, *k2 = k1 + HD, *k3 = k2 + HD;
#define QKSTEP(A, KP, J, QQ) { int8x16_t kk = vld1q_s8(KP + 16 * J);                                                              \
                A = vpadalq_s16(A, vmull_s8(vget_low_s8(QQ), vget_low_s8(kk))); A = vpadalq_s16(A, vmull_s8(vget_high_s8(QQ), vget_high_s8(kk))); }
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
            for (; s < S; s++) { float v = (float)dot64_s8(q, K + (size_t)s * HD) * fac; sc[s] = v; mx = v > mx ? v : mx; }
            float sum = 0.f; s = 0;
#if HAVE_NEON
            {
                float32x4_t vsum = vdupq_n_f32(0.f), vmxx = vdupq_n_f32(mx), v127 = vdupq_n_f32(127.0f);
                for (; s + 8 <= S; s += 8) {
                    float32x4_t e0 = exp4(vsubq_f32(vld1q_f32(sc + s), vmxx)), e1 = exp4(vsubq_f32(vld1q_f32(sc + s + 4), vmxx));
                    vsum = vaddq_f32(vsum, vaddq_f32(e0, e1));
                    int16x8_t pp = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(e0, v127))), vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(e1, v127))));
                    vst1_s8(p8 + s, vqmovn_s16(pp));
                }
                float t[4]; vst1q_f32(t, vsum); sum = t[0] + t[1] + t[2] + t[3];
            }
#endif
            for (; s < S; s++) { float e = exp_poly(sc[s] - mx); sum += e; float v = e * 127.0f; v = (v + 12582912.0f) - 12582912.0f; p8[s] = (int8_t)(int)v; }
            if (S & 1) p8[S] = 0;
            float spv = 1.0f / (127.0f * sum);
            float *o = O + h * HD;
#if HAVE_NEON
            int32x4_t acc[16]; for (int j = 0; j < 16; j++) acc[j] = vdupq_n_s32(0);
            for (s = 0; s + 1 < S; s += 2) {
                int8x8_t pa = vdup_n_s8(p8[s]), pb = vdup_n_s8(p8[s + 1]);
                const int8_t *va = V + (size_t)s * HD, *vb = va + HD;
                for (int j = 0; j < 8; j++) {                           /* 8 vector int8x8 = 64 kenh; |p|<=127,|v|<=128 -> 2 tich < 32767 */
                    int16x8_t t = vmull_s8(vld1_s8(va + 8 * j), pa);
                    t = vmlal_s8(t, vld1_s8(vb + 8 * j), pb);
                    acc[2 * j]     = vaddw_s16(acc[2 * j],     vget_low_s16(t));
                    acc[2 * j + 1] = vaddw_s16(acc[2 * j + 1], vget_high_s16(t));
                }
            }
            if (s < S) {
                int8x8_t pa = vdup_n_s8(p8[s]); const int8_t *va = V + (size_t)s * HD;
                for (int j = 0; j < 8; j++) { int16x8_t t = vmull_s8(vld1_s8(va + 8 * j), pa);
                    acc[2 * j] = vaddw_s16(acc[2 * j], vget_low_s16(t)); acc[2 * j + 1] = vaddw_s16(acc[2 * j + 1], vget_high_s16(t)); }
            }
            { float32x4_t vsp = vdupq_n_f32(spv);
              for (int j = 0; j < 16; j++) vst1q_f32(o + 4 * j, vmulq_f32(vmulq_f32(vcvtq_f32_s32(acc[j]), vsp), vld1q_f32(vscale + kv * HD + 4 * j))); }
#else
            int32_t tmp[64]; memset(tmp, 0, sizeof tmp);
            for (int s2 = 0; s2 < S; s2++) { const int8_t *v = V + (size_t)s2 * HD; int p = p8[s2]; for (int d = 0; d < HD; d++) tmp[d] += p * v[d]; }
            for (int d = 0; d < HD; d++) o[d] = (float)tmp[d] * spv * vscale[kv * HD + d];
#endif
        }
    }
}
/* tham chieu (vo huong, libm) de kiem tra */
static void attn_ref(const int8_t *q8, const int8_t *K8, const int8_t *V8, int S, float fac, const float *vscale, float *O) {
    float *sc = malloc(sizeof(float) * S);
    for (int kv = 0; kv < NKV; kv++) for (int hh = 0; hh < REP; hh++) {
        int h = kv * REP + hh; double mx = -1e30;
        for (int s = 0; s < S; s++) { sc[s] = (float)dot64_s8_ref(q8 + h * HD, K8 + ((size_t)kv * S + s) * HD) * fac; if (sc[s] > mx) mx = sc[s]; }
        double sum = 0; for (int s = 0; s < S; s++) { sc[s] = (float)exp(sc[s] - mx); sum += sc[s]; }
        double spv = 1.0 / (127.0 * sum);
        for (int d = 0; d < HD; d++) { long a = 0; for (int s = 0; s < S; s++) a += (long)(int)nearbyintf(sc[s] * 127.0f) * V8[((size_t)kv * S + s) * HD + d];
            O[h * HD + d] = (float)(a * spv * vscale[kv * HD + d]); }
    }
    free(sc);
}

/* ---- ma tran-vector fallback FP: y[N] = sw[N] * sum_k x[k]*B8[k][N] + bias ---- (B8 [K][N] int8, hang-chinh) */
static void matvec_fp_ref(const float *x, const int8_t *B, const float *sw, const float *bias, float *y, int K, int N) {
    for (int n = 0; n < N; n++) { double a = 0; for (int k = 0; k < K; k++) a += (double)x[k] * B[(size_t)k * N + n]; y[n] = (float)(a * sw[n] + bias[n]); }
}
/* khoi 64 cot (= 1 dong cache) giu 16 bo cong trong thanh ghi, doc B theo hang, khong ghi/doc tmp trong vong K (A53 in-order). */
static void matvec_fp(const float *restrict x, const int8_t *restrict B, const float *restrict sw, const float *restrict bias,
                      float *restrict y, int K, int N, float *restrict tmp /* [1024], dung cho phan du */) {
    int n0 = 0;
#if HAVE_NEON
    for (; n0 + 64 <= N; n0 += 64) {
        float32x4_t acc[16]; for (int j = 0; j < 16; j++) acc[j] = vdupq_n_f32(0.f);
        const int8_t *b = B + n0;
        for (int k = 0; k < K; k++, b += N) {
            __builtin_prefetch(b + 8 * (size_t)N);
            float xk = x[k];
            for (int j = 0; j < 4; j++) {
                int8x16_t v = vld1q_s8(b + 16 * j);
                int16x8_t l = vmovl_s8(vget_low_s8(v)), h = vmovl_s8(vget_high_s8(v));
                acc[4 * j]     = vmlaq_n_f32(acc[4 * j],     vcvtq_f32_s32(vmovl_s16(vget_low_s16(l))),  xk);
                acc[4 * j + 1] = vmlaq_n_f32(acc[4 * j + 1], vcvtq_f32_s32(vmovl_s16(vget_high_s16(l))), xk);
                acc[4 * j + 2] = vmlaq_n_f32(acc[4 * j + 2], vcvtq_f32_s32(vmovl_s16(vget_low_s16(h))),  xk);
                acc[4 * j + 3] = vmlaq_n_f32(acc[4 * j + 3], vcvtq_f32_s32(vmovl_s16(vget_high_s16(h))), xk);
            }
        }
        for (int j = 0; j < 16; j++) vst1q_f32(y + n0 + 4 * j, vmlaq_f32(vld1q_f32(bias + n0 + 4 * j), acc[j], vld1q_f32(sw + n0 + 4 * j)));
    }
#endif
    for (; n0 < N; n0 += 1024) {                                  /* phan du (hoac may khong NEON): ban don gian */
        int nb = N - n0 < 1024 ? N - n0 : 1024;
        memset(tmp, 0, sizeof(float) * nb);
        for (int k = 0; k < K; k++) { const int8_t *b = B + (size_t)k * N + n0; float xk = x[k]; for (int n = 0; n < nb; n++) tmp[n] += xk * (float)b[n]; }
        for (int n = 0; n < nb; n++) y[n0 + n] = tmp[n] * sw[n0 + n] + bias[n0 + n];
    }
}

/* ---- fallback SO NGUYEN: x lam tron 12 bit theo hang (|xq|<=2047), w int8, cong int32 (|acc| <= 2047*128*K < 2^31 khi K <= 8192).
        y[n] = acc[n]*xs*sw[n] + bias[n].  NEON: sxtl (int8->int16) + smlal (int16*int16 -> int32, 4 lan/lenh), khong qua float. ---- */
static void fb16_quant(const float *x, int K, int16_t *xq, float *xs_out) {
    float amax = 0.f; int k = 0;
#if HAVE_NEON
    float32x4_t m = vdupq_n_f32(0.f);
    for (; k + 4 <= K; k += 4) m = vmaxq_f32(m, vabsq_f32(vld1q_f32(x + k)));
    float t[4]; vst1q_f32(t, m); amax = fmaxf(fmaxf(t[0], t[1]), fmaxf(t[2], t[3]));
#endif
    for (; k < K; k++) amax = fmaxf(amax, fabsf(x[k]));
    float inv = amax > 0.f ? 2047.0f / amax : 0.f;
    *xs_out = amax > 0.f ? amax / 2047.0f : 1.0f;
    k = 0;
#if HAVE_NEON
    float32x4_t vi = vdupq_n_f32(inv);
    for (; k + 4 <= K; k += 4) vst1_s16(xq + k, vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + k), vi))));
#endif
    for (; k < K; k++) xq[k] = (int16_t)nearbyintf(x[k] * inv);
}
static void matvec_fb16_ref(const float *x, const int8_t *B, const float *sw, const float *bias, float *y, int K, int N) {
    int16_t xq[8192]; float xs; fb16_quant(x, K, xq, &xs);               /* cung buoc luong tu voi ban NEON; tong chinh xac bang int64 */
    for (int n = 0; n < N; n++) { long long a = 0; for (int k = 0; k < K; k++) a += (long long)xq[k] * B[(size_t)k * N + n]; y[n] = (float)((double)a * xs * sw[n] + bias[n]); }
}
static void matvec_fb16(const float *restrict x, const int8_t *restrict B, const float *restrict sw, const float *restrict bias,
                        float *restrict y, int K, int N, float *restrict tmp) {
    (void)tmp;
    static int16_t xq[8192]; float xs; fb16_quant(x, K, xq, &xs);
    int n0 = 0;
#if HAVE_NEON
    float32x4_t vxs = vdupq_n_f32(xs);
    for (; n0 + 32 <= N; n0 += 32) {
        int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        const int8_t *b = B + n0;
        for (int k = 0; k < K; k++, b += N) {
            __builtin_prefetch(b + 8 * (size_t)N);
            int16_t xk = xq[k];
            int8x16_t v0 = vld1q_s8(b), v1 = vld1q_s8(b + 16);
            int16x8_t l0 = vmovl_s8(vget_low_s8(v0)), h0 = vmovl_s8(vget_high_s8(v0)), l1 = vmovl_s8(vget_low_s8(v1)), h1 = vmovl_s8(vget_high_s8(v1));
            a0 = vmlal_n_s16(a0, vget_low_s16(l0), xk); a1 = vmlal_n_s16(a1, vget_high_s16(l0), xk);
            a2 = vmlal_n_s16(a2, vget_low_s16(h0), xk); a3 = vmlal_n_s16(a3, vget_high_s16(h0), xk);
            a4 = vmlal_n_s16(a4, vget_low_s16(l1), xk); a5 = vmlal_n_s16(a5, vget_high_s16(l1), xk);
            a6 = vmlal_n_s16(a6, vget_low_s16(h1), xk); a7 = vmlal_n_s16(a7, vget_high_s16(h1), xk);
        }
        int32x4_t r[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
        for (int j = 0; j < 8; j++) vst1q_f32(y + n0 + 4 * j, vmlaq_f32(vld1q_f32(bias + n0 + 4 * j), vmulq_f32(vcvtq_f32_s32(r[j]), vxs), vld1q_f32(sw + n0 + 4 * j)));
    }
#endif
    for (; n0 < N; n0++) { int32_t a = 0; for (int k = 0; k < K; k++) a += (int32_t)xq[k] * B[(size_t)k * N + n0]; y[n0] = (float)a * xs * sw[n0] + bias[n0]; }
}
/* lm_head hau xu ly: y = acc*scale[n], argmax, (phat lap lai ap sau). Tra argmax. */
static int lm_post(const int32_t *restrict acc, const float *restrict scale, int n, float *restrict y) {
    int i = 0;
#if HAVE_NEON
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y + i,     vmulq_f32(vcvtq_f32_s32(vld1q_s32(acc + i)),     vld1q_f32(scale + i)));
        vst1q_f32(y + i + 4, vmulq_f32(vcvtq_f32_s32(vld1q_s32(acc + i + 4)), vld1q_f32(scale + i + 4)));
    }
#endif
    for (; i < n; i++) y[i] = (float)acc[i] * scale[i];
    /* argmax (lan dau xuat hien neu bang nhau) */
    float m = y[0]; int im = 0; i = 0;
#if HAVE_NEON
    {
        float32x4_t bv = vdupq_n_f32(-3.0e38f); uint32x4_t bi = vdupq_n_u32(0);
        uint32_t lane0[4] = {0, 1, 2, 3}; uint32x4_t vi = vld1q_u32(lane0), four = vdupq_n_u32(4);
        for (; i + 4 <= n; i += 4) { float32x4_t v = vld1q_f32(y + i); uint32x4_t gt = vcgtq_f32(v, bv); bv = vbslq_f32(gt, v, bv); bi = vbslq_u32(gt, vi, bi); vi = vaddq_u32(vi, four); }
        float tv[4]; uint32_t ti[4]; vst1q_f32(tv, bv); vst1q_u32(ti, bi);
        im = 0; m = -3.0e38f;
        for (int l = 0; l < 4; l++) { int idx = (int)ti[l]; if (tv[l] > m || (tv[l] == m && idx < im)) { m = tv[l]; im = idx; } }
    }
#endif
    for (; i < n; i++) if (y[i] > m) { m = y[i]; im = i; }
    return im;
}

/* ============================ TU KIEM TRA ============================ */
static int selftest(void) {
    int bad = 0;
    { /* exp_poly */
        double me = 0; for (float x = -80; x < 20; x += 0.37f) { double e = fabs(exp_poly(x) - expf(x)) / expf(x); if (e > me) me = e; }
        printf("  exp_poly: sai so tuong doi lon nhat vs expf = %.2e %s\n", me, me < 5e-6 ? "OK" : "SAI"); bad += me >= 5e-6;
    }
    { /* quant */
        float x[1003]; int8_t q[1003], q2[1003]; fill_f(x, 1003, -300, 300); for (int i = 0; i < 20; i++) x[i] = i - 10 + 0.5f;   /* gom ca nua so */
        x[40] = 1e9f; x[41] = -1e9f; x[42] = 127.5f; x[43] = -128.5f; x[44] = 128.5f; x[45] = -127.5f;
        float inv = 1.0f; quant_i8(x, inv, q, 1003); quant_i8_ref(x, inv, q2, 1003); int nb = 0;
        for (int i = 0; i < 1003; i++) nb += q[i] != q2[i];
        inv = 1.0f / 2.3f; quant_i8(x, inv, q, 1003); quant_i8_ref(x, inv, q2, 1003);
        for (int i = 0; i < 1003; i++) nb += q[i] != q2[i];
        printf("  quant_i8 vs clip(rint): %d sai %s\n", nb, nb ? "SAI" : "OK"); bad += nb != 0;
    }
    { /* flag_maxabs + lm_post */
        enum { NN = 1003 }; static int32_t a[NN], b[NN]; static float k[NN], y1[NN], y2[NN];
        for (int i = 0; i < NN; i++) { a[i] = (int32_t)(rnd() & 0xFFFFF) - 0x80000; b[i] = (int32_t)(rnd() & 0xFF) - 128; }
        fill_f(k, NN, 0.0001f, 0.01f);
        double mr = 0; for (int i = 0; i < NN; i++) mr = fmax(mr, fabs((double)(a[i] + b[i]) * k[i]));
        float mf = flag_maxabs(a, b, k, NN); int ok = fabs(mf - mr) <= 1e-5 * mr; printf("  flag_maxabs: %.6g vs %.6g %s\n", mf, mr, ok ? "OK" : "SAI"); bad += !ok;
        int nbad = 0;
        for (int t = 0; t < 40; t++) {
            int n = 1 + (int)(rnd() % NN); for (int i = 0; i < n; i++) k[i] = frand(-0.01f, 0.01f);
            if (t % 4 == 1) { for (int i = 0; i < n; i++) a[i] = 7; }                       /* hoa tat ca bang -> lan dau */
            int im = lm_post(a, k, n, y1); float m = (float)a[0] * k[0]; int jr = 0; for (int i = 0; i < n; i++) { y2[i] = (float)a[i] * k[i]; if (y2[i] > m) { m = y2[i]; jr = i; } }
            nbad += im != jr; for (int i = 0; i < n; i++) nbad += y1[i] != y2[i];
            for (int i = 0; i < NN; i++) a[i] = (int32_t)(rnd() & 0xFFFFF) - 0x80000;
        }
        printf("  lm_post (dequant+argmax, ties, n ngau nhien): %d sai %s\n", nbad, nbad ? "SAI" : "OK"); bad += nbad != 0;
    }
    { /* dot64 */
        int8_t a[64], b[64]; int nb = 0; for (int t = 0; t < 2000; t++) { fill_i8(a, 64); fill_i8(b, 64); nb += dot64_s8(a, b) != dot64_s8_ref(a, b); }
        printf("  dot64_s8 vs tham chieu: %d sai %s\n", nb, nb ? "SAI" : "OK"); bad += nb != 0;
    }
    { /* attention, S le/chan, gom ca cuc tri +-127/-128 */
        int Ss[] = {1, 2, 3, 4, 5, 7, 8, 9, 33, 64, 129, 300};
        for (unsigned si = 0; si < sizeof Ss / sizeof *Ss; si++) {
            int S = Ss[si]; int8_t q[NH * HD]; int8_t *K = amalloc((size_t)NKV * S * HD), *V = amalloc((size_t)NKV * S * HD);
            float vs[NKV * HD], O1[NH * HD], O2[NH * HD], sc[512]; int8_t p8[512];
            fill_i8(q, NH * HD); fill_i8(K, (size_t)NKV * S * HD); fill_i8(V, (size_t)NKV * S * HD); fill_f(vs, NKV * HD, 0.01f, 0.05f);
            for (int i = 0; i < 64; i++) { V[i] = -128; if (S > 1) V[HD + i] = 127; }
            float fac = 0.0007f; attn_i8(q, K, V, S, fac, vs, O1, sc, p8); attn_ref(q, K, V, S, fac, vs, O2);
            double md = 0, mo = 0; for (int i = 0; i < NH * HD; i++) { md = fmax(md, fabs(O1[i] - O2[i])); mo = fmax(mo, fabs(O2[i])); }
            int ok = md <= 2e-3 * (mo + 1e-6) + 1e-4; printf("  attn_i8 S=%-4d max|d|=%.2e (|O|max=%.2e) %s\n", S, md, mo, ok ? "OK" : "SAI"); bad += !ok;
            free(K); free(V);
        }
    }
    { /* matvec fp */
        int K = 70, N = 2500; float x[70], sw[2500], bi[2500], y1[2500], y2[2500], tmp[1024]; int8_t *B = amalloc((size_t)K * N);
        fill_f(x, K, -3, 3); fill_f(sw, N, 0.001f, 0.01f); fill_f(bi, N, -1, 1); fill_i8(B, (size_t)K * N);
        matvec_fp(x, B, sw, bi, y1, K, N, tmp); matvec_fp_ref(x, B, sw, bi, y2, K, N);
        double md = 0; for (int i = 0; i < N; i++) md = fmax(md, fabs(y1[i] - y2[i]));
        printf("  matvec_fp max|d|=%.2e %s\n", md, md < 1e-3 ? "OK" : "SAI"); bad += md >= 1e-3; free(B);
    }
    { /* fallback so nguyen 12 bit: (1) trung khit tham chieu so nguyen, (2) sai so so voi FP64 */
        int K = 133, N = 2501; float x[133], sw[2501], bi[2501], y1[2501], y2[2501], y3[2501], tmp[1024]; int8_t *B = amalloc((size_t)K * N);
        fill_f(x, K, -3, 3); x[7] = 40.f; fill_f(sw, N, 0.001f, 0.01f); fill_f(bi, N, -1, 1); fill_i8(B, (size_t)K * N);
        matvec_fb16(x, B, sw, bi, y1, K, N, tmp); matvec_fb16_ref(x, B, sw, bi, y2, K, N); matvec_fp_ref(x, B, sw, bi, y3, K, N);
        double md = 0, mo = 0, me = 0; for (int i = 0; i < N; i++) { md = fmax(md, fabs(y1[i] - y2[i])); me = fmax(me, fabs(y1[i] - y3[i])); mo = fmax(mo, fabs(y3[i])); }
        printf("  matvec_fb16: vs tham chieu so nguyen max|d|=%.2e %s | vs FP64 max|d|=%.2e (|y|max=%.2f, %.1e tuong doi) %s\n", md, md < 1e-4 * mo ? "OK" : "SAI", me, mo, me / mo, me < 2e-3 * mo ? "OK" : "SAI");
        bad += !(md < 1e-4 * mo) + !(me < 2e-3 * mo); free(B);
    }
    { /* rmsnorm + silu + lm_post (so voi tham chieu double) */
        float x[HID], w[HID], o[HID]; fill_f(x, HID, -5, 5); fill_f(w, HID, 0.5f, 1.5f); rmsnorm(x, w, o, HID, 1e-6f);
        double ss = 0; for (int i = 0; i < HID; i++) ss += (double)x[i] * x[i]; double r = 1 / sqrt(ss / HID + 1e-6), md = 0;
        for (int i = 0; i < HID; i++) md = fmax(md, fabs(o[i] - x[i] * r * w[i]));
        printf("  rmsnorm max|d|=%.2e %s\n", md, md < 1e-4 ? "OK" : "SAI"); bad += md >= 1e-4;
        float g[INTER], u[INTER], a[INTER], b[INTER]; fill_f(g, INTER, -12, 12); fill_f(u, INTER, -2, 2); silu_mul_poly(g, u, a, INTER); silu_mul_libm(g, u, b, INTER);
        md = 0; for (int i = 0; i < INTER; i++) md = fmax(md, fabs(a[i] - b[i]) / (fabs(b[i]) + 1e-3)); printf("  silu_mul poly vs libm: %.2e %s\n", md, md < 1e-4 ? "OK" : "SAI"); bad += md >= 1e-4;
    }
    printf("TU KIEM TRA: %s\n", bad ? "CO LOI - DUNG, khong tin so do" : "TAT CA DUNG");
    return bad;
}

/* ============================ DO THOI GIAN ============================ */
typedef void (*fn_t)(void *);
static double bench_us(fn_t f, void *ctx, double budget_s) {   /* min cua 5 lan do, moi lan >= budget/5 */
    f(ctx); f(ctx);
    double best = 1e30; long it = 1;
    for (int trial = 0; trial < 5; trial++) {
        double t0, dt;
        for (;;) { t0 = now_s(); for (long i = 0; i < it; i++) f(ctx); dt = now_s() - t0; if (dt >= budget_s / 5 || it > (1L << 26)) break; it *= 2; }
        double us = dt / it * 1e6; if (us < best) best = us;
    }
    return best;
}

static struct {
    float *x, *w, *y, *y2, *g, *u, *c, *s, *k, *vs, *sc, *tmp, *scale; int8_t *q, *q2, *K8, *V8, *p8, *B; int32_t *acc, *bias; int S, K, N; float *lm_y;
} G;
static void f_rms(void *c) { (void)c; rmsnorm(G.x, G.w, G.y, HID, 1e-6f); g_sink_f = G.y[3]; }
static void f_q896(void *c) { (void)c; quant_i8(G.x, 0.37f, G.q, HID); g_sink_i = G.q[5]; }
static void f_q4864(void *c) { (void)c; quant_i8(G.g, 0.37f, G.q, INTER); g_sink_i = G.q[5]; }
static void f_dq(void *c) { int n = (int)(intptr_t)c; dequant_i8(G.q2, G.k, G.y, n); g_sink_f = G.y[3]; }
static void f_flag(void *c) { int n = (int)(intptr_t)c; g_sink_f = flag_maxabs(G.acc, G.bias, G.k, n); }
static void f_rope(void *c) { (void)c; rope(G.y, NH + NKV, G.c, G.s); g_sink_f = G.y[3]; }
static void f_silu_p(void *c) { (void)c; silu_mul_poly(G.g, G.u, G.y2, INTER); g_sink_f = G.y2[3]; }
static void f_silu_l(void *c) { (void)c; silu_mul_libm(G.g, G.u, G.y2, INTER); g_sink_f = G.y2[3]; }
static void f_add(void *c) { (void)c; add_inplace(G.y, G.x, HID); g_sink_f = G.y[3]; }
static void f_attn(void *c) { (void)c; attn_i8(G.q, G.K8, G.V8, G.S, 0.0007f, G.vs, G.y, G.sc, G.p8); g_sink_f = G.y[3]; }
static void f_mv(void *c) { (void)c; matvec_fp(G.x, G.B, G.scale, G.k, G.y2, G.K, G.N, G.tmp); g_sink_f = G.y2[3]; }
static void f_mv16(void *c) { (void)c; matvec_fb16(G.x, G.B, G.scale, G.k, G.y2, G.K, G.N, G.tmp); g_sink_f = G.y2[3]; }
static void f_lm(void *c) { (void)c; g_sink_i = lm_post(G.acc, G.k, VOCAB, G.lm_y); }
static void f_ev(void *c) { (void)c; /* doc 640 KB (C32 cua lm_head) */ long s = 0; for (int i = 0; i < VOCAB; i += 16) s += G.acc[i]; g_sink_i = (int)s; }

static double cpu_freq_mhz(void) { FILE *f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r"); double v = -1; if (f) { if (fscanf(f, "%lf", &v) == 1) v /= 1000.0; fclose(f); } return v; }

int main(int argc, char **argv) {
    int quick = 0, test_only = 0;
    for (int i = 1; i < argc; i++) { if (!strcmp(argv[i], "-q")) quick = 1; else if (!strcmp(argv[i], "-t")) test_only = 1; }
    printf("ps_bench: NEON=%d  CPU: %s\n", HAVE_NEON,
#if defined(__aarch64__)
           "aarch64"
#else
           "KHONG phai aarch64 (so do chi mang tinh tham khao)"
#endif
    );
    printf("  compiler: %s | ", __VERSION__);
#ifdef __OPTIMIZE__
    printf("toi uu: BAT (__OPTIMIZE__)");
#else
    printf("toi uu: TAT  <<< CANH BAO: bien dich KHONG co -O2/-O3, so do vo nghia, hay bien dich lai dung lenh o dau file >>>");
#endif
    printf("\n");
    double mhz = cpu_freq_mhz(); if (mhz > 0) printf("  xung CPU0 ~ %.0f MHz\n", mhz);
    double la[3] = {0}; if (getloadavg(la, 3) > 0) printf("  load average: %.2f %.2f %.2f  (cao -> co nguoi khac dang dung CPU, so do bi cham di)\n", la[0], la[1], la[2]);
    printf("[1] tu kiem tra kernel\n");
    if (selftest()) return 1;
    if (test_only) return 0;

    double bud = quick ? 0.25 : 0.6;
    int SMAX = 1024;
    G.x = amalloc(4 * INTER * 4); G.w = amalloc(4 * INTER); G.y = amalloc(4 * 12000); G.y2 = amalloc(4 * INTER); G.g = amalloc(4 * INTER); G.u = amalloc(4 * INTER);
    G.c = amalloc(4 * 64); G.s = amalloc(4 * 64); G.k = amalloc(4 * VOCAB); G.vs = amalloc(4 * NKV * HD); G.sc = amalloc(4 * (SMAX + 4)); G.tmp = amalloc(4 * 1024); G.scale = amalloc(4 * VOCAB);
    G.q = amalloc(INTER + 64); G.q2 = amalloc(12000); G.K8 = amalloc((size_t)NKV * SMAX * HD); G.V8 = amalloc((size_t)NKV * SMAX * HD); G.p8 = amalloc(SMAX + 8);
    G.acc = amalloc(4 * VOCAB); G.bias = amalloc(4 * VOCAB); G.lm_y = amalloc(4 * VOCAB);
    fill_f(G.x, HID, -3, 3); fill_f(G.w, HID, 0.5f, 1.5f); fill_f(G.g, INTER, -8, 8); fill_f(G.u, INTER, -2, 2); fill_f(G.c, 64, -1, 1); fill_f(G.s, 64, -1, 1);
    fill_f(G.k, VOCAB, 0.001f, 0.01f); fill_f(G.vs, NKV * HD, 0.01f, 0.05f); fill_f(G.scale, VOCAB, 0.001f, 0.01f); fill_f(G.y, 12000, -1, 1);
    fill_i8(G.q2, 12000); fill_i8(G.K8, (size_t)NKV * SMAX * HD); fill_i8(G.V8, (size_t)NKV * SMAX * HD); fill_i8((int8_t *)G.q, HID);
    for (int i = 0; i < VOCAB; i++) { G.acc[i] = (int32_t)(rnd() & 0xFFFFF) - 0x80000; G.bias[i] = (int32_t)(rnd() & 0xFF) - 128; }

    printf("\n[2] do tung kernel (us/lan goi, min cua 5 lan)\n");
    double t_rms = bench_us(f_rms, 0, bud), t_q896 = bench_us(f_q896, 0, bud), t_q4864 = bench_us(f_q4864, 0, bud);
    int dqn[4] = {QKVN, HID, 2 * INTER, HID};
    double t_dq[4], t_fl[4];
    for (int i = 0; i < 4; i++) { t_dq[i] = bench_us(f_dq, (void *)(intptr_t)dqn[i], bud); t_fl[i] = bench_us(f_flag, (void *)(intptr_t)dqn[i], bud); }
    double t_rope = bench_us(f_rope, 0, bud), t_sp = bench_us(f_silu_p, 0, bud), t_sl = bench_us(f_silu_l, 0, bud), t_add = bench_us(f_add, 0, bud);
    double t_lm = bench_us(f_lm, 0, bud), t_ev = bench_us(f_ev, 0, bud);
    printf("  rmsnorm 896              %8.2f\n  quant_i8 896             %8.2f\n  quant_i8 4864            %8.2f\n", t_rms, t_q896, t_q4864);
    const char *nm[4] = {"qkv(1152)", "o_proj(896)", "gate_up(9728)", "down(896)"};
    for (int i = 0; i < 4; i++) printf("  dequant %-13s    %8.2f   | kiem tra ngoai lai tu INT32 %8.2f\n", nm[i], t_dq[i], t_fl[i]);
    printf("  rope q14+k2              %8.2f\n  silu*up 4864 (poly exp)  %8.2f\n  silu*up 4864 (libm expf) %8.2f\n  cong residual 896        %8.2f\n", t_rope, t_sp, t_sl, t_add);
    printf("  lm_head: dequant+argmax 156032 (doc C32 cache nong) %8.2f   | doc 624KB tu DDR ~ %8.2f\n", t_lm, t_ev);

    int Ss[] = {32, 64, 128, 256, 512, 1024}; double t_att[6];
    printf("\n[3] attention INT8 1 lop (14 dau, 2 kv) theo do dai ngu canh S (us/lop)\n");
    for (int i = 0; i < 6; i++) { G.S = Ss[i]; t_att[i] = bench_us(f_attn, 0, bud); printf("  S=%-5d %9.2f us   (%.2f ns/ (dau*vi tri))\n", Ss[i], t_att[i], t_att[i] * 1e3 / (NH * Ss[i])); }

    printf("\n[4] ma tran-vector fallback FP (hang ngoai lai tinh tren PS: int8 -> float) - thong luong\n");
    struct { const char *n; int K, N; } mv[] = {{"qkv 896x1152", HID, QKVN}, {"o_proj 896x896", HID, HID}, {"gate_up 896x9728", HID, 2 * INTER}, {"down 4864x896", INTER, HID}};
    int8_t *Bm = amalloc((size_t)HID * 2 * INTER > (size_t)INTER * HID ? (size_t)HID * 2 * INTER : (size_t)INTER * HID); fill_i8(Bm, (size_t)INTER * HID);
    G.B = Bm; double mv_us[4];
    for (int i = 0; i < 4; i++) { G.K = mv[i].K; G.N = mv[i].N; mv_us[i] = bench_us(f_mv, 0, bud); printf("  %-18s %10.1f us  = %.2f GB/s (trong so int8)\n", mv[i].n, mv_us[i], (double)mv[i].K * mv[i].N / mv_us[i] / 1e3); }
    double bw = (double)HID * 2 * INTER / mv_us[2] / 1e3;           /* GB/s */
    printf("  -- fallback SO NGUYEN 12 bit (NEON smlal), cung kich thuoc --\n");
    double mv16_us[4];
    for (int i = 0; i < 4; i++) { G.K = mv[i].K; G.N = mv[i].N; mv16_us[i] = bench_us(f_mv16, 0, bud); printf("  %-18s %10.1f us  = %.2f GB/s (trong so int8) | nhanh %.1fx\n", mv[i].n, mv16_us[i], (double)mv[i].K * mv[i].N / mv16_us[i] / 1e3, mv_us[i] / mv16_us[i]); }
    double bw16 = (double)HID * 2 * INTER / mv16_us[2] / 1e3;
    printf("  => lm_head 896x156032 (140 MB) neu 1 hang bi ngoai lai: ~ %.0f ms\n", 896.0 * 156032 / (bw * 1e9) * 1e3);

    printf("\n[5] TONG PS / token (don luong, chua tinh fallback hang ngoai lai & chi phi khoi chay NPU)\n");
    double per_layer_const = 2 * t_rms + 3 * t_q896 + t_q4864 + t_dq[0] + t_dq[1] + t_dq[2] + t_dq[3] + t_fl[0] + t_fl[1] + t_fl[2] + t_fl[3] + t_rope + 2 * t_add;
    for (int pass = 0; pass < 2; pass++) {
        double silu = pass ? t_sl : t_sp;
        printf("  exp = %s:\n", pass ? "libm expf (don gian)" : "da thuc vector (khuyen nghi)");
        printf("    %-6s %12s %12s %12s %12s\n", "S", "PS ms/tok", "PL ms/tok", "tong ms", "token/s");
        for (int i = 0; i < 6; i++) {
            double ps = (NL * (per_layer_const + silu + t_att[i]) + t_rms + t_lm + t_ev) / 1e3;
            double pl = 204.3;
            printf("    %-6d %12.2f %12.1f %12.1f %12.2f\n", Ss[i], ps, pl, ps + pl, 1000.0 / (ps + pl));
        }
    }
    printf("\n  (PL 204.3 ms/token = do that tren KR260: attention-proj 20.3 + MLP 128.5 + lm_head 55.5)\n");

    /* [6] chi phi fallback KY VONG / token decode, tu ti le do bang ref_runtime.py golden --fbstats (5 clip, 227 token decode, fb-scope member) */
    {
        /* {GEMM, so lan goi/token, p (co FB), MB/su kien, index mv[]} ; lm_head ngoai le */
        struct { const char *n; double calls, p, mb; int idx; } ev[] = {
            {"qkv", NL, 0.0075, 1.03, 0}, {"o_proj", NL, 0.0114, 0.80, 1}, {"gate_up", NL, 0.0039, 5.81, 2}, {"down_proj", NL, 0.0790, 4.36, 3}, {"lm_head", 1, 0.0044, 139.80, 2}};
        printf("\n[6] fallback hang ngoai lai KY VONG / token decode (ti le tu fbstats: 227 token, 5 clip)\n");
        printf("  %-10s %10s %9s %11s %14s %14s\n", "GEMM", "su kien/tok", "MB/su ki", "FP (ms/tok)", "int12 (ms/tok)", "int12 1 su kien");
        double tot_fp = 0, tot_i = 0;
        for (int i = 0; i < 5; i++) {
            double e = ev[i].calls * ev[i].p, bwf = (double)mv[ev[i].idx].K * mv[ev[i].idx].N / mv_us[ev[i].idx] / 1e3, bwi = (double)mv[ev[i].idx].K * mv[ev[i].idx].N / mv16_us[ev[i].idx] / 1e3;
            if (i == 4) { bwf = bw; bwi = bw16; }
            double t_fp = e * ev[i].mb / bwf, t_i = e * ev[i].mb / bwi;               /* MB / (GB/s) = ms */
            tot_fp += t_fp; tot_i += t_i;
            printf("  %-10s %10.3f %9.2f %11.2f %14.2f %12.1f ms\n", ev[i].n, e, ev[i].mb, t_fp, t_i, ev[i].mb / bwi);
        }
        printf("  %-10s %10s %9s %11.2f %14.2f\n", "TONG", "", "", tot_fp, tot_i);
        printf("  => token/s (S=256, PS da thuc) : FP %.2f | int12 %.2f   (PL 204.3 ms; chua tinh khoi chay NPU)\n",
               1000.0 / (204.3 + (NL * (per_layer_const + t_sp + t_att[3]) + t_rms + t_lm + t_ev) / 1e3 + tot_fp),
               1000.0 / (204.3 + (NL * (per_layer_const + t_sp + t_att[3]) + t_rms + t_lm + t_ev) / 1e3 + tot_i));
    }
    printf("  Phan con thieu: (a) hang ngoai lai ~1%% hang/GEMM -> them ~ p x (byte trong so)/%.2f GB/s;  (b) khoi chay NPU + bao tri cache + doc C32.\n", bw);
    return 0;
}