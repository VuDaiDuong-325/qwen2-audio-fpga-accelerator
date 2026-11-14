// =============================================================================
// npu_mbench.c - Do thoi gian phan cung cua 1 luot GEMM theo M (1,2,3,4,8,16) va che do dau ra (INT8 / INT32),
//                trong so da 'dc cvac' MOT LAN (skip_clean_b). Tra loi 3 cau hoi cho runtime C++:
//   (1) M=2/3 co ton bang M=1 khong?  (ky vong: co - 1 tile M <= 16, thoi gian bi chan boi luong doc trong so)
//   (2) INT32 co cham hon INT8 khong?
//   (3) chi phi 'dc cvac' trong so moi lan chay (skip_clean_b=0) la bao nhieu.
// Build:  gcc -O2 -Wall -DNPU_DMAHEAP -o npu_mbench npu_mbench.c
// Chay :  sudo env NPU_PL_OK=1 NPU_DDR_MB=32 ./npu_mbench [--repeat 20] [--shapes down,qkv,o,gate_up,lm]
// CHI chay khi khong ai khac dung NPU. Khong doi bitstream/cau hinh he thong. Du lieu ngau nhien (chi do thoi gian).
// =============================================================================
#include "kr260_npu_driver.h"

typedef struct { const char *n; uint32_t K, N; } shape_t;
static const shape_t SH[] = {{"qkv", 896, 1152}, {"o", 896, 896}, {"gate_up", 896, 9728}, {"down", 4864, 896}, {"lm", 896, 20000}};

int main(int argc, char **argv) {
    int rep = 20; const char *want = "qkv,o,gate_up,down";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--repeat") && i + 1 < argc) rep = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) want = argv[++i];
    }
    npu_t d;
    if (npu_open(&d) < 0) return 2;
    uint32_t Ms[] = {1, 2, 3, 4, 8, 16};
    printf("%-8s %6s %5s | %9s %9s %8s | %s\n", "shape", "M", "mode", "hw_ms", "wall_ms", "GB/s(w)", "ghi chu");
    int bad = 0;
    for (size_t si = 0; si < sizeof SH / sizeof SH[0]; si++) {
        const shape_t *s = &SH[si];
        if (!strstr(want, s->n) && strcmp(s->n, "lm")) continue;
        if (!strcmp(s->n, "lm") && !strstr(want, "lm")) continue;
        uint32_t K = s->K, N = s->N;
        npu_layout_t l = npu_layout(16, K, N);
        if (l.total > d.ddr_size) { printf("%-8s bo qua: can %.1f MB > %.1f MB (tang NPU_DDR_MB)\n", s->n, l.total / 1048576.0, d.ddr_size / 1048576.0); continue; }
        int8_t *A = malloc((size_t)16 * K), *B = malloc((size_t)K * N); npu_qparam_t *qp = malloc(sizeof(npu_qparam_t) * N);
        uint32_t r = 12345; for (size_t i = 0; i < (size_t)16 * K; i++) { r = r * 1664525u + 1013904223u; A[i] = (int8_t)(r >> 24); }
        for (size_t i = 0; i < (size_t)K * N; i++) { r = r * 1664525u + 1013904223u; B[i] = (int8_t)(r >> 24); }
        for (uint32_t n = 0; n < N; n++) { qp[n].bias = 0; qp[n].mult = 1; qp[n].shift = 12; qp[n].zp = 0; }
        npu_load_A(&d, &l, A, 16, K); npu_load_B(&d, &l, B, K, N); npu_load_params(&d, &l, qp, N);
        double t0 = npu_now(); npu_cache_clean(d.ddr + l.off_b, (size_t)K * l.ldb); double t_clean = (npu_now() - t0) * 1e3;
        printf("%-8s       dc cvac cua B (%.2f MB): %.2f ms (lam 1 lan)\n", s->n, (double)K * l.ldb / 1048576.0, t_clean);
        int32_t *c1 = malloc(sizeof(int32_t) * N), *cm = malloc(sizeof(int32_t) * 16 * N);
        for (int skip = 1; skip >= 0; skip--) {
            d.skip_clean_b = skip;
            for (int mo = 0; mo < 2; mo++) {
                uint32_t mode = mo ? NPU_OUT_INT32 : NPU_OUT_INT8;
                if (!skip && mo == 0) continue;
                for (size_t mi = 0; mi < sizeof Ms / sizeof Ms[0]; mi++) {
                    uint32_t M = Ms[mi];
                    if (!skip && M != 1) continue;
                    double best_hw = 1e9, best_wall = 1e9;
                    for (int it = 0; it < rep; it++) {
                        double w0 = npu_now();
                        int rc = npu_run(&d, &l, M, K, N, NPU_KDIM_AUTO, mode, 5.0);
                        double w = (npu_now() - w0) * 1e3;
                        if (rc < 0) { printf("  LOI npu_run rc=%d\n", rc); bad++; break; }
                        if (d.last_hw_ms < best_hw) best_hw = d.last_hw_ms;
                        if (w < best_wall) best_wall = w;
                    }
                    const char *note = "";
                    char buf[96]; buf[0] = 0;
                    if (mode == NPU_OUT_INT32 && skip) {            // hang 0 phai y het nhau o moi M (cac hang doc lap)
                        npu_read_C32(&d, &l, cm, M, N);
                        if (M == 1) memcpy(c1, cm, sizeof(int32_t) * N);
                        else if (memcmp(c1, cm, sizeof(int32_t) * N)) { snprintf(buf, sizeof buf, "HANG 0 LECH so voi M=1 !"); bad++; }
                        else snprintf(buf, sizeof buf, "hang 0 = M=1 OK");
                        note = buf;
                    }
                    if (!skip) note = "KHONG skip_clean_b (nhu driver cu: clean B moi lan)";
                    printf("%-8s %6u %5s | %9.3f %9.3f %8.2f | %s\n", s->n, M, mo ? "I32" : "I8", best_hw, best_wall, (double)K * N / best_hw / 1e6, note);
                }
            }
        }
        free(A); free(B); free(qp); free(c1); free(cm);
    }
    npu_close(&d);
    printf(bad ? "\nCO LOI (%d)\n" : "\nXONG\n", bad);
    return bad ? 1 : 0;
}
