// =============================================================================
// npu_run_case.c - Chay 1 phep GEMM (A[MxK] x B[KxN] + requant theo kenh) tu FILE tren KV260,
//                  so sanh bit-true voi file golden do Python (Kaggle) xuat ra.
//
// Build tren board:   gcc -O2 -o npu_run_case npu_run_case.c
// Chay (can root):    sudo ./npu_run_case --A fc1_A.bin --B fc1_B.bin --params fc1_params.bin
//                          --M 1500 --K 1280 --N 5120 --golden8 golden_fc1_C8.bin
//                          --golden32 golden_fc1_C32.bin --out8 hw_fc1_C8.bin
//
// Tuy chon:
//   --kdim auto|N        k_dim ghi vao thanh ghi KDIM (luy thua 2, <=256). Mac dinh auto = npu_pick_kdim(K)
//   --mode both|int8|int32   (mac dinh both)  -> out_mode 3 | 1 | 2
//   --out8 / --out32     ghi ket qua phan cung ra file
//   --golden8/--golden32 so sanh bit-true (file int8 / int32 little-endian, row-major MxN)
//   --lut lut.bin --out_lut g.bin [--golden_lut g_ref.bin]
//                        ap LUT 256 muc (index = c8+128) len ket qua INT8 -> dung cho GELU cua FFN
//   --repeat R           chay R lan (lan dau + nhung lan sau), in thoi gian HW min/avg
//   --verbose
// Ma thoat: 0 = OK/bit-true, 1 = lech bit, 2 = loi phan cung/tham so.
// Dong cuoi cung cua stdout la 1 dong JSON tom tat (de run_on_kv260.py doc).
//
// Bien dich che do mo phong phan mem (KHONG can phan cung, chi de kiem tra luong chuong trinh):
//   gcc -O2 -DNPU_SOFTSIM -o npu_run_case_sim npu_run_case.c
// =============================================================================
#ifdef NPU_SOFTSIM
#define NPU_DDR_SIZE 0x8000000UL
#define npu_open  npu_open_hw_unused
#define npu_run   npu_run_hw_unused
#define npu_close npu_close_hw_unused
#endif

#include "kr260_npu_driver.h"

#ifdef NPU_SOFTSIM
#undef npu_open
#undef npu_run
#undef npu_close
static inline int npu_open(npu_t *d) {
    memset(d, 0, sizeof(*d));
    d->ddr = (uint8_t *)calloc(1, NPU_DDR_SIZE);
    d->ddr_pa = NPU_DDR_PA;
    d->ddr_size = NPU_DDR_SIZE;
    return d->ddr ? 0 : -1;
}
static inline void npu_close(npu_t *d) { free(d->ddr); }
// Mo phong phan mem: acc INT32 wrap (uint32) + npu_requant_ref cua driver.
static inline int npu_run(npu_t *d, const npu_layout_t *l, uint32_t M, uint32_t K, uint32_t N,
                          uint32_t kdim, uint32_t mode, double timeout_s) {
    (void)timeout_s;
    if (kdim == NPU_KDIM_AUTO) kdim = npu_pick_kdim(K);
    if (kdim < 1 || kdim > 256 || (kdim & (kdim - 1)) || !(mode & 3)) return -1;
    const npu_qparam_t *qp = (const npu_qparam_t *)(d->ddr + l->off_p);
    for (uint32_t m = 0; m < M; m++) {
        const int8_t *a = (const int8_t *)(d->ddr + l->off_a + (uint64_t)m * l->lda);
        for (uint32_t n = 0; n < N; n++) {
            uint32_t acc = 0;
            for (uint32_t k = 0; k < K; k++) {
                int8_t b = (int8_t)d->ddr[l->off_b + (uint64_t)k * l->ldb + n];
                acc += (uint32_t)(int32_t)((int32_t)a[k] * (int32_t)b);
            }
            if (mode & NPU_OUT_INT32)
                memcpy(d->ddr + l->off_c32 + (uint64_t)m * l->ldc32 + (uint64_t)n * 4, &acc, 4);
            if (mode & NPU_OUT_INT8)
                *(int8_t *)(d->ddr + l->off_c + (uint64_t)m * l->ldc + n) = npu_requant_ref((int32_t)acc, qp[n]);
        }
    }
    d->last_hw_ms = 0.0;
    return 0;
}
#endif

static void *read_file(const char *path, size_t expect) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[run] khong mo duoc %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (expect && (size_t)sz != expect) {
        fprintf(stderr, "[run] %s: kich thuoc %ld, can %zu\n", path, sz, expect);
        fclose(f);
        return NULL;
    }
    void *buf = malloc(sz ? sz : 1);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fprintf(stderr, "[run] doc loi %s\n", path); free(buf); fclose(f); return NULL; }
    fclose(f);
    return buf;
}
static int write_file(const char *path, const void *buf, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[run] khong ghi duoc %s\n", path); return -1; }
    fwrite(buf, 1, n, f);
    fclose(f);
    return 0;
}

// Dem phan tu lech; in phan tu lech dau tien. elem = 1 (int8) hoac 4 (int32)
static uint64_t compare_buf(const char *name, const void *ref, const void *got, uint64_t count, uint32_t N, int elem) {
    uint64_t bad = 0, first = (uint64_t)-1;
    for (uint64_t i = 0; i < count; i++) {
        int diff = elem == 1 ? ((const int8_t *)ref)[i] != ((const int8_t *)got)[i]
                             : ((const int32_t *)ref)[i] != ((const int32_t *)got)[i];
        if (diff) { if (first == (uint64_t)-1) first = i; bad++; }
    }
    if (bad) {
        long long r = elem == 1 ? ((const int8_t *)ref)[first] : ((const int32_t *)ref)[first];
        long long g = elem == 1 ? ((const int8_t *)got)[first] : ((const int32_t *)got)[first];
        fprintf(stderr, "[run] %s: LECH %llu/%llu phan tu; dau tien o (m=%llu,n=%llu) golden=%lld hw=%lld\n", name,
                (unsigned long long)bad, (unsigned long long)count, (unsigned long long)(first / N),
                (unsigned long long)(first % N), r, g);
    } else {
        printf("[run] %s: BIT-TRUE (%llu phan tu)\n", name, (unsigned long long)count);
    }
    return bad;
}

static void usage(void) {
    fprintf(stderr, "dung: npu_run_case --A f --B f --params f --M m --K k --N n [--kdim auto|N] [--mode both|int8|int32]\n"
                    "      [--out8 f] [--out32 f] [--golden8 f] [--golden32 f] [--lut f --out_lut f [--golden_lut f]]\n"
                    "      [--repeat R] [--verbose]\n");
}

int main(int argc, char **argv) {
    const char *fA = NULL, *fB = NULL, *fP = NULL, *out8 = NULL, *out32 = NULL, *g8 = NULL, *g32 = NULL;
    const char *fLut = NULL, *outLut = NULL, *gLut = NULL;
    uint32_t M = 0, K = 0, N = 0, kdim = NPU_KDIM_AUTO, mode = NPU_OUT_INT8 | NPU_OUT_INT32;
    int repeat = 1, verbose = 0;
    for (int i = 1; i < argc; i++) {
#define OPT(name) (!strcmp(argv[i], name) && i + 1 < argc)
        if (OPT("--A")) fA = argv[++i];
        else if (OPT("--B")) fB = argv[++i];
        else if (OPT("--params")) fP = argv[++i];
        else if (OPT("--M")) M = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (OPT("--K")) K = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (OPT("--N")) N = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (OPT("--kdim")) { i++; kdim = !strcmp(argv[i], "auto") ? NPU_KDIM_AUTO : (uint32_t)strtoul(argv[i], NULL, 0); }
        else if (OPT("--mode")) {
            i++;
            mode = !strcmp(argv[i], "int8") ? NPU_OUT_INT8 : !strcmp(argv[i], "int32") ? NPU_OUT_INT32 : 3;
        }
        else if (OPT("--out8")) out8 = argv[++i];
        else if (OPT("--out32")) out32 = argv[++i];
        else if (OPT("--golden8")) g8 = argv[++i];
        else if (OPT("--golden32")) g32 = argv[++i];
        else if (OPT("--lut")) fLut = argv[++i];
        else if (OPT("--out_lut")) outLut = argv[++i];
        else if (OPT("--golden_lut")) gLut = argv[++i];
        else if (OPT("--repeat")) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose")) verbose = 1;
        else { usage(); return 2; }
    }
    if (!fA || !fB || !fP || !M || !K || !N) { usage(); return 2; }
    if ((out8 || g8 || fLut) && !(mode & NPU_OUT_INT8)) { fprintf(stderr, "[run] can --mode co int8\n"); return 2; }
    if ((out32 || g32) && !(mode & NPU_OUT_INT32)) { fprintf(stderr, "[run] can --mode co int32\n"); return 2; }
    if (repeat < 1) repeat = 1;

    int8_t *A = (int8_t *)read_file(fA, (size_t)M * K);
    int8_t *B = (int8_t *)read_file(fB, (size_t)K * N);
    npu_qparam_t *qp = (npu_qparam_t *)read_file(fP, (size_t)N * sizeof(npu_qparam_t));
    if (!A || !B || !qp) return 2;
    for (uint32_t n = 0; n < N; n++)
        if (qp[n].shift < 0 || qp[n].shift > 62) { fprintf(stderr, "[run] shift kenh %u = %d ngoai [0,62]\n", n, qp[n].shift); return 2; }

    npu_t dev;
    if (npu_open(&dev)) return 2;
    npu_layout_t l = npu_layout(M, K, N);
    if (l.total > dev.ddr_size) { fprintf(stderr, "[run] can %llu byte DDR > %zu\n", (unsigned long long)l.total, dev.ddr_size); return 2; }

    double t0 = npu_now();
    npu_load_A(&dev, &l, A, M, K);
    npu_load_B(&dev, &l, B, K, N);
    npu_load_params(&dev, &l, qp, N);
    double load_ms = (npu_now() - t0) * 1e3;

    if (kdim == NPU_KDIM_AUTO) kdim = npu_pick_kdim(K);      // ghi ra bao cao k_dim THUC SU da dung
    double hw_min = 1e30, hw_sum = 0;
    for (int r = 0; r < repeat; r++) {
        int rc = npu_run(&dev, &l, M, K, N, kdim, mode, 600.0);
        if (rc) { fprintf(stderr, "[run] npu_run loi %d\n", rc); npu_close(&dev); return 2; }
        if (dev.last_hw_ms < hw_min) hw_min = dev.last_hw_ms;
        hw_sum += dev.last_hw_ms;
        if (verbose) printf("[run] lan %d: hw %.3f ms\n", r, dev.last_hw_ms);
    }
    uint64_t cnt = (uint64_t)M * N;
    int8_t *C8 = NULL; int32_t *C32 = NULL;
    t0 = npu_now();
    if (mode & NPU_OUT_INT8)  { C8  = (int8_t *)malloc(cnt);        npu_read_C8(&dev, &l, C8, M, N); }
    if (mode & NPU_OUT_INT32) { C32 = (int32_t *)malloc(cnt * 4);   npu_read_C32(&dev, &l, C32, M, N); }
    double read_ms = (npu_now() - t0) * 1e3;

    uint64_t bad8 = 0, bad32 = 0, badl = 0;
    int cmp8 = 0, cmp32 = 0, cmpl = 0;
    if (out8)  write_file(out8, C8, cnt);
    if (out32) write_file(out32, C32, cnt * 4);
    if (g8) {
        int8_t *ref = (int8_t *)read_file(g8, cnt);
        if (!ref) return 2;
        bad8 = compare_buf("C8  (INT8)", ref, C8, cnt, N, 1); cmp8 = 1; free(ref);
    }
    if (g32) {
        int32_t *ref = (int32_t *)read_file(g32, cnt * 4);
        if (!ref) return 2;
        bad32 = compare_buf("C32 (INT32)", ref, C32, cnt, N, 4); cmp32 = 1; free(ref);
    }
    if (fLut) {
        int8_t *lut = (int8_t *)read_file(fLut, 256);
        if (!lut) return 2;
        int8_t *G = (int8_t *)malloc(cnt);
        for (uint64_t i = 0; i < cnt; i++) G[i] = lut[(int)C8[i] + 128];
        if (outLut) write_file(outLut, G, cnt);
        if (gLut) {
            int8_t *ref = (int8_t *)read_file(gLut, cnt);
            if (!ref) return 2;
            badl = compare_buf("LUT(GELU)", ref, G, cnt, N, 1); cmpl = 1; free(ref);
        }
        free(G); free(lut);
    }

    double ops = 2.0 * M * (double)K * N;
    printf("[run] M=%u K=%u N=%u kdim=%u mode=%u | load DDR %.2f ms | HW min %.3f ms avg %.3f ms (%.2f GOPS) | read %.2f ms\n",
           M, K, N, kdim, mode, load_ms, hw_min, hw_sum / repeat, (hw_min > 0 ? ops / (hw_min * 1e6) : 0.0), read_ms);
    printf("{\"M\":%u,\"K\":%u,\"N\":%u,\"kdim\":%u,\"mode\":%u,\"repeat\":%d,\"load_ms\":%.3f,\"hw_ms_min\":%.4f,"
           "\"hw_ms_avg\":%.4f,\"read_ms\":%.3f,\"c8_compared\":%d,\"c8_mismatch\":%llu,\"c32_compared\":%d,"
           "\"c32_mismatch\":%llu,\"lut_compared\":%d,\"lut_mismatch\":%llu}\n",
           M, K, N, kdim, mode, repeat, load_ms, hw_min, hw_sum / repeat, read_ms, cmp8, (unsigned long long)bad8, cmp32,
           (unsigned long long)bad32, cmpl, (unsigned long long)badl);
    free(A); free(B); free(qp); free(C8); free(C32);
    npu_close(&dev);
    return (bad8 || bad32 || badl) ? 1 : 0;
}
