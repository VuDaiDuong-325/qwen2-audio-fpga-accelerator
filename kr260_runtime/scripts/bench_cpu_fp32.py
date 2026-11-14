#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_cpu_fp32.py - (TUY CHON) baseline CPU FP32 cho cung hai GEMM cua FFN, dung numpy (OpenBLAS).
Day la cach model thuc su chay tren CPU (FP32), bo sung cho baseline INT8 trong npu_bench.

    pip3 install numpy          # neu board chua co
    python3 bench_cpu_fp32.py --bundle ./bundle --repeat 3
    OPENBLAS_NUM_THREADS=1 python3 bench_cpu_fp32.py ...   # de do 1 luong

Chi do phep nhan ma tran (A[M,K] @ B[K,N]) tren du lieu int8 doi sang float32, KHONG gom GELU/requant,
nen day la can duoi thoi gian cua duong FP32 CPU. Khong co y nghia bit-true (FP32).
"""
import argparse
import json
import os
import statistics
import time

try:
    import numpy as np
except ImportError:
    raise SystemExit("Can numpy: pip3 install numpy")


def bench(a, b, repeat):
    a @ b  # khoi dong
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        a @ b
        ts.append((time.perf_counter() - t) * 1e3)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="./bundle")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json", default="bench_cpu_fp32.json")
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.bundle, "manifest.json"), encoding="utf-8"))
    M, K, H = man["M"], man["K"], man["H"]
    P = lambda n: os.path.join(args.bundle, n)
    cases = [("fc1", M, K, H, "fc1_A.bin", "fc1_B.bin"), ("fc2", M, H, K, "golden_gelu_G8.bin", "fc2_B.bin")]
    out = {"numpy": np.__version__, "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"), "gemms": {}}
    total = 0.0
    for name, m, k, n, fa, fb in cases:
        a = np.fromfile(P(fa), dtype=np.int8).reshape(m, k).astype(np.float32)
        b = np.fromfile(P(fb), dtype=np.int8).reshape(k, n).astype(np.float32)
        ts = bench(a, b, args.repeat)
        med = statistics.median(ts)
        total += med
        gflops = 2.0 * m * k * n / (med * 1e6)
        out["gemms"][name] = {"median_ms": med, "min_ms": min(ts), "gflops": gflops}
        print("%s  M=%d K=%d N=%d : FP32 numpy median %.1f ms (min %.1f)  -> %.2f GFLOPS" % (name, m, k, n, med, min(ts), gflops))
    out["ffn_matmul_total_median_ms"] = total
    print("FFN (chi 2 phep nhan): %.1f ms" % total)
    json.dump(out, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
