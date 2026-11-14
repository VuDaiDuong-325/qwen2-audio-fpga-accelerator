#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qffn_common.py - Module dung chung (chi can numpy) cho thi nghiem
"FFN cua audio-encoder Qwen2-Audio:  fc1 -> GELU -> fc2" tren bo tang toc INT8 GEMM v6.

Co 2 duong tinh INT8, doc lap nhau, dung de so sanh bit-true:

  1) gemm_golden()    : "100% Python" - nhan chinh xac (float64 BLAS theo khoi K=256, cong int64),
                        wrap INT32 mot lan, requant bang so nguyen lon cua Python (giong het
                        npu_requant_ref cua npu_driver.h, dung __int128).
  2) gemm_rtl_model() : MO HINH BIT-ACCURATE CUA RTL v6 - pad M,N len boi 16, K len boi k_dim,
                        cong don INT32 (wrap) theo tung chunk k_dim, requant theo dung 7 tang
                        cua int32_int8_requant_pc (do rong bit 33/65/66/67 duoc assert).
  (3) Phan cung that tren KV260 se cho ket qua thu 3 - xem npu_run_case.c / run_on_kv260.py.

Quy uoc:
  A  : [M,K] int8, row-major          B : [K,N] int8, row-major (= W.T cua nn.Linear)
  qp : [N,4] int32 = (bias, mult, shift, zp) moi cot N  (16 byte/kenh, little-endian,
       trung layout npu_qparam_t)
  Requant:  t=acc+bias; p=t*mult; q=(p+(shift?1<<(shift-1):0))>>shift (floor); out=clamp(q+zp,-128,127)
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TILE = 16
KDIM_DEFAULT = 32
I32_MOD = 1 << 32


# --------------------------------------------------------------------------------------
# Tien ich so nguyen
# --------------------------------------------------------------------------------------
def ceil_to(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def pick_kdim(K: int) -> int:
    """Chon k_dim theo K, GIONG HET npu_pick_kdim() trong npu_driver.h.
    Chi phi 1 tile ~ ceil(K/kd) * max(kd,48) chu ky (RTL v8: moi chunk chay it nhat 48 chu ky, K pad len boi kd).
    Chon kd trong {32,64,128} co chi phi nho nhat; hoa -> kd lon hon (128 la gia tri da do bit-true tren board)."""
    best, best_cost = 128, None
    for kd in (32, 64, 128):
        cost = -(-int(K) // kd) * max(kd, 48)
        if best_cost is None or cost <= best_cost:
            best, best_cost = kd, cost
    return best


def wrap_i32(x) -> np.ndarray:
    """Wrap hai-bu 32 bit (giong thanh ghi tich luy INT32 cua PE)."""
    x = np.asarray(x, dtype=np.int64)
    return (((x + (1 << 31)) & (I32_MOD - 1)) - (1 << 31)).astype(np.int32)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(a).tobytes())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read_bin(path, dtype, shape) -> np.ndarray:
    a = np.fromfile(path, dtype=dtype)
    n = int(np.prod(shape))
    if a.size != n:
        raise ValueError(f"{path}: can {n} phan tu {dtype}, doc duoc {a.size}")
    return a.reshape(shape)


def write_bin(path, arr, dtype) -> str:
    arr = np.ascontiguousarray(arr, dtype=dtype)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    return sha256_file(path)


def compare_int(ref: np.ndarray, got: np.ndarray) -> dict:
    ref = np.asarray(ref)
    got = np.asarray(got)
    if ref.shape != got.shape:
        return {"bit_true": False, "reason": f"shape {ref.shape} != {got.shape}"}
    d = ref.astype(np.int64) - got.astype(np.int64)
    bad = np.flatnonzero(d.reshape(-1))
    out = {"bit_true": bad.size == 0, "elements": int(ref.size), "mismatches": int(bad.size),
           "max_abs_diff": int(np.abs(d).max()) if ref.size else 0,
           "sha256_ref": sha256_array(ref), "sha256_got": sha256_array(got)}
    if bad.size:
        i = int(bad[0])
        idx = np.unravel_index(i, ref.shape)
        out["first_mismatch"] = {"index": [int(v) for v in idx], "ref": int(ref.reshape(-1)[i]),
                                 "got": int(got.reshape(-1)[i])}
    return out


# --------------------------------------------------------------------------------------
# Luong tu hoa
# --------------------------------------------------------------------------------------
def quant_absmax(x: np.ndarray, pct: float = 100.0):
    """Per-tensor, doi xung, round-half-even. Tra ve (int8, scale float64)."""
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("gia tri khong huu han")
    amax = float(np.percentile(np.abs(x), pct)) if pct < 100.0 else float(np.max(np.abs(x))) if x.size else 0.0
    scale = amax / 127.0 if amax > 0 else 1.0
    q = np.clip(np.rint(x / scale), -128, 127).astype(np.int8)
    return q, scale


def quant_weight_per_channel(w_kn: np.ndarray):
    """w_kn: [K,N] (da transpose). Scale rieng cho moi cot N. Tra ve (int8 [K,N], scale [N])."""
    w = np.asarray(w_kn, dtype=np.float64)
    amax = np.max(np.abs(w), axis=0)
    scale = np.where(amax > 0, amax / 127.0, 1.0)
    q = np.clip(np.rint(w / scale[None, :]), -128, 127).astype(np.int8)
    return q, scale


def make_qparams(bias_fp, s_in: float, s_w: np.ndarray, s_out: float, zp: int = 0) -> np.ndarray:
    """
    Tao bang qparam [N,4] int32 (bias, mult, shift, zp) sao cho
        out_q ~= (acc + bias_q) * (s_in * s_w[n] / s_out) + zp
    mult < 2^31, shift in [0,62].
    """
    s_w = np.asarray(s_w, dtype=np.float64)
    n = s_w.size
    qp = np.zeros((n, 4), dtype=np.int32)
    real = s_in * s_w / s_out
    for j in range(n):
        r = float(real[j])
        if not (r > 0 and math.isfinite(r)):
            raise ValueError(f"he so requant khong hop le o kenh {j}: {r}")
        shift = min(62, int(math.floor(math.log2((2 ** 31 - 1) / r))))
        if shift < 0:
            raise ValueError(f"he so requant qua lon ({r}) o kenh {j}")
        mult = int(round(math.ldexp(r, shift)))
        while mult > 2 ** 31 - 1 and shift > 0:
            shift -= 1
            mult = int(round(math.ldexp(r, shift)))
        if mult < 1:
            raise ValueError(f"he so requant qua nho ({r}) o kenh {j}")
        qp[j, 1] = mult
        qp[j, 2] = shift
    if bias_fp is not None:
        b = np.rint(np.asarray(bias_fp, dtype=np.float64) / (s_in * s_w))
        qp[:, 0] = np.clip(b, -(2 ** 31), 2 ** 31 - 1).astype(np.int32)
    qp[:, 3] = zp
    return qp


# --------------------------------------------------------------------------------------
# GEMM INT8 -> INT32 (golden) va requant golden (so nguyen lon Python)
# --------------------------------------------------------------------------------------
def gemm_acc_exact(A: np.ndarray, B: np.ndarray, kblock: int = 256) -> np.ndarray:
    """Tich chinh xac int64 (float64 BLAS chinh xac vi |tong khoi| <= 256*16384 << 2^53)."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    acc = np.zeros((M, N), dtype=np.int64)
    for k0 in range(0, K, kblock):
        a = A[:, k0:k0 + kblock].astype(np.float64)
        b = B[k0:k0 + kblock, :].astype(np.float64)
        acc += np.rint(a @ b).astype(np.int64)
    return acc


def requant_golden(acc32: np.ndarray, qp: np.ndarray, block: int = 128) -> np.ndarray:
    """Tuong duong tung dong voi npu_requant_ref (C, __int128): dung int Python (vo han bit)."""
    M, N = acc32.shape
    bias = np.array([int(v) for v in qp[:, 0]], dtype=object)
    mult = np.array([int(v) for v in qp[:, 1]], dtype=object)
    shift = np.array([int(v) for v in qp[:, 2]], dtype=object)
    zp = np.array([int(v) for v in qp[:, 3]], dtype=object)
    rnd = np.array([(1 << (int(s) - 1)) if int(s) > 0 else 0 for s in qp[:, 2]], dtype=object)
    out = np.empty((M, N), dtype=np.int8)
    for m0 in range(0, M, block):
        a = np.array(acc32[m0:m0 + block].tolist(), dtype=object)
        v = (a + bias) * mult + rnd
        v = v >> shift
        v = v + zp
        v = np.minimum(np.maximum(v, -128), 127)
        out[m0:m0 + block] = np.array(v.tolist(), dtype=np.int64).astype(np.int8)
    return out


def gemm_golden(A: np.ndarray, B: np.ndarray, qp: np.ndarray | None):
    """100% Python. Tra ve (c8 hoac None, c32)."""
    c32 = wrap_i32(gemm_acc_exact(A, B))
    c8 = requant_golden(c32, qp) if qp is not None else None
    return c8, c32


# --------------------------------------------------------------------------------------
# MO HINH BIT-ACCURATE CUA RTL v6
# --------------------------------------------------------------------------------------
def _assert_width(name: str, lo: int, hi: int, bits: int):
    if lo < -(1 << (bits - 1)) or hi > (1 << (bits - 1)) - 1:
        raise AssertionError(f"{name} vuot {bits} bit co dau: [{lo},{hi}]")


def gemm_acc_rtl(A: np.ndarray, B: np.ndarray, k_dim: int = KDIM_DEFAULT) -> np.ndarray:
    """
    Tich luy INT32 kieu PE output-stationary: pad M,N -> boi 16, K -> boi k_dim (so 0),
    cong don (wrap INT32) tung chunk k_dim. Tich 8x8 nam trong int16 (mac_core: m_r 16 bit).
    Tra ve acc [M,N] int32 (da bo pad).
    """
    if k_dim < 1 or k_dim > 256 or (k_dim & (k_dim - 1)):
        raise ValueError("k_dim phai la luy thua 2 trong [1,256]")
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    Mp, Np, Kp = ceil_to(M, TILE), ceil_to(N, TILE), ceil_to(K, k_dim)
    Af = np.zeros((Mp, Kp), dtype=np.float64)
    Bf = np.zeros((Kp, Np), dtype=np.float64)
    Af[:M, :K] = A
    Bf[:K, :N] = B
    acc = np.zeros((Mp, Np), dtype=np.int32)
    for k0 in range(0, Kp, k_dim):
        part = np.rint(Af[:, k0:k0 + k_dim] @ Bf[k0:k0 + k_dim, :]).astype(np.int32)  # <= 256*16384
        acc = acc + part  # int32 + int32 -> wrap (khong canh bao voi mang)
    return acc[:M, :N]


def gemm_acc_tilewise_slow(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Mo hinh cham, tung tile 16x16, tung k tuan tu, moi phep cong wrap int32 - dung de tu kiem tra."""
    M, K = A.shape
    N = B.shape[1]
    Mp, Np = ceil_to(M, TILE), ceil_to(N, TILE)
    Ap = np.zeros((Mp, K), dtype=np.int8)
    Bp = np.zeros((K, Np), dtype=np.int8)
    Ap[:M] = A
    Bp[:, :N] = B
    C = np.zeros((Mp, Np), dtype=np.int32)
    for i in range(0, Mp, TILE):
        for j in range(0, Np, TILE):
            t = np.zeros((TILE, TILE), dtype=np.int32)
            for k in range(K):
                prod = (Ap[i:i + TILE, k].astype(np.int16)[:, None] * Bp[k, j:j + TILE].astype(np.int16)[None, :])
                t = t + prod.astype(np.int32)
            C[i:i + TILE, j:j + TILE] = t
    return C[:M, :N]


def requant_rtl(acc32: np.ndarray, qp: np.ndarray, block: int = 256) -> np.ndarray:
    """
    Mo hinh tung tang cua int32_int8_requant_pc (S1..S7):
      S1 t=acc+bias (33b) | S2/S3 p=t*mult (65b) | S4 r=p+rnd (66b) | S5 q=r>>>s
      S6 y=q+zp (67b)     | S7 clamp -> int8
    Dung int64 khi chac chan khong tran, nguoc lai dung int Python (object).
    """
    M, N = acc32.shape
    bias = qp[:, 0].astype(np.int64)
    mult = qp[:, 1].astype(np.int64)
    shift = (qp[:, 2].astype(np.int64) & 0x3F)  # RTL chi dung 6 bit thap
    zp = qp[:, 3].astype(np.int64)
    rnd_py = [(1 << (int(s) - 1)) if int(s) > 0 else 0 for s in shift]
    out = np.empty((M, N), dtype=np.int8)
    for m0 in range(0, M, block):
        a = acc32[m0:m0 + block].astype(np.int64)
        t_max = int(np.abs(a).max() if a.size else 0) + int(np.abs(bias).max())
        p_bound = t_max * int(np.abs(mult).max()) + (1 << 61)   # |p| + |rnd| toi da
        if p_bound >= (1 << 62):   # co the tran int64 -> dung so nguyen Python
            a_, bias_, mult_, shift_, zp_ = (v.astype(object) for v in (a, bias, mult, shift, zp))
            rnd_ = np.array(rnd_py, dtype=object)
        else:
            a_, bias_, mult_, shift_, zp_ = a, bias, mult, shift, zp
            rnd_ = np.array(rnd_py, dtype=np.int64)
        t = a_ + bias_                      # S1
        _assert_width("t", int(t.min()), int(t.max()), 33)
        p = t * mult_                       # S2/S3
        _assert_width("p", int(p.min()), int(p.max()), 65)
        r = p + rnd_                        # S4
        _assert_width("r", int(r.min()), int(r.max()), 66)
        q = r >> shift_                     # S5 (dich so hoc)
        y = q + zp_                         # S6
        _assert_width("y", int(y.min()), int(y.max()), 67)
        y = np.minimum(np.maximum(y, -128), 127)  # S7
        out[m0:m0 + block] = np.array(y.tolist(), dtype=np.int64).astype(np.int8)
    return out


def gemm_rtl_model(A: np.ndarray, B: np.ndarray, qp: np.ndarray | None, k_dim: int = KDIM_DEFAULT):
    c32 = gemm_acc_rtl(A, B, k_dim)
    c8 = requant_rtl(c32, qp) if qp is not None else None
    return c8, c32


# --------------------------------------------------------------------------------------
# FFN:  fc1 (GEMM+requant) -> GELU (LUT int8->int8) -> fc2 (GEMM+requant)
# --------------------------------------------------------------------------------------
def _erf_array(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erf  # type: ignore
        return erf(x)
    except Exception:  # scipy khong co
        return np.array([math.erf(float(v)) for v in x.reshape(-1)]).reshape(x.shape)


def gelu_exact(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + _erf_array(x / math.sqrt(2.0)))


def build_gelu_lut(s_in: float, s_out: float) -> np.ndarray:
    """LUT 256 muc (chi so q+128): int8 -> GELU -> int8. Tinh bang math.erf (float64) => xac dinh."""
    lut = np.empty(256, dtype=np.int8)
    for i in range(256):
        x = (i - 128) * s_in
        g = 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
        lut[i] = int(max(-128, min(127, round(g / s_out))))  # round() cua Python = half-even
    return lut


def apply_lut(c8: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[c8.astype(np.int16) + 128]


@dataclass
class FFNBundle:
    """Toan bo tham so luong tu hoa cua mot FFN (fc1 -> GELU -> fc2)."""
    W1: np.ndarray            # [K,H] int8
    qp1: np.ndarray           # [H,4] int32
    W2: np.ndarray            # [H,K] int8
    qp2: np.ndarray           # [K,4] int32
    lut: np.ndarray           # [256] int8
    scales: dict = field(default_factory=dict)


PCT_NAMES = ("x", "z1", "g", "z2")


def _as_pcts(act_pct):
    """act_pct: so (dung chung) hoac bo 4 phan tram cho (x, z1, g, z2)."""
    if isinstance(act_pct, (int, float)):
        return (float(act_pct),) * 4
    t = tuple(float(v) for v in act_pct)
    if len(t) != 4:
        raise ValueError("act_pct phai la so hoac bo 4 gia tri (x, z1, g, z2)")
    return t


def fp_reference(x_fp, W1_hk, b1, W2_kh, b2) -> dict:
    """Tham chieu float64: z1 = x W1^T + b1 ; g = GELU(z1) ; z2 = g W2^T + b2."""
    x = np.asarray(x_fp, dtype=np.float64)
    W1 = np.asarray(W1_hk, dtype=np.float64)
    W2 = np.asarray(W2_kh, dtype=np.float64)
    b1 = np.zeros(W1.shape[0]) if b1 is None else np.asarray(b1, dtype=np.float64)
    b2 = np.zeros(W2.shape[0]) if b2 is None else np.asarray(b2, dtype=np.float64)
    z1 = x @ W1.T + b1
    g = gelu_exact(z1)
    z2 = g @ W2.T + b2
    return {"z1": z1, "g": g, "z2": z2}


def calibrate_ffn(x_fp, W1_hk, b1, W2_kh, b2, act_pct=100.0, ref: dict | None = None):
    """
    x_fp [M,K] float; W1_hk [H,K] va W2_kh [K,H] theo layout nn.Linear (out,in); b1 [H], b2 [K].
    act_pct: percentile (100 = absmax) dung chung hoac bo 4 gia tri cho (x, z1, g, z2).
    ref: ket qua fp_reference da tinh san (de quet nhieu percentile khong tinh lai).
    Tra ve (FFNBundle, x_q int8 [M,K], fp_ref dict). Scale la TINH (calibrate tu chinh x_fp nay).
    """
    px, pz1, pg, pz2 = _as_pcts(act_pct)
    x = np.asarray(x_fp, dtype=np.float64)
    W1 = np.asarray(W1_hk, dtype=np.float64)
    W2 = np.asarray(W2_kh, dtype=np.float64)
    b1 = np.zeros(W1.shape[0]) if b1 is None else np.asarray(b1, dtype=np.float64)
    b2 = np.zeros(W2.shape[0]) if b2 is None else np.asarray(b2, dtype=np.float64)
    if ref is None:
        ref = fp_reference(x, W1, b1, W2, b2)
    z1, g, z2 = ref["z1"], ref["g"], ref["z2"]

    x_q, sx = quant_absmax(x, px)
    W1q, sw1 = quant_weight_per_channel(W1.T)
    W2q, sw2 = quant_weight_per_channel(W2.T)
    _, s_z1 = quant_absmax(z1, pz1)
    _, s_g = quant_absmax(g, pg)
    _, s_z2 = quant_absmax(z2, pz2)

    qp1 = make_qparams(b1, sx, sw1, s_z1, 0)
    lut = build_gelu_lut(s_z1, s_g)
    qp2 = make_qparams(b2, s_g, sw2, s_z2, 0)
    bundle = FFNBundle(W1q, qp1, W2q, qp2, lut,
                       {"s_x": sx, "s_z1": s_z1, "s_g": s_g, "s_z2": s_z2,
                        "s_w1_min": float(sw1.min()), "s_w1_max": float(sw1.max()),
                        "s_w2_min": float(sw2.min()), "s_w2_max": float(sw2.max()),
                        "act_percentiles": {"x": px, "z1": pz1, "g": pg, "z2": pz2}, "zero_point": 0})
    return bundle, x_q, ref


def sweep_percentiles(x_fp, W1_hk, b1, W2_kh, b2, y_ref, grid=(100.0, 99.999, 99.99, 99.9, 99.5),
                      start=100.0, rounds=1, gemm_fn=None, verbose=True):
    """
    Tim bo percentile (x, z1, g, z2) cho sai so thap nhat so voi y_ref (FP32 cua model) bang
    tim kiem toa do: lan luot toi uu tung tensor trong khi giu cac tensor con lai.
    Tra ve (pcts, metrics_tot_nhat, lich_su).
    """
    gemm_fn = gemm_fn or gemm_golden
    ref = fp_reference(x_fp, W1_hk, b1, W2_kh, b2)
    y_ref = np.asarray(y_ref, dtype=np.float64)

    def evaluate(p):
        b, xq, _ = calibrate_ffn(x_fp, W1_hk, b1, W2_kh, b2, act_pct=tuple(p), ref=ref)
        r = ffn_forward(xq, b, gemm_fn)
        return fp_error_metrics(y_ref, dequant_output(r["fc2_c8"], b))

    cur = list(_as_pcts(start))
    best = evaluate(cur)
    hist = [{"pcts": list(cur), **best}]
    if verbose:
        print(f"  start {dict(zip(PCT_NAMES, cur))}: rel_l2={best['rel_l2_err']:.4f} cos={best['cosine']:.5f}")
    for _ in range(rounds):
        for i, name in enumerate(PCT_NAMES):
            for v in grid:
                if v == cur[i]:
                    continue
                trial = list(cur)
                trial[i] = v
                m = evaluate(trial)
                hist.append({"pcts": list(trial), **m})
                better = m["rel_l2_err"] < best["rel_l2_err"]
                if verbose:
                    print(f"  {name}={v:<7} -> rel_l2={m['rel_l2_err']:.4f} cos={m['cosine']:.5f}{'  <== tot hon' if better else ''}")
                if better:
                    best, cur = m, trial
    return tuple(cur), best, hist


def ffn_forward(x_q: np.ndarray, b: FFNBundle, gemm_fn) -> dict:
    """gemm_fn(A,B,qp) -> (c8,c32). Dung chung cho golden / mo hinh RTL."""
    h8, h32 = gemm_fn(x_q, b.W1, b.qp1)
    g8 = apply_lut(h8, b.lut)
    y8, y32 = gemm_fn(g8, b.W2, b.qp2)
    return {"fc1_c8": h8, "fc1_c32": h32, "gelu_q": g8, "fc2_c8": y8, "fc2_c32": y32}


def dequant_output(y8: np.ndarray, b: FFNBundle) -> np.ndarray:
    return (y8.astype(np.float64) * b.scales["s_z2"]).astype(np.float32)


def fp_error_metrics(ref, got) -> dict:
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    d = got - ref
    rn = float(np.linalg.norm(ref))
    dn = float(np.linalg.norm(d))
    cos = float(np.sum(ref * got) / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-30))
    return {"max_abs_err": float(np.abs(d).max()), "mean_abs_err": float(np.abs(d).mean()),
            "rel_l2_err": dn / (rn + 1e-30), "cosine": cos,
            "sqnr_db": float(20 * math.log10((rn + 1e-30) / (dn + 1e-30)))}


# --------------------------------------------------------------------------------------
# Giao dien cap FFN: float vao -> float ra, backend INT8 co the thay (Python / mo hinh RTL / phan cung)
# --------------------------------------------------------------------------------------
class PyInt8FFN:
    """Backend INT8 bang Python: x_q [M,K] int8 -> y8 [M,K] int8 (golden hoac mo hinh RTL)."""

    def __init__(self, bundle: "FFNBundle", gemm_fn=None):
        self.bundle = bundle
        self.gemm_fn = gemm_fn or gemm_golden

    def __call__(self, x_q: np.ndarray) -> np.ndarray:
        return ffn_forward(x_q, self.bundle, self.gemm_fn)["fc2_c8"]


class QuantFFN:
    """FFN luong tu hoa: float32 vao -> (quantize tinh) -> backend INT8 -> dequant -> float32 ra."""

    def __init__(self, bundle: "FFNBundle", int8_ffn):
        self.bundle = bundle
        self.int8_ffn = int8_ffn
        self.s_x = float(bundle.scales["s_x"])
        self.s_out = float(bundle.scales["s_z2"])

    def quantize(self, x2d: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(np.asarray(x2d, dtype=np.float64) / self.s_x), -128, 127).astype(np.int8)

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x)
        shape = x.shape
        xq = np.ascontiguousarray(self.quantize(x.reshape(-1, shape[-1])))
        y8 = self.int8_ffn(xq)
        return (y8.astype(np.float64) * self.s_out).astype(np.float32).reshape(shape)


def load_bundle(bundle_dir):
    """Doc goi file do export_bundle tao ra. Tra ve (FFNBundle, manifest)."""
    d = Path(bundle_dir)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    K, H = man["K"], man["H"]
    b = FFNBundle(
        W1=read_bin(d / "fc1_B.bin", np.int8, (K, H)),
        qp1=read_bin(d / "fc1_params.bin", "<i4", (H, 4)).astype(np.int32),
        W2=read_bin(d / "fc2_B.bin", np.int8, (H, K)),
        qp2=read_bin(d / "fc2_params.bin", "<i4", (K, 4)).astype(np.int32),
        lut=read_bin(d / "gelu_lut.bin", np.int8, (256,)),
        scales=man["scales"])
    return b, man


# --------------------------------------------------------------------------------------
# Xuat goi file cho KV260
# --------------------------------------------------------------------------------------
def export_bundle(outdir, x_q, b: FFNBundle, golden: dict, extra: dict | None = None,
                  y_fp_ref: np.ndarray | None = None, k_dim: int = KDIM_DEFAULT) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    M, K = x_q.shape
    H = b.W1.shape[1]
    files = {}

    def put(name, arr, dtype):
        files[name] = {"sha256": write_bin(out / name, arr, dtype), "dtype": str(np.dtype(dtype)),
                       "shape": list(np.shape(arr))}

    put("fc1_A.bin", x_q, np.int8)
    put("fc1_B.bin", b.W1, np.int8)
    put("fc1_params.bin", b.qp1, "<i4")
    put("gelu_lut.bin", b.lut, np.int8)
    put("fc2_B.bin", b.W2, np.int8)
    put("fc2_params.bin", b.qp2, "<i4")
    put("golden_fc1_C8.bin", golden["fc1_c8"], np.int8)
    put("golden_fc1_C32.bin", golden["fc1_c32"], "<i4")
    put("golden_gelu_G8.bin", golden["gelu_q"], np.int8)
    put("golden_fc2_C8.bin", golden["fc2_c8"], np.int8)
    put("golden_fc2_C32.bin", golden["fc2_c32"], "<i4")
    put("golden_out_dequant_fp32.bin", dequant_output(golden["fc2_c8"], b), "<f4")
    if y_fp_ref is not None:
        put("fp32_reference_out.bin", np.asarray(y_fp_ref, dtype=np.float32), "<f4")
    manifest = {"schema": "qffn_bundle_v1", "function": "audio_encoder_FFN: fc1 -> GELU(LUT int8) -> fc2",
                "M": int(M), "K": int(K), "H": int(H), "k_dim": int(k_dim),
                "gemms": {"fc1": {"M": int(M), "K": int(K), "N": int(H)},
                          "fc2": {"M": int(M), "K": int(H), "N": int(K)}},
                "requant": "t=acc+bias; q=(t*mult+(1<<(shift-1)))>>shift; out=clamp(q+zp,-128,127)",
                "layout": "row-major, int8 / little-endian int32, qparam = (bias,mult,shift,zp) int32 x N",
                "scales": b.scales, "files": files}
    if extra:
        manifest.update(extra)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------------------
# Tu kiem tra (du lieu tong hop, chay duoc o bat ky dau: python3 qffn_common.py)
# --------------------------------------------------------------------------------------
def selftest(M=45, K=48, H=200, seed=1, verbose=True) -> bool:
    rng = np.random.default_rng(seed)
    # 1) acc: mo hinh chunk == mo hinh tile tuan tu == chinh xac (ke ca tran int32)
    A = rng.integers(-128, 128, size=(21, 37), dtype=np.int8)
    B = rng.integers(-128, 128, size=(37, 19), dtype=np.int8)
    assert np.array_equal(gemm_acc_rtl(A, B, 32), gemm_acc_tilewise_slow(A, B))
    assert np.array_equal(gemm_acc_rtl(A, B, 8), wrap_i32(gemm_acc_exact(A, B)))
    A = np.full((16, 140000), -128, dtype=np.int8)       # gay tran: K*16384 > 2^31
    B = np.full((140000, 16), -128, dtype=np.int8)
    assert np.array_equal(gemm_acc_rtl(A, B, 256), wrap_i32(gemm_acc_exact(A, B)))
    assert int(gemm_acc_rtl(A, B, 256)[0, 0]) != 140000 * 16384  # that su da wrap
    # 2) requant: RTL model == golden, bien (shift 0, 62, mult am, bao hoa)
    acc = rng.integers(-2 ** 31, 2 ** 31, size=(40, 24), dtype=np.int64).astype(np.int32)
    qp = np.zeros((24, 4), dtype=np.int32)
    qp[:, 0] = rng.integers(-2 ** 31, 2 ** 31, size=24, dtype=np.int64)
    qp[:, 1] = rng.integers(-2 ** 31, 2 ** 31, size=24, dtype=np.int64)
    qp[:, 2] = rng.integers(0, 63, size=24)
    qp[:, 3] = rng.integers(-128, 128, size=24)
    qp[0, 2] = 0
    qp[1, 2] = 62
    assert np.array_equal(requant_rtl(acc, qp), requant_golden(acc, qp))
    small = rng.integers(-5000, 5000, size=(40, 24)).astype(np.int32)
    qp2 = qp.copy()
    qp2[:, 1] = rng.integers(-1000, 1000, size=24)
    qp2[:, 2] = rng.integers(0, 20, size=24)
    qp2[:, 0] = rng.integers(-100, 100, size=24)
    assert np.array_equal(requant_rtl(small, qp2), requant_golden(small, qp2))
    # 3) FFN day du
    x = rng.standard_normal((M, K)) * 1.5
    W1 = rng.standard_normal((H, K)) * 0.1
    W2 = rng.standard_normal((K, H)) * 0.07
    b1 = rng.standard_normal(H) * 0.05
    b2 = rng.standard_normal(K) * 0.05
    bundle, xq, ref = calibrate_ffn(x, W1, b1, W2, b2)
    # percentile rieng cho tung tensor + QuantFFN phai cho ket qua bit-giong ffn_forward
    b_p, xq_p, _ = calibrate_ffn(x, W1, b1, W2, b2, act_pct=(100.0, 99.99, 99.9, 99.99))
    qf = QuantFFN(b_p, PyInt8FFN(b_p))
    assert np.array_equal(qf.quantize(x), xq_p)
    y_q = qf(x)
    assert y_q.shape == x.shape and y_q.dtype == np.float32
    g = ffn_forward(xq, bundle, gemm_golden)
    r = ffn_forward(xq, bundle, gemm_rtl_model)
    ok = all(np.array_equal(g[k], r[k]) for k in g)
    met = fp_error_metrics(ref["z2"], dequant_output(g["fc2_c8"], bundle))
    if verbose:
        print(f"selftest: golden == rtl_model: {ok}; sai so so voi FP: {met}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
