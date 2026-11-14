#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ref_runtime.py - Runtime THAM CHIEU (numpy) doc goi INT8 `qwen2audio-int8-pkg/1` (pkg_v1) tu FILE va tai tao luong suy luan INT8
cua mo phong `reference/qint8_sim.py` (cau hinh P1.5).  KHONG can torch / HuggingFace / GPU, chi can numpy (+ scipy cho erf).

Muc dich
  1. Chung minh goi du de dung runtime doc lap (khong can model goc / simulator).
  2. Lam dac ta thuc thi cho runtime C++ tren KR260 (cung thu tu phep tinh, cung quy uoc lam tron).

Lenh
  python3 ref_runtime.py vectors --pkg <pkg_v1>             tai tao C8/C32 cua moi vector kiem thu GEMM tu A,B,params (khong can model)
  python3 ref_runtime.py golden  --pkg <pkg_v1> [--clips 0,2]   dau-cuoi: so voi golden/clip_XX (token INT8, top-5 logit teacher-forced)
  python3 ref_runtime.py mel     --pkg <pkg_v1> --wav a.wav       log-mel numpy; neu co golden: so voi input_features cua clip
  python3 ref_runtime.py run     --pkg <pkg_v1> --wav a.wav       wav -> mel -> encoder -> LLM -> text (can transformers de token hoa/giai ma)

Ghi chu ve do trung khop
  * GEMM INT8 -> INT32 va requant la SO NGUYEN CHINH XAC (giong RTL / simulator).
  * norm / RoPE / softmax / GELU / SiLU chay float32 bang numpy; co the lech vai ulp so voi torch -> hiem khi lat 1 muc lam tron INT8,
    nen token co the lech muon (bao cao "lech dau tai token"); so top-5 teacher-forced de do muc lech.
  * --fb-scope member (mac dinh) tai tao dung mo phong: co fallback hang tinh RIENG cho q, k, v (va gate, up). 'gemm' = co chung cho
    ca GEMM gop (de xuat C++ neu muon don gian; hang FP nhieu hon mot chut, chinh xac hon).
"""
import argparse
import json
import math
import mmap
import os
import re
import sys
import time

import numpy as np

try:
    from scipy.special import erf as _scipy_erf
except Exception:          # pragma: no cover
    _scipy_erf = None

F32, F64 = np.float32, np.float64
HP = False                 # True: cac phep float (norm, softmax, RoPE, GELU, SiLU) tinh bang float64 roi ha ve float32 -> "mot mau nhieu ulp khac" (do do nhay, xem golden --sens)
FB = None                  # neu la dict: ghi lai cac su kien fallback theo (pha, loai GEMM) -> [so lan goi, so lan co hang FP, so byte trong so can doc]
FBBITS = 0                 # >0: hang ngoai lai tinh bang so nguyen (x lam tron ve FBBITS bit theo tung hang, w int8, cong int64 chinh xac) thay vi FP32 -> mo phong fallback NEON int16 tren PS
FBDIG = 0                  # >0: hang ngoai lai tinh bang FBDIG "chu so" int8 tren PL (x = sum x_i*s_i, moi x_i la 1 hang int8, cung 1 luot doc trong so, dau ra INT32) thay vi FP32 tren PS
FBPHASE = "enc"            # enc | prefill | decode
KCHUNK = 1024              # 1024*127*128 < 2^24 -> tong tung khoi float32 la so nguyen chinh xac


# ----------------------------------------------------------------------------- toan hoc float32
def erf32(x):
    if _scipy_erf is not None:
        return _scipy_erf(x).astype(F32)
    # du phong (sai so ~1.5e-7, Abramowitz-Stegun 7.1.26): cai scipy de co ket qua tot hon
    s = np.sign(x)
    a = np.abs(x).astype(F64)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return (s * y).astype(F32)


def gelu(x):               # F.gelu(approximate='none'), float32
    if HP:
        x64 = x.astype(F64)
        return (x64 * 0.5 * (1.0 + _scipy_erf(x64 * 0.70710678118654752440))).astype(F32)
    return (x * F32(0.5) * (F32(1.0) + erf32(x * F32(0.70710678118654752440)))).astype(F32)


def silu(x):
    if HP:
        x64 = x.astype(F64)
        return (x64 / (1.0 + np.exp(-x64))).astype(F32)
    return (x / (F32(1.0) + np.exp(-x))).astype(F32)


def layer_norm(x, w, b, eps):
    if HP:
        x64 = x.astype(F64); xc = x64 - x64.mean(-1, keepdims=True)
        return ((xc / np.sqrt((xc * xc).mean(-1, keepdims=True) + eps)) * w.astype(F64) + b.astype(F64)).astype(F32)
    m = x.mean(-1, keepdims=True, dtype=F32)
    xc = x - m
    var = (xc * xc).mean(-1, keepdims=True, dtype=F32)
    return ((xc / np.sqrt(var + F32(eps))) * w + b).astype(F32)


def rms_norm(x, w, eps):   # Qwen2RMSNorm
    if HP:
        x64 = x.astype(F64)
        return (w.astype(F64) * (x64 / np.sqrt((x64 * x64).mean(-1, keepdims=True) + eps))).astype(F32)
    var = (x * x).mean(-1, keepdims=True, dtype=F32)
    return (w * (x * (F32(1.0) / np.sqrt(var + F32(eps))))).astype(F32)


def softmax32(s):
    if HP:
        s64 = s.astype(F64); e = np.exp(s64 - s64.max(-1, keepdims=True))
        return (e / e.sum(-1, keepdims=True)).astype(F32)
    m = s.max(-1, keepdims=True)
    e = np.exp(s - m)
    return (e / e.sum(-1, keepdims=True, dtype=F32)).astype(F32)


def rope_cos_sin(pos, head_dim, theta):
    if HP:
        inv = 1.0 / np.power(float(theta), np.arange(0, head_dim, 2, dtype=F64) / head_dim)
        fr = pos.astype(F64)[:, None] * inv[None, :]
        emb = np.concatenate([fr, fr], axis=-1)
        return np.cos(emb).astype(F32), np.sin(emb).astype(F32)
    inv = F32(1.0) / np.power(F32(theta), np.arange(0, head_dim, 2, dtype=np.int64).astype(F32) / F32(head_dim))
    fr = pos.astype(F32)[:, None] * inv[None, :]
    emb = np.concatenate([fr, fr], axis=-1)
    return np.cos(emb).astype(F32), np.sin(emb).astype(F32)


def apply_rope(x, cos, sin):          # x [H,T,D]; cos,sin [T,D]
    h = x.shape[-1] // 2
    rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)
    return (x * cos[None] + rot * sin[None]).astype(F32)


# ----------------------------------------------------------------------------- goi
def _dt(s):
    return np.dtype({"float32": "<f4", "float64": "<f8", "float16": "<f2", "int32": "<i4", "int8": "i1", "uint8": "u1"}[s])


class Pkg:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.m = json.load(open(os.path.join(self.root, "MANIFEST.json")))
        assert self.m["format"] == "qwen2audio-int8-pkg/1", self.m["format"]
        self.scales = json.load(open(os.path.join(self.root, "scales.json")))
        self._mm = {}
        self.dims, self.dec = self.m["dims"], self.m["decode"]
        self.ent = {g["name"]: g for g in self.m["gemms"]}
        self._gc, self._tc = {}, {}

    def _buf(self, fname):
        if fname not in self._mm:
            f = open(os.path.join(self.root, fname), "rb")
            self._mm[fname] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mm[fname]

    def arr(self, e):
        dt = _dt(e["dtype"])
        return np.ndarray(tuple(e["shape"]), dtype=dt, buffer=self._buf(e["file"]), offset=e["offset"])

    def tensor(self, key):
        if key not in self._tc:
            self._tc[key] = np.array(self.arr(self.m["tensors"][key]))      # copy vao RAM (nho)
        return self._tc[key]

    def gemm(self, name):
        if name not in self._gc:
            self._gc[name] = Gemm(self, self.ent[name])
        return self._gc[name]

    def embed_rows(self, ids):
        e = self.m["embedding"]
        if "_emb" not in self.__dict__:
            self._emb = self.arr(e)
        return self._emb[np.asarray(ids, dtype=np.int64)].astype(F32)

    def attn(self, lk):
        return self.m["attention"][lk]


def int_mm(x8, B):
    """x8 [M,K] float32 (so nguyen), B [K,N] int8 -> [M,N] float64 (so nguyen chinh xac), khoi K = 1024."""
    M, K = x8.shape
    N = B.shape[1]
    out = np.empty((M, N), F64)
    nb = max(128, (16 << 20) // (4 * min(K, KCHUNK)))
    for n0 in range(0, N, nb):
        n1 = min(N, n0 + nb)
        acc = None
        for k0 in range(0, K, KCHUNK):
            k1 = min(K, k0 + KCHUNK)
            part = (x8[:, k0:k1] @ B[k0:k1, n0:n1].astype(F32)).astype(F64)
            acc = part if acc is None else acc + part
        out[:, n0:n1] = acc
    return out


def requant(acc, bi, mult, shift, rnd, zp):
    t = acc.astype(np.int64) + bi[None, :]
    v = np.right_shift(t * mult[None, :] + rnd[None, :], shift[None, :]) + zp[None, :]
    return np.clip(v, -128, 127)


def _fb_rec(name, calls, events, nbytes, by_in=0):
    if FB is None:
        return
    kind = "lm_head" if name == "lm_head" else re.sub(r"^(enc|llm)\.\d+\.", r"\1.", name)
    r = FB.setdefault((FBPHASE, kind), [0, 0, 0, 0])
    r[0] += calls; r[1] += events; r[2] += nbytes; r[3] += by_in


class Gemm:
    def __init__(self, pkg, e):
        self.name, self.K, self.N, self.sin = e["name"], e["K"], e["N"], float(e["sin"])
        self.int32, self.rowfb = e["out"] == "int32", bool(e["rowfb"])
        self.members = e["members"]
        self.B = pkg.arr(e["B"])                                   # [K,N] int8 (mmap)
        self.sw32 = np.array(pkg.arr(e["sw"]))
        self.bias = np.array(pkg.arr(e["bias_f32"]))
        if self.int32:
            self.sw64 = np.array(pkg.arr(e["sw_f64"]))
            self.sinsw = self.sin * self.sw64                          # float64 [N]
        else:
            qp = np.array(pkg.arr(e["qparams"]))
            self.bi, self.mult, self.shift = qp[:, 0].astype(np.int64), qp[:, 1].astype(np.int64), qp[:, 2].astype(np.int64)
            self.zp = qp[:, 3].astype(np.int64)
            self.rnd = np.where(self.shift > 0, np.left_shift(np.int64(1), np.maximum(self.shift - 1, 0)), 0).astype(np.int64)
            self.sout = np.array(pkg.arr(e["sout"]))                   # float32 [N]
            self.rawk = self.sin * self.sw32.astype(F64) / self.sout.astype(F64)   # he so cho co bao hoa dau ra
        self.thr_in = F32(127.5 * self.sin)

    def dg_rows(self, xf, c0, c1):
        """Hang ngoai lai bang nhieu chu so int8 (chay tren PL, dau ra INT32, M = FBDIG hang cho moi hang ngoai lai, cung 1 luot doc trong so):
        s1 = max|x|/127, x1 = rint(x/s1); phan du r = x - x1*s1 (|r| <= s1/2); s2 = s1/254, x2 = rint(r/s2); ... (moi chu so them ~7 bit)
        y = (sum_i int_mm(x_i, B) * s_i) * sw + bias  (cong INT32 chinh xac, hau xu ly float tren PS)."""
        R = xf.shape[0]
        x = xf.astype(F64)
        amax = np.abs(x).max(1, keepdims=True)
        s = np.where(amax > 0, amax / 127.0, 1.0)                      # [R,1] buoc cua chu so 1
        digs, scs = [], []
        r = x
        for i in range(FBDIG):
            xi = np.clip(np.rint(r / s), -127, 127)
            digs.append(xi.astype(F32)); scs.append(s)
            r = r - xi * s
            s = s / 254.0                                              # |r| <= s_prev/2 = 127*s_new
        out = np.empty((R, c1 - c0), F32)
        Bs = self.B[:, c0:c1]
        acc = 0
        for xi, si in zip(digs, scs):
            acc = acc + int_mm(xi, Bs) * si
        out[:] = (acc * self.sw32[c0:c1].astype(F64)[None, :] + self.bias[c0:c1].astype(F64)[None, :]).astype(F32)
        return out

    def fp_rows(self, xf, c0, c1):
        """Hang ngoai lai tinh FP tren PS: x_fp32 @ (B_int8 * sw) + bias, cac cot [c0,c1). Chia khoi cot de khong cap phat ca ma tran (lm_head ~0.5 GB)."""
        if FBDIG:
            return self.dg_rows(xf, c0, c1)
        out = np.empty((xf.shape[0], c1 - c0), F32)
        if FBBITS:
            qmax = (1 << (FBBITS - 1)) - 1
            amax = np.abs(xf).max(1, keepdims=True).astype(F64)
            xs = np.where(amax > 0, amax / qmax, 1.0)                    # [R,1] buoc luong tu tung hang
            xq = np.rint(xf.astype(F64) / xs)                            # so nguyen |.| <= qmax
            step = max(256, (32 << 20) // (8 * self.K))
            for n0 in range(c0, c1, step):
                n1 = min(c1, n0 + step)
                acc = xq @ self.B[:, n0:n1].astype(F64)                  # chinh xac (|acc| < 2^53)
                out[:, n0 - c0:n1 - c0] = (acc * xs * self.sw32[n0:n1].astype(F64)[None, :] + self.bias[n0:n1].astype(F64)[None, :]).astype(F32)
            return out
        step = max(256, (32 << 20) // (4 * self.K))
        for n0 in range(c0, c1, step):
            n1 = min(c1, n0 + step)
            Wd = self.B[:, n0:n1].astype(F32) * self.sw32[n0:n1][None, :]
            out[:, n0 - c0:n1 - c0] = xf @ Wd + self.bias[n0:n1][None, :]
        return out

    def __call__(self, x, stats=None, scope="member"):
        """x [M,K] float32 -> (y [M,N] float32 da giai luong tu, [co hang FP cua tung thanh vien])"""
        x = np.ascontiguousarray(x, dtype=F32)
        x8 = np.clip(np.rint(x / F32(self.sin)), -128, 127).astype(F32)
        acc = int_mm(x8, self.B)
        infl = (np.abs(x).max(1) > self.thr_in) if self.rowfb else None
        if self.int32:
            y = (acc * self.sinsw[None, :]).astype(F32)
            flags = []
            if self.rowfb:
                flags = [infl]
                if infl.any():
                    y[infl] = self.fp_rows(x[infl], 0, self.N)
                if stats is not None:
                    stats.setdefault(self.name, [0, 0])[0] += int(infl.sum()); stats[self.name][1] += len(infl)
                _fb_rec(self.name, len(infl), int(infl.sum()), int(infl.sum()) * self.K * self.N, int(infl.sum()))
            return y, flags
        v = requant(acc, self.bi, self.mult, self.shift, self.rnd, self.zp)
        y = (v.astype(F32) * self.sout[None, :]).astype(F32)
        flags = []
        if self.rowfb:
            sat = (np.abs((acc + self.bi[None, :]) * self.rawk[None, :]) > 127.5)
            for mb in self.members:
                c0, c1 = mb["cols"]
                flags.append(infl | sat[:, c0:c1].any(1))
            if scope == "gemm":
                u = np.logical_or.reduce(flags)
                flags = [u] * len(flags)
                if u.any():
                    y[u] = self.fp_rows(x[u], 0, self.N)
            else:
                for mb, fb in zip(self.members, flags):
                    if fb.any():
                        c0, c1 = mb["cols"]
                        y[np.ix_(fb, np.arange(c0, c1))] = self.fp_rows(x[fb], c0, c1)
            if FB is not None:
                if scope == "gemm":
                    nb = int(flags[0].sum()) * self.K * self.N
                else:
                    nb = sum(int(fb.sum()) * self.K * (mb["cols"][1] - mb["cols"][0]) for mb, fb in zip(self.members, flags))
                _fb_rec(self.name, len(infl), int(np.logical_or.reduce(flags).sum()), nb, int((infl & np.logical_or.reduce(flags)).sum()))
            if stats is not None:
                fb_any = np.logical_or.reduce(flags)
                s = stats.setdefault(self.name, [0, 0]); s[0] += int(fb_any.sum()); s[1] += len(fb_any)
        return y, flags


# ----------------------------------------------------------------------------- attention luong tu (kind: enc | llm)
def chunked_av(p8, v8):
    S = p8.shape[-1]
    acc = None
    for s0 in range(0, S, KCHUNK):
        part = np.matmul(p8[..., s0:s0 + KCHUNK], v8[:, s0:s0 + KCHUNK, :]).astype(F64)
        acc = part if acc is None else acc + part
    return acc


def attention(att, q, k, v, vscale, pf_q, pf_k, mask, scaling):
    """q [H,T,d]; k,v [Hkv,S,d] (cung RoPE-sau cho k); vscale [Hkv,1,d] (float32, theo kenh); pf_q [T], pf_k [S] bool (vi tri ngoai lai);
       mask [T,S] cong (0/-inf) hoac None; scaling: he so cua duong FP (enc: 1.0 vi q da nhan 0.125; llm: head_dim^-0.5).
       Tra [H,T,d] float32.  Tuong duong make_attention (quant, score_mode='int32', row_fb)."""
    H, Hkv = q.shape[0], k.shape[0]
    rep = H // Hkv
    kk = np.repeat(k, rep, 0) if rep > 1 else k
    vv = np.repeat(v, rep, 0) if rep > 1 else v
    vs = np.repeat(vscale, rep, 0) if rep > 1 else vscale
    sq, sk, fac = float(att["sq"]), float(att["sk"]), float(att["fac"])
    q8 = np.clip(np.rint(q / F32(sq)), -128, 127).astype(F32)
    k8 = np.clip(np.rint(kk / F32(sk)), -128, 127).astype(F32)
    v8 = np.clip(np.rint(vv / vs), -128, 127).astype(F32)
    acc = np.matmul(q8, k8.transpose(0, 2, 1))                       # <= 64*128*128 < 2^24: chinh xac
    sc = (acc * F32(fac)).astype(F32)
    qrow = np.abs(q).max(-1) > F32(127.5 * sq)                       # [H,T]
    kcol = (np.abs(kk).max(-1) > F32(127.5 * sk)) | (np.abs(vv) > (F32(127.5) * vs)).any(-1)   # [H,S]
    qrow = qrow | pf_q[None, :]
    kcol = kcol | pf_k[None, :]
    if qrow.any() or kcol.any():
        fpm = qrow[:, :, None] | kcol[:, None, :]
        sc_fp = (np.matmul(q, kk.transpose(0, 2, 1)) * F32(scaling)).astype(F32)
        sc = np.where(fpm, sc_fp, sc)
    else:
        kcol = None
    if mask is not None:
        sc = sc + mask[None]
    p = softmax32(sc)
    pn = p if kcol is None else np.where(kcol[:, None, :], F32(0.0), p).astype(F32)
    sp = np.maximum(pn.max(-1, keepdims=True) / F32(127.0), F32(1e-30))
    p8 = np.rint(pn / sp).astype(F32)
    o = chunked_av(p8, v8) * sp.astype(F64) * vs.astype(F64)
    if kcol is not None:
        pfp = np.where(kcol[:, None, :], p, F32(0.0)).astype(F32)
        o = o + np.matmul(pfp, vv).astype(F64)
    return o.astype(F32)


# ----------------------------------------------------------------------------- mel
def log_mel(pkg, wave, n_valid_out=False):
    """wave float32 [n] (16 kHz) -> input_features [128,3000] float32 (+ so khung hop le)."""
    a = pkg.m["dims"]["audio"]
    n_fft, hop, ns = a["n_fft"], a["hop"], a["n_samples"]
    wave = np.asarray(wave, F32)[:ns]
    n = len(wave)
    w = np.zeros(ns, F32)
    w[:n] = wave
    L = int(math.ceil(n / hop))
    p = np.pad(w, (n_fft // 2, n_fft // 2), mode="reflect")
    nfr = 1 + (len(p) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nfr)[:, None]
    frames = p[idx].astype(F64) * pkg.tensor("hann_window").astype(F64)[None, :]
    spec = np.fft.rfft(frames, axis=-1)                                                          # float64 nhu HF
    mag = (spec.real ** 2 + spec.imag ** 2)[:-1].T                                               # [201, 3000]
    mel = pkg.tensor("mel_filters").astype(F64).T @ mag                                          # [128,3000]
    ls = np.log10(np.maximum(mel, 1e-10))
    ls = np.maximum(ls, ls.max() - 8.0)
    out = ((ls + 4.0) / 4.0).astype(F32)
    return (out, L) if n_valid_out else out


# ----------------------------------------------------------------------------- mo hinh
class Model:
    def __init__(self, pkg, scope="member", verbose=True):
        self.p, self.scope, self.verbose = pkg, scope, verbose
        self.stats = {}
        d = pkg.dims
        self.E, self.Lm = d["enc"], d["llm"]
        self.cos_sin = {}

    # ---- encoder: chi tinh cac hang HOP LE (mel L -> L2 -> Lp), tuong duong chay 1500 hang co mat na vi moi phep tinh theo hang va
    #      cac khoa ngoai vung hop le bi che (-inf)
    def encode(self, feats, L):
        global FBPHASE
        FBPHASE = "enc"
        p, E = self.p, self.E
        L2 = (L - 1) // 2 + 1
        Lp = (L2 - 2) // 2 + 1
        R1 = min(feats.shape[1], 2 * L2)
        t0 = time.time()
        # conv1 (k=3, pad=1): hang R1-1 can khung R1 THAT (khung dem sau vung hop le, KHONG phai 0) -> lay R1+1 khung neu con
        x = feats[:, :R1 + 1].astype(F32)                               # [128, R1+1] (hoac R1 neu het 3000 khung)
        xp = np.pad(x, ((0, 0), (1, 1)))
        cols = [xp[:, k:k + R1].T for k in range(3)]
        y, _ = p.gemm("conv1")(np.concatenate(cols, axis=1), self.stats)         # [R1,1280], cot k*C + c
        y = gelu(y)                                                    # = conv1 -> GELU (theo hang)
        yp = np.pad(y, ((1, 1), (0, 0)))
        cols = [yp[k: k + 2 * (L2 - 1) + 1: 2] for k in range(3)]
        y2, _ = p.gemm("conv2")(np.concatenate(cols, axis=1), self.stats)        # [L2,1280]
        h = gelu(y2) + p.tensor("audio_tower.embed_positions.weight")[:L2]
        h = h.astype(F32)
        H, D = E["heads"], E["d_model"] // E["heads"]
        for i in range(E["layers"]):
            lk = f"enc.{i}"
            r = h
            a_in = layer_norm(h, p.tensor(f"audio_tower.layers.{i}.self_attn_layer_norm.weight"),
                              p.tensor(f"audio_tower.layers.{i}.self_attn_layer_norm.bias"), E["ln_eps"])
            g = p.gemm(f"{lk}.qkv")
            qkv, fl = g(a_in, self.stats, self.scope)
            d_ = E["d_model"]
            q = (qkv[:, :d_] * F32(E["q_scaling"])).astype(F32)
            k, v = qkv[:, d_:2 * d_], qkv[:, 2 * d_:]
            pf = fl[0] | fl[1] | fl[2]
            vscale = g.sout[2 * d_:].reshape(H, 1, D)
            qh, kh, vh = (z.reshape(L2, H, D).transpose(1, 0, 2) for z in (q, k, v))
            o = attention(p.attn(lk), qh, kh, vh, vscale, pf, pf, None, 1.0)
            o = o.transpose(1, 0, 2).reshape(L2, d_)
            h = r + p.gemm(f"{lk}.out_proj")(o, self.stats, self.scope)[0]
            r = h
            f_in = layer_norm(h, p.tensor(f"audio_tower.layers.{i}.final_layer_norm.weight"),
                              p.tensor(f"audio_tower.layers.{i}.final_layer_norm.bias"), E["ln_eps"])
            f1 = gelu(p.gemm(f"{lk}.fc1")(f_in, self.stats, self.scope)[0])
            h = (r + p.gemm(f"{lk}.fc2")(f1, self.stats, self.scope)[0]).astype(F32)
        hp = (F32(0.5) * (h[0:2 * Lp:2] + h[1:2 * Lp:2])).astype(F32)               # avg-pool k=2,s=2
        hp = layer_norm(hp, p.tensor("audio_tower.layer_norm.weight"), p.tensor("audio_tower.layer_norm.bias"), E["ln_eps"])
        out, _ = p.gemm("proj")(hp, self.stats, self.scope)
        if self.verbose:
            print(f"   encoder: mel {L} -> {L2} -> {Lp} hang | {time.time() - t0:.1f}s")
        return out.astype(F32)

    # ---- LLM
    def new_cache(self, maxlen):
        Lm = self.Lm
        Hkv, d = Lm["kv_heads"], Lm["head_dim"]
        self.kc = [np.zeros((Hkv, maxlen, d), F32) for _ in range(Lm["layers"])]
        self.vc = [np.zeros((Hkv, maxlen, d), F32) for _ in range(Lm["layers"])]
        self.pfc = [np.zeros(maxlen, bool) for _ in range(Lm["layers"])]
        self.n = 0

    def llm_rows(self, x):
        """x [T,hidden] float32: T hang moi (noi tiep cache). Tra hang cuoi cung sau chuan cuoi (truoc lm_head)."""
        p, Lm = self.p, self.Lm
        T = x.shape[0]
        H, Hkv, d, hid = Lm["heads"], Lm["kv_heads"], Lm["head_dim"], Lm["hidden"]
        s0 = self.n
        S = s0 + T
        pos = np.arange(s0, S)
        cs = self.cos_sin.get((s0, T))
        cos, sin = rope_cos_sin(pos, d, Lm["rope_theta"])
        mask = None
        if T > 1:
            mask = np.where(np.arange(S)[None, :] > pos[:, None], F32(-np.inf), F32(0.0)).astype(F32)
        for j in range(Lm["layers"]):
            lk = f"llm.{j}"
            pre = f"language_model.model.layers.{j}"
            h = rms_norm(x, p.tensor(f"{pre}.input_layernorm.weight"), Lm["rms_eps"])
            g = p.gemm(f"{lk}.qkv")
            qkv, fl = g(h, self.stats, self.scope)
            q, k, v = qkv[:, :hid], qkv[:, hid:hid + Hkv * d], qkv[:, hid + Hkv * d:]
            pf = fl[0] | fl[1] | fl[2]
            vscale = g.sout[hid + Hkv * d:].reshape(Hkv, 1, d)
            qh = q.reshape(T, H, d).transpose(1, 0, 2)
            kh = k.reshape(T, Hkv, d).transpose(1, 0, 2)
            vh = v.reshape(T, Hkv, d).transpose(1, 0, 2)
            qh, kh = apply_rope(qh, cos, sin), apply_rope(kh, cos, sin)
            self.kc[j][:, s0:S], self.vc[j][:, s0:S], self.pfc[j][s0:S] = kh, vh, pf
            o = attention(p.attn(lk), qh, self.kc[j][:, :S], self.vc[j][:, :S], vscale, pf, self.pfc[j][:S], mask, d ** -0.5)
            o = o.transpose(1, 0, 2).reshape(T, hid)
            x = (x + p.gemm(f"{lk}.o_proj")(o, self.stats, self.scope)[0]).astype(F32)
            h = rms_norm(x, p.tensor(f"{pre}.post_attention_layernorm.weight"), Lm["rms_eps"])
            gu = p.gemm(f"{lk}.gate_up")(h, self.stats, self.scope)[0]
            act = (silu(gu[:, :Lm["inter"]]) * gu[:, Lm["inter"]:]).astype(F32)
            act = (act * p.tensor(f"{lk}.down_inv")[None, :]).astype(F32)
            x = (x + p.gemm(f"{lk}.down_proj")(act, self.stats, self.scope)[0]).astype(F32)
            if getattr(self, "trace", None) is not None:
                self.trace.append(x.copy())
        self.n = S
        return x

    def logits(self, x):
        """x [R,hidden] (dau ra cua lop cuoi) -> logits [R,V] float32 (norm cuoi + lm_head INT32)."""
        h = rms_norm(x, self.p.tensor("language_model.model.norm.weight"), self.Lm["rms_eps"])
        return self.p.gemm("lm_head")(h, self.stats, self.scope)[0]

    def embed_inputs(self, ids, audio_rows):
        x = self.p.embed_rows(ids)
        pos = np.nonzero(np.asarray(ids) == self.p.dec["audio_token_id"])[0]
        assert len(pos) == len(audio_rows), f"so <|AUDIO|> = {len(pos)} != so hang audio {len(audio_rows)}"
        x[pos] = audio_rows
        return x

    def teacher_forced(self, ids, audio_rows, gen):
        """Mot luot qua [prompt + gen] (nhu forward_logits cua mo phong). Tra logits [len(gen), V] cho cac vi tri du doan gen."""
        global FBPHASE
        FBPHASE = "tf"
        full = np.concatenate([np.asarray(ids), np.asarray(gen)]).astype(np.int64)
        n0 = len(ids)
        self.new_cache(len(full))
        x = self.llm_rows(self.embed_inputs(full, audio_rows))
        return self.logits(x[n0 - 1:len(full) - 1])

    def generate(self, ids, audio_rows, max_new, penalty, eos):
        ids = np.asarray(ids, np.int64)
        global FBPHASE
        self.new_cache(len(ids) + max_new + 1)
        t0 = time.time()
        FBPHASE = "prefill"
        x = self.llm_rows(self.embed_inputs(ids, audio_rows))
        t_pre = time.time() - t0
        hist, out = list(ids), []
        t1 = time.time()
        FBPHASE = "decode"
        self.tok_fbytes = []
        for step in range(max_new):
            b0 = sum(v[2] for (ph, _), v in FB.items() if ph == "decode") if FB is not None else 0
            lg = self.logits(x[-1:])[0].copy()
            u = np.unique(np.asarray(hist))
            vv = lg[u]
            lg[u] = np.where(vv > 0, vv / F32(penalty), vv * F32(penalty))
            tok = int(np.argmax(lg))
            out.append(tok)
            hist.append(tok)
            if tok == eos:
                if FB is not None:
                    self.tok_fbytes.append(sum(v[2] for (ph, _), v in FB.items() if ph == "decode") - b0)
                break
            x = self.llm_rows(self.p.embed_rows([tok]))
            if FB is not None:
                self.tok_fbytes.append(sum(v[2] for (ph, _), v in FB.items() if ph == "decode") - b0)
        FBPHASE = "enc"
        t_dec = time.time() - t1
        if self.verbose:
            print(f"   LLM: prefill {len(ids)} hang {t_pre:.1f}s | decode {len(out)} token {t_dec:.1f}s ({len(out) / max(t_dec, 1e-9):.2f} token/s numpy)")
        return out


# ----------------------------------------------------------------------------- tien ich
def apply_penalty_row(lg, hist, penalty):
    lg = lg.copy()
    u = np.unique(np.asarray(hist))
    v = lg[u]
    lg[u] = np.where(v > 0, v / F32(penalty), v * F32(penalty))
    return lg


def read_bin(path, dtype, shape=None):
    a = np.fromfile(path, dtype=dtype)
    return a.reshape(shape) if shape else a


# ----------------------------------------------------------------------------- lenh: vectors
def cmd_vectors(a):
    root = os.path.join(a.pkg, "vectors")
    idx = json.load(open(os.path.join(root, "INDEX.json")))
    fdir = os.path.join(root, "functions")
    fns = sorted(os.listdir(fdir))
    n = bad = 0
    for fn in fns:
        fj = os.path.join(fdir, fn, "function.json")
        if not os.path.exists(fj):
            continue
        for c in json.load(open(fj))["cases"]:
            if a.functions and not any(w in fn or w in c["id"] for w in a.functions.split(",")):
                continue
            P = lambda k: os.path.join(fdir, fn, c["files"][k]["path"])
            M, K, N = c["M"], c["K"], c["N"]
            A = read_bin(P("A"), np.int8, (M, K))
            B = read_bin(P("B"), np.int8, (K, N))
            qp = read_bin(P("params"), "<i4", (N, 4)).astype(np.int64)
            t0 = time.time()
            acc = int_mm(A.astype(F32), B)
            ok = True
            msg = []
            if "golden32" in c["files"]:
                g32 = read_bin(P("golden32"), "<i4", (M, N))
                e = acc.astype(np.int64).astype(np.int32)       # RTL: INT32 quay vong
                ok32 = bool((e == g32).all()); ok &= ok32; msg.append("C32 " + ("OK" if ok32 else "LECH"))
            if "golden8" in c["files"]:
                g8 = read_bin(P("golden8"), np.int8, (M, N))
                bi, mult, shift, zp = qp[:, 0], qp[:, 1], qp[:, 2], qp[:, 3]
                rnd = np.where(shift > 0, np.left_shift(np.int64(1), np.maximum(shift - 1, 0)), 0)
                v = requant(acc, bi, mult, shift, rnd, zp).astype(np.int8)
                ok8 = bool((v == g8).all()); ok &= ok8; msg.append("C8 " + ("OK" if ok8 else "LECH"))
            n += 1
            bad += (not ok)
            print("%-28s %-18s %4dx%-5dx%-6d %s  [%.2fs]" % (fn, c["id"], M, K, N, " | ".join(msg), time.time() - t0))
    print("\nKET QUA: %s (%d case)" % ("TAT CA KHOP" if bad == 0 else f"{bad} case LECH", n))
    return 1 if bad else 0


# ----------------------------------------------------------------------------- lenh: golden
def fb_report(fb, toks, bw_gbs, label):
    """Bao cao chi phi fallback hang ngoai lai (tinh tren PS): fb[(pha,loai)] = [so lan goi, so lan co hang FP, byte trong so doc].
    Chi phi ~ byte / bw. toks = list byte fallback cua tung token decode."""
    print(f"   [fbstats {label}] hang ngoai lai theo GEMM (p = ti le lan-goi-co-fallback; MB = trong so INT8 phai doc lai tren PS moi su kien)")
    print(f"     {'pha':8s} {'GEMM':14s} {'lan goi':>9s} {'co FB':>8s} {'p %':>7s} {'MB/su kien':>11s} {'do x>nguong %':>14s}")
    for (ph, kd), (c, e, b, bi) in sorted(fb.items()):
        if ph == "tf":
            continue
        print(f"     {ph:8s} {kd:14s} {c:9d} {e:8d} {100 * e / max(c, 1):7.2f} {(b / e / 1e6) if e else 0:11.2f} {100 * bi / max(e, 1):14.0f}")
    if toks:
        t = np.asarray(toks, np.float64) / (bw_gbs * 1e9) * 1e3          # ms / token
        print(f"     decode {len(t)} token, fallback ms/token o {bw_gbs:.2f} GB/s: trung binh {t.mean():.2f} | p50 {np.percentile(t, 50):.2f} | p90 {np.percentile(t, 90):.2f} | "
              f"p99 {np.percentile(t, 99):.2f} | max {t.max():.2f} | token co FB {int((t > 0).sum())}/{len(t)}")


def cmd_golden(a):
    global HP, FB, FBBITS, FBDIG
    FBDIG = int(getattr(a, "fb_digits", 0))
    FBBITS = int(getattr(a, "fb_bits", 0))
    HP = bool(getattr(a, "hp", False))
    FB = {} if getattr(a, "fbstats", False) else None
    fb_all, tok_all = {}, []
    pkg = Pkg(a.pkg)
    m = Model(pkg, a.fb_scope)
    gdir = os.path.join(a.pkg, "golden")
    clips = sorted(d for d in os.listdir(gdir) if d.startswith("clip_"))
    if a.clips:
        want = {f"clip_{int(c):02d}" for c in a.clips.split(",")}
        clips = [c for c in clips if c in want]
    eos, pen = pkg.dec["eos_id"], pkg.dec["repetition_penalty"]
    tot_tf = tot_n = tot_sf = tot_rf = 0
    allok = True
    for cl in clips:
        d = os.path.join(gdir, cl)
        meta = json.load(open(os.path.join(d, "meta.json")))
        sh = meta["shapes"]
        feats = read_bin(os.path.join(d, "input_features.f32"), "<f4", tuple(sh["input_features"]))[0]
        ids = read_bin(os.path.join(d, "input_ids.i32"), "<i4").astype(np.int64)
        fam = read_bin(os.path.join(d, "feature_attention_mask.i32"), "<i4")
        g8 = read_bin(os.path.join(d, "int8_gen_ids.i32"), "<i4").astype(np.int64)
        gf = read_bin(os.path.join(d, "fp32_gen_ids.i32"), "<i4").astype(np.int64)
        tid = read_bin(os.path.join(d, "tf_int8_top5_ids.i32"), "<i4").reshape(-1, 5)
        tvl = read_bin(os.path.join(d, "tf_int8_top5_vals.f32"), "<f4").reshape(-1, 5)
        L = int(fam.sum())
        print(f"\n=== {cl}: {os.path.basename(meta['wav'])} | mel {L} | prompt {len(ids)} | golden INT8 {len(g8)} token, FP32 {len(gf)} ===")
        m.stats = {}
        if FB is not None:
            FB.clear()
        t0 = time.time()
        rows = m.encode(feats, L)
        assert len(rows) == meta["valid"]["proj"], (len(rows), meta["valid"])
        # teacher-forced (theo token FP32), so top-5 INT8 cua mo phong
        if not a.no_tf:
            lg = m.teacher_forced(ids, rows, gf)
            top1 = int(sum(int(np.argmax(apply_penalty_row(lg[t], np.concatenate([ids, gf[:t]]), pen))) ==
                           int(np.argmax(apply_penalty_row(_dense(tid[t], tvl[t], lg.shape[1]), np.concatenate([ids, gf[:t]]), pen)))
                           for t in range(len(gf))))
            raw1 = int(sum(int(np.argmax(lg[t])) == int(tid[t, 0]) for t in range(len(gf))))
            top5 = int(sum(set(np.argsort(-lg[t])[:5]) == set(tid[t]) for t in range(len(gf))))
            dv = float(np.max(np.abs(np.sort(lg, axis=1)[:, ::-1][:, :5] - np.sort(tvl, axis=1)[:, ::-1])))
            print(f"   teacher-forced {len(gf)} vi tri: top-1 trung {raw1}/{len(gf)} | tap top-5 trung {top5}/{len(gf)} | max|d top-5 logit| {dv:.3e}")
            tot_tf += raw1
            tot_n += len(gf)
            t1f = read_bin(os.path.join(d, "tf_fp32_top5_ids.i32"), "<i4").reshape(-1, 5)[:, 0]
            sf, rf = int((tid[:, 0] == t1f).sum()), int((np.argmax(lg, 1) == t1f).sum())
            tot_sf += sf; tot_rf += rf
            print(f"   so voi FP32 (top-1 tho): mo phong INT8 {sf}/{len(gf)} | ref {rf}/{len(gf)}")
            if a.sens:
                HP = True
                try:
                    r64 = m.encode(feats, L)
                    lg64 = m.teacher_forced(ids, r64, gf)
                finally:
                    HP = False
                t5 = lambda z: np.argsort(-z, 1)[:, :5]
                s_a = int(sum(np.argmax(lg[t]) == np.argmax(lg64[t]) for t in range(len(gf))))
                s_b = int(sum(np.argmax(lg64[t]) == tid[t, 0] for t in range(len(gf))))
                s5 = int(sum(set(t5(lg)[t]) == set(t5(lg64)[t]) for t in range(len(gf))))
                dd = float(np.max(np.abs(np.sort(lg, 1)[:, ::-1][:, :5] - np.sort(lg64, 1)[:, ::-1][:, :5])))
                print(f"   [do nhay] float32 vs float64 (CUNG goi, chi doi do chinh xac phep float): top-1 trung {s_a}/{len(gf)} | top-5 trung {s5}/{len(gf)} | max|d top-5| {dd:.3e}"
                      f" | float64-ref vs mo phong top-1 {s_b}/{len(gf)}")
        if not a.no_gen:
            out = np.array(m.generate(ids, rows, a.max_new or pkg.dec["max_new"], pen, eos))
            same = len(out) == len(g8) and bool((out == g8).all())
            fd = next((t for t in range(min(len(out), len(g8))) if out[t] != g8[t]), min(len(out), len(g8)))
            print(f"   sinh tu do: {len(out)} token | GIONG HET golden INT8: {same} | lech dau tai token {fd}")
            allok &= same
            if a.text:
                try:
                    from transformers import AutoTokenizer
                    tk = AutoTokenizer.from_pretrained(os.path.join(a.pkg, "hf_processor"))
                    print("   text:", tk.decode(out, skip_special_tokens=True).strip()[:200])
                except Exception as e:
                    print("   (khong giai ma duoc text:", type(e).__name__, ")")
        fbs = {k: v for k, v in m.stats.items()}
        ft = sum(v[0] for v in fbs.values()); nt = sum(v[1] for v in fbs.values())
        print(f"   hang FP: {ft}/{nt} = {100 * ft / max(nt, 1):.2f}% | tong {time.time() - t0:.0f}s")
        if FB is not None and not a.no_gen:
            fb_report(FB, m.tok_fbytes, a.fb_bw, cl)
            for k, v in FB.items():
                r = fb_all.setdefault(k, [0, 0, 0, 0]); [r.__setitem__(i, r[i] + v[i]) for i in range(4)]
            tok_all += m.tok_fbytes
    if FB is not None and tok_all:
        print("\n=== TONG fbstats (tat ca clip) ===")
        fb_report(fb_all, tok_all, a.fb_bw, "tong")
    if tot_n:
        print(f"\nTONG teacher-forced top-1 so voi mo phong: {tot_tf}/{tot_n} = {100 * tot_tf / tot_n:.2f}%")
        print(f"TONG top-1 so voi FP32: mo phong INT8 {tot_sf}/{tot_n} = {100 * tot_sf / tot_n:.2f}% | ref {tot_rf}/{tot_n} = {100 * tot_rf / tot_n:.2f}%")
    return 0 if allok else 1


def _dense(ids5, vals5, V):
    z = np.full(V, -1e30, F32)
    z[ids5] = vals5
    return z


# ----------------------------------------------------------------------------- lenh: mel
def read_wav(path):
    import wave
    try:
        import soundfile as sf
        w, sr = sf.read(path, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
    except Exception:
        with wave.open(path, "rb") as f:
            sr = f.getframerate()
            raw = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(F32) / 32768.0
            w = raw.reshape(-1, f.getnchannels()).mean(1) if f.getnchannels() > 1 else raw
    if sr != 16000:
        n = int(round(len(w) * 16000 / sr))
        w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(F32)
    return w.astype(F32)


def cmd_mel(a):
    pkg = Pkg(a.pkg)
    feats, L = log_mel(pkg, read_wav(a.wav), True)
    print("input_features", feats.shape, "khung hop le", L)
    gdir = os.path.join(a.pkg, "golden")
    if a.clip is not None:
        d = os.path.join(gdir, f"clip_{int(a.clip):02d}")
        g = read_bin(os.path.join(d, "input_features.f32"), "<f4").reshape(128, -1)
        print("so voi golden clip %02d: max|d| = %.3e" % (int(a.clip), np.abs(g - feats).max()))
    return 0


# ----------------------------------------------------------------------------- lenh: run
def cmd_run(a):
    global HP
    HP = bool(getattr(a, "hp", False))
    pkg = Pkg(a.pkg)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(os.path.join(a.pkg, "hf_processor"))
    feats, L = log_mel(pkg, read_wav(a.wav), True)
    if getattr(a, "dump_mel", None):
        np.ascontiguousarray(feats[:, :min(feats.shape[1], L + 5)], np.float32).tofile(a.dump_mel)
    m = Model(pkg, a.fb_scope)
    rows = m.encode(feats, L)
    if getattr(a, "dump_rows", None):
        np.ascontiguousarray(rows, np.float32).tofile(a.dump_rows); print("da ghi %s: %s" % (a.dump_rows, tuple(np.asarray(rows).shape)))
    if getattr(a, "rows_only", False):
        return 0
    prompt = pkg.dec["prompt"].replace("<|AUDIO|>", "<|AUDIO|>" * len(rows))
    ids = np.array(tk(prompt, add_special_tokens=False)["input_ids"], np.int64)
    assert int((ids == pkg.dec["audio_token_id"]).sum()) == len(rows)
    out = m.generate(ids, rows, a.max_new or pkg.dec["max_new"], pkg.dec["repetition_penalty"], pkg.dec["eos_id"])
    print("\n>>", tk.decode(out, skip_special_tokens=True).strip())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("vectors", "golden", "mel", "run"):
        s = sub.add_parser(nm)
        s.add_argument("--pkg", required=True)
        s.add_argument("--fb-scope", choices=["member", "gemm"], default="member")
        if nm == "vectors":
            s.add_argument("--functions", default="")
        if nm == "golden":
            s.add_argument("--clips", default="")
            s.add_argument("--max-new", type=int, default=0)
            s.add_argument("--no-gen", action="store_true")
            s.add_argument("--no-tf", action="store_true")
            s.add_argument("--text", action="store_true")
            s.add_argument("--sens", action="store_true", help="do do nhay: so logit tinh bang float32 voi logit tinh bang float64 cho cac phep float (cung goi, khong lien quan mo phong)")
            s.add_argument("--fbstats", action="store_true", help="thong ke hang ngoai lai (fallback) theo GEMM va pha (enc/prefill/decode) + chi phi fallback ms/token tren PS (can sinh tu do, khong dung --no-gen)")
            s.add_argument("--fb-bits", type=int, default=0, help="0: fallback FP32 (mac dinh). N>0 (vd 12): hang ngoai lai tinh bang so nguyen N bit (x theo tung hang) x w int8, cong chinh xac -> mo phong fallback NEON int16 tren PS")
            s.add_argument("--fb-digits", type=int, default=0, help="N>0 (vd 2): hang ngoai lai tinh bang N chu so int8 tren PL (INT32, M=N hang/su kien, cung 1 luot doc trong so) - mo phong phuong an 'PL multi-digit'")
            s.add_argument("--fb-bw", type=float, default=0.47, help="thong luong fallback tren PS, GB/s trong so int8 (ps_bench: 0.45-0.49)")
            s.add_argument("--hp", action="store_true", help="chay toan bo bang HP (float64 cho norm/softmax/RoPE/GELU/SiLU)")
        if nm in ("mel", "run"):
            s.add_argument("--wav", required=True)
        if nm == "mel":
            s.add_argument("--clip", default=None)
        if nm == "run":
            s.add_argument("--max-new", type=int, default=0)
            s.add_argument("--dump-rows", default=None, help="ghi hang audio (sau projector) ra file f32 de so voi llm_runtime --dump-rows")
            s.add_argument("--dump-mel", default=None)
            s.add_argument("--rows-only", action="store_true", help="chi chay encoder (khong sinh chu)")
            s.add_argument("--hp", action="store_true", help="float64 cho norm/softmax/GELU... (do do nhay cua encoder)")
    a = ap.parse_args(argv)
    rc = ({"vectors": cmd_vectors, "golden": cmd_golden, "mel": cmd_mel, "run": cmd_run}[a.cmd](a))
    if argv is not None:          # goi tu notebook / Python: tra ma thoat, khong thoat tien trinh
        return rc
    sys.exit(rc)


if __name__ == "__main__" and "ipykernel" not in sys.modules:      # dan vao o notebook thi khong tu chay; goi main([...]) hoac !python3 ref_runtime.py ...
    main()
