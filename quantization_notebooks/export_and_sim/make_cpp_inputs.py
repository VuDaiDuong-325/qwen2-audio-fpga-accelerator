#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_cpp_inputs.py - Tao dau vao cho llm_runtime (C++) tu 1 clip golden cua pkg_v1, va so ket qua C++ voi ref_runtime.py.
Dat cung thu muc voi ref_runtime.py (Colab / may co numpy; KHONG chay tren board).

  python3 make_cpp_inputs.py make --pkg <pkg_v1> --clip 0 --out in_clip00 [--ref-logits]
      -> in_clip00/ids.i32        prompt (da co <|AUDIO|> lap lai)
         in_clip00/rows.f32       hang audio [n,hidden] (encoder+projector bang ref_runtime, fallback 3 chu so)
         in_clip00/gen_fp32.i32   token FP32 cua golden (dung lam --teacher)
         in_clip00/gen_int8.i32   token INT8 cua golden
         (--ref-logits) in_clip00/ref_top5.txt  top-5 teacher-forced cua ref_runtime (de so voi --dump-top5 cua C++)
  python3 make_cpp_inputs.py compare --pkg <pkg_v1> --clip 0 --in in_clip00 --cpp top5_cpp.txt
      so top-1 / top-5 giua C++ (--teacher gen_fp32.i32 --dump-top5 top5_cpp.txt) va ref_runtime.py, va voi top-1 FP32 cua golden.
  python3 make_cpp_inputs.py text --pkg <pkg_v1> --tokens tokens.txt        giai ma token -> text (can transformers)
"""
import argparse, json, os, sys
import numpy as np
import ref_runtime as R


def load_clip(pkg_dir, clip):
    d = os.path.join(pkg_dir, "golden", f"clip_{int(clip):02d}")
    meta = json.load(open(os.path.join(d, "meta.json")))
    sh = meta["shapes"]
    rb = R.read_bin
    feats = rb(os.path.join(d, "input_features.f32"), "<f4", tuple(sh["input_features"]))[0]
    ids = rb(os.path.join(d, "input_ids.i32"), "<i4").astype(np.int64)
    fam = rb(os.path.join(d, "feature_attention_mask.i32"), "<i4")
    g8 = rb(os.path.join(d, "int8_gen_ids.i32"), "<i4").astype(np.int64)
    gf = rb(os.path.join(d, "fp32_gen_ids.i32"), "<i4").astype(np.int64)
    t1f = rb(os.path.join(d, "tf_fp32_top5_ids.i32"), "<i4").reshape(-1, 5)[:, 0]
    return meta, feats, ids, int(fam.sum()), g8, gf, t1f


def top5_ref(lg):
    ix = np.argsort(-lg, axis=1, kind="stable")[:, :5]
    return ix, np.take_along_axis(lg, ix, 1)


def cmd_make(a):
    R.FBDIG = a.digits
    pkg = R.Pkg(a.pkg)
    meta, feats, ids, L, g8, gf, t1f = load_clip(a.pkg, a.clip)
    m = R.Model(pkg, "member")
    rows = m.encode(feats, L)
    os.makedirs(a.out, exist_ok=True)
    ids.astype("<i4").tofile(os.path.join(a.out, "ids.i32"))
    rows.astype("<f4").tofile(os.path.join(a.out, "rows.f32"))
    gf.astype("<i4").tofile(os.path.join(a.out, "gen_fp32.i32"))
    g8.astype("<i4").tofile(os.path.join(a.out, "gen_int8.i32"))
    print(f"ids {len(ids)} | rows {rows.shape} | gen_fp32 {len(gf)} | gen_int8 {len(g8)} -> {a.out}")
    if a.ref_logits:
        lg = m.teacher_forced(ids, rows, gf)
        ix, vv = top5_ref(lg)
        with open(os.path.join(a.out, "ref_top5.txt"), "w") as f:
            for t in range(len(gf)):
                f.write(str(t) + "".join(f" {ix[t, i]} {vv[t, i]:.6f}" for i in range(5)) + "\n")
        lg.astype("<f4").tofile(os.path.join(a.out, "ref_logits.f32")) if a.dense else None
        print("ref top5 ->", os.path.join(a.out, "ref_top5.txt"))


def read_top5(path):
    out = []
    for ln in open(path):
        p = ln.split()
        out.append(([int(p[1 + 2 * i]) for i in range(5)], [float(p[2 + 2 * i]) for i in range(5)]))
    return out


def cmd_compare(a):
    meta, feats, ids, L, g8, gf, t1f = load_clip(a.pkg, a.clip)
    ref = read_top5(os.path.join(a.inp, "ref_top5.txt"))
    cpp = read_top5(a.cpp)
    n = min(len(ref), len(cpp), len(gf))
    top1 = sum(ref[t][0][0] == cpp[t][0][0] for t in range(n))
    top5 = sum(set(ref[t][0]) == set(cpp[t][0]) for t in range(n))
    fp = sum(cpp[t][0][0] == t1f[t] for t in range(n))
    fpr = sum(ref[t][0][0] == t1f[t] for t in range(n))
    dv = max(max(abs(x - y) for x, y in zip(ref[t][1], cpp[t][1])) for t in range(n))
    print(f"{n} vi tri | top-1 C++ == ref_runtime: {top1}/{n} | tap top-5 trung: {top5}/{n} | max|d top-5 logit| {dv:.3e}")
    print(f"top-1 so voi FP32 golden: C++ {fp}/{n} | ref_runtime {fpr}/{n}")


def cmd_text(a):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(os.path.join(a.pkg, "hf_processor"))
    toks = [int(x) for x in open(a.tokens).read().split()]
    print(tk.decode(toks, skip_special_tokens=True).strip())


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("make", "compare", "text"):
        s = sp.add_parser(nm); s.add_argument("--pkg", required=True)
        if nm != "text":
            s.add_argument("--clip", type=int, default=0)
        if nm == "make":
            s.add_argument("--out", required=True); s.add_argument("--ref-logits", action="store_true")
            s.add_argument("--dense", action="store_true"); s.add_argument("--digits", type=int, default=3)
        if nm == "compare":
            s.add_argument("--in", dest="inp", required=True); s.add_argument("--cpp", required=True)
        if nm == "text":
            s.add_argument("--tokens", required=True)
    a = ap.parse_args()
    {"make": cmd_make, "compare": cmd_compare, "text": cmd_text}[a.cmd](a)


if __name__ == "__main__":
    main()
