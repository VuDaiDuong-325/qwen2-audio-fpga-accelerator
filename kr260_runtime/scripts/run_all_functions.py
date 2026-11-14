#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_functions.py - Chay MOI function (moi GEMM) trong goi kv260_all tren bo tang toc (KV260) va so BIT-TRUE voi golden.

    gcc -O2 -o npu_run_case npu_run_case.c            # kr260_npu_driver.h canh ben
    sudo python3 run_all_functions.py --pkg ./kv260_all --repeat 3 --fclk-mhz 100

Cach dung thuong gap:
    --list                              liet ke function / case roi thoat
    --functions ffn,projector           chi chay cac function co ten chua cac chuoi nay
    --cases qkv,lm_head                 loc theo ten case
    --layers 0                          chi cac lop nay (mac dinh: tat ca lop co trong goi)
    --kdim auto|32|64|128               k_dim (mac dinh auto = chon theo K, giong qffn_common.pick_kdim)
    --skip-int32                        khong chay them lan kiem INT32 (chi INT8, nhanh hon)

Voi moi case:  (1) chay che do INT8-only (duong suy luan that) --repeat lan, lay thoi gian HW nho nhat, so golden INT8;
               (2) neu goi co golden INT32: chay them che do INT32-only 1 lan va so golden INT32 (kiem tra truoc requant).
Case FFN co chuoi: fc1 -> LUT GELU (CPU) -> fc2, dau vao fc2 la KET QUA PHAN CUNG cua fc1.
Ket thuc in bang tong ket va UOC TINH thoi gian HW cho ca model (= thoi gian case x so lan x m_full/M; chua gom CPU/chuyen du lieu).
Ma thoat: 0 = tat ca bit-true, 1 = co lech, 2 = loi phan cung/tham so. Bao cao JSON: --json.
Kiem thu khong can phan cung:  gcc -O2 -DNPU_SOFTSIM -o npu_run_case_sim npu_run_case.c ; ... --runner ./npu_run_case_sim
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def run_runner(cmd, quiet):
    """Chay npu_run_case. Neu that bai vi KHONG XIN DUOC bo nho CMA (loi tam thoi tren board dung chung) thi thu lai,
    de 'LOI cap phat' khong bi nham voi 'LECH BIT'. So lan thu lai: bien moi truong RUN_RETRY (mac dinh 4)."""
    import time
    if not quiet:
        print("   $", " ".join(cmd), flush=True)
    tries = 1 + max(0, int(os.environ.get("RUN_RETRY", "4")))
    for att in range(tries):
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        alloc_fail = p.returncode == 2 and "CMA that bai" in p.stderr
        if not alloc_fail or att == tries - 1:
            break
        print("   (xin CMA that bai, thu lai %d/%d)" % (att + 1, tries - 1), flush=True)
        try:
            os.sync()
        except (AttributeError, OSError):
            pass
        time.sleep(1.0 + att)
    summary = None
    for line in reversed(p.stdout.strip().splitlines()):
        if line.startswith("{"):
            try:
                summary = json.loads(line)
            except ValueError:
                pass
            break
    if p.returncode != 0 or not quiet:
        err = p.stderr.strip()
        if err:
            print("   " + err.replace("\n", "\n   "))
    return p.returncode, summary


def _slice_cols(src, dst, rows, N, n0, n1, nbytes):
    import numpy as np
    mm = np.memmap(src, dtype=np.uint8, mode="r", shape=(rows, N * nbytes))
    np.ascontiguousarray(mm[:, n0 * nbytes:n1 * nbytes]).tofile(dst)
    del mm


def run_split(runner, c, F, A, kdim, repeat, mode, wd, tag, chunk, quiet):
    """Chay GEMM M x K x N thanh nhieu manh theo cot N (A dung chung). B,params,golden duoc cat theo cot vao file tam.
    Tra (rc, summary_gop). rc = 0 neu moi manh bit-true; 2 neu co loi tham so/phan cung; 1 neu lech bit."""
    M, K, N = c["M"], c["K"], c["N"]
    chunk = max(16, chunk // 16 * 16)
    agg = {"hw_ms_min": 0.0, "hw_ms_avg": 0.0, "load_ms": 0.0, "read_ms": 0.0, "kdim": None}
    worst = 0
    gkey, gbytes = ("golden32", 4) if mode == "int32" else ("golden8", 1)
    for i, n0 in enumerate(range(0, N, chunk)):
        n1 = min(N, n0 + chunk)
        tmp = {k: os.path.join(wd, "%s_chunk_%s.bin" % (tag, k)) for k in ("B", "params", gkey)}
        _slice_cols(F["B"], tmp["B"], K, N, n0, n1, 1)
        _slice_cols(F["params"], tmp["params"], 1, N, n0, n1, 16)
        _slice_cols(F[gkey], tmp[gkey], M, N, n0, n1, gbytes)
        cmd = [runner, "--A", A, "--B", tmp["B"], "--params", tmp["params"], "--M", str(M), "--K", str(K), "--N", str(n1 - n0),
               "--kdim", str(kdim), "--repeat", str(repeat), "--mode", mode, "--" + gkey, tmp[gkey]]
        if not quiet:
            print("   manh %d: cot [%d,%d)" % (i, n0, n1), flush=True)
        rc, s = run_runner(cmd, True)
        for f in tmp.values():
            try:
                os.remove(f)
            except OSError:
                pass
        if rc == 2 or s is None:
            return 2, None
        worst = max(worst, rc)
        for k in ("hw_ms_min", "hw_ms_avg", "load_ms", "read_ms"):
            agg[k] += s[k]
        agg["kdim"] = agg["kdim"] or s["kdim"]
    return worst, agg


def group_key(cid):
    return re.sub(r"_h\d+", "_h*", re.sub(r"^L\d+[./]", "", cid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="./kv260_all")
    ap.add_argument("--runner", default="./npu_run_case")
    ap.add_argument("--workdir", default="./hw_out_all")
    ap.add_argument("--functions", default="")
    ap.add_argument("--cases", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--exclude", default="", help="bo qua case/function co chuoi nay trong ten (vd: lm_head), phan cach bang dau phay")
    ap.add_argument("--split-n", type=int, default=0,
                    help="case co N > gia tri nay se duoc chay thanh nhieu GEMM theo cot (moi manh <= N cot, boi so cua 16). 0 = khong chia. "
                         "Ket qua van bit-true vi cac cot doc lap")
    ap.add_argument("--max-ddr-mb", type=float, default=0.0,
                    help="bo qua case can nhieu hon X MB DDR reserved (0 = khong gioi han). Dung de tranh case lon khi chua xac nhan vung reserved")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--kdim", default="auto")
    ap.add_argument("--fclk-mhz", type=float, default=100.0)
    ap.add_argument("--skip-int32", action="store_true")
    ap.add_argument("--skip-sha", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", default="report_all.json")
    args = ap.parse_args()

    idx = json.load(open(os.path.join(args.pkg, "INDEX.json"), encoding="utf-8"))
    want_f = [s for s in args.functions.split(",") if s]
    want_c = [s for s in args.cases.split(",") if s]
    want_l = {int(s) for s in args.layers.split(",") if s.strip() != ""}

    plan = []
    for fname, finfo in idx["functions"].items():
        if want_f and not any(w in fname for w in want_f):
            continue
        man = json.load(open(os.path.join(args.pkg, finfo["dir"], "function.json"), encoding="utf-8"))
        for c in man["cases"]:
            if want_c and not any(w in c["id"] for w in want_c):
                continue
            if want_l and c.get("layer") not in want_l and c.get("layer") is not None:
                continue
            plan.append((fname, finfo["dir"], c))

    excl = [s_ for s_ in args.exclude.split(",") if s_]

    def ddr_mb(c):                      # cung cong thuc npu_layout() trong kr260_npu_driver.h
        a4 = lambda x: (x + 4095) // 4096 * 4096
        M, K, N = c["M"], c["K"], c["N"]
        if args.split_n > 0 and N > args.split_n and not c.get("post_lut"):
            N = args.split_n // 16 * 16          # moi manh chi can DDR cho split_n cot
        c16 = lambda x: (x + 15) // 16 * 16
        o_b = a4(M * c16(K)); o_c = a4(o_b + K * c16(N)); o_c32 = a4(o_c + M * c16(N))
        o_p = a4(o_c32 + M * 4 * c16(N)); return a4(o_p + N * 16) / 1048576.0
    kept, skipped = [], []
    for fname, d, c in plan:
        mb = ddr_mb(c)
        if any(w in fname or w in c["id"] for w in excl) or (args.max_ddr_mb > 0 and mb > args.max_ddr_mb):
            skipped.append((fname, c["id"], mb))
        else:
            kept.append((fname, d, c))
    plan = kept
    for fname, cid, mb in skipped:
        print("BO QUA %s / %s (can %.1f MB DDR)" % (fname, cid, mb))

    if args.list:
        for fname, d, c in plan:
            print("%-38s %-22s %5dx%5dx%-6d kd_auto=%-3s mode=%-5s %s" % (fname, c["id"], c["M"], c["K"], c["N"], c.get("kdim_auto"),
                                                                       c.get("mode", "int8"), c.get("phase")))
        print(len(plan), "case")
        return 0
    if not plan:
        print("khong co case nao khop bo loc")
        return 2

    os.makedirs(args.workdir, exist_ok=True)
    report = {"pkg": os.path.abspath(args.pkg), "runner": args.runner, "kdim": args.kdim, "fclk_mhz": args.fclk_mhz,
              "time": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": []}
    sha_cache = {}
    bad_total, err_total, chain_out = 0, 0, {}

    for n, (fname, fdir, c) in enumerate(plan, 1):
        base = os.path.join(args.pkg, fdir)
        F = {k: os.path.join(base, v["path"]) for k, v in c["files"].items()}
        if not args.skip_sha:
            for k, v in c["files"].items():
                p = F[k]
                if p not in sha_cache:
                    sha_cache[p] = sha256_file(p)
                if sha_cache[p] != v["sha256"]:
                    print("LOI: sha256 khong khop (file hong khi copy?):", p)
                    return 2
        wd = os.path.join(args.workdir, fname)
        os.makedirs(wd, exist_ok=True)
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", c["id"])
        print("[%d/%d] %s / %s  M=%d K=%d N=%d  mode=%s" % (n, len(plan), fname, c["id"], c["M"], c["K"], c["N"], c.get("mode", "int8")), flush=True)
        A = F["A"]
        if c.get("A_from"):
            key = (fname, c.get("layer"), c["A_from"])
            if key in chain_out:
                A = chain_out[key]
                print("   (dau vao = ket qua PHAN CUNG cua '%s' qua LUT)" % c["A_from"])
        mode = c.get("mode", "int8")
        cmd = [args.runner, "--A", A, "--B", F["B"], "--params", F["params"], "--M", str(c["M"]), "--K", str(c["K"]),
               "--N", str(c["N"]), "--kdim", str(args.kdim), "--repeat", str(args.repeat)]
        out_lut = None
        if mode == "int32":
            cmd += ["--mode", "int32", "--golden32", F["golden32"]]
        else:
            cmd += ["--mode", "int8", "--golden8", F["golden8"]]
            if c.get("post_lut"):
                out_lut = os.path.join(wd, tag + "_G8.bin")
                cmd += ["--lut", os.path.join(base, c["post_lut"]["lut"]), "--out_lut", out_lut,
                        "--golden_lut", os.path.join(base, c["post_lut"]["golden_out"])]
        split = args.split_n > 0 and c["N"] > args.split_n and not c.get("post_lut")
        if split:
            print("   (N=%d > %d: chia thanh %d manh theo cot)" % (c["N"], args.split_n, -(-c["N"] // (args.split_n // 16 * 16))), flush=True)
            rc, s = run_split(args.runner, c, F, A, args.kdim, args.repeat, "int32" if mode == "int32" else "int8", wd, tag,
                              args.split_n, args.quiet)
        else:
            rc, s = run_runner(cmd, args.quiet)
        rec = {"function": fname, "id": c["id"], "M": c["M"], "K": c["K"], "N": c["N"], "layer": c.get("layer"), "phase": c.get("phase"),
               "calls_per_forward": c.get("calls_per_forward"), "m_full": c.get("m_full"), "exit_int8": rc, "summary": s}
        if rc == 2 or s is None:
            err_total += 1
            report["cases"].append(rec)
            print("   LOI phan cung/tham so")
            continue
        bad = rc != 0
        if out_lut and rc == 0:
            chain_out[(fname, c.get("layer"), c["name"])] = out_lut
        rec["bit_true_int8" if mode != "int32" else "bit_true_int32"] = rc == 0
        if mode != "int32" and "golden32" in F and not args.skip_int32:
            if split:
                rc2, s2 = run_split(args.runner, c, F, A, args.kdim, 1, "int32", wd, tag, args.split_n, True)
            else:
                rc2, s2 = run_runner([args.runner, "--A", A, "--B", F["B"], "--params", F["params"], "--M", str(c["M"]), "--K", str(c["K"]),
                                      "--N", str(c["N"]), "--kdim", str(args.kdim), "--mode", "int32", "--golden32", F["golden32"],
                                      "--repeat", "1"], args.quiet)
            rec["bit_true_int32"] = (rc2 == 0) if rc2 != 2 else None
            if rc2 == 2 or s2 is None:               # loi cap phat/tham so o luot int32: KHONG phai lech bit
                err_total += 1
                rec["error_int32"] = True
                print("   LOI phan cung/tham so (luot INT32) - khong ket luan duoc bit-true")
                report["cases"].append(rec)
                continue
            bad = bad or rc2 != 0
        hw = s["hw_ms_min"]
        tiles = -(-c["M"] // 16) * -(-c["N"] // 16)
        ideal = tiles * c["K"] / (args.fclk_mhz * 1e3)
        rec.update({"kdim_used": s["kdim"], "hw_ms_min": hw, "hw_ms_avg": s["hw_ms_avg"], "load_ms": s["load_ms"], "read_ms": s["read_ms"],
                    "gops": (2.0 * c["M"] * c["K"] * c["N"] / (hw * 1e6)) if hw > 0 else None,
                    "ideal_ms": ideal, "efficiency": (ideal / hw) if hw > 0 else None})
        bad_total += int(bad)
        report["cases"].append(rec)
        print("   -> %s | kd=%s | HW %.3f ms | %s GOPS | hieu suat vs ly tuong %s" % (
            "BIT-TRUE" if not bad else "LECH BIT", s["kdim"], hw, ("%.2f" % rec["gops"]) if rec["gops"] else "n/a",
            ("%.1f%%" % (100 * rec["efficiency"])) if rec["efficiency"] else "n/a"), flush=True)

    # ---------------- bang tong ket ----------------
    print("\n================ TONG KET ================")
    print("%-38s %-20s %-18s %3s %9s %7s %6s %4s" % ("function", "case", "MxKxN", "kd", "HW ms", "GOPS", "eff%", "bit"))
    for r in report["cases"]:
        if "hw_ms_min" not in r:
            print("%-38s %-20s %-18s  -- loi --" % (r["function"], r["id"], "%dx%dx%d" % (r["M"], r["K"], r["N"])))
            continue
        ok = all(v for k, v in r.items() if k.startswith("bit_true"))
        print("%-38s %-20s %-18s %3s %9.3f %7.2f %6.1f %4s" % (r["function"], r["id"], "%dx%dx%d" % (r["M"], r["K"], r["N"]), r["kdim_used"],
                                                               r["hw_ms_min"], r["gops"] or 0, 100 * (r["efficiency"] or 0), "OK" if ok else "LECH"))
    est = {}
    for r in report["cases"]:
        if "hw_ms_min" not in r:
            continue
        g = est.setdefault(r["function"], {}).setdefault(group_key(r["id"]), {"ms": [], "calls": r["calls_per_forward"], "phase": r["phase"]})
        g["ms"].append(r["hw_ms_min"] * (r["m_full"] / r["M"] if r["phase"] != "decode" else 1.0))
    print("\nUOC TINH thoi gian HW ca model (chi tinh toan tren PL; case dai dien theo lop; KHONG gom CPU/chuyen du lieu/softmax...):")
    tot = {"prefill": 0.0, "decode": 0.0}
    report["estimate_ms"] = {}
    for fname, groups in est.items():
        t = {"prefill": 0.0, "decode": 0.0}
        for k, g in groups.items():
            t["decode" if g["phase"] == "decode" else "prefill"] += sum(g["ms"]) / len(g["ms"]) * g["calls"]
        report["estimate_ms"][fname] = t
        for ph in t:
            tot[ph] += t[ph]
        print("  %-38s prefill/encoder %10.1f ms | decode/token %8.2f ms" % (fname, t["prefill"], t["decode"]))
    print("  %-38s prefill/encoder %10.1f ms | decode/token %8.2f ms" % ("TONG", tot["prefill"], tot["decode"]))
    report["estimate_total_ms"] = tot
    ok_all = bad_total == 0 and err_total == 0
    report["bit_true_all"] = ok_all
    print("\nTONG KET: %s   (bao cao: %s)" % ("BIT-TRUE 100%% tren %d case" % len(report["cases"]) if ok_all else
                                             "CO LECH BIT (%d) / LOI (%d)" % (bad_total, err_total), args.json))
    json.dump(report, open(args.json, "w"), indent=2)
    return 0 if ok_all else (2 if err_total else 1)


if __name__ == "__main__":
    sys.exit(main())
