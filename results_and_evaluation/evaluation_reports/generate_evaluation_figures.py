#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Publication-Ready Academic Figures (300 DPI PNG + Vector PDF)
for the Qwen2-Audio INT8 GEMM Hardware Accelerator on Xilinx Kria KR260.

Input JSON reports:
  1. compare_result/int8_ps_pl_compare/pspl_compare_report.json
  2. compare_result/fp32_vs_kr260_int8_report.json
  3. kr260/tools/report_all.json
  4. Notebook/qint8_report_quality_colab/quiz_report.json

Output directory:
  result_evaluation/figures/
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set clean academic styling (IEEE/ACM style)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PSPL_JSON = os.path.join(ROOT_DIR, "compare_result", "int8_ps_pl_compare", "pspl_compare_report.json")
FP32_JSON = os.path.join(ROOT_DIR, "compare_result", "fp32_vs_kr260_int8_report.json")
GEMM_JSON = os.path.join(ROOT_DIR, "kr260", "tools", "report_all.json")
QUIZ_JSON = os.path.join(ROOT_DIR, "Notebook", "qint8_report_quality_colab", "quiz_report.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")

os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(fig, basename):
    png_path = os.path.join(OUT_DIR, f"{basename}.png")
    pdf_path = os.path.join(OUT_DIR, f"{basename}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {png_path} and {pdf_path}")


def short_wav_label(w):
    return w.replace("_voiceassistant.wav", "_va").replace("_alpacaaudio.wav", "_alp").replace(".wav", "")


def plot_fig1_e2e_speedup(pspl_data):
    """Figure 1: Per-WAV End-to-End Inference Time (Mode 1 vs Mode 2) & Speedup Across 20 WAVs."""
    labels = [short_wav_label(d["wav"]) for d in pspl_data]
    m1_tot = [d["m1"]["total_s"] for d in pspl_data]
    m2_tot = [d["m2"]["total_s"] for d in pspl_data]
    speedups = [d["speedup"] for d in pspl_data]
    tokens = [d["m2"]["tokens"] for d in pspl_data]

    x = np.arange(len(labels))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7.2), gridspec_kw={"height_ratios": [1.55, 1.0]}, sharex=True
    )

    # Top subplot: Execution time bars
    rects1 = ax1.bar(
        x - width / 2, m1_tot, width, label="Mode 1: 100% PS (4-Core ARM Cortex-A53 NEON)",
        color="#4C72B0", edgecolor="black", linewidth=0.6
    )
    rects2 = ax1.bar(
        x + width / 2, m2_tot, width, label="Mode 2: PS + PL (KR260 16x16 INT8 GEMM @ 200 MHz)",
        color="#DD8452", edgecolor="black", linewidth=0.6
    )

    for r in rects1:
        h = r.get_height()
        ax1.annotate(
            f"{h:.0f}s", xy=(r.get_x() + r.get_width() / 2, h),
            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7, color="#1f3b6d"
        )
    for r in rects2:
        h = r.get_height()
        ax1.annotate(
            f"{h:.1f}s", xy=(r.get_x() + r.get_width() / 2, h),
            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#8c3b0c"
        )

    ax1.axvline(13.5, color="#333333", linestyle="-.", linewidth=1.2, alpha=0.8)
    ax1.text(6.5, max(m1_tot) * 0.96, "In-Domain Original Set (01 - 14)", ha="center", fontsize=9.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#eef4fa", edgecolor="#4C72B0", alpha=0.9))
    ax1.text(16.5, max(m1_tot) * 0.96, "Unseen OOD Set (15 - 20)", ha="center", fontsize=9.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#fdf2ea", edgecolor="#DD8452", alpha=0.9))

    ax1.set_ylabel("End-to-End Latency (seconds)")
    ax1.set_ylim(0, max(m1_tot) * 1.13)
    ax1.set_title("End-to-End Inference Latency & Hardware Speedup Across 20 Audio Prompts (1,451 Tokens, 100% Bit-True)")
    ax1.legend(loc="upper left", framealpha=0.95)

    # Bottom subplot: Speedup curve + Generated Token count annotations
    ax2.axvline(13.5, color="#333333", linestyle="-.", linewidth=1.2, alpha=0.8)
    bars_sp = ax2.bar(x, speedups, width=0.52, color="#55A868", alpha=0.85, edgecolor="black", linewidth=0.6, label="Per-WAV End-to-End Speedup (x)")
    avg_sp = sum(m1_tot) / sum(m2_tot)
    ax2.axhline(avg_sp, color="#C44E52", linestyle="--", linewidth=1.8, label=f"Overall Aggregate Speedup: {avg_sp:.2f}x (6445.3s -> 534.2s)")

    for idx, (b, sp, tok) in enumerate(zip(bars_sp, speedups, tokens)):
        ax2.annotate(
            f"{sp:.1f}x\n({tok}t)", xy=(b.get_x() + b.get_width() / 2, sp),
            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7, fontweight="bold"
        )

    ax2.set_ylabel("Speedup (x) [PS+PL vs PS]")
    ax2.set_xlabel("Audio Prompt ID (01-14: Original | 15-17: VoiceAssistant-400K OOD | 18-20: Alpaca-Audio OOD)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylim(0, max(speedups) * 1.25)
    ax2.legend(loc="upper right", framealpha=0.95)

    fig.tight_layout()
    save_fig(fig, "fig1_e2e_speedup_20wavs")


def plot_fig2_phase_and_runtime_breakdown(pspl_data):
    """Figure 2: Phase-by-Phase Speedup, Mode 2 Runtime Breakdown, and In-Domain vs OOD Comparison."""
    fig = plt.figure(figsize=(14, 4.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.25, 1.05, 1.15], wspace=0.28)

    # Calculate aggregates
    m1_enc = sum(d["m1"]["encode_s"] for d in pspl_data)
    m1_pre = sum(d["m1"]["prefill_s"] for d in pspl_data)
    m1_dec = sum(d["m1"]["decode_s"] for d in pspl_data)
    m1_tot = sum(d["m1"]["total_s"] for d in pspl_data)

    m2_enc = sum(d["m2"]["encode_s"] for d in pspl_data)
    m2_pre = sum(d["m2"]["prefill_s"] for d in pspl_data)
    m2_dec = sum(d["m2"]["decode_s"] for d in pspl_data)
    m2_tot = sum(d["m2"]["total_s"] for d in pspl_data)

    # Panel A: Phase Speedup (Log-scale latency + Speedup badges)
    ax1 = fig.add_subplot(gs[0])
    phases = ["Audio Encoder\n(M=1500/750)", "LLM Prefill\n(M=N_prompt)", "LLM Decode\n(M=1 GEMV)", "End-to-End\nTotal (20 WAVs)"]
    m1_vals = [m1_enc, m1_pre, m1_dec, m1_tot]
    m2_vals = [m2_enc, m2_pre, m2_dec, m2_tot]
    sp_vals = [m1_enc / m2_enc, m1_pre / m2_pre, m1_dec / m2_dec, m1_tot / m2_tot]

    y = np.arange(len(phases))
    h = 0.36
    ax1.barh(y + h / 2, m1_vals, h, label="Mode 1: 100% PS (ARM CPU)", color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax1.barh(y - h / 2, m2_vals, h, label="Mode 2: PS + PL (KR260 FPGA)", color="#DD8452", edgecolor="black", linewidth=0.6)
    ax1.set_xscale("log")
    ax1.set_xlim(20, 50000)
    ax1.set_yticks(y)
    ax1.set_yticklabels(phases)
    ax1.invert_yaxis()
    ax1.set_xlabel("Total Time Across 20 WAVs (seconds, log scale)")
    ax1.set_title("(a) Phase-by-Phase Latency & Speedup")
    ax1.legend(loc="center right", fontsize=8, framealpha=0.95)

    for i, (v1, v2, sp) in enumerate(zip(m1_vals, m2_vals, sp_vals)):
        ax1.text(v1 * 1.12, i + h / 2, f"{v1:.1f}s", va="center", fontsize=8, color="#1f3b6d")
        ax1.text(v2 * 1.12, i - h / 2, f"{v2:.1f}s ({sp:.2f}x)", va="center", fontsize=8, fontweight="bold", color="#8c3b0c")

    # Panel B: Mode 2 Runtime Breakdown (PL Compute vs PS Compute vs Xfer)
    ax2 = fig.add_subplot(gs[1])
    hw_s = sum(d["m2"]["hw_compute_s"] for d in pspl_data)
    ps_s = sum(d["m2"]["ps_compute_s"] for d in pspl_data)
    xf_s = sum(d["m2"]["xfer_s"] for d in pspl_data)

    pie_vals = [hw_s, ps_s, xf_s]
    pie_labels = [
        f"PL GEMM Compute\n({hw_s:.1f}s | {100*hw_s/m2_tot:.1f}%)",
        f"PS Non-GEMM Ops\n({ps_s:.1f}s | {100*ps_s/m2_tot:.1f}%)",
        f"PS<->PL DMA & Cache\n({xf_s:.1f}s | {100*xf_s/m2_tot:.1f}%)"
    ]
    pie_colors = ["#DD8452", "#4C72B0", "#55A868"]
    wedges, texts = ax2.pie(
        pie_vals, labels=pie_labels, colors=pie_colors, startangle=140,
        explode=(0.03, 0.03, 0.08),
        wedgeprops=dict(width=0.48, edgecolor="black", linewidth=0.8),
        textprops=dict(fontsize=8.2, fontweight="bold")
    )
    ax2.text(0, 0, f"Mode 2 Total\n534.20 s\n(26.71 s/wav)", ha="center", va="center", fontsize=9, fontweight="bold")
    ax2.set_title("(b) Mode 2 (PS + PL) Time Breakdown")

    # Panel C: Original 14 vs Unseen 6 OOD Speedup by Phase
    ax3 = fig.add_subplot(gs[2])
    orig = pspl_data[:14]
    ood = pspl_data[14:]

    orig_sp = [
        sum(d["m1"]["encode_s"] for d in orig) / sum(d["m2"]["encode_s"] for d in orig),
        sum(d["m1"]["prefill_s"] for d in orig) / sum(d["m2"]["prefill_s"] for d in orig),
        sum(d["m1"]["decode_s"] for d in orig) / sum(d["m2"]["decode_s"] for d in orig),
        sum(d["m1"]["total_s"] for d in orig) / sum(d["m2"]["total_s"] for d in orig),
    ]
    ood_sp = [
        sum(d["m1"]["encode_s"] for d in ood) / sum(d["m2"]["encode_s"] for d in ood),
        sum(d["m1"]["prefill_s"] for d in ood) / sum(d["m2"]["prefill_s"] for d in ood),
        sum(d["m1"]["decode_s"] for d in ood) / sum(d["m2"]["decode_s"] for d in ood),
        sum(d["m1"]["total_s"] for d in ood) / sum(d["m2"]["total_s"] for d in ood),
    ]

    cats = ["Encoder", "Prefill", "Decode", "End-to-End"]
    x3 = np.arange(len(cats))
    w3 = 0.35
    b1 = ax3.bar(x3 - w3 / 2, orig_sp, w3, label="Original Set (01-14)", color="#4C72B0", edgecolor="black", linewidth=0.6)
    b2 = ax3.bar(x3 + w3 / 2, ood_sp, w3, label="Unseen OOD Set (15-20)", color="#55A868", edgecolor="black", linewidth=0.6)

    for b in b1:
        ax3.annotate(f"{b.get_height():.1f}x", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(-2, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    for b in b2:
        ax3.annotate(f"{b.get_height():.1f}x", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(3, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax3.set_xticks(x3)
    ax3.set_xticklabels(cats)
    ax3.set_ylabel("Speedup Factor (x)")
    ax3.set_ylim(0, 37)
    ax3.set_title("(c) In-Domain vs. OOD Speedup Consistency")
    ax3.legend(loc="upper right", fontsize=8)

    save_fig(fig, "fig2_phase_and_runtime_breakdown")


def plot_fig3_fp32_vs_kr260_accuracy(fp32_data, quiz_data):
    """Figure 3: Numerical & Semantic Fidelity (KR260 INT8 vs PyTorch FP32 / BF16)."""
    labels = [short_wav_label(d["wav"]) for d in fp32_data]
    tf_top1 = [d["tf_top1_pct"] for d in fp32_data]
    tf_top5 = [d["tf_top5_pct"] for d in fp32_data]
    int8_sem = [d["kr260_vs_fp32"]["sem_cos_pct"] for d in fp32_data]
    bf16_sem = [d["bf16_vs_fp32"]["sem_cos_pct"] for d in fp32_data]

    fig = plt.figure(figsize=(14.5, 7.8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.48, 1.0], height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.25)

    x = np.arange(len(labels))

    # Top-Left: Per-WAV Teacher-Forced Top-1 & Top-5 Token Agreement
    ax1 = fig.add_subplot(gs[0, 0])
    w = 0.38
    ax1.bar(x - w / 2, tf_top5, w, label="Teacher-Forced Top-5 Agreement (Avg: 99.79% | OOD: 100.0%)", color="#64B5CD", edgecolor="black", linewidth=0.5)
    ax1.bar(x + w / 2, tf_top1, w, label="Teacher-Forced Top-1 Agreement (Avg: 90.69% | OOD: 93.90%)", color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax1.axvline(13.5, color="#333333", linestyle="-.", linewidth=1.1)
    ax1.axhline(90.69, color="#1f3b6d", linestyle="--", linewidth=1.2)
    ax1.set_ylim(60, 106)
    ax1.set_ylabel("Token Agreement (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_title("(a) Teacher-Forced Token Agreement: KR260 INT8 vs. FP32")
    ax1.legend(loc="lower right", fontsize=8)

    # Bottom-Left: Per-WAV Semantic Cosine Similarity (all-MiniLM-L6-v2)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(x - w / 2, bf16_sem, w, label="BF16 GPU vs. FP32 GPU (Avg: 94.75% | OOD: 96.56%)", color="#CCB974", edgecolor="black", linewidth=0.5)
    ax2.bar(x + w / 2, int8_sem, w, label="KR260 INT8 (PS+PL) vs. FP32 GPU (Avg: 83.98% | OOD: 91.57%)", color="#DD8452", edgecolor="black", linewidth=0.5)
    ax2.axvline(13.5, color="#333333", linestyle="-.", linewidth=1.1)
    ax2.axhline(91.57, color="#8c3b0c", linestyle="--", linewidth=1.2, label="KR260 INT8 OOD Avg Semantic Similarity (91.57%)")
    ax2.set_ylim(30, 108)
    ax2.set_ylabel("Semantic Cosine Similarity (%)")
    ax2.set_xlabel("Audio Prompt ID (01-14: Original | 15-20: Unseen Out-of-Distribution)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_title("(b) End-to-End Semantic Cosine Similarity (all-MiniLM-L6-v2)")
    ax2.legend(loc="lower right", fontsize=8)

    # Top-Right: Group Summary Comparison (All 20 vs Original 14 vs Unseen 6 OOD)
    ax3 = fig.add_subplot(gs[0, 1])
    groups = ["All 20 WAVs\n(1,428 tok)", "Original 14 WAVs\n(1,002 tok)", "Unseen 6 OOD\n(426 tok)"]
    top1_grp = [90.69, 89.32, 93.90]
    top5_grp = [99.79, 99.70, 100.00]
    sem_grp = [83.98, 80.73, 91.57]

    xg = np.arange(len(groups))
    wg = 0.25
    b_t5 = ax3.bar(xg - wg, top5_grp, wg, label="TF Top-5 (%)", color="#64B5CD", edgecolor="black", linewidth=0.6)
    b_t1 = ax3.bar(xg, top1_grp, wg, label="TF Top-1 (%)", color="#4C72B0", edgecolor="black", linewidth=0.6)
    b_sm = ax3.bar(xg + wg, sem_grp, wg, label="Semantic Cosine (%)", color="#DD8452", edgecolor="black", linewidth=0.6)

    for bars in [b_t5, b_t1, b_sm]:
        for b in bars:
            ax3.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax3.set_xticks(xg)
    ax3.set_xticklabels(groups)
    ax3.set_ylim(65, 108)
    ax3.set_ylabel("Score (%)")
    ax3.set_title("(c) Accuracy Generalization: In-Domain vs. Unseen OOD")
    ax3.legend(loc="lower right", fontsize=8)

    # Bottom-Right: 3-Tier Verification Summary (50-Clip Calibration + 150-Question Audio Quiz + KR260 Bit-True)
    ax4 = fig.add_subplot(gs[1, 1])
    tier_metrics = [
        "150-Quiz OOD\nFP32 Baseline",
        "150-Quiz OOD\nBF16 Autocast",
        "150-Quiz OOD\nINT8 P1.5",
        "50-Clip (3410t)\nTF Top-1 Match",
        "KR260 PS+PL\nvs PS Bit-True"
    ]
    tier_vals = [74.00, 71.33, 71.33, 91.96, 100.00]
    tier_colors = ["#4C72B0", "#CCB974", "#DD8452", "#55A868", "#8172B2"]

    xt = np.arange(len(tier_metrics))
    bt = ax4.bar(xt, tier_vals, width=0.52, color=tier_colors, edgecolor="black", linewidth=0.6)
    for b, val in zip(bt, tier_vals):
        ax4.annotate(f"{val:.1f}%", xy=(b.get_x() + b.get_width() / 2, val),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax4.annotate("McNemar p = 0.344\n(Statistically Indistinguishable\n& Identical to BF16: 107/150)",
                 xy=(2, 71.33), xytext=(1.1, 86),
                 arrowprops=dict(arrowstyle="->", color="#8c3b0c", lw=1.2),
                 fontsize=7.8, fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#fdf2ea", edgecolor="#DD8452"))

    ax4.set_xticks(xt)
    ax4.set_xticklabels(tier_metrics, fontsize=7.8)
    ax4.set_ylim(50, 110)
    ax4.set_ylabel("Accuracy / Agreement (%)")
    ax4.set_title("(d) Multi-Tier Quantization & Hardware Verification")

    save_fig(fig, "fig3_fp32_vs_kr260_accuracy")


def plot_fig4_roofline_and_gemm_gops(gemm_data):
    """Figure 4: Measured GOPS Across Model GEMM Shapes & Roofline Model of KR260 16x16 Accelerator."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.25, 1.05]})

    # Representative GEMM layers from report_all.json & bench.log
    layer_names = [
        "Enc Conv1\n(48x384x1280)",
        "Enc Conv2\n(48x3840x1280)",
        "Enc QKV\n(48x1280x3840)",
        "Enc OutProj\n(48x1280x1280)",
        "Enc FFN FC1\n(48x1280x5120)",
        "Enc FFN FC2\n(48x5120x1280)",
        "Enc Full FFN\n(1500x1280x5120)",
        "Audio Proj\n(48x1280x896)",
        "LLM Prefill QKV\n(48x896x1152)",
        "LLM Prefill FFN\n(48x896x9728)",
        "LLM Decode FFN\n(M=1x896x9728)",
        "LLM Decode Head\n(M=1x896x156032)"
    ]
    gops_vals = [
        94.49,
        100.80,
        100.67,
        96.25,
        101.02,
        100.83,
        101.29,
        94.82,
        95.14,
        100.41,
        2.32,
        2.38
    ]
    colors = ["#4C72B0"] * 8 + ["#55A868"] * 2 + ["#C44E52"] * 2

    y = np.arange(len(layer_names))
    bars = ax1.barh(y, gops_vals, color=colors, edgecolor="black", linewidth=0.6, height=0.68)
    ax1.axvline(102.4, color="#8c3b0c", linestyle="--", linewidth=1.8, label="Theoretical Peak @ 200 MHz: 102.4 GOPS")

    for b, g in zip(bars, gops_vals):
        eff = 100.0 * g / 102.4
        ax1.text(g + 1.5, b.get_y() + b.get_height() / 2, f"{g:.2f} GOPS ({eff:.1f}%)",
                 va="center", fontsize=8, fontweight="bold")

    ax1.set_yticks(y)
    ax1.set_yticklabels(layer_names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 142)
    ax1.set_xlabel("Measured Hardware Throughput (GOPS)")
    ax1.set_title("(a) Measured KR260 PL Throughput Across Model GEMM Layers")
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel B: Roofline Model (Log-Log Plot)
    ax2_oi = np.logspace(-0.5, 3.2, 300)
    peak_gops = 102.4          # 16x16x2 ops/cycle @ 200 MHz
    ddr_bw_gbs = 2.45          # Measured sustained AXI4 128-bit DMA bandwidth (~2.45 GB/s)
    m1_ceil_gops = 6.4         # M=1 single-row ceiling on 16x16 OS array (1/16 * 102.4 GOPS)

    roof_full = np.minimum(peak_gops, ddr_bw_gbs * ax2_oi)
    roof_m1 = np.minimum(m1_ceil_gops, ddr_bw_gbs * ax2_oi)

    ax2.loglog(ax2_oi, roof_full, color="#1f3b6d", linewidth=2.2, label="16x16 Systolic Roofline (102.4 GOPS Peak, 2.45 GB/s AXI)")
    ax2.loglog(ax2_oi, roof_m1, color="#C44E52", linestyle="--", linewidth=1.6, label="M=1 Single-Row Array Ceiling (6.4 GOPS = 1/16 Peak)")

    # Plot operating points
    # Operational Intensity OI = (2*M*N*K) / (M*K + K*N + 4*M*N)
    pts = [
        ("Enc Full FFN\n(101.3 GOPS)", (2.0 * 1500 * 1280 * 5120) / (1500 * 1280 + 1280 * 5120 + 4 * 1500 * 5120), 101.29, "#4C72B0", "o"),
        ("Enc QKV/Conv2\n(100.8 GOPS)", (2.0 * 48 * 1280 * 3840) / (48 * 1280 + 1280 * 3840 + 4 * 48 * 3840), 100.80, "#55A868", "s"),
        ("LLM Prefill FFN\n(100.4 GOPS)", (2.0 * 48 * 896 * 9728) / (48 * 896 + 896 * 9728 + 4 * 48 * 9728), 100.41, "#DD8452", "^"),
        ("LLM Decode (M=1)\n(2.38 GOPS)", (2.0 * 1 * 896 * 9728) / (1 * 896 + 896 * 9728 + 4 * 1 * 9728), 2.38, "#C44E52", "D"),
    ]
    for lbl, oi, g, col, mk in pts:
        ax2.scatter([oi], [g], color=col, s=75, marker=mk, zorder=5, edgecolors="black", linewidth=0.8)
        offset = (-18, -32) if "101.3" in lbl else ((8, 12) if "100.8" in lbl else ((-90, -24) if "100.4" in lbl else (10, -5)))
        ax2.annotate(lbl, xy=(oi, g), xytext=offset, textcoords="offset points",
                     fontsize=7.8, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=col, lw=0.9))

    ax2.set_xlabel("Operational Intensity (INT8 OPs / Byte Transferred)")
    ax2.set_ylabel("Attainable Throughput (GOPS, log scale)")
    ax2.set_ylim(0.8, 250)
    ax2.set_xlim(0.4, 1500)
    ax2.set_title("(b) Roofline Analysis of KR260 INT8 GEMM Accelerator")
    ax2.legend(loc="lower right", fontsize=7.8)

    fig.tight_layout()
    save_fig(fig, "fig4_roofline_and_gemm_gops")


def plot_fig5_fpga_resource_and_power():
    """Figure 5: Xilinx Kria KR260 (XCK26) FPGA Resource Utilization & On-Chip Power Breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={"width_ratios": [1.2, 1.0]})

    # Panel A: FPGA Resource Utilization (from gemm_system_wrapper_utilization_placed.rpt)
    res_names = [
        "CLB LUTs\n(22,693 / 117,120)",
        "CLB Flip-Flops\n(34,532 / 234,240)",
        "Block RAM (36Kb)\n(95.5 / 144)",
        "UltraRAM (288Kb)\n(8 / 64)",
        "DSP48E2 Slices\n(320 / 1,248)"
    ]
    used_pct = [19.38, 14.74, 66.32, 12.50, 25.64]
    avail_pct = [100.0 - u for u in used_pct]

    x = np.arange(len(res_names))
    w = 0.52
    b_used = ax1.bar(x, used_pct, w, label="Used by 16x16 INT8 GEMM System", color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax1.bar(x, avail_pct, w, bottom=used_pct, label="Available Headroom on XCK26", color="#E2E8F0", edgecolor="black", linewidth=0.6)

    for b, u in zip(b_used, used_pct):
        ax1.annotate(f"{u:.2f}%", xy=(b.get_x() + b.get_width() / 2, u),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(res_names, fontsize=8.2)
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("FPGA Resource Utilization (%)")
    ax1.set_title("(a) Xilinx Kria KR260 (XCK26) Post-Place & Route Utilization")
    ax1.legend(loc="upper right", fontsize=8.5)

    # Panel B: On-Chip Power Breakdown (3.652 W Total @ 200 MHz, WNS = +0.361 ns)
    pwr_vals = [2.567, 0.777, 0.307]
    pwr_labels = [
        "PS Quad-Core ARM A53 + DDR\n(2.567 W | 70.3%)",
        "PL GEMM Accelerator @ 200MHz\n(0.777 W | 21.3%)",
        "Device Static Power\n(0.307 W | 8.4%)"
    ]
    pwr_colors = ["#4C72B0", "#DD8452", "#94A3B8"]

    ax2.pie(
        pwr_vals, labels=pwr_labels, colors=pwr_colors, startangle=125,
        explode=(0.02, 0.06, 0.02),
        wedgeprops=dict(width=0.48, edgecolor="black", linewidth=0.8),
        textprops=dict(fontsize=8.2, fontweight="bold")
    )
    ax2.text(
        0, 0,
        "Total Power\n3.652 W\n(28.0 GOPS/W)\nWNS: +0.361ns",
        ha="center", va="center", fontsize=8.8, fontweight="bold"
    )
    ax2.set_title("(b) On-Chip Power & Energy Efficiency @ 200 MHz")

    fig.tight_layout()
    save_fig(fig, "fig5_fpga_resource_and_power")


def main():
    with open(PSPL_JSON, "r", encoding="utf-8") as f:
        pspl_data = json.load(f)
    with open(FP32_JSON, "r", encoding="utf-8") as f:
        fp32_data = json.load(f)
    with open(GEMM_JSON, "r", encoding="utf-8") as f:
        gemm_data = json.load(f)
    with open(QUIZ_JSON, "r", encoding="utf-8") as f:
        quiz_data = json.load(f)

    plot_fig1_e2e_speedup(pspl_data)
    plot_fig2_phase_and_runtime_breakdown(pspl_data)
    plot_fig3_fp32_vs_kr260_accuracy(fp32_data, quiz_data)
    plot_fig4_roofline_and_gemm_gops(gemm_data)
    plot_fig5_fpga_resource_and_power()
    print("\n[ALL DONE] Generated 5 publication-ready figures (.png + .pdf) in:", OUT_DIR)


if __name__ == "__main__":
    main()
