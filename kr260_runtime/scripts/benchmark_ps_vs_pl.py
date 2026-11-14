#!/usr/bin/env python3
"""
Benchmark & Quality Comparison: Mode 1 (100% PS) vs Mode 2 (PS + PL GEMM)
on Xilinx Kria KR260 for Qwen2-Audio Dialogue (20 WAVs).
All console outputs are strictly in English.
"""

import argparse
import difflib
import gc
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

QUESTIONS = [
    "If you had to live in a video game universe for a year, which one would you choose?",
    "Would you rather be able to speak every language fluently or play every musical instrument perfectly?",
    "If animals could talk, which species do you think would be the rudest?",
    "If you found a suitcase with $1 million inside but no ID, would you keep it or turn it in?",
    "If you were a ghost, who would you haunt and how?",
    "Would you rather travel 100 years into the past or 100 years into the future?",
    "If you could only eat one cuisine (Italian, Japanese, Mexican, etc.) for the rest of your life, what would it be?",
    "If you were forced to participate in the Olympics, which sport would you have the best chance of not embarrassing yourself in?",
    "If you could cure one disease or solve one world problem, which would you pick?",
    "If you had to delete all apps from your phone except for three, which ones would you keep?",
    "What is the capital city of Vietnam?",
    "Vietnam is consistently one of the world's top two exporters of which caffeinated commodity?",
    "What is the ancient name of Hanoi, which translates to Ascending Dragon?",
    "In Vietnamese culture, what are the specific flowers used to decorate homes during the Tet holiday (Lunar New Year) in the North versus the South?",
    "What makes you different from other assistant models?",
    "Can you share a brief summary of your existence?",
    "Can you give me an introduction of yourself?",
    "Tell me about the movie Star Wars.",
    "Suggest ways to reduce daily water use by households.",
    "Classify the following movie genres: comedy, action, drama.",
]

QUESTION_BY_FILE = {
    "01.wav": QUESTIONS[0],
    "02.wav": QUESTIONS[1],
    "03.wav": QUESTIONS[2],
    "04.wav": QUESTIONS[3],
    "05.wav": QUESTIONS[4],
    "06.wav": QUESTIONS[5],
    "07.wav": QUESTIONS[6],
    "08.wav": QUESTIONS[7],
    "09.wav": QUESTIONS[8],
    "10.wav": QUESTIONS[9],
    "11.wav": QUESTIONS[10],
    "12.wav": QUESTIONS[11],
    "13.wav": QUESTIONS[12],
    "14.wav": QUESTIONS[13],
    "15_voiceassistant.wav": QUESTIONS[14],
    "15.wav": QUESTIONS[14],
    "16_voiceassistant.wav": QUESTIONS[15],
    "16.wav": QUESTIONS[15],
    "17_voiceassistant.wav": QUESTIONS[16],
    "17.wav": QUESTIONS[16],
    "18_alpacaaudio.wav": QUESTIONS[17],
    "18.wav": QUESTIONS[17],
    "19_alpacaaudio.wav": QUESTIONS[18],
    "19.wav": QUESTIONS[18],
    "20_alpacaaudio.wav": QUESTIONS[19],
    "20.wav": QUESTIONS[19],
}


def get_question_for_wav(wav_name, idx):
    if wav_name in QUESTION_BY_FILE:
        return QUESTION_BY_FILE[wav_name]
    m = re.match(r"^(\d+)", wav_name)
    if m:
        num = int(m.group(1))
        if 1 <= num <= len(QUESTIONS):
            return QUESTIONS[num - 1]
    return QUESTIONS[idx] if idx < len(QUESTIONS) else "N/A"

DEFAULT_INSTRUCTION = (
    "You are a virtual assistant, answering questions and providing information "
    "in a clear and polite manner."
)


def normalize_words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def word_error_rate(ref_text, hyp_text):
    r = normalize_words(ref_text)
    h = normalize_words(hyp_text)
    if not r:
        return 0.0 if not h else 100.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return 100.0 * d[len(r)][len(h)] / len(r)


def similarity_metrics(ref_text, hyp_text):
    char_sim = 100.0 * difflib.SequenceMatcher(None, ref_text.lower(), hyp_text.lower()).ratio()
    rw = normalize_words(ref_text)
    hw = normalize_words(hyp_text)
    word_sim = 100.0 * difflib.SequenceMatcher(None, rw, hw).ratio() if (rw or hw) else 100.0
    wer = word_error_rate(ref_text, hyp_text)
    return word_sim, char_sim, wer


def parse_llm_runtime_log(log_text, backend="npu"):
    res = {
        "load_s": 0.0,
        "logmel_s": 0.0,
        "encode_s": 0.0,
        "prefill_s": 0.0,
        "decode_s": 0.0,
        "xfer_s": 0.0,
        "hw_compute_s": 0.0,
        "ps_compute_s": 0.0,
        "proc_s": 0.0,
        "total_s": 0.0,
        "tokens": 0,
        "tok_per_s": 0.0,
        "ms_per_tok": 0.0,
        "text": "",
    }
    m = re.search(r"\[pkg\].*doc manifest\s+([0-9.]+)\s+s", log_text)
    if m:
        res["load_s"] += float(m.group(1))
    m = re.search(r"\[pkg\] encoder:.*nap manifest\s+([0-9.]+)\s+s", log_text)
    if m:
        res["load_s"] += float(m.group(1))
    m = re.search(r"\[npu\] prepare\s+([0-9.]+)\s+s", log_text)
    if m:
        res["load_s"] += float(m.group(1))

    m = re.search(r"\[encoder\] mel .*log-mel\s+([0-9.]+)\s+s\s*\|\s*encoder\s+([0-9.]+)\s+s", log_text)
    if m:
        res["logmel_s"] = float(m.group(1))
        res["encode_s"] = float(m.group(2))

    enc_dyn = 0.0
    enc_hw = 0.0
    enc_copy = 0.0
    m = re.search(
        r"\[encoder\] GEMM .*PL\s+([0-9.]+)\s+\(nap dong\s+([0-9.]+),\s*PL chay thuc\s+([0-9.]+),\s*chep A/C\s+([0-9.]+)\)",
        log_text,
    )
    if m:
        enc_dyn = float(m.group(2))
        enc_hw = float(m.group(3))
        enc_copy = float(m.group(4))

    m = re.search(r"\[prefill\]\s+\d+\s+hang:\s+([0-9.]+)\s+s", log_text)
    if m:
        res["prefill_s"] = float(m.group(1))

    m = re.search(r"DECODE:\s+(\d+)\s+token trong\s+([0-9.]+)\s+s\s*=\s*([0-9.]+)\s+token/s", log_text)
    if m:
        res["tokens"] = int(m.group(1))
        res["decode_s"] = float(m.group(2))
        res["tok_per_s"] = float(m.group(3))
        if res["tokens"] > 0:
            res["ms_per_tok"] = 1000.0 * res["decode_s"] / res["tokens"]

    dec_pl_wall_ms = 0.0
    dec_pl_hw_ms = 0.0
    m = re.search(r"PL wall\s+([0-9.]+)\s+ms/token\s*\(phan cung\s+([0-9.]+)\s+ms/token", log_text)
    if m:
        dec_pl_wall_ms = float(m.group(1))
        dec_pl_hw_ms = float(m.group(2))

    res["total_s"] = res["logmel_s"] + res["encode_s"] + res["prefill_s"] + res["decode_s"]
    if backend == "npu":
        dec_xfer_s = max(0.0, (dec_pl_wall_ms - dec_pl_hw_ms) * res["tokens"] / 1000.0)
        dec_hw_s = dec_pl_hw_ms * res["tokens"] / 1000.0
        res["xfer_s"] = enc_dyn + enc_copy + dec_xfer_s
        res["hw_compute_s"] = enc_hw + dec_hw_s
        res["proc_s"] = max(0.0, res["total_s"] - res["xfer_s"])
        res["ps_compute_s"] = max(0.0, res["proc_s"] - res["hw_compute_s"])
    else:
        res["xfer_s"] = 0.0
        res["hw_compute_s"] = 0.0
        res["proc_s"] = res["total_s"]
        res["ps_compute_s"] = res["total_s"]

    m = re.search(r"===== VAN BAN SINH RA =====\n(.*?)\n===========================", log_text, re.S)
    if m:
        res["text"] = " ".join(m.group(1).strip().split())
    return res


class LiveProgress:
    """Thread-safe terminal progress bar with live elapsed heartbeat (no external deps)."""

    def __init__(self, mode_label, total_wavs):
        self.mode_label = mode_label
        self.total_wavs = max(1, total_wavs)
        self.idx = 0
        self.wav_name = ""
        self.phase = "Init"
        self.t0 = time.perf_counter()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._spin = "|/-\\"
        self._tick = 0

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def update(self, idx=None, wav_name=None, phase=None, reset_timer=False):
        with self._lock:
            if idx is not None:
                self.idx = idx
            if wav_name is not None:
                self.wav_name = wav_name
            if phase is not None:
                self.phase = phase
            if reset_timer:
                self.t0 = time.perf_counter()

    def finish_wav(self, wav_name, summary_str):
        with self._lock:
            dt = time.perf_counter() - self.t0
            sys.stdout.write("\r\033[K")
            print(f"[{self.mode_label}] [{self.idx}/{self.total_wavs}] {wav_name} DONE in {dt:.2f}s | {summary_str}")
            sys.stdout.flush()

    def _render(self):
        with self._lock:
            pct = 100.0 * self.idx / self.total_wavs
            filled = int(14 * self.idx / self.total_wavs)
            bar = "#" * filled + "-" * (14 - filled)
            sp = self._spin[self._tick % 4]
            self._tick += 1
            dt = time.perf_counter() - self.t0
            line = (
                f"\r\033[K[{self.mode_label}] [{bar}] {self.idx}/{self.total_wavs} ({pct:4.0f}%) "
                f"{sp} {self.wav_name} | {self.phase} | {dt:5.1f}s"
            )
            sys.stdout.write(line)
            sys.stdout.flush()

    def _loop(self):
        while not self._stop.wait(0.5):
            self._render()


def run_llm_runtime(bin_path, pkg_dir, wav_path, template_path, backend, max_new, threads, out_txt, out_log, progress=None):
    cmd = [
        bin_path,
        "--pkg", pkg_dir,
        "--wav", wav_path,
        "--template", template_path,
        "--backend", backend,
        "--max-new", str(max_new),
        "--threads", str(threads),
        "--text-out", out_txt,
        "--verbose",
    ]
    if backend == "npu":
        cmd = ["sudo", "env", "NPU_PL_OK=1"] + cmd

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if progress is not None:
            if line.startswith("[npu] nap trong so"):
                m = re.search(r"(\d+%)", line)
                progress.update(phase=f"Load weights {m.group(1) if m else ''}".strip())
            elif line.startswith("[wav]"):
                progress.update(phase="Audio Encoding")
            elif line.startswith("[encoder]"):
                progress.update(phase="Prefill")
            elif line.startswith("[prefill]"):
                progress.update(phase=f"Decoding 0/{max_new}")
            elif line.strip().startswith("token "):
                m = re.search(r"token\s+(\d+)\s+->", line)
                if m:
                    progress.update(phase=f"Decoding {int(m.group(1)) + 1}/{max_new}")
    rc = proc.wait()
    wall_s = time.perf_counter() - t0
    full_out = "".join(lines)
    with open(out_log, "w", encoding="utf-8") as f:
        f.write(full_out)
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}\nSee {out_log}")
    parsed = parse_llm_runtime_log(full_out, backend=backend)
    if os.path.exists(out_txt):
        with open(out_txt, "r", encoding="utf-8") as f:
            txt = " ".join(f.read().strip().split())
            if txt:
                parsed["text"] = txt
    parsed["wall_s"] = wall_s
    return parsed


def run_torch_ps_all(asc_dir, wav_files, max_new, threads, torch_dtype="bfloat16"):
    sys.path.insert(0, asc_dir)
    import torch
    import librosa
    from models.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration
    from models.processing_qwen2_audio import Qwen2AudioProcessor

    torch.set_num_threads(threads)
    device = torch.device("cpu")
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[torch_dtype]
    ckpt = os.path.join(asc_dir, "qwen2-audio-dialogue")

    prog = LiveProgress("Mode 1: 100% PS", len(wav_files))
    prog.update(idx=0, wav_name="model", phase=f"Loading {torch_dtype} model...", reset_timer=True)
    prog.start()

    t_load0 = time.perf_counter()
    processor = Qwen2AudioProcessor.from_pretrained(ckpt)
    try:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            ckpt, dtype=dtype, low_cpu_mem_usage=True
        ).to(device).eval()
    except TypeError:
        try:
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                ckpt, torch_dtype=dtype, low_cpu_mem_usage=True
            ).to(device).eval()
        except Exception:
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                ckpt, torch_dtype=dtype
            ).to(device).eval()
    except Exception:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            ckpt, dtype=dtype
        ).to(device).eval()
    load_s = time.perf_counter() - t_load0
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    target_sr = processor.feature_extractor.sampling_rate

    phase = {"enc_s": 0.0, "pre_s": 0.0, "dec_s": 0.0, "in_llm_pre": False, "dec_step": 0}

    def hook_enc_pre(_m, _inp):
        phase["_t_enc"] = time.perf_counter()
        prog.update(phase="Audio Encoding")

    def hook_enc_post(_m, _inp, _out):
        phase["enc_s"] += time.perf_counter() - phase["_t_enc"]

    def hook_llm_pre(_m, args, kwargs):
        phase["_t_llm"] = time.perf_counter()
        past = kwargs.get("past_key_values", None)
        is_pre = False
        if past is None:
            is_pre = True
        elif hasattr(past, "get_seq_length"):
            is_pre = past.get_seq_length() == 0
        elif isinstance(past, (tuple, list)) and len(past) == 0:
            is_pre = True
        phase["in_llm_pre"] = is_pre
        if is_pre:
            prog.update(phase="LLM Prefill")
        else:
            phase["dec_step"] += 1
            prog.update(phase=f"Decoding {phase['dec_step']}/{max_new}")

    def hook_llm_post(_m, _inp, _out):
        dt = time.perf_counter() - phase["_t_llm"]
        if phase["in_llm_pre"]:
            phase["pre_s"] += dt
        else:
            phase["dec_s"] += dt

    h1 = model.audio_tower.register_forward_pre_hook(hook_enc_pre)
    h2 = model.audio_tower.register_forward_hook(hook_enc_post)
    h3 = model.multi_modal_projector.register_forward_pre_hook(hook_enc_pre)
    h4 = model.multi_modal_projector.register_forward_hook(hook_enc_post)
    h5 = model.language_model.register_forward_pre_hook(hook_llm_pre, with_kwargs=True)
    h6 = model.language_model.register_forward_hook(hook_llm_post)

    results = []
    try:
        for idx, wav_path in enumerate(wav_files, 1):
            wname = os.path.basename(wav_path)
            phase["enc_s"] = 0.0
            phase["pre_s"] = 0.0
            phase["dec_s"] = 0.0
            phase["dec_step"] = 0
            prog.update(idx=idx, wav_name=wname, phase="Preprocessing WAV", reset_timer=True)

            t_prep0 = time.perf_counter()
            audio, _ = librosa.load(wav_path, sr=target_sr)
            prompt = (
                f"<|audio_bos|><|AUDIO|><|audio_eos|>"
                f"<|im_start|>user\n"
                f"Audio content provided. {DEFAULT_INSTRUCTION}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs = processor(
                text=prompt, audio=audio, return_tensors="pt",
                sampling_rate=target_sr, padding=True
            )
            inputs = {
                k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device))
                for k, v in inputs.items()
            }
            logmel_s = time.perf_counter() - t_prep0

            t_gen0 = time.perf_counter()
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    repetition_penalty=1.1,
                    eos_token_id=im_end_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    use_cache=True,
                )
            gen_s = time.perf_counter() - t_gen0
            in_len = inputs["input_ids"].size(1)
            out_ids = gen_ids[:, in_len:]
            n_tok = int(out_ids.size(1))
            text = processor.batch_decode(
                out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            text = " ".join(text.strip().split())

            enc_s = phase["enc_s"]
            pre_s = phase["pre_s"]
            dec_s = phase["dec_s"]
            other_s = max(0.0, gen_s - (enc_s + pre_s + dec_s))
            dec_s_adj = dec_s + other_s
            total_s = logmel_s + gen_s
            tps = (n_tok / dec_s_adj) if dec_s_adj > 0 else 0.0
            prog.finish_wav(wname, f"{n_tok} tokens | {tps:.2f} tok/s | total {total_s:.2f}s")
            results.append({
                "load_s": load_s if idx == 1 else 0.0,
                "load_model_s": load_s,
                "logmel_s": logmel_s,
                "encode_s": enc_s,
                "prefill_s": pre_s,
                "decode_s": dec_s_adj,
                "xfer_s": 0.0,
                "proc_s": total_s,
                "total_s": total_s,
                "tokens": n_tok,
                "tok_per_s": tps,
                "ms_per_tok": (1000.0 * dec_s_adj / n_tok) if n_tok > 0 else 0.0,
                "text": text,
            })
    finally:
        prog.stop()
        for h in (h1, h2, h3, h4, h5, h6):
            h.remove()
    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_kr260 = os.path.dirname(script_dir)
    default_asc = os.path.join(os.path.dirname(default_kr260), "Audio_Speech_Conversation")
    if not os.path.isdir(default_kr260):
        default_kr260 = os.path.expanduser("~/dduong/kr260")
    if not os.path.isdir(default_asc):
        default_asc = os.path.expanduser("~/dduong/Audio_Speech_Conversation")

    ap = argparse.ArgumentParser(description="Compare Mode 1 (100% PS) vs Mode 2 (PS + PL GEMM) on KR260")
    ap.add_argument("--kr260-dir", default=default_kr260)
    ap.add_argument("--asc-dir", default=default_asc)
    ap.add_argument("--wav-dir", default="", help="Optional explicit path to WAV folder")
    ap.add_argument("--ps-mode", choices=["torch", "soft"], default="soft",
                    help="Mode 1 backend: 'soft' (INT8 C++ on PS with pkg_v1, default) or 'torch' (PyTorch on PS)")
    ap.add_argument("--torch-dtype", choices=["bfloat16", "float32", "float16"], default="bfloat16",
                    help="PyTorch dtype for Mode 1 if --ps-mode torch is selected")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--wav", default="",
                    help="Run specific WAV(s): e.g. '16', '16_voiceassistant.wav', '15,16,18', or '15-20'")
    ap.add_argument("--limit", type=int, default=0, help="Run only first N wavs (0 = all)")
    ap.add_argument("--out-dir", default="ps_pl_compare")
    args = ap.parse_args()

    if args.wav_dir:
        wav_dir = os.path.expanduser(args.wav_dir)
    else:
        cand1 = os.path.join(args.asc_dir, "LLM_Question")
        cand2 = os.path.join(args.asc_dir, "LLM_Questions")
        if os.path.isdir(cand2) and glob.glob(os.path.join(cand2, "*.wav")):
            wav_dir = cand2
        else:
            wav_dir = cand1
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        sys.exit(f"ERROR: No .wav files found in {wav_dir}")

    if args.wav:
        wanted_nums = set()
        wanted_names = set()
        for part in args.wav.split(","):
            part = part.strip()
            if not part:
                continue
            m_rng = re.match(r"^(\d+)-(\d+)$", part)
            if m_rng:
                for k in range(int(m_rng.group(1)), int(m_rng.group(2)) + 1):
                    wanted_nums.add(k)
            elif part.isdigit():
                wanted_nums.add(int(part))
            else:
                wanted_names.add(part.lower())
        filtered = []
        for wf in wav_files:
            bname = os.path.basename(wf)
            bl = bname.lower()
            m_num = re.match(r"^(\d+)", bname)
            fnum = int(m_num.group(1)) if m_num else -1
            if fnum in wanted_nums or bl in wanted_names or any(w in bl for w in wanted_names):
                filtered.append(wf)
        if not filtered:
            sys.exit(f"ERROR: No .wav file matching --wav '{args.wav}' found in {wav_dir}")
        wav_files = filtered

    if args.limit > 0:
        wav_files = wav_files[: args.limit]

    tools_dir = os.path.join(args.kr260_dir, "tools")
    pkg_dir = os.path.join(args.kr260_dir, "pkg_v1")
    bin_path = os.path.join(tools_dir, "llm_runtime")
    tmpl_path = os.path.join(tools_dir, "in_clip00", "ids.i32")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Running benchmark on {len(wav_files)} WAV files | Mode 1: 100% PS ({args.ps_mode}) | Mode 2: PS + PL (npu)")
    print(f"Parameters: max_new={args.max_new}, threads={args.threads}")

    # Run Mode 1 (100% PS)
    m1_results = []
    if args.ps_mode == "torch":
        print(f"\n[Step 1/2] Running Mode 1: 100% PS (PyTorch {args.torch_dtype} on ARM CPU)...")
        m1_results = run_torch_ps_all(args.asc_dir, wav_files, args.max_new, args.threads, args.torch_dtype)
        gc.collect()
    else:
        print("\n[Step 1/2] Running Mode 1: 100% PS (C++ INT8 SoftBackend on ARM CPU)...")
        prog1 = LiveProgress("Mode 1: 100% PS", len(wav_files))
        prog1.start()
        try:
            for idx, wf in enumerate(wav_files, 1):
                wname = os.path.basename(wf)
                base = os.path.splitext(wname)[0]
                prog1.update(idx=idx, wav_name=wname, phase="Starting...", reset_timer=True)
                r = run_llm_runtime(
                    bin_path, pkg_dir, wf, tmpl_path, "soft", args.max_new, args.threads,
                    os.path.join(args.out_dir, f"{base}_ps.txt"),
                    os.path.join(args.out_dir, f"{base}_ps.log"),
                    progress=prog1,
                )
                prog1.finish_wav(wname, f"{r['tokens']} tokens | {r['tok_per_s']:.2f} tok/s | total {r['total_s']:.2f}s")
                m1_results.append(r)
        finally:
            prog1.stop()

    # Run Mode 2 (PS + PL)
    print("\n[Step 2/2] Running Mode 2: PS + PL (INT8 GEMM Accelerator on KR260)...")
    m2_results = []
    prog2 = LiveProgress("Mode 2: PS + PL", len(wav_files))
    prog2.start()
    try:
        for idx, wf in enumerate(wav_files, 1):
            wname = os.path.basename(wf)
            base = os.path.splitext(wname)[0]
            prog2.update(idx=idx, wav_name=wname, phase="Starting...", reset_timer=True)
            r = run_llm_runtime(
                bin_path, pkg_dir, wf, tmpl_path, "npu", args.max_new, args.threads,
                os.path.join(args.out_dir, f"{base}_pl.txt"),
                os.path.join(args.out_dir, f"{base}_pl.log"),
                progress=prog2,
            )
            prog2.finish_wav(wname, f"{r['tokens']} tokens | {r['tok_per_s']:.2f} tok/s | total {r['total_s']:.2f}s")
            m2_results.append(r)
    finally:
        prog2.stop()

    # Print per-WAV Q&A Comparison in English
    print("\n" + "=" * 80)
    print("PER-WAV QUALITATIVE COMPARISON (MODE 1: 100% PS vs MODE 2: PS + PL)")
    print("=" * 80)
    rows = []
    for i, wf in enumerate(wav_files):
        name = os.path.basename(wf)
        q = get_question_for_wav(name, i)
        r1 = m1_results[i]
        r2 = m2_results[i]
        wsim, csim, wer = similarity_metrics(r1["text"], r2["text"])
        speedup = (r1["total_s"] / r2["total_s"]) if r2["total_s"] > 0 else 0.0
        tok_diff = r2["tokens"] - r1["tokens"]

        print("-----")
        print(f"WAV: {name}")
        print(f"Question: {q}")
        print(f"Answer (Mode 1 - 100% PS): {r1['text']}")
        print(f"Answer (Mode 2 - PS + PL): {r2['text']}")
        print("-----")

        rows.append({
            "wav": name,
            "question": q,
            "m1": r1,
            "m2": r2,
            "speedup": speedup,
            "word_sim_pct": wsim,
            "char_sim_pct": csim,
            "wer_pct": wer,
            "tok_diff": tok_diff,
        })

    # Print Performance & Deviation Table
    print("\n" + "=" * 154)
    print("BENCHMARK & DEVIATION COMPARISON TABLE: MODE 1 (100% PS) vs MODE 2 (PS + PL)")
    print("=" * 154)
    hdr = (
        f"{'WAV':<22} | {'M1 Load':>7} {'M2 Load':>7} | {'M2 Xfer':>7} | "
        f"{'M1 Proc':>8} {'M2 Proc':>7} | {'M1 Enc/Pre/Dec':>17} | {'M2 Enc/Pre/Dec':>17} | "
        f"{'M1 Tok/s':>8} {'M2 Tok/s':>8} | {'M1 Tot':>8} {'M2 Tot':>7} | "
        f"{'Speedup':>7} | {'WordSim':>7} {'WER':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for row in rows:
        r1, r2 = row["m1"], row["m2"]
        m1_epd = f"{r1['encode_s']:.1f}/{r1['prefill_s']:.1f}/{r1['decode_s']:.1f}"
        m2_epd = f"{r2['encode_s']:.1f}/{r2['prefill_s']:.1f}/{r2['decode_s']:.1f}"
        m1_ld = r1.get("load_model_s", r1["load_s"])
        print(
            f"{row['wav']:<22} | {m1_ld:6.2f}s {r2['load_s']:6.2f}s | {r2['xfer_s']:6.2f}s | "
            f"{r1['proc_s']:7.2f}s {r2['proc_s']:6.2f}s | {m1_epd:>17} | {m2_epd:>17} | "
            f"{r1['tok_per_s']:8.2f} {r2['tok_per_s']:8.2f} | {r1['total_s']:7.2f}s {r2['total_s']:6.2f}s | "
            f"{row['speedup']:6.2f}x | {row['word_sim_pct']:6.1f}% {row['wer_pct']:5.1f}%"
        )

    print("-" * len(hdr))
    n = len(rows)
    avg_m1_tot = sum(r["m1"]["total_s"] for r in rows) / n
    avg_m2_tot = sum(r["m2"]["total_s"] for r in rows) / n
    tot_m1 = sum(r["m1"]["total_s"] for r in rows)
    tot_m2 = sum(r["m2"]["total_s"] for r in rows)
    overall_speedup = tot_m1 / tot_m2 if tot_m2 > 0 else 0.0
    avg_m1_tps = sum(r["m1"]["tok_per_s"] for r in rows) / n
    avg_m2_tps = sum(r["m2"]["tok_per_s"] for r in rows) / n
    avg_wsim = sum(r["word_sim_pct"] for r in rows) / n
    avg_csim = sum(r["char_sim_pct"] for r in rows) / n
    avg_wer = sum(r["wer_pct"] for r in rows) / n
    avg_xfer = sum(r["m2"]["xfer_s"] for r in rows) / n

    print("\nSUMMARY METRICS:")
    print(f"  Total WAV files processed       : {n}")
    print(f"  Mode 1 (100% PS) Total Time     : {tot_m1:.2f} s (Avg {avg_m1_tot:.2f} s/wav)")
    print(f"  Mode 2 (PS + PL) Total Time     : {tot_m2:.2f} s (Avg {avg_m2_tot:.2f} s/wav)")
    print(f"  Overall End-to-End Speedup      : {overall_speedup:.2f}x faster (PS+PL vs 100% PS)")
    print(f"  Avg Decode Speed (Mode 1 vs 2)  : {avg_m1_tps:.2f} tok/s vs {avg_m2_tps:.2f} tok/s")
    print(f"  Avg Mode 2 Data Transfer Time   : {avg_xfer:.2f} s/wav")
    print(f"  Avg Answer Word Similarity      : {avg_wsim:.2f}%")
    print(f"  Avg Answer Char Similarity      : {avg_csim:.2f}%")
    print(f"  Avg Word Error Rate (WER)       : {avg_wer:.2f}%")

    report_path = os.path.join(args.out_dir, "pspl_compare_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nDetailed JSON report saved to: {report_path}")


if __name__ == "__main__":
    main()
