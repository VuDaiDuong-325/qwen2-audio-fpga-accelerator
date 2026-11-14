import os
import glob
import time
import torch
import librosa
from models.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration
from models.processing_qwen2_audio import Qwen2AudioProcessor
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- 1. Tối ưu cấu hình chạy trên CPU (PS của FPGA) ---
device = torch.device("cpu")
num_cores = os.cpu_count() or 4
torch.set_num_threads(num_cores)  # Ép PyTorch tận dụng tối đa số nhân CPU của PS
logger.info(f"Running baseline on FPGA PS (ARM CPU) with {num_cores} threads...")

checkpoint = "qwen2-audio-dialogue"
logger.info(f"Loading checkpoint: {checkpoint}")

processor = Qwen2AudioProcessor.from_pretrained(checkpoint)
model = Qwen2AudioForConditionalGeneration.from_pretrained(checkpoint)
model = model.to(device).eval()

# Ensure we use the correct End-Of-Sequence token
IM_END_ID = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

# --- Data Configuration for "In the Wild" ---
audio_folder = "./LLM_Questions" if os.path.isdir("./LLM_Questions") else "./LLM_Question"
logger.info(f"Scanning directory: {audio_folder}")

# Get all .wav files and sort them
audio_files = sorted(glob.glob(os.path.join(audio_folder, "*.wav")))

if not audio_files:
    logger.error(" No .wav files found in the directory!")
    exit()

logger.info(f"Found {len(audio_files)} files. Starting Inference Benchmark on PS...")

text_sources = [
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
    "What is the ancient name of Hanoi, which translates to Ascending Dragon",
    "In Vietnamese culture, what are the specific flowers used to decorate homes during the Tết holiday (Lunar New Year) in the North versus the South?",
    "What makes you different from other assistant models?",
    "Can you share a brief summary of your existence?",
    "Can you give me an introduction of yourself?",
    "Tell me about the movie Star Wars.",
    "Suggest ways to reduce daily water use by households.",
    "Classify the following movie genres: comedy, action, drama."
]

DEFAULT_INSTRUCTION = "You are a virtual assistant, answering questions and providing information in a clear and polite manner."

# Khai báo biến tích lũy tổng thời gian để tổng kết Benchmark
total_gen_time_all = 0.0
total_tokens_all = 0

# --- Inference Loop with Benchmark ---
for index, audio_path in enumerate(audio_files):
    filename = os.path.basename(audio_path)
    question_text = text_sources[index] if index < len(text_sources) else "N/A"

    t_total_start = time.perf_counter()

    # --- 1. Audio Loading & Preprocessing Benchmark ---
    t_prep_start = time.perf_counter()
    target_sr = processor.feature_extractor.sampling_rate
    audio, _ = librosa.load(audio_path, sr=target_sr)

    text_prompt = (
        f"<|audio_bos|><|AUDIO|><|audio_eos|>"
        f"<|im_start|>user\n"
        f"Audio content provided. {DEFAULT_INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = processor(
        text=text_prompt,
        audio=audio,
        return_tensors="pt",
        sampling_rate=target_sr,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t_prep_end = time.perf_counter()

    # --- 2. Model Generation Benchmark ---
    t_gen_start = time.perf_counter()
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=IM_END_ID,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True
        )
    t_gen_end = time.perf_counter()

    # --- 3. Decode Output & Calculate Metrics ---
    input_len = inputs['input_ids'].size(1)
    generated_ids_trimmed = generate_ids[:, input_len:]
    num_generated_tokens = generated_ids_trimmed.size(1)

    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    t_total_end = time.perf_counter()

    # TÍNH TOÁN CÁC CHỈ SỐ
    prep_time = t_prep_end - t_prep_start
    gen_time = t_gen_end - t_gen_start
    total_time = t_total_end - t_total_start
    tokens_per_sec = (num_generated_tokens / gen_time) if gen_time > 0 else 0.0

    total_gen_time_all += gen_time
    total_tokens_all += num_generated_tokens

    # --- IN KẾT QUẢ BENCHMARK MỖI FILE ---
    print("\n" + "=" * 50)
    print(f"File [{index+1}/{len(audio_files)}]: {filename}")
    print(f"Question        : {question_text}")
    print(f"Generated Answer: {result.strip()}")
    print("-" * 50)
    print(f"Preprocessing Time : {prep_time:.4f} s")
    print(f"Generation Time    : {gen_time:.4f} s")
    print(f"Total Time         : {total_time:.4f} s")
    print(f"Generated Tokens   : {num_generated_tokens} tokens")
    print(f"Generation Speed   : {tokens_per_sec:.2f} tokens/sec")
    print("=" * 50)

# --- IN TỔNG KẾT TOÀN BỘ BỘ DỮ LIỆU ---
if total_gen_time_all > 0:
    avg_speed = total_tokens_all / total_gen_time_all
    print("\n" + "=" * 11 + " SUMMARY BENCHMARK (PS CPU) " + "=" * 11)
    print(f"Total Files Processed   : {len(audio_files)}")
    print(f"Total Generated Tokens  : {total_tokens_all} tokens")
    print(f"Total Generation Time   : {total_gen_time_all:.2f} s")
    print(f"Average Generation Speed: {avg_speed:.2f} tokens/sec")
    print("=" * 50)