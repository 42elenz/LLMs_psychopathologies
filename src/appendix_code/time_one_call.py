"""
Measure wall-clock time for a single API call per model.
Uses 1 transcript, 1 chunk (10 definitions), temperature=0.
Then extrapolates to full majority-voting setup (10 chunks × 10 runs = 100 calls).
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

import tiktoken
from utils.model_config import models_to_use  # not used directly; we define all models below
from utils import helper_inference
from dotenv import load_dotenv

load_dotenv(dotenv_path="../api_keys/.env")

# ── All models that were used in the study ──────────────────────────
ALL_MODELS = [
    ("open_source", "Qwen/Qwen3-Next-80B-A3B-Thinking"),
    ("open_source", "deepseek-ai/DeepSeek-R1"),
    ("open_source", "moonshotai/Kimi-K2-Thinking"),
    ("gemini",      "gemini-3-pro-preview"),
    ("anthropic",   "claude-sonnet-4-5"),
    ("openai",      "gpt-5.1"),
    ("gemini",      "gemini-2.5-flash"),
    ("mistral",     "mistral-large-latest"),
    ("open_source", "openai/gpt-oss-20b"),
    ("openai",      "gpt-4-turbo"),
]

# ── CLI filter ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="Run only models whose name contains this substring")
args = parser.parse_args()

if args.model:
    ALL_MODELS = [(f, m) for f, m in ALL_MODELS if args.model.lower() in m.lower()]
    if not ALL_MODELS:
        print(f"No models matching '{args.model}'")
        sys.exit(1)

# ── Settings ────────────────────────────────────────────────────────
PROMPT_DIR       = "../data/01_prompts/"
TRANSCRIPT_DIR   = "../data/02_transcripts_example/"  # has 1 transcript
REFERENCE_FILE   = "../data/00_ratings/reference.csv"
PROMPT_LANGUAGE  = "german"
NUM_DEFS         = 10          # definitions per chunk
TEMPERATURE      = 0.0
NUM_CHUNKS       = 10          # 99 definitions / 10 ≈ 10 chunks
NUM_OUTER_RUNS   = 10          # majority-voting runs (for extrapolation only)

# ── Preparation (done once) ─────────────────────────────────────────
schema_instruction, psy_schema, Psychopathologies = helper_inference.get_schema_instruction(PROMPT_LANGUAGE)
prompt, basic_prompt, definitions = helper_inference.load_prompts(PROMPT_DIR, PROMPT_LANGUAGE)
transkripts = helper_inference.build_transcripts_dict(TRANSCRIPT_DIR, PROMPT_LANGUAGE)
instruction_prompts_list = helper_inference.generate_instructions_prompts(
    definitions, NUM_DEFS, prompt, basic_prompt
)

# Pick the first transcript + first chunk for the timing call
txt_name = list(transkripts.keys())[0]
meta = transkripts[txt_name]
with open(meta["path"], encoding="utf-8") as f:
    transcript_text = f.read().strip()
chunk = instruction_prompts_list[0]
chunk_prompt = chunk["prompt"]

# ── Token counting ──────────────────────────────────────────────────
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def input_tokens_one_chunk(api_family: str) -> int:
    """Input tokens for ONE chunk (system/instructions + transcript)."""
    if api_family in ("open_source", "mistral", "openai"):
        sys_tok = count_tokens(chunk_prompt + schema_instruction)
    else:  # gemini, anthropic — schema enforced via API, not in prompt
        sys_tok = count_tokens(chunk_prompt)
    return sys_tok + count_tokens(transcript_text)

def input_tokens_full_run(api_family: str) -> int:
    """Input tokens summed across ALL chunks for one full run."""
    total = 0
    for ch in instruction_prompts_list:
        cp = ch["prompt"]
        if api_family in ("open_source", "mistral", "openai"):
            sys_tok = count_tokens(cp + schema_instruction)
        else:
            sys_tok = count_tokens(cp)
        total += sys_tok + count_tokens(transcript_text)
    return total

print(f"Transcript: {txt_name}  ({len(transcript_text)} chars)")
print(f"Chunk: first 10 definitions")
print(f"Chunks total: {len(instruction_prompts_list)}")
print(f"Extrapolation: {NUM_CHUNKS} chunks × {NUM_OUTER_RUNS} runs = {NUM_CHUNKS * NUM_OUTER_RUNS} calls")
print("=" * 80)

results = []

for api_family, model_name in ALL_MODELS:
    print(f"\n--- {api_family} / {model_name} ---")
    # Initialize only the client we need
    try:
        clients = helper_inference.initialize_clients([(api_family, model_name)])
    except Exception as e:
        print(f"  SKIP (client init failed): {e}")
        results.append((api_family, model_name, None, None))
        continue

    try:
        # Warmup call to establish connection (not timed)
        print(f"  warmup...")
        helper_inference.call_api(
            api_family, model_name, chunk_prompt, transcript_text,
            TEMPERATURE, 0, txt_name, 0,
            clients, schema_instruction, Psychopathologies, psy_schema
        )

        # Timed call (connection already warm)
        t0 = time.time()
        rating_text = helper_inference.call_api(
            api_family, model_name, chunk_prompt, transcript_text,
            TEMPERATURE, 1, txt_name, 1,
            clients, schema_instruction, Psychopathologies, psy_schema
        )
        elapsed = time.time() - t0

        total_est = elapsed * NUM_CHUNKS * NUM_OUTER_RUNS

        # Token counts
        in_tok_1chunk = input_tokens_one_chunk(api_family)
        in_tok_run    = input_tokens_full_run(api_family)
        out_tok_1chunk = count_tokens(rating_text) if rating_text else 0

        results.append((api_family, model_name, elapsed, total_est,
                        in_tok_1chunk, in_tok_run, out_tok_1chunk))
        print(f"  1 call: {elapsed:.1f}s  |  estimated {NUM_CHUNKS}×{NUM_OUTER_RUNS}: {total_est:.0f}s ({total_est/60:.1f}min)")
        print(f"  input tokens (1 chunk): {in_tok_1chunk:,}  |  full run: {in_tok_run:,}")
        print(f"  output tokens (1 chunk): {out_tok_1chunk:,}")

    except Exception as e:
        print(f"  FAILED: {e}")
        results.append((api_family, model_name, None, None, None, None, None))

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"{'Model':<45} {'1 call (s)':>10} {'100 calls (min)':>16} {'In tok/chunk':>14} {'In tok/run':>12} {'Out tok/chunk':>15}")
print("-" * 115)
for api_family, model_name, t1, t100, in1, in_run, out1 in results:
    label = f"{api_family}/{model_name}"
    if t1 is not None:
        print(f"{label:<45} {t1:>10.1f} {t100/60:>16.1f} {in1:>14,} {in_run:>12,} {out1:>15,}")
    else:
        print(f"{label:<45} {'FAILED':>10} {'—':>16} {'—':>14} {'—':>12} {'—':>15}")
print("-" * 115)
print(f"Note: 100 calls = {NUM_CHUNKS} chunks × {NUM_OUTER_RUNS} outer runs (majority voting)")
print(f"      per transcript. Multiply by number of transcripts for full dataset.")
print(f"      Token counts use cl100k_base tokenizer as approximation.")

# ── Save to CSV ─────────────────────────────────────────────────────
import csv
out_path = "../outputs/tables/inference_timing.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["api_family", "model_name", "time_1_call_s", "estimated_100_calls_s", "estimated_100_calls_min",
                 "input_tokens_1chunk", "input_tokens_per_run", "output_tokens_1chunk"])
    for api_family, model_name, t1, t100, in1, in_run, out1 in results:
        w.writerow([
            api_family, model_name,
            round(t1, 1) if t1 is not None else "",
            round(t100, 0) if t100 is not None else "",
            round(t100 / 60, 1) if t100 is not None else "",
            in1 if in1 is not None else "",
            in_run if in_run is not None else "",
            out1 if out1 is not None else "",
        ])
print(f"\nSaved to {out_path}")
