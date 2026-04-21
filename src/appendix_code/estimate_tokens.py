"""
Estimate input & output token counts for ALL models.

Input tokens:  system prompt + user prompt (transcript) per chunk, summed across all chunks.
               Uses native tokenizer per model where available.
Output tokens: mean Begründung token count from saved raw responses.

Merges results into inference_timing.csv.
"""
import sys, os, json, glob, statistics, csv
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))

import tiktoken
from transformers import AutoTokenizer
from utils import helper_inference

# ── All models (matching inference_timing.csv) ─────────────────────
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

# ── Paths ───────────────────────────────────────────────────────────
PROMPT_DIR       = "../data/01_prompts/"
TRANSCRIPT_DIR   = "../data/02_transcripts_example/"
RAW_RESPONSES    = (
    "/zi/home/esra.lenz/Documents/00_HITKIP/01_GPTS/99_Git_share/"
    "LLM_PPB/00_AMDP/01_code/02_Commercial_models/raw_responses"
)
TIMING_CSV       = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables", "inference_timing.csv")
PROMPT_LANGUAGE  = "german"
NUM_DEFS         = 10

# ── Tokenizers ──────────────────────────────────────────────────────
# Native tokenizer per model; fallback to tiktoken cl100k_base for commercial APIs
TOKENIZER_MAP = {
    # open_source (Together AI) — use HF tokenizers
    "Qwen/Qwen3-Next-80B-A3B-Thinking": "Qwen/Qwen2.5-7B-Instruct",  # same Qwen tokenizer family
    "deepseek-ai/DeepSeek-R1":          "deepseek-ai/DeepSeek-R1",
    "moonshotai/Kimi-K2-Thinking":      "moonshotai/Kimi-K2-Thinking",
    "openai/gpt-oss-20b":               None,  # tiktoken (OpenAI family)
    # mistral
    "mistral-large-latest":             "mistralai/Mistral-Nemo-Instruct-2407",  # same Tekken tokenizer as Mistral Large
    # commercial — tiktoken cl100k_base as approximation
    "gemini-3-pro-preview":             None,
    "claude-sonnet-4-5":                None,
    "gpt-5.1":                          None,
    "gemini-2.5-flash":                 None,
    "gpt-4-turbo":                      None,
}

print("Loading tokenizers...")
enc_tiktoken = tiktoken.get_encoding("cl100k_base")
tokenizers = {}
for model_name, hf_name in TOKENIZER_MAP.items():
    if hf_name:
        try:
            tokenizers[model_name] = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            print(f"  {model_name}: HF tokenizer ({hf_name}, vocab={tokenizers[model_name].vocab_size})")
        except Exception as e:
            print(f"  {model_name}: HF load failed ({e}), using tiktoken")
            tokenizers[model_name] = None
    else:
        tokenizers[model_name] = None
        print(f"  {model_name}: tiktoken cl100k_base")

def count_tokens(text: str, model_name: str = None) -> int:
    tok = tokenizers.get(model_name) if model_name else None
    if tok:
        return len(tok.encode(text))
    return len(enc_tiktoken.encode(text))


# ═══════════════════════════════════════════════════════════════════
# 1. BUILD PROMPTS (exactly as the pipeline does)
# ═══════════════════════════════════════════════════════════════════
schema_instruction, psy_schema, Psychopathologies = \
    helper_inference.get_schema_instruction(PROMPT_LANGUAGE)

prompt, basic_prompt, definitions = \
    helper_inference.load_prompts(PROMPT_DIR, PROMPT_LANGUAGE)

transkripts = helper_inference.build_transcripts_dict(TRANSCRIPT_DIR, PROMPT_LANGUAGE)

instruction_prompts_list = helper_inference.generate_instructions_prompts(
    definitions, NUM_DEFS, prompt, basic_prompt
)

# Load transcript text
txt_name = list(transkripts.keys())[0]
meta = transkripts[txt_name]
with open(meta["path"], encoding="utf-8") as f:
    transcript_text = f.read().strip()

num_chunks = len(instruction_prompts_list)

# ═══════════════════════════════════════════════════════════════════
# 2. INPUT TOKEN ESTIMATION (per model, accounting for prompt differences)
# ═══════════════════════════════════════════════════════════════════
print("=" * 80)
print(f"INPUT TOKEN ESTIMATION  (NUM_DEFS={NUM_DEFS}, chunks={num_chunks})")
print(f"Transcript: {txt_name}  ({len(transcript_text)} chars)")
print("=" * 80)

def input_tokens_for_model(api_family: str, model_name: str) -> int:
    """
    Sum input tokens across all chunks for one run, using native tokenizer.
    Prompt structure differs by api_family:
      - open_source / mistral / openai: system = chunk_prompt + schema_instruction
      - gemini, anthropic: system = chunk_prompt (schema via API params)
    """
    total = 0
    for chunk in instruction_prompts_list:
        cp = chunk["prompt"]
        if api_family in ("open_source", "mistral", "openai"):
            sys_tok = count_tokens(cp + schema_instruction, model_name)
        else:
            sys_tok = count_tokens(cp, model_name)
        total += sys_tok + count_tokens(transcript_text, model_name)
    return total

input_by_model = {}
print(f"\n{'Model':<50} {'Tokenizer':<15} {'Input tokens / run':>20}")
print("-" * 87)
for api_family, model_name in ALL_MODELS:
    tok = input_tokens_for_model(api_family, model_name)
    input_by_model[model_name] = tok
    tok_type = "native" if tokenizers.get(model_name) else "cl100k"
    print(f"{model_name:<50} {tok_type:<15} {tok:>20,}")

# ═══════════════════════════════════════════════════════════════════
# 3. OUTPUT TOKEN ESTIMATION (Begründungen from raw responses)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("OUTPUT TOKEN ESTIMATION  (mean Begründung tokens from raw responses)")
print(f"Source: {RAW_RESPONSES}")
print("=" * 80)

def raw_file_prefix(api_family: str, model_name: str) -> str:
    """Build the filename prefix used by save_raw_response()."""
    key = model_name.replace("/", "_")
    if api_family == "open_source":
        return f"response_open_source_{key}"
    return f"response_{api_family}_{key}"

output_by_model = {}  # model_name -> {mean_begr, mean_full_resp, ...}

for api_family, model_name in ALL_MODELS:
    prefix = raw_file_prefix(api_family, model_name)
    pattern = os.path.join(RAW_RESPONSES, f"{prefix}_*.txt")
    files = sorted(glob.glob(pattern))

    begr_tokens = []
    full_resp_tokens = []

    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                raw = f.read()
            data = json.loads(raw)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        full_resp_tokens.append(count_tokens(raw, model_name))
        for item in data.get("psychopathologies", []):
            bg = item.get("Begründung") or item.get("Reasoning") or ""
            if bg:
                begr_tokens.append(count_tokens(bg, model_name))

    output_by_model[model_name] = {
        "n_files": len(files),
        "n_begr": len(begr_tokens),
        "mean_begr": round(statistics.mean(begr_tokens), 1) if begr_tokens else None,
        "mean_full_resp": round(statistics.mean(full_resp_tokens), 1) if full_resp_tokens else None,
    }

    print(f"\n--- {model_name} ---")
    print(f"  Raw response files       : {len(files)}")
    if begr_tokens:
        print(f"  Total Begründungen       : {len(begr_tokens)}")
        print(f"  Mean tokens / Begründung : {statistics.mean(begr_tokens):>8.1f}")
        print(f"  Median                   : {statistics.median(begr_tokens):>8.1f}")
        print(f"  Min / Max                : {min(begr_tokens):>5} / {max(begr_tokens):>5}")
    if full_resp_tokens:
        print(f"  Mean tokens / full resp  : {statistics.mean(full_resp_tokens):>8.1f}")
    if not files:
        print("  (no raw response files found)")

# ═══════════════════════════════════════════════════════════════════
# 4. MERGE INTO inference_timing.csv
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)

# Read existing CSV
with open(TIMING_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Add new columns
for row in rows:
    mn = row["model_name"]
    row["input_tokens_per_run"] = input_by_model.get(mn, "")
    info = output_by_model.get(mn, {})
    row["mean_output_tokens_per_begruendung"] = info.get("mean_begr", "") or ""
    row["mean_output_tokens_per_response"] = info.get("mean_full_resp", "") or ""

# Write back
fieldnames = list(rows[0].keys())
with open(TIMING_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated: {os.path.abspath(TIMING_CSV)}")

# ═══════════════════════════════════════════════════════════════════
# 5. COST CALCULATION (1 run = 10 chunks × 10 defs = all 100 items)
# ═══════════════════════════════════════════════════════════════════
PRICES_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "03_ressources", "model_prices.csv")
COST_CSV   = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables", "cost_per_run.csv")

print("\n" + "=" * 80)
print("COST PER RUN  (1 transcript × 10 chunks × 10 defs = 100 items)")
print("=" * 80)

# Load prices
prices = {}
with open(PRICES_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        prices[row["model_name"]] = {
            "input_price": float(row["input_price_per_1M"]),
            "output_price": float(row["output_price_per_1M"]),
            "source": row["price_source"],
        }

NUM_CHUNKS_TOTAL = num_chunks  # 10

cost_rows = []
print(f"\n{'Model':<45} {'In tok':>10} {'Out tok':>10} {'In cost $':>10} {'Out cost $':>10} {'Total $':>10}")
print("-" * 100)

for api_family, model_name in ALL_MODELS:
    in_tok = input_by_model.get(model_name)
    info = output_by_model.get(model_name, {})
    mean_resp = info.get("mean_full_resp")

    # Output tokens for full run = mean tokens per response × number of chunks
    out_tok = round(mean_resp * NUM_CHUNKS_TOTAL, 0) if mean_resp else None

    p = prices.get(model_name, {})
    in_price = p.get("input_price", 0)
    out_price = p.get("output_price", 0)

    in_cost  = (in_tok * in_price / 1_000_000) if in_tok else None
    out_cost = (out_tok * out_price / 1_000_000) if out_tok else None
    total    = ((in_cost or 0) + (out_cost or 0)) if (in_cost is not None or out_cost is not None) else None

    cost_rows.append({
        "api_family": api_family,
        "model_name": model_name,
        "input_tokens_per_run": in_tok or "",
        "output_tokens_per_run": int(out_tok) if out_tok else "",
        "input_price_per_1M": in_price,
        "output_price_per_1M": out_price,
        "input_cost_usd": round(in_cost, 4) if in_cost else "",
        "output_cost_usd": round(out_cost, 4) if out_cost else "",
        "total_cost_usd": round(total, 4) if total else "",
    })

    in_s  = f"${in_cost:.4f}" if in_cost else "—"
    out_s = f"${out_cost:.4f}" if out_cost else "—"
    tot_s = f"${total:.4f}" if total else "—"
    print(f"{model_name:<45} {in_tok or '—':>10} {int(out_tok) if out_tok else '—':>10} {in_s:>10} {out_s:>10} {tot_s:>10}")

# Write cost CSV
os.makedirs(os.path.dirname(COST_CSV), exist_ok=True)
cost_fields = list(cost_rows[0].keys())
with open(COST_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cost_fields)
    writer.writeheader()
    writer.writerows(cost_rows)

print(f"\nSaved: {os.path.abspath(COST_CSV)}")
print("=" * 80)
