"""
Calculate cost per run (1 transcript × 10 chunks = all 100 AMDP items).

Data sources (in priority order):
  1. inference_timing.csv   — measured input_tokens_per_run & output_tokens_1chunk
  2. token_estimation.csv   — estimated tokens from raw responses (fills gaps)
  3. model_prices.csv       — $/1M tokens

Output: outputs/tables/cost_per_run.csv
"""
import csv, os

BASE = os.path.join(os.path.dirname(__file__), "..")
TIMING   = os.path.join(BASE, "outputs", "tables", "inference_timing.csv")
ESTIM    = os.path.join(BASE, "outputs", "tables", "token_estimation.csv")
PRICES   = os.path.join(BASE, "data", "03_ressources", "model_prices.csv")
OUT_CSV  = os.path.join(BASE, "outputs", "tables", "cost_per_run.csv")

NUM_CHUNKS = 10  # 100 items / 10 defs per chunk
NUM_OUTER_RUNS = 10  # majority-voting repetitions

# ── Load timing (measured) ──────────────────────────────────────────
timing = {}
with open(TIMING, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        timing[r["model_name"]] = r

# ── Load estimation (from raw responses, open_source only) ──────────
estim = {}
with open(ESTIM, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        estim[r["model_name"]] = r

# ── Load prices ─────────────────────────────────────────────────────
prices = {}
with open(PRICES, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        prices[r["model_name"]] = r

# ── Merge & calculate ──────────────────────────────────────────────
def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

rows = []
for model_name, p in prices.items():
    t = timing.get(model_name, {})
    e = estim.get(model_name, {})

    api_family = t.get("api_family") or p.get("api_family", "")

    # Input tokens: prefer measured (timing), fallback to estimation
    in_tok = safe_float(t.get("input_tokens_per_run")) or safe_float(e.get("input_tokens_per_run"))

    # Output tokens per run:
    #   timing has output_tokens_1chunk (measured for 1 chunk) → multiply by NUM_CHUNKS
    #   estimation has mean_tokens_per_full_response (mean per chunk from raw files) → multiply by NUM_CHUNKS
    out_1chunk = safe_float(t.get("output_tokens_1chunk"))
    if out_1chunk:
        out_tok = out_1chunk * NUM_CHUNKS
    else:
        mean_resp = safe_float(e.get("mean_tokens_per_full_response"))
        out_tok = mean_resp * NUM_CHUNKS if mean_resp else None

    in_price  = float(p["input_price_per_1M"])
    out_price = float(p["output_price_per_1M"])

    in_cost  = (in_tok * in_price / 1_000_000) if in_tok else None
    out_cost = (out_tok * out_price / 1_000_000) if out_tok else None
    total    = ((in_cost or 0) + (out_cost or 0)) if (in_cost is not None or out_cost is not None) else None

    total_10 = round(total * NUM_OUTER_RUNS, 4) if total else None

    rows.append({
        "api_family": api_family,
        "model_name": model_name,
        "input_tokens_per_run": int(in_tok) if in_tok else "",
        "output_tokens_per_run": int(out_tok) if out_tok else "",
        "input_price_per_1M": in_price,
        "output_price_per_1M": out_price,
        "input_cost_usd": round(in_cost, 4) if in_cost else "",
        "output_cost_usd": round(out_cost, 4) if out_cost else "",
        "total_cost_1run_usd": round(total, 4) if total else "",
        "total_cost_10runs_usd": total_10 if total_10 else "",
    })

# ── Print ───────────────────────────────────────────────────────────
print(f"{'Model':<45} {'In tok':>10} {'Out tok':>10} {'1 run $':>10} {'10 runs $':>10}")
print("-" * 90)
for r in rows:
    t1  = f"${r['total_cost_1run_usd']:.4f}" if r['total_cost_1run_usd'] != "" else "—"
    t10 = f"${r['total_cost_10runs_usd']:.4f}" if r['total_cost_10runs_usd'] != "" else "—"
    print(f"{r['model_name']:<45} {r['input_tokens_per_run']:>10} {r['output_tokens_per_run']:>10} {t1:>10} {t10:>10}")

# ── Save CSV ────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
fields = list(rows[0].keys())
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"\nSaved: {OUT_CSV}")
