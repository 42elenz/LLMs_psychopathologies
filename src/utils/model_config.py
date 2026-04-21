models_to_use = [
    #("open_source", "Qwen/Qwen3-Next-80B-A3B-Thinking"),
    #("open_source", "deepseek-ai/DeepSeek-R1"),
    #("open_source", "moonshotai/Kimi-K2-Thinking"),
    #("gemini", "gemini-3-pro-preview"),
    #("anthropic", "claude-sonnet-4-5"),
    #("openai", "gpt-5.1"),
    #("gemini", "gemini-2.5-flash"),
    #("mistral", "mistral-large-latest"),
    #("open_source", "openai/gpt-oss-20b"),
    #("openai", "gpt-4-turbo"),
     ("openai", "gpt-5.1"),
     #("gemini", "gemini-3.1-pro-preview"),

    # ── Local vLLM models ──────────────────────────────────────────────
    # Deploy with:  vllm serve Qwen/Qwen2.5-32B-Instruct --tensor-parallel-size <N_GPUS>
    # Then set VLLM_BASE_URL in .env (default: http://localhost:8000/v1)
    #("vllm", "Qwen/Qwen2.5-32B-Instruct"),
]