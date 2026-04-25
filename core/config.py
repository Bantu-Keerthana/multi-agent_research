"""
Configuration for models, API keys, and model tiering strategy.

Model Tiering:
  - Lightweight tasks (routing, planning) → Groq Llama 3.3 70B (free, fast)
  - Heavy tasks (synthesis, reasoning)   → Gemini 2.5 Flash (free, powerful)
"""

import os
from dotenv import load_dotenv

load_dotenv()


# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")

# --- Model Configuration ---
# Tier 1: Fast & lightweight (routing, planning, classification)
FAST_MODEL = "llama-3.3-70b-versatile"  # via Groq — free tier
FAST_MODEL_PROVIDER = "groq"

# Tier 2: Powerful (synthesis, deep reasoning, long-context)
# NOTE: "gemini-2.0-flash" was retired for new users.
# "gemini-2.5-flash" is the current free-tier model.
POWER_MODEL = "gemini-2.5-flash"  # via Google — free tier
POWER_MODEL_PROVIDER = "google"

# --- Task → Model Mapping ---
MODEL_TIER_MAP: dict[str, str] = {
    "planner": FAST_MODEL,
    "router": FAST_MODEL,
    "web_search": POWER_MODEL,
    "data_analysis": POWER_MODEL,
    "summarization": POWER_MODEL,
    "code_generation": POWER_MODEL,
    "aggregator": POWER_MODEL,
}


def get_model_for_task(task_type: str) -> tuple[str, str]:
    """Return (model_name, provider) for a given task type."""
    model = MODEL_TIER_MAP.get(task_type, POWER_MODEL)
    provider = FAST_MODEL_PROVIDER if model == FAST_MODEL else POWER_MODEL_PROVIDER
    return model, provider


def validate_config() -> list[str]:
    """Check that required API keys are set. Returns list of warnings."""
    warnings = []
    if not GOOGLE_API_KEY:
        warnings.append("GOOGLE_API_KEY not set — Gemini agents will fail.")
    if not GROQ_API_KEY:
        warnings.append(
            "GROQ_API_KEY not set — falling back to Gemini for all tasks."
        )
    return warnings