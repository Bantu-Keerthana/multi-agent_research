"""
LLM Factory — centralizes model initialization, invocation, and tracing.

Model Tiering:
  - Groq Llama 3.3 70B  → fast tasks (planning, routing)
  - Gemini 2.0 Flash     → heavy tasks (synthesis, search analysis, summarization)

If Groq key is missing, all tasks fall back to Gemini.
If both keys are missing, raises a clear error.

Every LLM call is traced via invoke_llm() for the model tiering report.
"""

import time
import logging
from functools import lru_cache

from langchain_core.messages import BaseMessage

from core.config import (
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    FAST_MODEL,
    POWER_MODEL,
    get_model_for_task,
)

logger = logging.getLogger(__name__)

_warned_groq_fallback = False


@lru_cache(maxsize=4)
def get_llm(task_type: str, temperature: float = 0.2):
    """
    Return the appropriate LangChain ChatModel for a given task type.
    Falls back to Gemini if Groq key is unavailable.
    """
    global _warned_groq_fallback

    model_name, provider = get_model_for_task(task_type)

    # --- Try Groq ---
    if provider == "groq" and GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model=model_name,
                api_key=GROQ_API_KEY,
                temperature=temperature,
                max_tokens=2048,
            )
            logger.info(f"[{task_type}] Using Groq/{model_name}")
            return llm
        except Exception as e:
            logger.warning(f"Groq init failed ({e}), falling back to Gemini")

    if provider == "groq" and not GROQ_API_KEY and not _warned_groq_fallback:
        logger.warning("GROQ_API_KEY not set — using Gemini for all tasks")
        _warned_groq_fallback = True

    # --- Gemini (primary or fallback) ---
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "No LLM available. Set at least GOOGLE_API_KEY in your .env file.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=POWER_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            max_output_tokens=4096,
        )
        logger.info(f"[{task_type}] Using Gemini/{POWER_MODEL}")
        return llm
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini: {e}")


def invoke_llm(task_type: str, messages: list[BaseMessage]) -> str:
    """
    Invoke the LLM for a task AND log the call to the tracer.

    This is the main entry point agents should use instead of
    calling get_llm().invoke() directly.

    Returns the response content as a string.
    """
    from core.tracer import tracer  # deferred to avoid circular import

    model_name = get_model_name(task_type)
    _, provider = get_model_for_task(task_type)
    # If falling back, correct the provider
    if provider == "groq" and not GROQ_API_KEY:
        provider = "google"
        model_name = POWER_MODEL

    llm = get_llm(task_type)
    start = time.time()
    success = True
    error_msg = ""

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
    except Exception as e:
        success = False
        error_msg = str(e)
        content = ""
        raise
    finally:
        latency = (time.time() - start) * 1000
        tracer.log(
            agent=task_type,
            model=model_name,
            provider=provider,
            latency_ms=latency,
            success=success,
            error=error_msg,
        )

    return content


def get_model_name(task_type: str) -> str:
    """Return the model name string that will be used for a task."""
    model_name, provider = get_model_for_task(task_type)
    if provider == "groq" and not GROQ_API_KEY:
        return POWER_MODEL
    return model_name
