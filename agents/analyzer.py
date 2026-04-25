"""
Data Analysis Agent — analyzes data, computes metrics, and generates insights.

Uses Gemini 2.0 Flash to reason about data from prior agent outputs.
Can generate and describe analytical approaches, extract metrics,
and produce structured evaluations.
"""

import time
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from core.state import AgentState, AgentResult
from core.llm import invoke_llm, get_model_name

logger = logging.getLogger(__name__)

ANALYZER_SYSTEM_PROMPT = """You are a data analyst and critical thinker. Given research context and an 
analysis task, provide structured analytical insights.

Your job:
1. Extract quantitative data points from the provided context
2. Identify patterns, trends, and comparisons
3. Compute or estimate metrics when exact data isn't available (clearly label estimates)
4. Assess feasibility, risks, and opportunities
5. Present findings in a structured format with clear sections

Output format:
- Start with a brief "Key Metrics" section with bullet points
- Follow with "Analysis" — your detailed reasoning
- End with "Assessment" — your overall evaluation (use a 1-10 score if applicable)

Be specific. Use numbers when available. Clearly distinguish facts from estimates.
If the context lacks sufficient data, state what's missing and provide best estimates."""


def analyzer_node(state: AgentState) -> dict:
    """Execute all data_analysis subtasks using prior results as context."""
    start = time.time()
    plan = state.get("plan", [])

    analysis_tasks = [t for t in plan if t["task_type"] == "data_analysis"]
    if not analysis_tasks:
        return {"current_step": "analyzer"}

    logger.info(f"Analyzer processing {len(analysis_tasks)} tasks")

    # Gather context from previous search results
    prior_results = state.get("results", [])
    context = "\n\n---\n\n".join(
        f"[{r['agent']}] {r['content']}" for r in prior_results
    )

    model_used = get_model_name("data_analysis")
    results: list[AgentResult] = []

    for task in analysis_tasks:
        task_start = time.time()

        try:
            content = _execute_analysis(task["description"], context)
        except Exception as e:
            logger.error(f"Analysis failed for task {task['id']}: {e}")
            content = f"Analysis failed: {str(e)}"

        latency = (time.time() - task_start) * 1000

        results.append(
            AgentResult(
                task_id=task["id"],
                agent="data_analysis",
                content=content,
                sources=[],
                model_used=model_used,
                latency_ms=round(latency, 1),
            )
        )

    total_latency = (time.time() - start) * 1000
    logger.info(f"Analyzer completed in {total_latency:.0f}ms")

    return {
        "results": results,
        "current_step": "analyzer",
        "metadata": {
            **state.get("metadata", {}),
            "analyzer_latency_ms": round(total_latency, 1),
            "analyzer_model": model_used,
        },
    }


def _execute_analysis(description: str, context: str) -> str:
    """Use the LLM to perform data analysis on prior context."""
    max_context_chars = 12000
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n\n[... context truncated ...]"

    messages = [
        SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Analysis Task: {description}\n\n"
                f"Available Context (from prior research):\n{context}\n\n"
                f"Provide your structured analysis."
            )
        ),
    ]

    return invoke_llm("data_analysis", messages)
