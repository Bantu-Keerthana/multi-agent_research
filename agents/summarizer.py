"""
Summarization Agent — condenses all prior outputs into structured summaries.

Leverages Gemini 2.0 Flash's 1M token context window for processing
large volumes of agent output into concise, actionable summaries.
"""

import time
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from core.state import AgentState, AgentResult
from core.llm import invoke_llm, get_model_name

logger = logging.getLogger(__name__)

SUMMARIZER_SYSTEM_PROMPT = """You are an expert research summarizer. Given outputs from multiple 
research agents, create a clear, well-structured summary.

Guidelines:
- Organize by theme or topic, not by source agent
- Lead with the most important findings
- Preserve key data points, statistics, and specific facts
- Highlight areas of agreement and contradiction between sources
- Note information gaps or areas needing further research
- Use clear headers and bullet points for readability
- Keep the summary to 300-500 words
- End with 2-3 key takeaways

Write in a professional, objective tone. This summary feeds into the final report."""


def summarizer_node(state: AgentState) -> dict:
    """Execute all summarization subtasks using all prior results."""
    start = time.time()
    plan = state.get("plan", [])

    summary_tasks = [t for t in plan if t["task_type"] == "summarization"]
    if not summary_tasks:
        return {"current_step": "summarizer"}

    logger.info(f"Summarizer processing {len(summary_tasks)} tasks")

    # Gather ALL prior results as input context
    prior_results = state.get("results", [])
    context = "\n\n---\n\n".join(
        f"[Agent: {r['agent']} | Task: {r['task_id']}]\n{r['content']}"
        for r in prior_results
    )

    model_used = get_model_name("summarization")
    results: list[AgentResult] = []

    for task in summary_tasks:
        task_start = time.time()

        try:
            content = _execute_summary(task["description"], context, state["query"])
        except Exception as e:
            logger.error(f"Summary failed for task {task['id']}: {e}")
            content = f"Summarization failed: {str(e)}"

        latency = (time.time() - task_start) * 1000

        results.append(
            AgentResult(
                task_id=task["id"],
                agent="summarization",
                content=content,
                sources=[],
                model_used=model_used,
                latency_ms=round(latency, 1),
            )
        )

    total_latency = (time.time() - start) * 1000
    logger.info(f"Summarizer completed in {total_latency:.0f}ms")

    return {
        "results": results,
        "current_step": "summarizer",
        "metadata": {
            **state.get("metadata", {}),
            "summarizer_latency_ms": round(total_latency, 1),
            "summarizer_model": model_used,
            "summarizer_context_chars": len(context),
        },
    }


def _execute_summary(description: str, context: str, original_query: str) -> str:
    """Use the LLM to generate a structured summary."""
    messages = [
        SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original Research Query: {original_query}\n\n"
                f"Summarization Task: {description}\n\n"
                f"Agent Outputs to Summarize:\n{context}\n\n"
                f"Create a structured summary of the findings."
            )
        ),
    ]

    return invoke_llm("summarization", messages)
