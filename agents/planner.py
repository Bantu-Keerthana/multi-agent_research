"""
Planner Agent — decomposes a user query into actionable subtasks.

Uses the FAST model (Groq Llama 3.3 70B) for quick task decomposition.
Falls back to rule-based planning if the LLM call fails.
"""

import json
import time
import uuid
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from core.state import AgentState, SubTask
from core.llm import invoke_llm, get_model_name

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a strategic research planner. Given a user query, decompose it into 
a list of concrete subtasks that specialist agents can execute.

Each subtask must have:
- task_type: one of "web_search", "data_analysis", "summarization", "code_generation"
- description: a clear, actionable description of what to do

Guidelines:
- Use "web_search" for finding current information, news, competitors, market data, facts
- Use "data_analysis" for analyzing numbers, comparing data, computing metrics, evaluating feasibility
- Use "summarization" for condensing large amounts of information into key insights
- Use "code_generation" for creating scripts, tools, or technical implementations
- Aim for 2-5 subtasks — enough to be thorough but not excessive
- Order tasks logically (search first, then analyze, then summarize)
- Make descriptions specific and actionable, not vague

Respond with ONLY a valid JSON array of objects. No markdown fences, no explanation, no preamble.

Example:
[
  {"task_type": "web_search", "description": "Search for recent AI startup funding rounds in 2024-2025"},
  {"task_type": "web_search", "description": "Find major competitors and their product offerings in the AI agent space"},
  {"task_type": "data_analysis", "description": "Compare funding amounts and valuations across top 10 AI startups"},
  {"task_type": "summarization", "description": "Synthesize key trends and investment patterns in AI startups"}
]"""


def planner_node(state: AgentState) -> dict:
    """
    Decompose the user query into subtasks using the LLM.
    Falls back to rule-based planning if the LLM fails.
    """
    start = time.time()
    query = state["query"]
    logger.info(f"Planner processing query: {query[:100]}...")

    model_used = get_model_name("planner")

    try:
        plan = _call_planner_llm(query)
        logger.info(f"LLM planner produced {len(plan)} tasks")
    except Exception as e:
        logger.warning(f"LLM planner failed ({e}), using rule-based fallback")
        plan = _rule_based_plan(query)
        model_used = "rule-based-fallback"

    latency = (time.time() - start) * 1000

    subtasks: list[SubTask] = []
    for i, task in enumerate(plan):
        subtasks.append(
            SubTask(
                id=f"task_{i}_{uuid.uuid4().hex[:6]}",
                task_type=task["task_type"],
                description=task["description"],
                status="pending",
            )
        )

    logger.info(f"Planner created {len(subtasks)} subtasks in {latency:.0f}ms (model: {model_used})")

    return {
        "plan": subtasks,
        "current_step": "planner",
        "metadata": {
            **state.get("metadata", {}),
            "planner_latency_ms": round(latency, 1),
            "planner_model": model_used,
            "planner_task_count": len(subtasks),
        },
    }


def _call_planner_llm(query: str) -> list[dict]:
    """Call the LLM to decompose the query. Parses JSON with error recovery."""
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Decompose this research query into subtasks:\n\n{query}"),
    ]

    raw = invoke_llm("planner", messages)

    # Strip markdown fences if the model wraps in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    plan = json.loads(raw)

    if not isinstance(plan, list) or len(plan) == 0:
        raise ValueError("Planner returned empty or non-list response")

    valid_types = {"web_search", "data_analysis", "summarization", "code_generation"}
    validated = []
    for task in plan:
        if not isinstance(task, dict):
            continue
        if task.get("task_type") not in valid_types:
            tt = task.get("task_type", "").lower().replace(" ", "_")
            if "search" in tt:
                task["task_type"] = "web_search"
            elif "analy" in tt:
                task["task_type"] = "data_analysis"
            elif "summar" in tt:
                task["task_type"] = "summarization"
            elif "code" in tt:
                task["task_type"] = "code_generation"
            else:
                task["task_type"] = "web_search"
        if "description" not in task or not task["description"]:
            continue
        validated.append(task)

    if not validated:
        raise ValueError("No valid tasks after validation")

    return validated


def _rule_based_plan(query: str) -> list[dict]:
    """Fallback planner that creates a sensible default plan."""
    query_lower = query.lower()
    tasks = []

    tasks.append({
        "task_type": "web_search",
        "description": f"Search the web for current information about: {query}",
    })

    data_keywords = ["data", "compare", "analysis", "numbers", "statistics", "metric",
                     "trend", "market", "size", "growth", "revenue", "price", "cost",
                     "feasibility", "score", "evaluate", "assess"]
    if any(kw in query_lower for kw in data_keywords):
        tasks.append({
            "task_type": "data_analysis",
            "description": f"Analyze and evaluate key metrics related to: {query}",
        })

    code_keywords = ["code", "build", "implement", "script", "program", "app",
                     "tool", "api", "function", "algorithm"]
    if any(kw in query_lower for kw in code_keywords):
        tasks.append({
            "task_type": "code_generation",
            "description": f"Generate code or technical solution for: {query}",
        })

    tasks.append({
        "task_type": "summarization",
        "description": f"Summarize all findings into a coherent research report about: {query}",
    })

    return tasks
