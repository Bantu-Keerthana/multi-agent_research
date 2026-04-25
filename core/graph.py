"""
Multi-Agent Research Pipeline — LangGraph StateGraph

Graph flow:
  User Query → Planner → [Specialists fan-out] → Human Review → Aggregator → Report

Human-in-the-Loop:
  When enabled, the graph pauses via interrupt_before=["human_review"].
  The UI or CLI collects approval, then resumes the graph.
  If rejected, the user can modify the plan and re-run.

Streaming:
  graph.stream() yields state updates after each node, enabling
  real-time progress display in the Chainlit UI and SSE API.
"""

import logging
import uuid
import time
from typing import AsyncIterator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import AgentState
from core.tracer import tracer
from agents.planner import planner_node
from agents.web_search import web_search_node
from agents.analyzer import analyzer_node
from agents.summarizer import summarizer_node
from agents.code_gen import code_gen_node
from agents.aggregator import aggregator_node

logger = logging.getLogger(__name__)


# ============================================================
# Human Review Node
# ============================================================

def human_review_node(state: AgentState) -> dict:
    """
    Human-in-the-loop gate.

    When interrupt_before=["human_review"] is set, LangGraph pauses
    execution HERE. The caller inspects state["results"] (specialist
    outputs collected so far) and decides whether to proceed.

    To approve: update state with human_approved=True and resume.
    To reject:  update state with human_approved=False and error message.

    In non-interactive mode (enable_human_review=False), this node
    is skipped entirely — specialists connect directly to aggregator.
    """
    results = state.get("results", [])
    logger.info(
        f"Human Review: {len(results)} specialist results ready for inspection"
    )

    # Log a summary of what's been collected
    agents_done = set(r["agent"] for r in results)
    total_sources = sum(len(r["sources"]) for r in results)

    logger.info(f"  Agents completed: {', '.join(sorted(agents_done))}")
    logger.info(f"  Total sources found: {total_sources}")
    logger.info(f"  Awaiting human approval to proceed to aggregation...")

    return {
        "current_step": "human_review",
        "metadata": {
            **state.get("metadata", {}),
            "human_review_results_count": len(results),
            "human_review_agents": list(agents_done),
        },
    }


# ============================================================
# Router
# ============================================================

def _route_after_planner(state: AgentState) -> list[str]:
    """
    Conditional router: inspect the plan and decide which specialist
    agents to invoke. Returns a list of node names for fan-out.
    """
    plan = state.get("plan", [])
    task_types = {t["task_type"] for t in plan}

    destinations = []

    if "web_search" in task_types:
        destinations.append("web_search")
    if "data_analysis" in task_types:
        destinations.append("analyzer")
    if "summarization" in task_types:
        destinations.append("summarizer")
    if "code_generation" in task_types:
        destinations.append("code_gen")

    if not destinations:
        destinations.append("aggregator")

    logger.info(f"Router dispatching to: {destinations}")
    return destinations


# ============================================================
# Graph Builder
# ============================================================

def build_graph(enable_human_review: bool = False):
    """
    Construct the multi-agent research graph.

    Args:
        enable_human_review: If True, inserts a human_review node between
            specialists and aggregator, with interrupt_before for pausing.

    Returns:
        Compiled LangGraph ready to invoke/stream.
    """
    graph = StateGraph(AgentState)

    # --- Add nodes ---
    graph.add_node("planner", planner_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("code_gen", code_gen_node)
    graph.add_node("aggregator", aggregator_node)

    # --- Define edges ---
    graph.add_edge(START, "planner")

    if enable_human_review:
        # With human review: specialists → human_review → aggregator
        graph.add_node("human_review", human_review_node)

        graph.add_conditional_edges(
            "planner",
            _route_after_planner,
            {
                "web_search": "web_search",
                "analyzer": "analyzer",
                "summarizer": "summarizer",
                "code_gen": "code_gen",
                "aggregator": "aggregator",
            },
        )

        # All specialists → human_review
        graph.add_edge("web_search", "human_review")
        graph.add_edge("analyzer", "human_review")
        graph.add_edge("summarizer", "human_review")
        graph.add_edge("code_gen", "human_review")

        # human_review → aggregator
        graph.add_edge("human_review", "aggregator")
    else:
        # Without human review: specialists → aggregator directly
        graph.add_conditional_edges(
            "planner",
            _route_after_planner,
            {
                "web_search": "web_search",
                "analyzer": "analyzer",
                "summarizer": "summarizer",
                "code_gen": "code_gen",
                "aggregator": "aggregator",
            },
        )

        graph.add_edge("web_search", "aggregator")
        graph.add_edge("analyzer", "aggregator")
        graph.add_edge("summarizer", "aggregator")
        graph.add_edge("code_gen", "aggregator")

    graph.add_edge("aggregator", END)

    # --- Compile ---
    checkpointer = MemorySaver()
    compile_kwargs = {"checkpointer": checkpointer}

    if enable_human_review:
        compile_kwargs["interrupt_before"] = ["human_review"]

    compiled = graph.compile(**compile_kwargs)
    logger.info(
        f"Graph compiled (human_review={'enabled' if enable_human_review else 'disabled'})"
    )

    return compiled


# ============================================================
# State Helpers
# ============================================================

def create_initial_state(query: str) -> AgentState:
    """Create a fresh AgentState for a new query."""
    return AgentState(
        query=query,
        plan=[],
        results=[],
        final_report="",
        human_approved=False,
        metadata={},
        error="",
        current_step="initialized",
    )


def get_review_summary(state: dict) -> str:
    """
    Format specialist results for human review.
    Returns a readable summary the user can approve or reject.
    """
    results = state.get("results", [])
    plan = state.get("plan", [])

    lines = []
    lines.append(f"Query: {state.get('query', 'N/A')}")
    lines.append(f"Plan: {len(plan)} subtasks")
    lines.append(f"Results: {len(results)} completed")
    lines.append("")

    for r in results:
        preview = r["content"][:200].replace("\n", " ")
        lines.append(f"  [{r['agent']}] ({r['model_used']}, {r['latency_ms']:.0f}ms)")
        lines.append(f"    {preview}...")
        if r["sources"]:
            lines.append(f"    Sources: {', '.join(r['sources'][:3])}")
        lines.append("")

    return "\n".join(lines)
