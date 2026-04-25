"""
State schema for the Multi-Agent Research pipeline.

The AgentState flows through:
  User Query → Planner → Specialists → Aggregator → Final Report
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict


class SubTask(TypedDict):
    """A single subtask produced by the Planner agent."""

    id: str
    task_type: str  # "web_search" | "data_analysis" | "summarization" | "code_generation"
    description: str
    status: str  # "pending" | "in_progress" | "completed" | "failed"


class AgentResult(TypedDict):
    """Output from a specialist agent."""

    task_id: str
    agent: str
    content: str
    sources: list[str]
    model_used: str
    latency_ms: float


class AgentState(TypedDict):
    """
    Central state that flows through the LangGraph pipeline.

    Attributes:
        query:          Original user query.
        plan:           List of subtasks from the Planner.
        results:        Accumulated specialist outputs (append-only via operator.add).
        final_report:   Synthesized Markdown report from the Aggregator.
        human_approved: Gate flag — the graph pauses here for human review.
        metadata:       Flexible dict for tracing info (models used, latencies, etc.).
        error:          Error message if any step fails.
        current_step:   Which node is currently executing (for UI progress).
    """

    query: str
    plan: list[SubTask]
    results: Annotated[list[AgentResult], operator.add]  # append-only
    final_report: str
    human_approved: bool
    metadata: dict[str, Any]
    error: str
    current_step: str
