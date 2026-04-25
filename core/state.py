"""
State schema for the Multi-Agent Research pipeline.

The AgentState flows through:
  User Query → Planner → Specialists → Aggregator → Final Report

Note: Fields written by parallel agents (fan-out) must use Annotated
reducers. Without this, LangGraph raises InvalidUpdateError when
multiple nodes write to the same key in a single step.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict


def _last_value(existing: str, new: str) -> str:
    """Reducer that keeps the latest value (for parallel writes)."""
    return new


def _merge_dicts(existing: dict, new: dict) -> dict:
    """Reducer that merges dicts (for parallel metadata writes)."""
    merged = {**existing}
    merged.update(new)
    return merged


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
        metadata:       Flexible dict for tracing info (merge on parallel writes).
        error:          Error message if any step fails.
        current_step:   Which node is currently executing (last-write-wins on parallel).
    """

    query: str
    plan: list[SubTask]
    results: Annotated[list[AgentResult], operator.add]  # append-only
    final_report: str
    human_approved: bool
    metadata: Annotated[dict[str, Any], _merge_dicts]  # merge parallel writes
    error: str
    current_step: Annotated[str, _last_value]  # last-write-wins on parallel