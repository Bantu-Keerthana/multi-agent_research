"""
Tests for the Multi-Agent Research Pipeline — Milestones 1–4.

Test categories:
  - Config & State: model tiering, state schema
  - Tracer: logging, summary output
  - Planner: rule-based fallback
  - Router: conditional fan-out logic
  - Agent nodes: skip-when-empty, result structure
  - Human review: graph pausing and resumption
  - Graph integration: full pipeline end-to-end
  - API: endpoint response structure

Run: python -m pytest tests/ -v
"""

import sys
import os
import uuid
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state import AgentState, SubTask, AgentResult
from core.config import get_model_for_task, FAST_MODEL, POWER_MODEL


# ============================================================
# Config Tests
# ============================================================

class TestConfig:
    def test_planner_uses_fast_model(self):
        model, provider = get_model_for_task("planner")
        assert model == FAST_MODEL
        assert provider == "groq"

    def test_aggregator_uses_power_model(self):
        model, provider = get_model_for_task("aggregator")
        assert model == POWER_MODEL
        assert provider == "google"

    def test_unknown_task_defaults_to_power(self):
        model, _ = get_model_for_task("unknown_xyz")
        assert model == POWER_MODEL


# ============================================================
# Tracer Tests
# ============================================================

class TestTracer:
    def test_log_and_retrieve(self):
        from core.tracer import PipelineTracer
        t = PipelineTracer()
        t.reset("test query")
        t.log("planner", "llama-3.3-70b", "groq", 85.2)
        t.log("web_search", "gemini-2.0-flash", "google", 1200.0)

        assert len(t.entries) == 2
        assert t.total_latency_ms == 85.2 + 1200.0

    def test_by_provider(self):
        from core.tracer import PipelineTracer
        t = PipelineTracer()
        t.reset()
        t.log("planner", "llama", "groq", 80.0)
        t.log("search", "gemini", "google", 500.0)
        t.log("analyzer", "gemini", "google", 600.0)

        by_prov = t.by_provider()
        assert len(by_prov["groq"]) == 1
        assert len(by_prov["google"]) == 2

    def test_summary_table_not_empty(self):
        from core.tracer import PipelineTracer
        t = PipelineTracer()
        t.reset()
        t.log("planner", "llama", "groq", 80.0)
        table = t.summary_table()
        assert "planner" in table
        assert "groq" in table

    def test_summary_json_structure(self):
        from core.tracer import PipelineTracer
        t = PipelineTracer()
        t.reset("query")
        t.log("planner", "llama", "groq", 80.0, success=True)
        t.log("search", "gemini", "google", 500.0, success=False, error="timeout")

        j = t.summary_json()
        assert j["total_calls"] == 2
        assert j["total_latency_ms"] == 580.0
        assert len(j["entries"]) == 2
        assert j["entries"][1]["success"] is False

    def test_reset_clears(self):
        from core.tracer import PipelineTracer
        t = PipelineTracer()
        t.log("x", "y", "z", 1.0)
        t.reset("new")
        assert len(t.entries) == 0
        assert t.pipeline_query == "new"

    def test_empty_tracer_summary(self):
        from core.tracer import PipelineTracer
        t = PipelineTracer()
        t.reset()
        assert t.summary_table() == "No model invocations traced."


# ============================================================
# State Tests
# ============================================================

class TestState:
    def test_initial_state(self):
        from core.graph import create_initial_state
        state = create_initial_state("Test")
        assert state["query"] == "Test"
        assert state["plan"] == []
        assert state["results"] == []
        assert state["human_approved"] is False
        assert state["error"] == ""

    def test_subtask_fields(self):
        task = SubTask(id="t0", task_type="web_search",
                       description="search", status="pending")
        assert task["task_type"] == "web_search"

    def test_agent_result_fields(self):
        result = AgentResult(
            task_id="t0", agent="web_search",
            content="data", sources=["https://x.com"],
            model_used="gemini", latency_ms=100.0,
        )
        assert result["agent"] == "web_search"
        assert len(result["sources"]) == 1


# ============================================================
# Planner Tests (rule-based — no LLM)
# ============================================================

class TestPlanner:
    def test_basic_query(self):
        from agents.planner import _rule_based_plan
        plan = _rule_based_plan("What is quantum computing?")
        types = [t["task_type"] for t in plan]
        assert "web_search" in types
        assert "summarization" in types

    def test_data_query(self):
        from agents.planner import _rule_based_plan
        plan = _rule_based_plan("Compare market statistics")
        types = [t["task_type"] for t in plan]
        assert "data_analysis" in types

    def test_code_query(self):
        from agents.planner import _rule_based_plan
        plan = _rule_based_plan("Build a REST API")
        types = [t["task_type"] for t in plan]
        assert "code_generation" in types

    def test_minimum_two_tasks(self):
        from agents.planner import _rule_based_plan
        plan = _rule_based_plan("Hello")
        assert len(plan) >= 2

    def test_node_returns_metadata(self):
        from agents.planner import planner_node
        from core.graph import create_initial_state
        state = create_initial_state("Test AI trends")
        result = planner_node(state)
        assert "plan" in result
        assert result["current_step"] == "planner"
        assert "planner_latency_ms" in result["metadata"]
        assert "planner_model" in result["metadata"]


# ============================================================
# Router Tests
# ============================================================

class TestRouter:
    def _make_state(self, task_types: list[str]) -> AgentState:
        plan = [
            SubTask(id=f"t{i}", task_type=tt, description="x", status="pending")
            for i, tt in enumerate(task_types)
        ]
        return AgentState(
            query="test", plan=plan, results=[], final_report="",
            human_approved=False, metadata={}, error="", current_step="",
        )

    def test_search_routes_correctly(self):
        from core.graph import _route_after_planner
        dests = _route_after_planner(self._make_state(["web_search"]))
        assert "web_search" in dests

    def test_code_routes_correctly(self):
        from core.graph import _route_after_planner
        dests = _route_after_planner(self._make_state(["code_generation"]))
        assert "code_gen" in dests

    def test_empty_plan_routes_to_aggregator(self):
        from core.graph import _route_after_planner
        dests = _route_after_planner(self._make_state([]))
        assert "aggregator" in dests

    def test_mixed_plan_fans_out(self):
        from core.graph import _route_after_planner
        dests = _route_after_planner(
            self._make_state(["web_search", "data_analysis", "summarization"])
        )
        assert len(dests) == 3
        assert set(dests) == {"web_search", "analyzer", "summarizer"}


# ============================================================
# Agent Node Skip-When-Empty Tests
# ============================================================

class TestAgentSkips:
    def _empty_state(self) -> AgentState:
        return AgentState(
            query="test", plan=[], results=[], final_report="",
            human_approved=False, metadata={}, error="", current_step="",
        )

    def test_web_search_skips(self):
        from agents.web_search import web_search_node
        result = web_search_node(self._empty_state())
        assert result["current_step"] == "web_search"
        assert "results" not in result

    def test_analyzer_skips(self):
        from agents.analyzer import analyzer_node
        result = analyzer_node(self._empty_state())
        assert result["current_step"] == "analyzer"

    def test_summarizer_skips(self):
        from agents.summarizer import summarizer_node
        result = summarizer_node(self._empty_state())
        assert result["current_step"] == "summarizer"

    def test_code_gen_skips(self):
        from agents.code_gen import code_gen_node
        result = code_gen_node(self._empty_state())
        assert result["current_step"] == "code_gen"


# ============================================================
# Human Review Node Tests
# ============================================================

class TestHumanReview:
    def test_review_node_logs_results(self):
        from core.graph import human_review_node
        state = AgentState(
            query="test",
            plan=[],
            results=[
                AgentResult(task_id="t0", agent="web_search", content="data",
                           sources=["https://x.com"], model_used="gemini",
                           latency_ms=100.0),
            ],
            final_report="", human_approved=False,
            metadata={}, error="", current_step="",
        )
        result = human_review_node(state)
        assert result["current_step"] == "human_review"
        assert result["metadata"]["human_review_results_count"] == 1

    def test_review_summary_format(self):
        from core.graph import get_review_summary
        state = {
            "query": "test query",
            "plan": [SubTask(id="t0", task_type="web_search",
                           description="search", status="pending")],
            "results": [
                AgentResult(task_id="t0", agent="web_search",
                           content="Found relevant data about the topic",
                           sources=["https://example.com"],
                           model_used="gemini", latency_ms=150.0),
            ],
        }
        summary = get_review_summary(state)
        assert "test query" in summary
        assert "web_search" in summary
        assert "gemini" in summary


# ============================================================
# Aggregator Tests
# ============================================================

class TestAggregator:
    def test_fallback_summary(self):
        from agents.aggregator import _fallback_summary
        results = [
            AgentResult(task_id="t0", agent="web_search", content="data",
                       sources=[], model_used="gemini", latency_ms=0),
        ]
        summary = _fallback_summary("test query", results)
        assert "test query" in summary
        assert "1" in summary


# ============================================================
# Graph Integration (full pipeline — uses rule-based fallback)
# ============================================================

class TestGraphIntegration:
    def _run(self, query: str, review: bool = False) -> dict:
        from core.graph import build_graph, create_initial_state
        graph = build_graph(enable_human_review=review)
        state = create_initial_state(query)
        config = {"configurable": {"thread_id": uuid.uuid4().hex[:8]}}
        return graph.invoke(state, config)

    def test_full_pipeline_produces_report(self):
        result = self._run("What are AI trends?")
        assert result["final_report"] != ""
        assert "Research Report" in result["final_report"]

    def test_pipeline_has_results(self):
        result = self._run("What is quantum computing?")
        assert len(result["results"]) >= 1

    def test_pipeline_preserves_query(self):
        query = "Pipeline validation test"
        result = self._run(query)
        assert query in result["final_report"]

    def test_metadata_tracks_planner(self):
        result = self._run("Quick test")
        metadata = result.get("metadata", {})
        assert "planner_model" in metadata or "planner_latency_ms" in metadata

    def test_streaming_mode_works(self):
        """Test that graph.stream() yields step updates."""
        from core.graph import build_graph, create_initial_state
        graph = build_graph(enable_human_review=False)
        state = create_initial_state("Streaming test")
        config = {"configurable": {"thread_id": uuid.uuid4().hex[:8]}}

        steps = list(graph.stream(state, config, stream_mode="updates"))
        assert len(steps) >= 2  # at minimum: planner + aggregator

        # Last step should contain the report
        last_step = steps[-1]
        has_report = any(
            "final_report" in output and output["final_report"]
            for output in last_step.values()
        )
        assert has_report


# ============================================================
# API Tests (structure only — no actual HTTP server)
# ============================================================

class TestAPIStructure:
    def test_sse_event_format(self):
        from api.server import _sse_event
        event = _sse_event("agent_done", {"agent": "planner", "status": "done"})
        assert event.startswith("event: agent_done\n")
        assert "data: " in event
        data_line = event.split("data: ")[1].strip()
        parsed = json.loads(data_line)
        assert parsed["agent"] == "planner"

    def test_safe_metadata_handles_non_serializable(self):
        from api.server import _safe_metadata
        meta = {
            "normal": "value",
            "number": 42,
            "bad": object(),
        }
        safe = _safe_metadata(meta)
        assert safe["normal"] == "value"
        assert safe["number"] == 42
        assert isinstance(safe["bad"], str)  # converted to string


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
