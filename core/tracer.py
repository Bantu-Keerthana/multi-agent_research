"""
Model Tiering Tracer — tracks which model handled each task and the latency.

Provides a singleton tracker that accumulates metrics across a pipeline run,
then generates a summary table for the final report and README.

Usage:
    from core.tracer import tracer

    tracer.reset()  # start of a new pipeline run
    tracer.log("planner", "llama-3.3-70b-versatile", "groq", latency_ms=85.2)
    tracer.log("web_search", "gemini-2.0-flash", "google", latency_ms=1200.5)

    print(tracer.summary_table())
    print(tracer.summary_json())
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class TraceEntry:
    """A single traced model invocation."""

    agent: str
    model: str
    provider: str
    latency_ms: float
    timestamp: str
    success: bool = True
    error: str = ""
    token_estimate: int = 0  # rough estimate for cost tracking


@dataclass
class PipelineTracer:
    """
    Accumulates model usage traces across a pipeline run.

    Thread-safe enough for sequential LangGraph execution.
    For true parallelism, wrap mutations in a lock.
    """

    entries: list[TraceEntry] = field(default_factory=list)
    pipeline_start: float = 0.0
    pipeline_query: str = ""

    def reset(self, query: str = ""):
        """Clear all traces for a new pipeline run."""
        self.entries.clear()
        self.pipeline_start = time.time()
        self.pipeline_query = query

    def log(
        self,
        agent: str,
        model: str,
        provider: str,
        latency_ms: float,
        success: bool = True,
        error: str = "",
        token_estimate: int = 0,
    ):
        """Record a model invocation."""
        entry = TraceEntry(
            agent=agent,
            model=model,
            provider=provider,
            latency_ms=round(latency_ms, 1),
            timestamp=datetime.now(timezone.utc).isoformat(),
            success=success,
            error=error,
            token_estimate=token_estimate,
        )
        self.entries.append(entry)
        logger.info(
            f"[Trace] {agent:15s} | {provider}/{model:30s} | "
            f"{latency_ms:>8.1f}ms | {'OK' if success else 'FAIL'}"
        )

    @property
    def total_latency_ms(self) -> float:
        """Total accumulated latency across all model calls."""
        return sum(e.latency_ms for e in self.entries)

    @property
    def pipeline_wall_time_ms(self) -> float:
        """Wall-clock time since pipeline start."""
        if self.pipeline_start:
            return (time.time() - self.pipeline_start) * 1000
        return 0.0

    def by_provider(self) -> dict[str, list[TraceEntry]]:
        """Group entries by provider (groq vs google)."""
        groups: dict[str, list[TraceEntry]] = {}
        for e in self.entries:
            groups.setdefault(e.provider, []).append(e)
        return groups

    def by_model(self) -> dict[str, list[TraceEntry]]:
        """Group entries by model name."""
        groups: dict[str, list[TraceEntry]] = {}
        for e in self.entries:
            groups.setdefault(e.model, []).append(e)
        return groups

    def summary_table(self) -> str:
        """Generate a Markdown table of model usage for the report."""
        if not self.entries:
            return "No model invocations traced."

        lines = [
            "| # | Agent | Provider | Model | Latency | Status |",
            "|---|-------|----------|-------|---------|--------|",
        ]
        for i, e in enumerate(self.entries, 1):
            status = "✅" if e.success else f"❌ {e.error[:30]}"
            lines.append(
                f"| {i} | {e.agent} | {e.provider} | {e.model} | "
                f"{e.latency_ms:.0f}ms | {status} |"
            )

        # Summary row
        by_prov = self.by_provider()
        lines.append("")
        lines.append("**Tiering Summary:**")
        for provider, entries in sorted(by_prov.items()):
            avg = sum(e.latency_ms for e in entries) / len(entries)
            total = sum(e.latency_ms for e in entries)
            lines.append(
                f"- **{provider}**: {len(entries)} calls, "
                f"avg {avg:.0f}ms, total {total:.0f}ms"
            )

        lines.append(f"- **Pipeline wall time**: {self.pipeline_wall_time_ms:.0f}ms")

        return "\n".join(lines)

    def summary_json(self) -> dict:
        """Return structured summary for API responses."""
        by_prov = self.by_provider()
        provider_stats = {}
        for provider, entries in by_prov.items():
            provider_stats[provider] = {
                "call_count": len(entries),
                "avg_latency_ms": round(
                    sum(e.latency_ms for e in entries) / len(entries), 1
                ),
                "total_latency_ms": round(
                    sum(e.latency_ms for e in entries), 1
                ),
                "models": list(set(e.model for e in entries)),
            }

        return {
            "total_calls": len(self.entries),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "pipeline_wall_time_ms": round(self.pipeline_wall_time_ms, 1),
            "providers": provider_stats,
            "entries": [
                {
                    "agent": e.agent,
                    "model": e.model,
                    "provider": e.provider,
                    "latency_ms": e.latency_ms,
                    "success": e.success,
                }
                for e in self.entries
            ],
        }


# Singleton tracer instance — import and use across the pipeline
tracer = PipelineTracer()
