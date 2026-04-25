"""
FastAPI server for the Multi-Agent Research System.

Endpoints:
  POST /research           — Run pipeline, returns SSE stream of agent progress
  POST /research/sync      — Run pipeline synchronously, returns JSON
  POST /research/review    — Run with human-in-the-loop (pauses for approval)
  POST /research/approve   — Resume a paused pipeline after human review
  GET  /health             — Health check
  GET  /config             — Show model tiering configuration

SSE Event Format:
  event: agent_start   data: {"agent": "planner", "status": "running"}
  event: agent_done    data: {"agent": "planner", "status": "done", "latency_ms": 85}
  event: report        data: {"report": "# Research Report..."}
  event: error         data: {"error": "..."}
  event: done          data: {}
"""

import json
import uuid
import asyncio
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.graph import build_graph, create_initial_state, get_review_summary
from core.config import MODEL_TIER_MAP, FAST_MODEL, POWER_MODEL, validate_config
from core.tracer import tracer

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Agent Research API",
    description="A multi-agent research pipeline powered by LangGraph, Gemini, and Groq",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for paused pipeline states (human review)
_paused_sessions: dict[str, dict] = {}


# ============================================================
# Request / Response Models
# ============================================================

class ResearchRequest(BaseModel):
    query: str

class ApproveRequest(BaseModel):
    session_id: str
    approved: bool = True
    feedback: str = ""


# ============================================================
# SSE Streaming
# ============================================================

def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_pipeline(query: str) -> AsyncGenerator[str, None]:
    """
    Run the pipeline and yield SSE events as each agent completes.

    Uses graph.stream() to get state updates after each node.
    """
    tracer.reset(query)

    graph = build_graph(enable_human_review=False)
    state = create_initial_state(query)
    thread_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}

    yield _sse_event("pipeline_start", {
        "query": query,
        "thread_id": thread_id,
    })

    try:
        # stream() yields {node_name: state_update} after each node
        for step in graph.stream(state, config, stream_mode="updates"):
            for node_name, node_output in step.items():
                current_step = node_output.get("current_step", node_name)

                # Emit agent progress event
                yield _sse_event("agent_done", {
                    "agent": current_step,
                    "status": "done",
                    "metadata": _safe_metadata(node_output.get("metadata", {})),
                    "results_count": len(node_output.get("results", [])),
                })

                # If this is the aggregator, emit the report
                if "final_report" in node_output and node_output["final_report"]:
                    yield _sse_event("report", {
                        "report": node_output["final_report"],
                    })

            # Small delay so the client can process events
            await asyncio.sleep(0.05)

        # Emit tiering summary
        yield _sse_event("tiering", tracer.summary_json())

        yield _sse_event("done", {"status": "complete"})

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        yield _sse_event("error", {"error": str(e)})


def _safe_metadata(metadata: dict) -> dict:
    """Ensure metadata is JSON-serializable."""
    safe = {}
    for k, v in metadata.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)
    return safe


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
async def health():
    """Health check with config validation."""
    warnings = validate_config()
    return {
        "status": "ok" if not warnings else "degraded",
        "version": "0.2.0",
        "warnings": warnings,
    }


@app.get("/config")
async def config():
    """Show the current model tiering configuration."""
    return {
        "fast_model": FAST_MODEL,
        "power_model": POWER_MODEL,
        "tier_map": MODEL_TIER_MAP,
    }


@app.post("/research")
async def research_stream(request: ResearchRequest):
    """
    Run the research pipeline with SSE streaming.

    Returns a stream of Server-Sent Events showing real-time agent progress.
    Connect with EventSource in the browser or any SSE client.
    """
    return StreamingResponse(
        _stream_pipeline(request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/research/sync")
async def research_sync(request: ResearchRequest):
    """Run the pipeline synchronously and return the full result as JSON."""
    tracer.reset(request.query)

    graph = build_graph(enable_human_review=False)
    state = create_initial_state(request.query)
    config = {"configurable": {"thread_id": uuid.uuid4().hex[:8]}}

    result = graph.invoke(state, config)

    return {
        "report": result.get("final_report", ""),
        "metadata": _safe_metadata(result.get("metadata", {})),
        "tiering": tracer.summary_json(),
    }


@app.post("/research/review")
async def research_with_review(request: ResearchRequest):
    """
    Run pipeline with human-in-the-loop.

    Runs planner + specialists, then pauses. Returns a session_id
    and review summary. Call /research/approve to continue.
    """
    tracer.reset(request.query)

    graph = build_graph(enable_human_review=True)
    state = create_initial_state(request.query)
    session_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": session_id}}

    # Run until interrupt
    paused_state = graph.invoke(state, config)

    # Store for later resumption
    _paused_sessions[session_id] = {
        "graph": graph,
        "config": config,
        "state": paused_state,
        "query": request.query,
    }

    return {
        "session_id": session_id,
        "status": "awaiting_review",
        "review_summary": get_review_summary(paused_state),
        "results_count": len(paused_state.get("results", [])),
    }


@app.post("/research/approve")
async def approve_review(request: ApproveRequest):
    """Resume a paused pipeline after human review."""
    session = _paused_sessions.pop(request.session_id, None)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id} not found or already completed",
        )

    graph = session["graph"]
    config = session["config"]

    if not request.approved:
        return {
            "status": "rejected",
            "message": "Pipeline rejected by reviewer.",
        }

    # Update state with approval
    graph.update_state(
        config,
        {
            "human_approved": True,
            "metadata": {
                "human_feedback": request.feedback or "approved",
            },
        },
    )

    # Resume pipeline
    final_state = graph.invoke(None, config)

    return {
        "status": "complete",
        "report": final_state.get("final_report", ""),
        "metadata": _safe_metadata(final_state.get("metadata", {})),
        "tiering": tracer.summary_json(),
    }


# ============================================================
# Run with: uvicorn api.server:app --reload
# ============================================================
