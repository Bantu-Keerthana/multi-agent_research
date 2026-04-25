"""
Chainlit UI for the Multi-Agent Research System.

Shows real-time agent progress as each step completes:
  "🗺️ Planner: decomposing..." → "🔍 Web Search: found 5 sources..."
  → "🔒 Review: awaiting approval..." → "📊 Aggregator: synthesizing..."

Run: chainlit run ui/app.py --host 0.0.0.0 --port 8080
"""

import uuid
import logging
import chainlit as cl

from core.graph import build_graph, create_initial_state, get_review_summary
from core.tracer import tracer

logger = logging.getLogger(__name__)

# Agent display config
AGENT_ICONS = {
    "planner": "🗺️",
    "web_search": "🔍",
    "analyzer": "📊",
    "summarizer": "📝",
    "code_gen": "💻",
    "human_review": "🔒",
    "aggregator": "🧩",
}

AGENT_LABELS = {
    "planner": "Planner",
    "web_search": "Web Search",
    "analyzer": "Data Analysis",
    "summarizer": "Summarizer",
    "code_gen": "Code Generator",
    "human_review": "Human Review",
    "aggregator": "Aggregator",
}


@cl.on_chat_start
async def on_start():
    """Initialize the chat session."""
    cl.user_session.set("pending_review", False)
    await cl.Message(
        content=(
            "## 🔬 Multi-Agent Research System\n\n"
            "Enter a research query and I'll dispatch a team of AI agents to investigate.\n\n"
            "**Agents available:**\n"
            "- 🗺️ Planner — decomposes your query into subtasks\n"
            "- 🔍 Web Search — searches DuckDuckGo & Wikipedia\n"
            "- 📊 Data Analysis — analyzes metrics and trends\n"
            "- 📝 Summarizer — condenses findings\n"
            "- 💻 Code Generator — creates technical solutions\n"
            "- 🧩 Aggregator — synthesizes the final report\n\n"
            "**Tip:** Start your query with `/review` to enable human-in-the-loop approval.\n"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Single message handler — routes between new queries and review responses.

    Chainlit only supports one @cl.on_message, so we check session state
    to determine if this message is a review response or a new query.
    """
    # --- Handle pending review response ---
    if cl.user_session.get("pending_review"):
        await _handle_review_response(message)
        return

    # --- Handle new query ---
    query = message.content.strip()
    if not query:
        return

    # Check for review mode prefix
    enable_review = False
    if query.startswith("/review "):
        enable_review = True
        query = query[8:].strip()
        await cl.Message(
            content="🔒 **Human-in-the-loop enabled** — I'll pause for your approval before generating the report."
        ).send()

    # Initialize tracer
    tracer.reset(query)

    # Create progress message
    progress_msg = cl.Message(content="⏳ Starting research pipeline...")
    await progress_msg.send()

    try:
        if enable_review:
            await _run_with_review(query, progress_msg)
        else:
            await _run_full_pipeline(query, progress_msg)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        await cl.Message(content=f"❌ **Pipeline error:** {str(e)}").send()


# ============================================================
# Pipeline Runners
# ============================================================

async def _run_full_pipeline(query: str, progress_msg: cl.Message):
    """Run the full pipeline with live progress updates."""
    graph = build_graph(enable_human_review=False)
    state = create_initial_state(query)
    thread_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}

    progress_lines = ["**Pipeline Progress:**\n"]
    final_report = ""

    for step in graph.stream(state, config, stream_mode="updates"):
        for node_name, node_output in step.items():
            progress_lines.append(_format_step(node_output, node_name))
            progress_msg.content = "\n".join(progress_lines)
            await progress_msg.update()

            if "final_report" in node_output and node_output["final_report"]:
                final_report = node_output["final_report"]

    if final_report:
        await cl.Message(content=final_report).send()

    await _send_tiering_summary()


async def _run_with_review(query: str, progress_msg: cl.Message):
    """Run with human-in-the-loop — pause for approval before aggregation."""
    graph = build_graph(enable_human_review=True)
    state = create_initial_state(query)
    thread_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}

    # Store graph + config in session for resumption
    cl.user_session.set("review_graph", graph)
    cl.user_session.set("review_config", config)

    progress_lines = ["**Pipeline Progress (Review Mode):**\n"]

    # Phase 1: Run until interrupt (planner + specialists)
    for step in graph.stream(state, config, stream_mode="updates"):
        for node_name, node_output in step.items():
            progress_lines.append(_format_step(node_output, node_name))
            progress_msg.content = "\n".join(progress_lines)
            await progress_msg.update()

    # Get paused state and show review summary
    current_state = graph.get_state(config)
    review = get_review_summary(current_state.values)

    await cl.Message(
        content=(
            "## 🔒 Human Review Required\n\n"
            "The specialist agents have completed their work. "
            "Review the results below and reply with:\n"
            "- **approve** — to generate the final report\n"
            "- **reject** — to cancel\n"
            "- Or type **feedback** that will be included in the report\n\n"
            f"```\n{review}\n```"
        )
    ).send()

    # Flag that the next message should be treated as a review response
    cl.user_session.set("pending_review", True)


# ============================================================
# Review Handler
# ============================================================

async def _handle_review_response(message: cl.Message):
    """Process the user's approve/reject/feedback response."""
    cl.user_session.set("pending_review", False)

    response = message.content.strip().lower()
    graph = cl.user_session.get("review_graph")
    config = cl.user_session.get("review_config")

    if not graph or not config:
        await cl.Message(content="❌ No pending review found. Start a new query.").send()
        return

    if response in ("reject", "no", "cancel"):
        await cl.Message(content="❌ Pipeline rejected. No report generated.").send()
        return

    # Determine feedback
    feedback = "approved" if response in ("approve", "yes", "y", "") else message.content.strip()

    # Update state with approval
    graph.update_state(
        config,
        {
            "human_approved": True,
            "metadata": {"human_feedback": feedback},
        },
    )

    progress_msg = cl.Message(content="⏳ Resuming pipeline after approval...")
    await progress_msg.send()

    # Phase 2: Resume the graph (human_review → aggregator)
    final_report = ""
    for step in graph.stream(None, config, stream_mode="updates"):
        for node_name, node_output in step.items():
            if "final_report" in node_output and node_output["final_report"]:
                final_report = node_output["final_report"]

    if final_report:
        await cl.Message(content=final_report).send()
    else:
        await cl.Message(content="⚠️ Pipeline completed but no report was generated.").send()

    await _send_tiering_summary()


# ============================================================
# Helpers
# ============================================================

def _format_step(node_output: dict, node_name: str) -> str:
    """Format a graph step into a progress line."""
    current = node_output.get("current_step", node_name)
    icon = AGENT_ICONS.get(current, "⚙️")
    label = AGENT_LABELS.get(current, current)

    info = f"{icon} **{label}** — done"

    # Add latency from metadata
    metadata = node_output.get("metadata", {})
    latency_key = f"{current}_latency_ms"
    if latency_key in metadata:
        info += f" ({metadata[latency_key]:.0f}ms)"

    # Add result count
    if "results" in node_output and node_output["results"]:
        count = len(node_output["results"])
        sources = sum(len(r.get("sources", [])) for r in node_output["results"])
        info += f" — {count} results, {sources} sources"

    return f"- {info}"


async def _send_tiering_summary():
    """Send the model tiering summary as a final message."""
    tiering = tracer.summary_table()
    if tiering and tiering != "No model invocations traced.":
        await cl.Message(
            content=f"### ⚡ Model Tiering Summary\n\n{tiering}"
        ).send()
