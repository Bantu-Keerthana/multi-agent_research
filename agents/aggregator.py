"""
Aggregator Agent — synthesizes all specialist outputs into a final Markdown report.

Uses Gemini 2.0 Flash to generate a polished executive summary and
assemble findings from all specialist agents into a structured report.

This is the final node in the pipeline. In Milestone 3, an 
interrupt_before gate will pause here for human approval.
"""

import time
import logging
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage

from core.state import AgentState, AgentResult
from core.llm import invoke_llm, get_model_name

logger = logging.getLogger(__name__)

AGGREGATOR_SYSTEM_PROMPT = """You are a senior research analyst producing a final report.
Given outputs from multiple specialist agents (web search, data analysis, summarization),
write a polished executive summary that:

1. Opens with a 1-2 sentence answer to the original query
2. Highlights 3-5 key findings with supporting evidence
3. Notes any risks, uncertainties, or gaps in the research
4. Ends with actionable recommendations or next steps

Write in a clear, professional tone. Be concise — aim for 200-400 words.
Do NOT include headers or formatting — just flowing prose paragraphs.
This will be placed under an "Executive Summary" header in the report."""


def aggregator_node(state: AgentState) -> dict:
    """Synthesize all specialist outputs into a final Markdown report."""
    start = time.time()
    query = state["query"]
    results = state.get("results", [])
    metadata = state.get("metadata", {})

    logger.info(f"Aggregator synthesizing {len(results)} results")

    # Group results by agent type
    by_agent: dict[str, list[AgentResult]] = {}
    for r in results:
        by_agent.setdefault(r["agent"], []).append(r)

    # Build detailed findings section
    findings_parts = []
    for agent_name, agent_results in by_agent.items():
        findings_parts.append(f"### {agent_name.replace('_', ' ').title()}\n")
        for r in agent_results:
            findings_parts.append(r["content"])
            findings_parts.append("")

    # Collect all sources
    all_sources = []
    for r in results:
        all_sources.extend(r["sources"])
    unique_sources = list(dict.fromkeys(all_sources))

    sources_md = (
        "\n".join(f"- [{s}]({s})" for s in unique_sources)
        if unique_sources
        else "- No external sources cited"
    )

    # Generate executive summary with LLM
    model_used = get_model_name("aggregator")
    try:
        exec_summary = _generate_executive_summary(query, results)
    except Exception as e:
        logger.warning(f"LLM summary failed ({e}), using fallback")
        exec_summary = _fallback_summary(query, results)
        model_used = "fallback"

    # Build model usage table
    model_rows = []
    for r in results:
        model_rows.append(
            f"| {r['agent']} | {r['model_used']} | {r['latency_ms']:.0f}ms |"
        )

    total_latency = (time.time() - start) * 1000

    # Assemble the final report
    report = f"""# Research Report

**Query:** {query}
**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
**Agents Used:** {", ".join(sorted(by_agent.keys()))}

---

## Executive Summary

{exec_summary}

## Detailed Findings

{"".join(findings_parts)}

## Sources

{sources_md}

---

## Pipeline Metadata

| Agent | Model | Latency |
|-------|-------|---------|
{chr(10).join(model_rows)}

| Metric | Value |
|--------|-------|
| Total Agents | {len(by_agent)} |
| Total Subtasks | {len(results)} |
| Pipeline Latency | {sum(r['latency_ms'] for r in results) + total_latency:.0f}ms |
"""

    logger.info(f"Aggregator completed — report is {len(report)} chars")

    return {
        "final_report": report,
        "current_step": "aggregator",
        "metadata": {
            **metadata,
            "aggregator_latency_ms": round(total_latency, 1),
            "aggregator_model": model_used,
            "report_length_chars": len(report),
            "models_used": list(set(r["model_used"] for r in results)),
        },
    }


def _generate_executive_summary(query: str, results: list[AgentResult]) -> str:
    """Use Gemini to synthesize a polished executive summary."""
    context = "\n\n---\n\n".join(
        f"[{r['agent']}]\n{r['content']}" for r in results
    )

    if len(context) > 15000:
        context = context[:15000] + "\n\n[... truncated ...]"

    messages = [
        SystemMessage(content=AGGREGATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original Query: {query}\n\n"
                f"Specialist Agent Outputs:\n{context}\n\n"
                f"Write the executive summary."
            )
        ),
    ]

    return invoke_llm("aggregator", messages)


def _fallback_summary(query: str, results: list[AgentResult]) -> str:
    """Simple fallback summary when the LLM is unavailable."""
    if not results:
        return "No results were produced by the specialist agents."

    agents = set(r["agent"] for r in results)
    return (
        f"This report addresses the query: *\"{query}\"*\n\n"
        f"The research pipeline executed {len(results)} subtasks across "
        f"{len(agents)} specialist agents ({', '.join(sorted(agents))}). "
        f"Key findings are detailed in the sections below."
    )
