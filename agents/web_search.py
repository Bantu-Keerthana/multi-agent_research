"""
Web Search Agent — finds and retrieves information from the web.

Search pipeline:
  1. DuckDuckGo search (free, no API key) → get top results
  2. Wikipedia API (free fallback) → supplementary context
  3. Gemini synthesis → extract and structure relevant findings

Optional: Firecrawl (500 pages/month free) for full page content extraction.
"""

import time
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from core.state import AgentState, AgentResult
from core.llm import invoke_llm, get_model_name

logger = logging.getLogger(__name__)

SEARCH_SYNTHESIS_PROMPT = """You are a research analyst. Given search results for a specific query,
extract and synthesize the most relevant findings.

Rules:
- Focus on facts, data points, and concrete information
- Include source attribution when possible
- Organize findings with clear structure
- If results are thin, note what information is missing
- Be concise but thorough — aim for 200-400 words
- Do NOT make up information not present in the results"""


def web_search_node(state: AgentState) -> dict:
    """Execute all web_search subtasks from the plan."""
    start = time.time()
    query = state["query"]
    plan = state.get("plan", [])

    search_tasks = [t for t in plan if t["task_type"] == "web_search"]
    if not search_tasks:
        return {"current_step": "web_search"}

    logger.info(f"Web Search processing {len(search_tasks)} tasks")
    model_used = get_model_name("web_search")
    results: list[AgentResult] = []

    for task in search_tasks:
        task_start = time.time()

        try:
            content, sources = _execute_search(task["description"])
        except Exception as e:
            logger.error(f"Search failed for task {task['id']}: {e}")
            content = f"Search failed: {str(e)}"
            sources = []

        latency = (time.time() - task_start) * 1000

        results.append(
            AgentResult(
                task_id=task["id"],
                agent="web_search",
                content=content,
                sources=sources,
                model_used=model_used,
                latency_ms=round(latency, 1),
            )
        )

    total_latency = (time.time() - start) * 1000
    logger.info(
        f"Web Search completed {len(results)} tasks in {total_latency:.0f}ms, "
        f"found {sum(len(r['sources']) for r in results)} sources"
    )

    return {
        "results": results,
        "current_step": "web_search",
        "metadata": {
            **state.get("metadata", {}),
            "web_search_latency_ms": round(total_latency, 1),
            "web_search_sources_count": sum(len(r["sources"]) for r in results),
        },
    }


def _execute_search(description: str) -> tuple[str, list[str]]:
    """
    Run the search pipeline:
      1. DuckDuckGo for web results
      2. Wikipedia as supplementary source
      3. Gemini to synthesize findings
    """
    raw_results = []
    sources = []

    # --- DuckDuckGo ---
    try:
        ddg_content, ddg_sources = _duckduckgo_search(description)
        raw_results.append(("DuckDuckGo", ddg_content))
        sources.extend(ddg_sources)
        logger.info(f"DuckDuckGo returned {len(ddg_sources)} results")
    except Exception as e:
        logger.warning(f"DuckDuckGo failed: {e}")

    # --- Wikipedia fallback ---
    try:
        wiki_content, wiki_sources = _wikipedia_search(description)
        if wiki_content:
            raw_results.append(("Wikipedia", wiki_content))
            sources.extend(wiki_sources)
            logger.info(f"Wikipedia returned content")
    except Exception as e:
        logger.debug(f"Wikipedia search skipped: {e}")

    if not raw_results:
        return "No search results found for this query.", []

    # --- Synthesize with LLM ---
    try:
        combined = "\n\n---\n\n".join(
            f"[Source: {name}]\n{content}" for name, content in raw_results
        )
        synthesis = _synthesize_results(description, combined)
        return synthesis, sources
    except Exception as e:
        logger.warning(f"Synthesis failed ({e}), returning raw results")
        combined = "\n\n".join(content for _, content in raw_results)
        return combined, sources


def _duckduckgo_search(query: str, max_results: int = 5) -> tuple[str, list[str]]:
    """Search using DuckDuckGo (free, no API key needed)."""
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        raise ValueError("No results from DuckDuckGo")

    content_parts = []
    sources = []
    for r in results:
        content_parts.append(f"**{r['title']}**\n{r['body']}")
        sources.append(r["href"])

    return "\n\n".join(content_parts), sources


def _wikipedia_search(query: str) -> tuple[str, list[str]]:
    """Search Wikipedia for supplementary context (free, no key)."""
    import wikipediaapi

    wiki = wikipediaapi.Wikipedia("MultiAgentResearch/1.0", "en")

    # Extract key terms from the query for better matching
    # Remove common words and try the core subject
    stop_words = {"search", "find", "about", "for", "the", "web", "current",
                  "information", "latest", "recent", "what", "how", "why"}
    terms = [w for w in query.lower().split() if w not in stop_words and len(w) > 2]
    search_term = " ".join(terms[:4])  # first 4 meaningful words

    page = wiki.page(search_term)
    if not page.exists():
        # Try with fewer terms
        if len(terms) >= 2:
            page = wiki.page(terms[0])
        if not page.exists():
            return "", []

    summary = page.summary[:2000]
    return summary, [page.fullurl]


def _synthesize_results(query: str, raw_results: str) -> str:
    """Use the LLM to synthesize raw search results into structured findings."""
    messages = [
        SystemMessage(content=SEARCH_SYNTHESIS_PROMPT),
        HumanMessage(
            content=(
                f"Query: {query}\n\n"
                f"Search Results:\n{raw_results}\n\n"
                f"Synthesize the most relevant findings."
            )
        ),
    ]

    return invoke_llm("web_search", messages)
