"""
Multi-Agent Research System — CLI Entry Point

Usage:
    python main.py "What are the latest trends in AI startups?"
    python main.py --interactive
    python main.py --review "Validate this startup idea: AI tutoring for kids"
"""

import sys
import logging
import uuid

from core.graph import build_graph, create_initial_state, get_review_summary
from core.config import validate_config
from core.tracer import tracer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        Multi-Agent Research & Automation System          ║
║  Planner → Specialists → Aggregator Pipeline             ║
║  Powered by Gemini 2.0 Flash + Groq Llama 3.3 70B       ║
╚══════════════════════════════════════════════════════════╝
"""


def run_query(query: str, human_review: bool = False) -> str:
    """
    Execute the full research pipeline for a given query.

    Args:
        query: The research question.
        human_review: If True, pauses for human approval before aggregation.

    Returns:
        The final Markdown report.
    """
    tracer.reset(query)

    logger.info(f"{'='*60}")
    logger.info("Starting research pipeline")
    logger.info(f"Query: {query}")
    logger.info(f"Human review: {'enabled' if human_review else 'disabled'}")
    logger.info(f"{'='*60}")

    graph = build_graph(enable_human_review=human_review)
    state = create_initial_state(query)
    thread_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}

    if not human_review:
        # Simple mode — run the full pipeline
        logger.info("Invoking graph (full pipeline)...")
        final_state = graph.invoke(state, config)
    else:
        # Human-in-the-loop mode
        logger.info("Invoking graph (will pause for review)...")

        # Phase 1: Run until interrupt (planner + specialists)
        final_state = graph.invoke(state, config)

        # Show review summary
        print("\n" + "=" * 60)
        print("  HUMAN REVIEW — Specialist results collected")
        print("=" * 60)
        summary = get_review_summary(final_state)
        print(summary)

        # Ask for approval
        print("-" * 60)
        approval = input("Approve and generate final report? [Y/n/feedback]: ").strip()

        if approval.lower() in ("n", "no", "reject"):
            print("Pipeline rejected. No report generated.")
            return "Pipeline rejected by human reviewer."

        # Phase 2: Resume the graph to run human_review → aggregator
        # Update state to mark as approved
        graph.update_state(
            config,
            {
                "human_approved": True,
                "metadata": {
                    **final_state.get("metadata", {}),
                    "human_feedback": approval if approval.lower() not in ("y", "yes", "") else "approved",
                },
            },
        )

        logger.info("Resuming pipeline after human approval...")
        final_state = graph.invoke(None, config)

    # Extract report
    report = final_state.get("final_report", "No report generated.")

    # Print model tiering summary
    print("\n" + "=" * 60)
    print("  MODEL TIERING SUMMARY")
    print("=" * 60)
    print(tracer.summary_table())

    logger.info(f"{'='*60}")
    logger.info("Pipeline complete!")
    logger.info(f"{'='*60}")

    return report


def interactive_mode():
    """Run the pipeline interactively with optional human review."""
    print(BANNER)
    print("Commands:")
    print('  Just type a query to run the full pipeline')
    print('  Prefix with /review to enable human-in-the-loop')
    print('  Type "quit" to exit')
    print()

    while True:
        try:
            raw = input("🔍 Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw or raw.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Check for /review prefix
        human_review = False
        query = raw
        if raw.startswith("/review "):
            human_review = True
            query = raw[8:].strip()
            print("🔒 Human-in-the-loop enabled — will pause for your approval\n")

        report = run_query(query, human_review=human_review)
        print("\n" + report)


def main():
    # Check config
    warnings = validate_config()
    for w in warnings:
        logger.warning(w)

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    elif len(sys.argv) > 1 and sys.argv[1] == "--review":
        query = " ".join(sys.argv[2:])
        if not query:
            print("Usage: python main.py --review \"Your query here\"")
            sys.exit(1)
        report = run_query(query, human_review=True)
        print("\n" + report)
    elif len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        report = run_query(query)
        print("\n" + report)
    else:
        print("Usage:")
        print('  python main.py "Your research query here"')
        print('  python main.py --review "Query with human approval"')
        print("  python main.py --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
