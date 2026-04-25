"""
Code Generation Agent — generates scripts, tools, and technical implementations.

Uses Gemini 2.0 Flash for code generation with structured output.
"""

import time
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from core.state import AgentState, AgentResult
from core.llm import invoke_llm, get_model_name

logger = logging.getLogger(__name__)

CODEGEN_SYSTEM_PROMPT = """You are an expert software engineer. Given a coding task description
and research context, generate clean, well-documented code.

Guidelines:
- Write production-quality Python code by default
- Include docstrings and type hints
- Add inline comments for complex logic
- If the task is conceptual, provide a code outline with pseudocode
- Include a brief explanation of the approach before the code
- If you need external libraries, note the pip install commands

Format your response as:
1. Brief explanation of approach (2-3 sentences)
2. Code block(s)
3. Usage example (if applicable)"""


def code_gen_node(state: AgentState) -> dict:
    """Execute all code_generation subtasks."""
    start = time.time()
    plan = state.get("plan", [])

    code_tasks = [t for t in plan if t["task_type"] == "code_generation"]
    if not code_tasks:
        return {"current_step": "code_gen"}

    logger.info(f"Code Gen processing {len(code_tasks)} tasks")

    prior_results = state.get("results", [])
    context = "\n\n".join(r["content"] for r in prior_results)

    model_used = get_model_name("code_generation")
    results: list[AgentResult] = []

    for task in code_tasks:
        task_start = time.time()

        try:
            content = _execute_codegen(task["description"], context)
        except Exception as e:
            logger.error(f"Code gen failed for task {task['id']}: {e}")
            content = f"Code generation failed: {str(e)}"

        latency = (time.time() - task_start) * 1000

        results.append(
            AgentResult(
                task_id=task["id"],
                agent="code_generation",
                content=content,
                sources=[],
                model_used=model_used,
                latency_ms=round(latency, 1),
            )
        )

    total_latency = (time.time() - start) * 1000
    logger.info(f"Code Gen completed in {total_latency:.0f}ms")

    return {
        "results": results,
        "current_step": "code_gen",
        "metadata": {
            **state.get("metadata", {}),
            "code_gen_latency_ms": round(total_latency, 1),
            "code_gen_model": model_used,
        },
    }


def _execute_codegen(description: str, context: str) -> str:
    """Use Gemini to generate code for the task."""
    max_context = 8000
    if len(context) > max_context:
        context = context[:max_context] + "\n\n[... truncated ...]"

    messages = [
        SystemMessage(content=CODEGEN_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Code Generation Task: {description}\n\n"
                f"Research Context:\n{context}\n\n"
                f"Generate the code solution."
            )
        ),
    ]

    return invoke_llm("code_generation", messages)
