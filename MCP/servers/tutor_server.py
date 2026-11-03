"""
Tutor agent MCP server
+ answer_question: runs the tutor graph in subagents/Tutor_agent/tutor_agent.py
"""



import inspect

import anyio
from mcp.server.fastmcp import FastMCP

from MCP.servers._common import quiet_stdout

mcp = FastMCP(
    "tutor-agent",
    instructions=(
        "Tutor for study questions. Math problems and English/writing/grammar questions are "
        "answered step by step, with a short explanation of the concept from Wikipedia."
    ),
)

_run_tutor = None


def _get_run_tutor():
    # imported lazily: config_tutor creates the Groq llm at import time
    global _run_tutor
    if _run_tutor is None:
        with quiet_stdout():
            from subagents.Tutor_agent import run_tutor
        _run_tutor = run_tutor
    return _run_tutor


@mcp.tool()
async def answer_question(question: str) -> str:
    """
    Answer a student's study question: math problems (step-by-step solution) and English questions
    (grammar, writing, vocabulary, literature). The answer starts with the main concept explained
    from Wikipedia and ends with the Wikipedia source link.
    """
    run_tutor = _get_run_tutor()
    if inspect.iscoroutinefunction(run_tutor):   # async run_tutor (graph.ainvoke)
        with quiet_stdout():
            return await run_tutor(question)

    def work() -> str:                           # sync run_tutor (graph.invoke)
        with quiet_stdout():
            return run_tutor(question)

    return await anyio.to_thread.run_sync(work)


if __name__ == "__main__":
    mcp.run()