from typing import Any, Dict

from langchain_core.messages import HumanMessage
from mcp.server.fastmcp import FastMCP

from MCP.servers._common import quiet_stdout
from MCP.servers.student_data_server import DEFAULT_STUDENT_ID, load_json_data


mcp = FastMCP(
    "academic-agent",
    instructions=(
        "Academic task agent for one student: study plans and schedules (planner), study notes in the "
        "student's learning style (notewrite), and advice on workload, exams and study habits (advisor)."
    ),
)
_graph = None

def _get_graph():
    """
    Imported lazily: academic_main creates the OpenAI llm 
    """

    global _graph
    if _graph is None:
        with quiet_stdout():
            from subagents.Academic_Task_agent import create_agent_graph, llm
            _graph = create_agent_graph(llm)
    return _graph


def _build_state(request: str, student_id: str) -> Dict[str, Any]:
    data = load_json_data()
    profiles = data["profile"].get("profiles", [])
    profile = next((p for p in profiles if p.get("id") == student_id), profiles[0] if profiles else {})
    return {
        "messages": [HumanMessage(content=request)],
        "profile": profile,
        "calendar": {"events": data["calendar"].get("events", [])},
        "tasks": {"tasks": data["task"].get("tasks", [])},
        "results": {},
    }


def _to_text(output: Any) -> str:
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, dict):
        return "\n\n".join(t for t in (_to_text(v) for v in output.values()) if t)
    if isinstance(output, (list, tuple)):
        return "\n\n".join(t for t in (_to_text(v) for v in output) if t)
    return str(getattr(output, "content", "") or "").strip()

@mcp.tool()
async def ask_academic_agent(request: str, student_id: str = DEFAULT_STUDENT_ID) -> str:
    """
    Send a request to the academic agent. Its coordinator picks the planner (study plans, schedules,
    deadlines), notewrite (study notes) and/or advisor (exam prep, workload, grades) and returns their answers.
    """
    with quiet_stdout():
        final = await _get_graph().ainvoke(_build_state(request, student_id), {"recursion_limit": 30})
    results = final.get("results", {})
    chosen = results.get("coordinator_analysis", {}).get("required_agents", [])
    sections = [f"## {name}\n\n{_to_text(output)}" for name, output in results.get("agent_outputs", {}).items()]
    return f"Agents used: {', '.join(chosen) or 'none'}\n\n" + "\n\n".join(sections)


@mcp.prompt()
def weekly_review(student_id: str = DEFAULT_STUDENT_ID) -> str:
    """Ready-made request: plan the week, with advice on the weakest course."""
    return (
        f"Use ask_academic_agent for student {student_id}: plan my study week around my calendar and tasks, "
        "and tell me which course needs the most attention and why."
    )


if __name__ == "__main__":
    mcp.run()