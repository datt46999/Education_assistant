"""
=======================================================
Test for the academic agent, based on run.py
 
Uses the workflow in academic_main.py (create_agent_graph) with the OpenAI llm
created there, and the three json files in data/ (next to subagents/):
    data/profile.json, data/calendar.json, data/task.json
 
Run like run.py (asks for your request):
    python -m subagents.Academic_Task_agent.test
    python -m subagents.Academic_Task_agent.test --query "Plan my week and give me advice for my midterm"
 
Run as tests (-s shows the agent output):
    pytest subagents/Academic_Task_agent/test.py -v -s
 
Run all commands from the project root (the folder that holds subagents/ and data/).
=======================================================
"""

import os
import re
import sys
import asyncio
import argparse
import importlib
from pathlib import Path


import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel



PROJECT_ROOT =  Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



load_dotenv()
from subagents.Academic_Task_agent import create_agent_graph, llm, DataManager
DATA_DIR = PROJECT_ROOT / "data"
STUDENT_ID = "student_123"
GRAPH_CONFIG = {"recursion_limit": 10}

FILE_PATTERNS = {
    "profile": r"^profile.*\.json$",
    "calendar": r"^calendar.*\.json$",
    "task": r"^task.*\.json$",
}





"""
========================
Load agent
========================
"""

def load_json_files(path_file: Path = DATA_DIR) -> dict:
    """Find profile/calendar/task json in path_file and return their text """
    files = os.listdir(path_file)
    found_files = {
        file_type: next((os.path.join(path_file, f) for f in files if re.match(pattern, f, re.IGNORECASE)), None)
        for file_type, pattern in FILE_PATTERNS.items()
    }
    missing = [k for k, v in found_files.items() if v is None]
    if missing:
        raise FileNotFoundError(f"Missing required files in {path_file}: {missing}")
 
    json_contents = {}
    for file_type, filename in found_files.items():
        with open(filename, "r", encoding="utf-8") as f:
            json_contents[file_type] = f.read()
    return json_contents

def build_state(json_contents: dict, user_input: str)-> dict:
    """Load json into DataManager and build initial state"""
    dm = DataManager()
    dm.load_data(json_contents["profile"], json_contents["calendar"], json_contents["task"])

    return {
        "messages": [HumanMessage(content=user_input)],  # User request
        "profile": dm.get_student_profile(STUDENT_ID),   # Student info
        "calendar": {"events": dm.get_upcoming_events()},  # Schedule
        "tasks": {"tasks": dm.get_active_tasks()},         # Active tasks
        "results": {},                                     # Will store agent outputs
    }



def _merge(base: dict,  update: dict)-> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


async def run_academic_agent(user_input: str, json_contents: dict, on_node= None)-> dict:
    """
    run academic agent 
    """
    state = build_state(json_contents, user_input)
    graph = create_agent_graph(llm)

    results = {}
    async for step in graph.astream(state, GRAPH_CONFIG, stream_mode="updates"):
        for node, update in step.items():
            if on_node:
                on_node(node)
            results = _merge(results, (update or {}).get("results", {}))

    return results

def output_to_text(output) -> str:
    """Collect the strings in an agent output."""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, dict):
        return "\n\n".join(t for t in (output_to_text(v) for v in output.values()) if t)
    if isinstance(output, (list, tuple)):
        return "\n\n".join(t for t in (output_to_text(v) for v in output) if t)
    return str(getattr(output, "content", "") or "").strip()




"""
=======================================================
pytest
=======================================================
"""
 
needs_openai = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
 
REQUESTS = {
    "planner": "Create a study plan for this week so I finish the ML exercise set before it is due.",
    "notewrite": "Write study notes on supervised learning for my ML exercise.",
    "advisor": "How should I prepare for my Advanced Algorithms midterm while keeping up with Calculus?",
    "all": "Plan my study week, write notes on supervised learning, and advise me on my midterm preparation.",
}
 
 
@pytest.fixture(scope="module")
def json_contents():
    return load_json_files(DATA_DIR)
 
 
def test_json_files_found(json_contents):
    assert set(json_contents) == {"profile", "calendar", "task"}
 
 
def test_state_has_the_json_data(json_contents):
    state = build_state(json_contents, "hello")
    assert state["profile"], "DataManager returned no profile for student_123"
    if not state["calendar"]["events"] or not state["tasks"]["tasks"]:
        print("\nWARNING: DataManager returned no upcoming events or active tasks. "
              "If it filters by today's date, the 2024 sample data is dropped.")
 
 
def test_graph_compiles():
    nodes = set(create_agent_graph(llm).get_graph().nodes)
    for node in ["coordinator", "profile_analyzer", "calendar_analyzer", "tasks_analyzer", "plan_generate",
                 "notewriter_analyzer", "notewriter_generate", "advisor_analyzer", "advisor_generate", "execute"]:
        assert node in nodes, f"graph has no node {node}"
 
 

@pytest.mark.parametrize("target", list(REQUESTS))
def test_academic_agent(target, json_contents):
    visited = []
    results = asyncio.run(run_academic_agent(REQUESTS[target], json_contents, visited.append))
 
    analysis = results.get("coordinator_analysis", {})
    required = [a.lower() for a in analysis.get("required_agents", [])]
    outputs = results.get("agent_outputs", {})
 
    print(f"\n===== {target}: {REQUESTS[target]}")
    print(f"nodes: {' -> '.join(visited)}")
    print(f"coordinator chose: {analysis.get('required_agents')}")
    for name, output in outputs.items():
        print(f"\n--- {name} ---\n{output_to_text(output)}")
 
    assert required, "coordinator chose no agents"
    assert "profile_analysis" in results, "profile_analyzer did not run"
    for agent in required:
        assert agent in outputs, f"{agent} was required but is missing from agent_outputs {list(outputs)}"
        assert len(output_to_text(outputs[agent])) > 50, f"{agent} returned almost no text"
 
 
"""
=======================================================
Manual run, like run.py
=======================================================
"""
 
def main():
    parser = argparse.ArgumentParser(description="Test the academic agent (academic_main.py workflow)")
    parser.add_argument("--query", default=None, help="your request (asked interactively when left out)")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="folder with profile/calendar/task json")
    args = parser.parse_args()
 
    console = Console()
    console.print("\n[bold magenta]🎓 Academic agent test[/bold magenta]")
    json_contents = load_json_files(Path(args.data_dir))
 
    user_input = args.query
    if not user_input:
        console.print("[bold green]Please enter your academic request:[/bold green]")
        user_input = input().strip() or REQUESTS["all"]
 
    with console.status("[bold green]Processing...", spinner="dots") as status:
        results = asyncio.run(run_academic_agent(user_input, json_contents,
                                                 lambda node: status.update(f"[bold green]finished {node}...")))
 
    console.print("\n[bold cyan]Selected Agents:[/bold cyan]")
    for agent in results.get("coordinator_analysis", {}).get("required_agents", []):
        console.print(f"• {agent}")
    for agent, output in results.get("agent_outputs", {}).items():
        console.print(Panel(Markdown(output_to_text(output) or "_no text output_"),
                            title=f"{agent.upper()} Output", border_style="blue"))
    console.print("\n[bold green]✓[/bold green] [bold]Task completed![/bold]")
 
 
if __name__ == "__main__":
    main()
 