"""
=======================================================
MCP coordinator
+ starts the MCP servers listed in MCP/mcp_config.json
+ loads all their tools into one LangGraph ReAct agent (the "coordinator" in the architecture)
+ the LLM picks researcher / academic / tutor tools for each request

=======================================================
"""
import os
import sys
import json
import asyncio
import argparse
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "MCP" / "mcp_config.json"

load_dotenv(PROJECT_ROOT / ".env")

COORDINATOR_PROMPT = """You are the coordinator of an education assistant for one student.
Pick the right tools for each request:
- researcher tools: finding scientific papers on arXiv
- academic tools: study plans (planner), study notes (notewrite), academic advice (advisor)
- tutor tools: math problems and English/writing/grammar questions, explained with the concept from Wikipedia
- student data tools: reading or changing the student's profile, calendar and tasks
You may call several tools. Answer with the tool results, do not invent schedules or papers."""


def load_connections(server_names: Optional[List[str]] = None, config_path: Path = CONFIG_PATH) -> Dict[str, dict]:
    """Read mcp_config.json; run stdio servers with this Python, from the project root, with this environment."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if server_names:
        unknown = set(server_names) - set(config)
        if unknown:
            raise ValueError(f"Unknown servers {sorted(unknown)}, known: {sorted(config)}")
        config = {name: config[name] for name in server_names}

    for connection in config.values():
        if connection.get("transport") == "stdio":
            if connection.get("command") in ("python", "python3"):
                connection["command"] = sys.executable
            connection.setdefault("cwd", str(PROJECT_ROOT))
            connection.setdefault("env", dict(os.environ))
    return config


class MCPCoordinator:
    """
    Keeps one session open per server for the whole conversation, so heavy servers
    (the researcher loads an embedding model) start only once.

        async with MCPCoordinator() as coordinator:
            print(await coordinator.ask("Plan my week"))
    """

    def __init__(self, server_names: Optional[List[str]] = None, model: Optional[str] = None, llm=None,
                 checkpointer=None, store=None):
        self.checkpointer = checkpointer  # AgentCore short-term memory (chat history per session)
        self.store = store                # AgentCore long-term memory (facts per student)
        self.connections = load_connections(server_names)
        self.client = MultiServerMCPClient(self.connections)
        self.model = model or os.getenv("COORDINATOR_MODEL", "gpt-4o-mini")
        self.llm = llm
        self.tools = []
        self.agent = None
        self._stack = AsyncExitStack()

    async def __aenter__(self):
        for name in self.connections:
            session = await self._stack.enter_async_context(self.client.session(name))
            self.tools += await load_mcp_tools(session)
        if self.llm is None:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=self.model, temperature=0)
        self.agent = create_react_agent(self.llm, self.tools, prompt=COORDINATOR_PROMPT,
                                        checkpointer=self.checkpointer, store=self.store)
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    async def ask(self, request: str, config: Optional[dict] = None) -> dict:
        """config = {"configurable": {"thread_id": session_id, "actor_id": user_id}}.
        The checkpointer keeps the history, so only the new message is sent."""
        return await self.agent.ainvoke({"messages": [HumanMessage(content=request)]}, config)


async def chat(server_names: Optional[List[str]] = None):
    from rich.console import Console
    from rich.markdown import Markdown

    try:  # AgentCore Memory if MEMORY_ID is set, otherwise in-process memory
        from memory import checkpointer, store, make_config
    except Exception as e:
        from langgraph.checkpoint.memory import InMemorySaver
        print(f"AgentCore memory not available ({e}); using in-process memory")
        checkpointer, store = InMemorySaver(), None
        make_config = lambda actor_id, session_id: {"configurable": {"thread_id": session_id, "actor_id": actor_id}}
    config = make_config("dat", "local-chat-session-00000000000000001")

    console = Console()
    async with MCPCoordinator(server_names, checkpointer=checkpointer, store=store) as coordinator:
        console.print(f"[bold magenta]Education assistant (MCP)[/bold magenta]  tools: "
                      f"{', '.join(t.name for t in coordinator.tools)}")
        while True:
            try:
                request = console.input("\n[bold green]You:[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if request.lower() in {"exit", "quit", "q"}:
                break
            if not request:
                continue
            with console.status("Thinking...", spinner="dots"):
                result = await coordinator.ask(request, config)
            messages = result["messages"]
            last_human = max(i for i, m in enumerate(messages) if getattr(m, "type", "") == "human")
            used = [m.name for m in messages[last_human:] if getattr(m, "type", "") == "tool"]
            if used:
                console.print(f"[dim]tools used: {', '.join(used)}[/dim]")
            for m in messages[last_human:]:  # show real tool errors instead of the LLM's "there was an issue"
                if getattr(m, "type", "") == "tool" and (getattr(m, "status", "") == "error" or "Error" in str(m.content)[:200]):
                    console.print(f"[red]{m.name} error:[/red] {str(m.content)[:1500]}")
            console.print(Markdown(messages[-1].content))


def main():
    parser = argparse.ArgumentParser(description="Education assistant coordinator over MCP")
    parser.add_argument("--servers", nargs="*", help="subset of servers from mcp_config.json")
    args = parser.parse_args()
    asyncio.run(chat(args.servers))


if __name__ == "__main__":
    main()
