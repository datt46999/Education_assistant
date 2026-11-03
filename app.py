"""Entrypoint for AWS Bedrock AgentCore Runtime.
 
Why the background loop: AgentCore runs every request in its own task/event loop.
The MCP stdio sessions (anyio task groups) must be opened, used and closed in ONE
long-lived task, otherwise you get
  "RuntimeError: Attempted to exit cancel scope in a different task than it was entered in".
So one owner task keeps the MCPCoordinator open for the life of the container, and
requests are handed to it through a queue.
 
Local test:  uv run python agentcore_app.py
             curl -X POST localhost:8080/invocations -H 'Content-Type: application/json' \
                  -d '{"prompt": "hello", "user_id": "dat"}'
"""
import asyncio
import concurrent.futures
import threading
 
from bedrock_agentcore.runtime import BedrockAgentCoreApp
 
from MCP.client.coordinator import MCPCoordinator
from memory import checkpointer, store, make_config
 
app = BedrockAgentCoreApp()
 
# ---- one event loop + one owner task for the MCP servers -------------------------
_loop = asyncio.new_event_loop()
_queue: "asyncio.Queue | None" = None
_ready = threading.Event()
 
 
async def _handle(coordinator, prompt, config, fut):
    try:
        fut.set_result(await coordinator.ask(prompt, config))
    except Exception as e:  # noqa: BLE001 - hand any error back to the request
        fut.set_exception(e)
 
 
async def _owner():
    global _queue
    _queue = asyncio.Queue()
    async with MCPCoordinator(checkpointer=checkpointer, store=store) as coordinator:
        _ready.set()
        while True:
            prompt, config, fut = await _queue.get()
            # each request in its own task, but all inside the task that owns the sessions
            asyncio.create_task(_handle(coordinator, prompt, config, fut))
 
 
def _run_loop():
    asyncio.set_event_loop(_loop)
    _loop.run_until_complete(_owner())
 
 
threading.Thread(target=_run_loop, daemon=True, name="mcp-owner").start()
 
 
async def ask_coordinator(prompt: str, config: dict) -> dict:
    await asyncio.to_thread(_ready.wait, 300)  # MCP servers + models start at container boot
    fut: concurrent.futures.Future = concurrent.futures.Future()
    _loop.call_soon_threadsafe(_queue.put_nowait, (prompt, config, fut))
    return await asyncio.wrap_future(fut)
 
 
# ---- AgentCore entrypoint ----------------------------------------------------------
@app.entrypoint
async def invoke(payload, context):
    actor_id = payload.get("user_id", "anonymous")
    session_id = getattr(context, "session_id", None) or payload.get(
        "session_id", "default-session-000000000000000000000"
    )
    result = await ask_coordinator(payload["prompt"], make_config(actor_id, session_id))
 
    messages = result["messages"]
    last_human = max(i for i, m in enumerate(messages) if getattr(m, "type", "") == "human")
    turn = messages[last_human:]
    tools_used = [m.name for m in turn if getattr(m, "type", "") == "tool"]
    tool_errors = [f"{m.name}: {str(m.content)[:1000]}" for m in turn
                   if getattr(m, "type", "") == "tool"
                   and (getattr(m, "status", "") == "error" or "Error" in str(m.content)[:200])]
    for err in tool_errors:  # also goes to CloudWatch
        print(f"TOOL ERROR {err}", flush=True)
    return {"result": messages[-1].content, "tools_used": tools_used, "tool_errors": tool_errors}
 
 
if __name__ == "__main__":
    app.run()
 