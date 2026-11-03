"""
=======================================================
Student data MCP server
+ exposes the profile, calendar and tasks json as MCP resources and tools
+ any agent (academic, tutor, an IDE, Claude Desktop) can read the same data
+ later swap this server for real Google Calendar / Notion / Trello MCP servers
  (the tools the profile lists) without touching the agents

Run:  python -m MCP.servers.student_data_server
=======================================================
"""
import os
import re
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from mcp.server.fastmcp import FastMCP

from MCP.servers._common import PROJECT_ROOT

DEFAULT_STUDENT_ID = "student_123"

# same file patterns as run.py
FILE_PATTERNS = {
    "profile": r"^profile.*\.json$",
    "calendar": r"^calendar.*\.json$",
    "task": r"^task.*\.json$",
}


def data_dir() -> Path:
    """ACADEMIC_DATA_DIR, else data/ next to subagents/ (what run.py uses)."""
    return Path(os.getenv("ACADEMIC_DATA_DIR", PROJECT_ROOT / "data"))


def find_data_files() -> Dict[str, Path]:
    folder = data_dir()
    files = sorted(os.listdir(folder))
    found = {k: next((folder / f for f in files if re.match(p, f, re.IGNORECASE)), None)
             for k, p in FILE_PATTERNS.items()}
    missing = [k for k, v in found.items() if v is None]
    if missing:
        raise FileNotFoundError(f"Missing {missing} json in {folder}")
    return found


def load_json_data() -> Dict[str, Any]:
    """{"profile": {...}, "calendar": {...}, "task": {...}}"""
    return {k: json.loads(path.read_text(encoding="utf-8")) for k, path in find_data_files().items()}


def get_profile(data: Dict[str, Any], student_id: str) -> Dict[str, Any]:
    profiles = data["profile"].get("profiles", [])
    return next((p for p in profiles if p.get("id") == student_id), profiles[0] if profiles else {})


mcp = FastMCP(
    "student-data",
    instructions="Read the student's profile, calendar events and tasks, and add or update tasks.",
)


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _event_start(event: Dict[str, Any]) -> Optional[datetime]:
    start = event.get("start", {})
    return _parse_time(start.get("dateTime") or start.get("date"))


def _write_tasks(tasks_json: Dict[str, Any]) -> None:
    path = find_data_files()["task"]
    path.write_text(json.dumps(tasks_json, indent=2, ensure_ascii=False), encoding="utf-8")


"""
=======================================================
Resources (read-only context the client can attach)
=======================================================
"""

@mcp.resource("student://profile/{student_id}", mime_type="application/json")
def profile_resource(student_id: str) -> str:
    """Full profile of one student."""
    return json.dumps(get_profile(load_json_data(), student_id), indent=2)


@mcp.resource("student://calendar", mime_type="application/json")
def calendar_resource() -> str:
    """All calendar events."""
    return json.dumps(load_json_data()["calendar"], indent=2)


@mcp.resource("student://tasks", mime_type="application/json")
def tasks_resource() -> str:
    """All tasks."""
    return json.dumps(load_json_data()["task"], indent=2)


"""
=======================================================
Tools
=======================================================
"""

@mcp.tool()
def get_student_profile(student_id: str = DEFAULT_STUDENT_ID) -> Dict[str, Any]:
    """Get a student's profile: courses, exams, grades, goals and learning preferences."""
    return get_profile(load_json_data(), student_id)


@mcp.tool()
def list_calendar_events(start: Optional[str] = None, end: Optional[str] = None,
                         subject: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List calendar events, optionally between two ISO datetimes (e.g. 2024-03-15T00:00:00Z)
    and/or for one subject (e.g. "Machine Learning").
    """
    start_dt, end_dt = _parse_time(start), _parse_time(end)
    events = []
    for event in load_json_data()["calendar"].get("events", []):
        event_start = _event_start(event)
        if start_dt and event_start and event_start < start_dt:
            continue
        if end_dt and event_start and event_start > end_dt:
            continue
        if subject and subject.lower() not in str(event.get("metadata", {}).get("subject", "")).lower():
            continue
        events.append(event)
    return events


@mcp.tool()
def list_tasks(status: Optional[Literal["needsAction", "completed"]] = None) -> List[Dict[str, Any]]:
    """List the student's tasks, optionally only those with a given status."""
    tasks = load_json_data()["task"].get("tasks", [])
    return [t for t in tasks if status is None or t.get("status") == status]


@mcp.tool()
def add_task(title: str, due: str, notes: str = "",
             priority: Literal["low", "medium", "high"] = "medium",
             estimated_duration: int = 60) -> Dict[str, Any]:
    """Add a task. `due` is an ISO datetime, `estimated_duration` is in minutes."""
    tasks_json = load_json_data()["task"]
    task = {
        "id": f"task_{uuid.uuid4().hex[:6]}",
        "title": title,
        "notes": notes,
        "due": due,
        "status": "needsAction",
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "priority": priority,
        "estimated_duration": estimated_duration,
    }
    tasks_json.setdefault("tasks", []).append(task)
    _write_tasks(tasks_json)
    return task


@mcp.tool()
def update_task_status(task_id: str, status: Literal["needsAction", "completed"]) -> Dict[str, Any]:
    """Mark a task as completed or as needing action again."""
    tasks_json = load_json_data()["task"]
    for task in tasks_json.get("tasks", []):
        if task.get("id") == task_id:
            task["status"] = status
            task["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _write_tasks(tasks_json)
            return task
    raise ValueError(f"No task with id {task_id}")


if __name__ == "__main__":
    mcp.run()
