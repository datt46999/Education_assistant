"""
Shared helpers for the MCP servers.
"""
import sys
import contextlib
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

import logging
for name in ("httpx", "google_genai", "google_genai.models", "sentence_transformers"):
    logging.getLogger(name).setLevel(logging.WARNING)
@contextlib.contextmanager
def quiet_stdout():
    """
    A stdio MCP server talks JSON-RPC over stdout, so any print() from the agents
    (e.g. "Procesing: ..." in knowledge_graph.py) would corrupt the stream.
    Send those prints to stderr while a tool runs.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield
