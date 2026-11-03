
import os

from typing import TypedDict, Annotated

from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "qwen/qwen3.8-27b"
llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0,
)
class TutorState(TypedDict):
    """State object for the tutor agent"""
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    concept: str
    concept_info: str
    source_url: str
    