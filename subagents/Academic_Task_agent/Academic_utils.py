from pydantic import BaseModel, Field
from typing import List, Dict, Annotated, Literal,  TypeVar, Any, TypedDict
from operator import add
from datetime import datetime, timedelta, timezone

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI



def dict_reducer(dict1: Dict[str, Any], dict2: Dict[str, Any])-> Dict[str, Any]:
    """
    Merge two dictionaries recusively
    exmaple
    dict1 = {"a": {"x": 1}, "b": 2}
    dict2 = {"a": {"y": 2}, "c": 3}
    result = {"a": {"x": 1, "y": 2}, "b": 2, "c": 3}

    """
    merge = dict1.copy()
    for key, val  in dict2.items():
        if key in merge and isinstance(merge[key], dict) and isinstance(val, dict):
            merge[key] = dict_reducer(merge[key], val)
        else:
            merge[key] = val
    return merge

class AcademicState(TypedDict):
    """
    Master state container acadamic system
    """
    messages: Annotated[list[BaseMessage],add]
    profile: Annotated[Dict, dict_reducer]
    calendar: Annotated[Dict, dict_reducer]
    tasks: Annotated[Dict, dict_reducer]
    results: Annotated[Dict[str, Any], dict_reducer]


ROLE_MAP = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
class OpenAILLM:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = ChatOpenAI(model=model, api_key=api_key)

    async def agenerate(self, messages: list[dict], temperature: float | None = None) -> str:
        lc_messages = [ROLE_MAP[m["role"]](content=m["content"]) for m in messages]
        client = self.client.bind(temperature=temperature) if temperature is not None else self.client
        return (await client.ainvoke(lc_messages)).content



class ReActAgent:
    """
    Base for class ReAct base agent implementing reason and action capability

    Feature:
        Tools managerment for specific actions
        Few-shot learning example 
        structure through process
        action execution framework
    """


    def __init__(self, llm):
        """
        Initialize ReAct with LM and available tools

        """

        self.llm = llm
        self.few_show_examples = []
        self.tools = {
            "search_calendar": self.search_calendar,
            "analysis_tasks": self.analysis_tasks,
            "check_learning_style": self.check_learning_style,
            "check_performance": self.check_performance
        }

    async def search_calendar(self, state: AcademicState) -> List[Dict]:
        """
            Search upcomming calendar event

            Args:
                state: curent academic state
            return:
                list[dict]: list upcomming event
        """
        events = state['calendar'].get('events', [])
        now = datetime.now(timezone.utc)


        return [e for e in events if datetime.fromisocalendar(e['start']['dateTime'])>now]


    async def analysis_tasks(self, state: AcademicState) ->List[dict]:
        """
        Analysis academic tasks
        args:
            state: curent academic state
        return:
            list[dict] list of tasks was analysis
        """

        return state['tasks'].get('tasks', [])

    async def check_learning_style(self, state: AcademicState) -> AcademicState:
        """
        Retrieval learning style of students

        Args:
            state: current acdemic state

        return:
            update state academic with learning style analysis
        """
        profile = state["profile"]

        learning_data = {
            "style" : profile.get("learning_preferences",{}).get("learning_style", {}),
            "patterns" : profile.get("learning_preferences",{}).get("learning_patterns", {})
        }

        if "results"  not in state:
            state["results"] = {}

        state['results']['learning_style'] = learning_data
        return state

    async def check_performance(self, state: AcademicState)-> AcademicState:
        """
        Check current acdemic performance across cource

        Args:
            state: Curent academic student
        Return:
            Update state academic with performance analysis
        
        """
        profile = state["profile"]

        course = profile.get("academic_info", {}).get("current_courses", [])

        if "results" not in state:
            state["results"] = {}

        state['results']["performance_analysis"] = course

        return state