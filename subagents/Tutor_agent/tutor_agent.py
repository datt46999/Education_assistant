import sys
import requests

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition

from subagents.Tutor_agent.toolagents import math, english
from subagents.Tutor_agent.config_tutor import llm, TutorState
from subagents.Tutor_agent.tutor_prompt import CONCEPT_EXTRACTION_PROMPT,  TUTOR_SYSTEM_PROMPT
TOOLS = [math, english]

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "education-assistant-tutor/0.1"}  # Wikipedia blocks requests without a UA
WIKI_MAX_CHARS = 1500


"""
# ---------------------------------------------------------------
# Wikipedia helper
# ---------------------------------------------------------------
"""
def get_concept_from_wikipedia(concept: str, max_chars: int = WIKI_MAX_CHARS) -> tuple[str, str]:
    """Search Wikipedia for `concept` and return (summary, url). Returns ("", "") if nothing found."""
    if not concept:
        return "", ""
    try:
        search = requests.get(WIKI_API, headers=WIKI_HEADERS, timeout=10, params={
            "action": "query", "list": "search", "srsearch": concept,
            "srlimit": 1, "format": "json",
        }).json()
        hits = search.get("query", {}).get("search", [])
        if not hits:
            return "", ""
        title = hits[0]["title"]
 
        page = requests.get(WIKI_API, headers=WIKI_HEADERS, timeout=10, params={
            "action": "query", "prop": "extracts|info", "exintro": 1, "explaintext": 1,
            "inprop": "url", "redirects": 1, "titles": title, "format": "json",
        }).json()
        info = next(iter(page["query"]["pages"].values()))
        summary = info.get("extract", "").strip()
        url = info.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
        return summary[:max_chars], url
    except Exception as e:  # network error, bad JSON... the tutor still answers without it
        print(f"[tutor] Wikipedia lookup failed for '{concept}': {e}", file=sys.stderr)
        return "", ""

"""
# ---------------------------------------------------------------
# Node
# ---------------------------------------------------------------
"""
async def extract_concept(state: TutorState)-> dict:
    question = state["question"]
    response = await llm.ainvoke([
        SystemMessage(content= CONCEPT_EXTRACTION_PROMPT), 
        HumanMessage(content = question)
    ])

    concept = response.content.strip().strip('"').splitlines()[0] if response.content else ""

    if concept.upper() == 'NONE':
        concept = ""
    return {'concept': concept}


def fetch_wikipedia(state: TutorState)-> dict:
    summary, url = get_concept_from_wikipedia(state.get("concept", ""))
    return {'concept_info': summary, "sourse_ulr": url}


llm_with_tools = llm.bind_tools(TOOLS)

async def tutor(state: TutorState)-> dict:
    system = SystemMessage(content = TUTOR_SYSTEM_PROMPT.format(
        concept=state.get("concept") or "(none)",
        concept_info=state.get("concept_info") or "(empty)",
        source_url=state.get("source_url") or "(none)",
    ))

    response = await llm.ainvoke([system]+ state['messages'])
    return {"messages": [response]}



"""
# ---------------------------------------------------------------
# graph
# ---------------------------------------------------------------
"""

def create_tutor_graph():
    graph = StateGraph(TutorState)
    graph.add_node("extract_concept", extract_concept)
    graph.add_node("fetch_wikipedia", fetch_wikipedia)
    graph.add_node("tutor", tutor)
    graph.add_node("tools", ToolNode(TOOLS))
 
    graph.add_edge(START, "extract_concept")
    graph.add_edge("extract_concept", "fetch_wikipedia")
    graph.add_edge("fetch_wikipedia", "tutor")

    graph.add_conditional_edges("tutor", 
                                tools_condition, 
                                {"tools": "tools", END: END})
    graph.add_edge("tools", "tutor")
    return graph.compile()



tutor_graph = create_tutor_graph()
 
async def run_tutor(question: str) -> str:
    result = await tutor_graph.ainvoke(
        {"question": question, "messages": [HumanMessage(content=question)]},
        {"recursion_limit": 12},
    )
    return result["messages"][-1].content