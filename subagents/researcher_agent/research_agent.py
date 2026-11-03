import os
import json

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph

from subagents.researcher_agent.retrieval import search_node, download_paper, gemini_llm, rank_papers,GEMINI_MODEL
from subagents.researcher_agent.config_research import *
from subagents.researcher_agent.research_prompts import *

KNOWLEDGE_GRAPH_MODULE = "subagents.Graph_RAG.knowledge_graph"

load_dotenv()


base_llm = gemini_llm()
decision_making_llm = base_llm.with_structured_output(DecisionMakingOutput)
judge_llm = base_llm.with_structured_output(JudgeOutput)
 
DEFAULT_MAX_RESULTS = 30        # papers fetched from arXiv before ranking
DEFAULT_TARGET_PAPERS = 5       # papers returned to the user
DEFAULT_RELEVANCE_THRESHOLD = 6 # minimum score (0-10) to keep a paper
DEFAULT_USE_KNOWLEDGE_GRAPH = True  # build a knowledge graph from the selected papers and answer from it
KG_VECTOR_TOP_K = 5             # entry entities found by vector search
KG_GRAPH_HOPS = 2     

def _get_user_query(state: ReviewerState)-> str:
    """
    The user query can come from 'user_query' or from the last human message
    """

    if state.get("user_query"):
        return state["user_query"]

    for message in reversed(state.get("message", [])):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def router(state: ReviewerState):
    """
    router ditection the user query to the  appropriate branch of the workflow
    """
    if state.get("requires_research", True):
        return "search"
    else: 
        return "end"

def decision_making_node(state: ReviewerState):
    """
    Entry point for user query.
    Base on user query, the model can either respond directly or perform a full research,
    routing workflow to the planning
    """
    system_prompt = SystemMessage(content = DECISION_MAKING_PROMPT)
    user_query = _get_user_query(state)
    messages = state.get("messages") or [HumanMessage(content = user_query)]
    try:

        response : DecisionMakingOutput = decision_making_llm.invoke([system_prompt]+ messages) 
    except Exception as error :
        print(f"Decision making failed, running research: {error}")
        response = DecisionMakingOutput(requires_research = True) 

    output = {"requires_research": response.requires_research, "user_query": user_query}

    if not state.get("messages"):
        output["messages"] = messages
    if not response.requires_research:
        output["messages"] = output.get("messages", []) + [AIMessage(content=response.answer or "Hello! Tell me the research topic and I will find arXiv papers for you.")]
    return output


def rank_node(state: ReviewerState ):
    """
    Keep the pappers that math the user query, best first
    """

    papers = state.get("found_papers")

    try: 
        ranked = rank_papers(
                state["user_query"],
                papers,
                relevance_threshold=state.get("relevance_threshold", DEFAULT_RELEVANCE_THRESHOLD),
                target_papers=state.get("target_papers", DEFAULT_TARGET_PAPERS),
            )
        status ="ranked"
    except Exception as error:
        print(f"Ranking failed, using arXiv order: {error}")
        ranked = [RankedPaper(paper=paper, score=0, reason="")
                  for paper in papers[: state.get("target_papers", DEFAULT_TARGET_PAPERS)]]
        status = "unranked"

    return {
        "ranked_papers": ranked,
        "selected_ids": [item.paper.arxiv_id for item in ranked],
        "status": status,
    }


def graph_node(state: ReviewerState):
    """
    Build knowledge from the select paper and generate attribute answer (title + abtract on document per paper).
    then answers to user from it will hydrid search (vector + multi hop search) ans [N] citations.
    """

    ranked = state.get("ranked_papers")
    if not state.get("use_knowledge_graph", DEFAULT_USE_KNOWLEDGE_GRAPH) or not ranked:
        return {"kg_answer": "", "kg_citations": []}


    import importlib
    KnowledgeGraph = importlib.import_module(KNOWLEDGE_GRAPH_MODULE).KnowledgeGraph 
    kg = KnowledgeGraph(model_name = GEMINI_MODEL)
    
    
    try: 
        for item in ranked:
            paper = item.paper
            text = (
                    f"Title: {paper.title}\n"
                    f"Authors: {', '.join(paper.authors)}\n"
                    f"Published: {paper.published}\n"
                    f"Abstract: {paper.abstract}"
                )
            kg.add_document(text, doc_name = paper.arxiv_id)
    
        context = kg.hybrid_search(state["user_query"], vector_top_k = KG_GRAPH_HOPS, graph_hops = KG_GRAPH_HOPS)
        if not context:
            return {"kg_answer": "", "kg_citations": []}
        attributed = kg.generate_attributed_answer(state["user_query"], context)
    except Exception as error:
        # The paper list is still useful without the graph, so never fail the whole research here.
        print(f"Knowledge graph step failed, returning the paper list only: {error}")
        return {"kg_answer": "", "kg_citations": []}

    paper_by_id = {item.paper.arxiv_id: item.paper for item in ranked}

    kg_citations = []
    for citation in attributed.citations:
        paper = paper_by_id.get(citation.source_document)
        kg_citations.append({
            "ref": citation.claim.replace("Reference ", ""),
            "arxiv_id": citation.source_document,
            "title": paper.title if paper else citation.source_document,
            "url": paper.entry_url if paper else "",
            "sentence": citation.source_sentence,
            "graph_path": " ".join(str(step) for step in citation.graph_path),
        })
    return {"kg_answer": attributed.answer, "kg_citations": kg_citations}


def answer_node(state: ReviewerState):
    """
    Format the selected paper as the final answer
    """

    ranked = state.get("ranked_papers", [])
    queries = ", ".join(f"`{query}`" for query in state.get("search_queries", []))


    if not ranked:
        markdown = (
            f"I could not find arXiv papers that match **{state.get('user_query', '')}**.\n\n"
            f"Searched with: {queries}. Try more specific technical keywords."
        )

    else:
        lines = [f"## arXiv papers for: {state.get('user_query', '')}", f"_Searched with: {queries}_", ""]
        for number, item in enumerate(ranked, start=1):
            paper = item.paper
            authors = ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "")
            lines.append(f"### {number}. {paper.title}")
            lines.append(f"- **Authors:** {authors}")
            lines.append(f"- **Published:** {paper.published}")
            lines.append(f"- **arXiv:** [{paper.arxiv_id}]({paper.entry_url}) · [PDF]({paper.pdf_url})")
            if item.score:
                lines.append(f"- **Relevance:** {item.score}/10")
            if item.reason:
                lines.append(f"- **Why it fits:** {item.reason}")
            lines.append("")
 
        # NEW: overview generated from the knowledge graph, with each [N] linked to its paper.
        if state.get("kg_answer"):
            lines.append("## Overview from the knowledge graph")
            lines.append(state["kg_answer"])
            lines.append("")
            if state.get("kg_citations"):
                lines.append("**Sources**")
                for citation in state["kg_citations"]:
                    link = f"[{citation['title']}]({citation['url']})" if citation["url"] else citation["title"]
                    lines.append(f"- {citation['ref']} {link}: _{citation['sentence']}_ (path: {citation['graph_path']})")
                lines.append("")
        markdown = "\n".join(lines)
 
    return {"markdown": markdown, "output": markdown, "messages": [AIMessage(content=markdown)]}


def judge_node(state: ReviewerState):
    """
    Node to let the LLM judge  the quality if its final answer
    """
    num_feedback_requests = state.get("num_feedback_requests", 0)
    if num_feedback_requests >= 2:
        return {"is_good_answer": True}
 
    system_prompt = SystemMessage(content=JUDGE_PROMPT)
    response: JudgeOutput = judge_llm.invoke([system_prompt] + state["messages"])
    output = {
        "is_good_answer": response.is_good_answer,
        "num_feedback_requests": num_feedback_requests + 1
    }
    if response.feedback:
        output["messages"] = [AIMessage(content=response.feedback)]
    return output



"""
=================================================
Final answer router function
=================================================
"""
def final_answer_router(state: ReviewerState):
    """
    Router to end the workflow or impore the the answer
    """

    if state["is_good_answer"]:
        return "end"
    else:
        return "planning"



def subgraph_research():
    workflow = StateGraph(ReviewerState)

    workflow.add_node("decision_making", decision_making_node)
    workflow.add_node("search", search_node)
    workflow.add_node("rank", rank_node)
    # NEW: knowledge-graph step between ranking and the answer.
    workflow.add_node("graph", graph_node)
    workflow.add_node("answer", answer_node)
 
    workflow.add_edge(START, "decision_making")
    workflow.add_conditional_edges(
        "decision_making",
        router,
        {
            "search": "search",
            "end": END,
        }
    )
    workflow.add_edge("search", "rank")
    # workflow.add_edge("rank", "answer")
    # NEW: rank -> graph -> answer.
    workflow.add_edge("rank", "graph")
    workflow.add_edge("graph", "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()





# NEW: helpers for the supervisor (academic_agent / tutor_agent setup).
_research_app = None
 
# def run_research(user_query: str, target_papers: int = DEFAULT_TARGET_PAPERS,
#                  max_results: int = DEFAULT_MAX_RESULTS) -> str:
# NEW: use_knowledge_graph switch.
def run_research(user_query: str, target_papers: int = DEFAULT_TARGET_PAPERS,
                 max_results: int = DEFAULT_MAX_RESULTS,
                 use_knowledge_graph: bool = DEFAULT_USE_KNOWLEDGE_GRAPH) -> str:
    """Run the researcher subgraph once and return the markdown answer."""
    global _research_app
    if _research_app is None:
        _research_app = subgraph_research()
    result = _research_app.invoke({
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "target_papers": target_papers,
        "max_results": max_results,
        "use_knowledge_graph": use_knowledge_graph,
    })
    return result.get("markdown") or result["messages"][-1].content
 
 
@tool("researcher_agent", args_schema=SearchPapersInput)
def researcher_tool(query: str, max_papers: int = 5) -> str:
    """Find arXiv papers that match a research need and return them with links and why they fit."""
    return run_research(query, target_papers=max_papers)
 





"""
======================================================================
test
======================================================================
"""
# if __name__ == "__main__":
#     print(run_research("I need to find some paper newest from STGCN"))