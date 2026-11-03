"""
Researcher MCP server

+ search_arxiv_papers: arxiv search -> ranking -> knowledge_graph answer with citations
"""
import anyio
from mcp.server.fastmcp import FastMCP
from MCP.servers._common import quiet_stdout


mcp = FastMCP(
    "researcher-agent",
    instructions = "Find arxiv papers for researcher need and summaries them from knowledge graph with citations",
)


_run_research = None


def _get_researcher():
    """Import lazy:
        researcher_agent  load Gemini client and the embedding 
    """

    global _run_research

    if _run_research is None:
        from subagents.researcher_agent import run_research
        _run_research = run_research
    return _run_research


# @mcp.tool()
# async def search_arxiv_papers(query: str, max_papers : int = 5, use_knowledge_graph: bool = True)-> str:
#     """
#     Find arxiv papers matching researcher need. 
#     Returns markdown with titles, authors,links, 
#     why each paper fits, and (if use_knowledge_graph) an overview with [N] citations.
#     """
#     run_research = _get_researcher()

#     def work()-> str:
#         with quiet_stdout():
#             return run_research(query, target_papers= max_papers, use_knowledge_graph=use_knowledge_graph)

#         return await anyio.to_thread.run_sync(work)


@mcp.tool()
async def search_arxiv_papers(query: str, max_papers: int = 5, use_knowledge_graph: bool = True) -> str:
    """
    Find arXiv papers matching a research need. Returns markdown with titles, authors,
    links, why each paper fits, and (if use_knowledge_graph) an overview with [N] citations.
    """
    run_research = _get_researcher()

    def work() -> str:
        with quiet_stdout():
            return run_research(query, target_papers=max_papers, use_knowledge_graph=use_knowledge_graph)

    # run_research is synchronous and slow, keep the server responsive
    return await anyio.to_thread.run_sync(work)

if __name__ == "__main__":
    mcp.run()