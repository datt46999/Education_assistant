import os
import time
 
import arxiv
import httpx
import pymupdf
import threading
 
 
from functools import lru_cache
from typing import TypeVar, List, Dict
from pydantic import BaseModel
 
from langchain_google_genai import ChatGoogleGenerativeAI
 
from subagents.researcher_agent.config_research import *

from subagents.researcher_agent.research_prompts import SEARCH_PLAN_PROMPT, RANKING_PROMPT
 
from dotenv import load_dotenv
 
load_dotenv()


MAX_PDF_BYTES = 50 * 1024 * 1024
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

T = TypeVar('T', bound = BaseModel)
_usage: list[tuple[int, int]] = []
_usage_lock = threading.Lock()


def record_tokens(usage_metadata: dict | None)-> None:
    """Store of input and output token counts respord for one call"""
 
 
    if not usage_metadata:
        return
    with _usage_lock:
        _usage.append(
            (
                int(usage_metadata.get("input_tokens", 0)),
                int(usage_metadata.get("output_tokens", 0)),
            )
        )


@lru_cache(maxsize=1)
def gemini_llm()-> ChatGoogleGenerativeAI:
    """
    Create a langChain Gemini chat model from configure API key
 
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
 
 
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key, temperature=0.0, max_retries=3)


def generate_structure(prompt: str, result_type: type[T] )-> T:
    """Send a prompt data to Gemini and validate the structure data.
 
    `include_raw` is what makes the token reachable: without it langChain returns the parsed object alone and discards `usage_metadata`.
    It also wrap the parser in a fallback, so a response that violates  the scheme is returned  under 'parsing_error' instead of being raised.
    Callers rely on that  exception to record a typed failured describing what the model got wrong, so it is re-raised here.
    """
    result = gemini_llm().with_structured_output(result_type, include_raw = True).invoke(prompt)
 
    if result['parsing_error'] is not None:
        raise result['parsing_error']
 
    record_tokens(getattr(result["raw"], "usage_metadata", None))
    if result["parsed"] is None:
        raise ValueError(f"Gemini returned no structured output for {result_type.__name__}.")
    return result_type.model_validate(result["parsed"], from_attributes=True)


def make_search_plan(user_query: str)-> SearchPlan:
    """Cover user query into arXiv search queries"""
 

    prompt = SEARCH_PLAN_PROMPT.format(user_query=user_query)
    return generate_structure(prompt, SearchPlan)



def search_arxiv(arxiv_query: str, max_results: int)-> List[PaperMetadata]:
    """Run one arxiv query and return  normalized paper metadata """
 

    client = arxiv.Client(page_size = max_results, delay_seconds = 3.0, num_retries = 3)
    search = arxiv.Search(
        query=arxiv_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
 
    pappers: List[PaperMetadata] = []
    seen_ids: set[str] = set()
 
    for result in client.results(search):
        arxiv_id = result.entry_id.rsplit("/", 1)[-1]
        if arxiv_id in seen_ids:
            continue
 
        seen_ids.add(arxiv_id)
        pappers.append(
            PaperMetadata(
                arxiv_id=arxiv_id,
                title=" ".join(result.title.split()),
                authors=[
                    getattr(author, "name", str(author)) for author in result.authors
                ],
                abstract=" ".join(result.summary.split()),
                published=result.published.date().isoformat(),
                pdf_url=result.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                entry_url=result.entry_id,
            )
        )
 
    return pappers


def download_paper(paper: PaperMetadata) -> bytes:
    """Fetch one paper PDF, rejecting responses that are not usable PDFs."""
 
    response = httpx.get(paper.pdf_url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    content = response.content
 
    if len(content) > MAX_PDF_BYTES:
        raise PaperUnusableError(
            f"PDF for {paper.arxiv_id} is {len(content)} bytes, "
            f"over the {MAX_PDF_BYTES} byte limit."
        )
    if not content.startswith(b"%PDF"):
        raise PaperUnusableError(f"Response for {paper.arxiv_id} is not a PDF.")
 
    return content



def search_node(state: ReviewerState):
    """Generate arXiv quiries, search arXiv and deduplicate papers"""

    user_query = state.get("user_query") or state["messages"][-1].content
    max_results = state.get("max_results", 30)
    try:
        search_plan = make_search_plan(user_query)
        arxiv_queries = [query.strip() for query in  search_plan.queries if query.strip()]
    except Exception as error:
        print(f"Search planning failed, using the raw query: {error}")
        arxiv_queries = []
 
    if not arxiv_queries:
        arxiv_queries = [user_query]
 
    found_papers : list[PaperMetadata]= []
    seen_ids :set[str] = set()
 
    for index, arxiv_query in enumerate(arxiv_queries):

        if len(found_papers) >= max_results:
            break
 
        if index >0:

            time.sleep(3.0)
 
        """
        The budget is shared across the planned queries, to the total have to cover all them:
        at 10 results and 3 queries each query saw about four hits, and reviews come out short.
        Measured at ten hits per query, Measured at ten hits per query, relevant papers
         went from 3 to 8 with on-topic density flat, because arXiv's ranking is level
        over the first ten. See `evals/results/search_depth.json`
        """

        remaining = max_results - len(found_papers)
        remaining_queries = len(arxiv_queries) - index
 
        query_limit = max(1, (remaining + remaining_queries - 1) // remaining_queries)
 

        try:
            results = search_arxiv(arxiv_query, query_limit)
        except Exception as error:
            print(f"arXiv search failed for '{arxiv_query}': {error}")
            continue
        for paper in results:
            if paper.arxiv_id in seen_ids:
                continue
            seen_ids.add(paper.arxiv_id)
            found_papers.append(paper)
            if len(found_papers) >= max_results:
                break
    return {
        "user_query": user_query,
        "search_queries": arxiv_queries,
        "found_papers": found_papers,
    }


def rank_papers(user_query: str, papers: List[PaperMetadata], relevance_threshold : int = 6,
                target_papers = 5)-> List[RankedPaper]:
    """
    Ask Genini to store each paper from its title and abtract and return the best one
    """
    if not papers:
        return []

    papers_text = "\n\n".join(
        f"[{paper.arxiv_id}] {paper.title} ({paper.published})\nAbstract: {paper.abstract[:1200]}"
        for paper in papers
    )

    prompt = RANKING_PROMPT.format(user_query = user_query, papers = papers_text)
    ranking = generate_structure(prompt, RankingOutput)

    by_id = {paper.arxiv_id: paper for paper in papers}
    ranked: list[RankedPaper] = []
    for item in ranking.rankings:
        paper = by_id.get(item.arxiv_id.strip())
        if paper is None or item.score < relevance_threshold:
            continue
        ranked.append(RankedPaper(paper=paper, score=item.score, reason=item.reason))
        by_id.pop(paper.arxiv_id)  # ignore duplicated rankings
 
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:target_papers]


def parse_paper(paper: PaperMetadata, content: bytes) -> ParsedPaper:
    """Extract text from a downloaded PDF one page at a time."""
 
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            pages = [
                ParsedPage(page_number=number, text=page.get_text())
                for number, page in enumerate(document, start=1)
            ]
            page_count = document.page_count
    except PaperUnusableError:
        raise
    except Exception as error:
        raise PaperUnusableError(
            f"Could not parse PDF for {paper.arxiv_id}: {error}"
        ) from error
 
    if not page_count or not any(page.text.strip() for page in pages):
        raise PaperUnusableError(f"No extractable text in {paper.arxiv_id}.")
 
    return ParsedPaper(
        arxiv_id=paper.arxiv_id,
        pages=pages,
        page_count=page_count,
    )
 
 
def fetch_parsed_paper(paper: PaperMetadata) -> ParsedPaper:
    """Download and parse one paper."""
 
    return parse_paper(paper, download_paper(paper))
