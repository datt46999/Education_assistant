
import operator


from typing import List,  TypedDict, Optional, Annotated
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool, tool
from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages


class SearchPlan(BaseModel):
    queries: List[str] = Field(min_length = 1, max_length =5, description="1 to 3 short arXiv keyword queries, most important first.",)


     
 
class PaperUnusableError(Exception):
    """Raised when a paper cannot be used and retrying would not help."""
 


class PaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    pdf_url: str
    entry_url: str


class ParsedPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str

class PaperRelevance(BaseModel):
    """Relevance judgement of one paper for the user query."""
    arxiv_id: str = Field(description="The arxiv_id of the paper exactly as given in the list.")
    score: int = Field(ge=0, le=10, description="0 = unrelated, 10 = exactly what the user needs.")
    reason: str = Field(description="One or two sentences on why this paper fits (or not) the user's need.")

class RankingOutput(BaseModel):
    """Output object of the ranking node."""
    rankings: list[PaperRelevance] = Field(description="One entry for every paper in the list.")
 
class ParsedPaper(BaseModel):
    arxiv_id: str
    pages: list[ParsedPage]
    page_count: int = Field(ge=1)
 
    @property
    def full_text(self) -> str:
        """Join every parsed page into a single string."""
 
        return "\n".join(page.text for page in self.pages)
 
class RankedPaper(BaseModel):
    paper: PaperMetadata
    score: int
    reason: str
    
class ReviewerState(TypedDict, total = False ):
    messages: Annotated[List[BaseMessage], add_messages]
    requires_research: bool
    user_query: str
    max_results: int
    target_paper: int
    relevant_threshold: int
    output : str
    thread_id : str
    data_dir: str
    retriever_kind: str
    top_k: int
    fetch_k: int
    multi_query: bool
    search_queries: List[str]
    found_papers: List[PaperMetadata]
    ranked_papers: List[RankedPaper]
    selected_ids: List[str]
    markdown: str
    status: str
    # NEW: knowledge-graph step (see graph_node in reseach_agent.py).
    use_knowledge_graph: bool
    kg_answer: str
    kg_citations: List[dict]


class SearchPapersInput(BaseModel):
    """Input object to search papers on arXiv."""
    query: str = Field(description="The research need, in natural language, to search for on arXiv.")
    max_papers: int = Field(description="The maximum number of papers to return.", default=5, ge=1, le=10)


class JudgeOutput(BaseModel):
    """Output object of the judge node."""
    is_good_answer: bool = Field(description="Whether the answer is good or not.")
    feedback: Optional[str] = Field(default=None, description="Detailed feedback about why the answer is not good. It should be None if the answer is good.")

 
class DecisionMakingOutput(BaseModel):
    """
    Output object of decision making node
    """ 
    requires_research: bool = Field(description = "Whether the user query requires research or not")
    answer: Optional[str] = Field(default = None, description ="The answer to the user query. It should be None if the user query requires research, otherwise it should be a direct answer to the answer query.")