import os
from typing import TypedDict, List, Annotated, Optional

from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# retrival
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,abstract,authors,year,publicationDate,externalIds,openAccessPdf,url,citationCount"
S2_MIN_INTERVAL = 1.1
MAX_PDF_BYTES = 50 * 1024 * 1024
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")






# agent
DEFAULT_MAX_RESULTS = 30       
DEFAULT_SOURCES = ["arxiv", "semantic_scholar"]  
DEFAULT_TARGET_PAPERS = 5     
DEFAULT_RELEVANCE_THRESHOLD = 6 
DEFAULT_USE_KNOWLEDGE_GRAPH = True 
KG_VECTOR_TOP_K = 5            
KG_GRAPH_HOPS = 2     







class PaperUnusableError(Exception):
    """Raised when a paper cannot be used and retrying would not help."""



class SearchPlan(BaseModel):
    queries: List[str] = Field(min_length = 1, max_length =5, description="1 to 5 short keyword queries for paper search (arXiv / Semantic Scholar), most important first.",)

 
class PaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    pdf_url: str          
    entry_url: str
    published: str
    source: str = "semantic_scholar"                 # "arxiv" or "semantic_scholar"
    citation_count: Optional[int] = None  
    
class PaperRelevance(BaseModel):
    """Relevance judgement of one paper for the user query."""
    arxiv_id: str = Field(description="The arxiv_id of the paper exactly as given in the list.")
    score: int = Field(ge=0, le=10, description="0 = unrelated, 10 = exactly what the user needs.")
    reason: str = Field(description="One or two sentences on why this paper fits (or not) the user's need.")



class RankingOutput(BaseModel):
    """output object of ranking node"""
    rankings: List[PaperRelevance] = Field(description = "One entry for every paper in the list")

class RankedPaper(BaseModel):
    paper: PaperMetadata
    score : int
    reason: str



class ParsedPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str

class ParsedPaper(BaseModel):
    arxiv_id: str
    pages: list[ParsedPage]
    page_count: int = Field(ge=1)
 
    @property
    def full_text(self) -> str:
        """Join every parsed page into a single string."""
 
        return "\n".join(page.text for page in self.pages)


 

class ReviewerState(TypedDict, total = False ):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    max_results: int
    sources: List[str]
    search_queries: List[str]
    found_papers: List[PaperMetadata]
    requires_research: bool
    
    ranked_papers: List[RankedPaper]
    selected_ids: List[str]
    status: str
    relevance_threshold: int
    target_papers:int

    use_knowledge_graph:bool
    kg_answer:str
    kg_citations: List[dict]

    markdown: str
    output : str

    num_feedback_requests: int
    is_good_answer: bool 




class JudgeOutput(BaseModel):
    """Output object of the judge node."""
    is_good_answer: bool = Field(description="Whether the answer is good or not.")
    feedback: Optional[str] = Field(default=None, description="Detailed feedback about why the answer is not good. It should be None if the answer is good.")

  
class DecisionMakingOutput(BaseModel):
    """
    Output object of decision making node
    """

    requires_research: bool = Field(description ="Wheater the user query requires research or not")
    answer:Optional[str] = Field(default = None, description ="The answer to the user query. It should be return None if the user query requires research, otherwise it should be a direct answer to the answer query.")
