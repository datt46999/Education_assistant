from langchain_core.tools import tool 
from langchain_core.messages import SystemMessage, HumanMessage
 
from dotenv import load_dotenv
from subagents.Tutor_agent.config_tutor import llm
from subagents.Tutor_agent.toolagents.EnglishAgent.english_prompt import ENGLISH_ASSISTANT_SYSTEM_PROMPT
 
load_dotenv()
 
@tool
def english(query: str)->str:
    """This is responsible for handling questions involving writing/ literature/ grammar
 
    Args:
        query (str): The user's English language/ literature/ writing question
 
    Returns:
        str: A helpful response addressing the user's query.
    """
    response = llm.invoke([
        SystemMessage(content=ENGLISH_ASSISTANT_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ])
    return response.content