from langchain_core.tools import tool 
from langchain_core.messages import SystemMessage, HumanMessage
 
from dotenv import load_dotenv
from subagents.Tutor_agent.config_tutor import llm
from subagents.Tutor_agent.toolagents.MathAgent.math_prompt import MATH_ASSISTANT_SYSTEM_PROMPT
 
load_dotenv()
 
@tool
def math(query: str)->str:
    """
    This responsible for handling math questions
    Args:
        query (str):A math question or problem for the user.
    Return:
        str: A detailer math answer with explanation of the steps take derive the answer 
    """
    response = llm.invoke([
        SystemMessage(content=MATH_ASSISTANT_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ])
    return response.content
 