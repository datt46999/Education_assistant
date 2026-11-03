import sys
import asyncio
from subagents.Tutor_agent import create_tutor_graph
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage


load_dotenv()
tutor_graph = create_tutor_graph()


async def run_tutor(question: str) -> str:
    result = await tutor_graph.ainvoke(
        {"question": question, "messages": [HumanMessage(content=question)]},
        {"recursion_limit": 12},
    )
    return result["messages"][-1].content

if __name__ == "__main__":
    print(asyncio.run(run_tutor(" ".join(sys.argv[1:]) or "What is the Pythagorean theorem?")))



# from groq import Groq
# from dotenv import load_dotenv
# load_dotenv()
# for m in Groq().models.list().data:
#     print(m.id)