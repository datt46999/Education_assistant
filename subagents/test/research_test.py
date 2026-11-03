import os
from subagents.researcher_agent.research_agent import run_research


from dotenv import load_dotenv

load_dotenv()
 

if __name__ == "__main__":
    print(run_research("I need to find some paper newest from STGCN"))