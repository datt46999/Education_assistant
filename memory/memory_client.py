import os
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore
from dotenv import load_dotenv
load_dotenv()


MEMORY_ID = os.getenv("MEMORY_ID")

REGION = os.getenv("AWS_REGION", "us-east-1")

checkpointer = AgentCoreMemorySaver(MEMORY_ID, region_name= REGION)
store = AgentCoreMemoryStore(memory_id= MEMORY_ID, region_name = REGION)

def make_config(actor_id: str, session_id: str)-> dict:

    return {"configurable": {"thread_id": session_id, "actor_id": actor_id}}
