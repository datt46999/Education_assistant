import os
import json
import asyncio
import re
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, Union, Literal

from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


from subagents.Academic_Task_agent.Academic_utils import AcademicState, OpenAILLM
from subagents.Academic_Task_agent.coordinator_agent import coordinator_agent
from subagents.Academic_Task_agent.planner import PlannerAgent
from subagents.Academic_Task_agent.notewrite import NoteWriteAgent
from subagents.Academic_Task_agent.advisor import AdvisorAgent
from subagents.Academic_Task_agent.system_prompt import *

load_dotenv()


openai_key = os.getenv("OPENAI_API_KEY")
llm = OpenAILLM(openai_key)

AGENT_ALIASES = {
    "PLANNER": "PLANNER", "PLANNER_AGENT": "PLANNER",
    "NOTEWRITE": "NOTEWRITE", "NOTEWRITER": "NOTEWRITE", "NOTE_WRITER": "NOTEWRITE", "NOTEWRITE_AGENT": "NOTEWRITE",
    "ADVISOR": "ADVISOR", "ADVISER": "ADVISOR", "ADVISOR_AGENT": "ADVISOR",
}
MAX_ROUNDS = 1
def normalize_agents(names) -> list:
    """Map the coordinator's agent names to PLANNER / NOTEWRITE / ADVISOR and drop unknown ones."""
    normalized = []
    for name in names or []:
        agent = AGENT_ALIASES.get(str(name).strip().upper().replace(" ", "_"))
        if agent and agent not in normalized:
            normalized.append(agent)
        elif not agent:
            print(f"Unknown agent from coordinator ignored: {name}")
    return normalized
class AgentExecutor:
    def __init__(self, llm):
        """
        Initialize executor with language morel and execution agent instance
        

        Attributes:
        self.llm: llm execute for agent system
        self.agents: subagents support for user query
        """

        self.llm = llm
        self.agents = {
            "PLANNER": PlannerAgent(self.llm),
            "NOTEWRITE": NoteWriteAgent(self.llm),
            "ADVISOR": AdvisorAgent(self.llm)
        }

    async def execute(self, state: AcademicState)-> dict:
        """
        Orchetrates concurrent executor of multiple agent base on analysis response

        This methob implement a sophisticated execute pattern
        1 Read coordination analisis to determine require agent
        2 Group agent and concurrent execute
        3 Execute agent group  in parallel
        4 Handle failure greatefule with fallback machenism

        ARGs:
            state: current academic state
        return:
            Dict merga  all of result  executes in return of aggent system
        """

        try:
            analysis = state["results"].get("coordinator_analysis", {})
            required_agents = analysis.get("required_agents", ["PLANNER"])  # PLANNER as default
            concurrent_groups = analysis.get("concurrent_groups", [])  
    
    
            results = {}
            for group in concurrent_groups:
                tasks = []
                for agent_name in group:
                    if agent_name in required_agents and agent_name in self.agents:
                        tasks.append(self.agents[agent_name](state))
    
                if tasks:
                    group_result = await asyncio.gather(*tasks, return_exceptions= True)
                    for agent_name, result in zip(group, group_result):
                            if not isinstance(result, Exception):
                                results[agent_name.lower()]= result
    
                    if not results and 'PLANNER' in self.agents:
                        planner_results = await self.agents['PLANNER'](state)
                        results["planner"] = planner_results
    
            print("agent output: ", results )
    
            return {
                'results':{
                    'agent_outputs': results
                }
            }
    
        except Exception as e:
            print(f"Execution error: {e}")
            # Emergency fallback with minimal response
            return {
                "results": {
                    "agent_outputs": {
                        "planner": {
                            "plan": "Emergency fallback plan: Please try again or contact support."
                        }
                    }
                }
            }

"""
=======================
agent action and agent output
=======================
"""
class AgentAction(BaseModel):
    """
    Model representing agent's action decision

    Attribute:
        action (str): the specific action to be taken(eg: search_calendar, analysis_task) 
        tools (Optional[str]): The action tools use for the action 
        thought (str): This reasoning process behind the action choise
        action_input (Optional[dict]): Input paramater for the action
    """
    action: str
    tools: Optional[str]
    thought: str
    action_input: Optional[dict]


class AgentOutput(BaseModel):
    """
    The model represented output for agent action

    Attributes:
        obsevation (str):the result or obsevation for excution the data
        output (Dict): Structured output data from action
    """
    obsevation: str
    output: dict

"""
=======================
Profile analyzer
=======================
"""

async def profile_analyzer(state: AcademicState)-> dict:
    """
    Analysis learning profile and interpret learning preference using ReAct framework


    This agent specilized in:
    1 Deep analysis od student learning profile
    2 Extract of learning preferences and patterns
    3 Interpretation of acadamic  history and tendencies
    4 Generate personlize learning insight 
    """

    profile = state['profile']
    prompt = PROFILE_ANALYZER_PROMPT.format(profile = profile)

    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content':json.dumps(profile)}
    ]

    response = await llm.agenerate(messages)


    return {
        'results':{
            'profile_analyzer':{
                "analysis": response
            }
        }
    }


"""
=====================================
Create agent workflow 
=====================================
"""

def create_agent_graph(llm)-> StateGraph:
    """
    Create coordinary workflow graph for multiagent

    This orchestraction system manager parallel excetion with three specialize agent:
    +planner_agent: Handel schedule and calendar managetment
    +notewrite: create personalizes study material
    +advisor: procide academic guidance and support
 
    """
    workflow = StateGraph(AcademicState)
    planner_agent = PlannerAgent(llm)
    notewrite = NoteWriteAgent(llm)
    advisor = AdvisorAgent(llm)
    executor = AgentExecutor(llm)


    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("execute", executor.execute)
    workflow.add_node("profile_analyzer", profile_analyzer)

    def route_parallel_agents(state: AcademicState)-> list[str]:
        """
        Determine which agent should be process with the current agent

        Analyzer coordinattion's output to route work to appropriable agent.
        Default planner with no specific agent are required

        Args: 
            state: current academic state wiht coordination analysis
        Returns:
            List is agent next
        """
        analysis = state['results'].get('coordinator_analysis', {})
        required_agents = analysis.get("required_agents", [])
        executed=  set(state['results'].get("agent_outputs", {}).keys())


        next_nodes = []

        if "PLANNER" in required_agents and 'planner' not in executed:
            next_nodes.append("calendar_analyzer")

        if "NOTEWRITE" in required_agents and "notewrite" not in executed:
            next_nodes.append('notewriter_analyzer')
        if "ADVISOR" in required_agents and "advisor" not in executed:
            next_nodes.append('advisor_analyzer')

        return next_nodes if next_nodes else []

    """
    Agent subgraph node
    """

    # planner agent workflow
    workflow.add_node('calendar_analyzer', planner_agent.calendar_analysis)
    workflow.add_node('tasks_analyzer', planner_agent.task_analysis)
    workflow.add_node('plan_generate', planner_agent.plan_generate)

    # notewrite agent workflow
    workflow.add_node("notewriter_analyzer", notewrite.learning_style_analysis)
    workflow.add_node("notewriter_generate", notewrite.generate_notes)

    # Advisor agent workflow
    workflow.add_node("advisor_analyzer", advisor.analyze_situation)
    workflow.add_node("advisor_generate", advisor.generate_guidance)

    """
    ===========================
    work connection
    ===========================
    """




    workflow.add_edge(START, "coordinator")  
    workflow.add_edge('coordinator', 'profile_analyzer')
 
    workflow.add_conditional_edges(
        'profile_analyzer',
        route_parallel_agents,
        ["calendar_analyzer", "notewriter_analyzer", "advisor_analyzer"] 
    )

    # Connect Planner agent's internal workflow
    workflow.add_edge("calendar_analyzer", "tasks_analyzer")
    workflow.add_edge("tasks_analyzer", "plan_generate")
    workflow.add_edge("plan_generate", "execute")


    # Connect NoteWriter agent's internal workflow
    workflow.add_edge("notewriter_analyzer", "notewriter_generate")
    workflow.add_edge("notewriter_generate", "execute")

    # Connect Advisor agent's internal workflow
    workflow.add_edge("advisor_analyzer", "advisor_generate")
    workflow.add_edge("advisor_generate", "execute")


    def should_end(state) -> Union[Literal['coordinator'], Literal[END]]:
        """
        Determine if all requirement have complete this their tasks.


        Compare set of completed agent output againts required agent 
        to decide whether to end or continute the workflow
        
        Args:
            State: Current academic state

        Return:
            Either 'coordinator' to continute or END to finish
        """

        analysis = state['results'].get("coordinator_analysis",{})
        executed = set(state['results'].get("agent_outputs", {}).keys())
        required = set(a.lower() for a in normalize_agents(analysis.get("required_agents")) or ["PLANNER"])
        rounds = state['results'].get("execute_rounds", 0)
        print(f"should_end: required={sorted(required)} executed={sorted(executed)} round={rounds}")
        if required.issubset(executed) or rounds >= MAX_ROUNDS:  # FIX: stop instead of looping forever
            return END
        return "coordinator"

    # Add conditional loop back to coordinator if needed
    workflow.add_conditional_edges(
        "execute",
        should_end,
        {
            "coordinator": "coordinator",  # Loop back if more work needed
            END: END  # End workflow if all agents complete
        }
    )

    # Compile and return the complete workflow
    return workflow.compile()