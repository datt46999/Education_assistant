import os
import json

from dotenv import load_dotenv
from typing import Dict, List
import traceback

from subagents.Academic_Task_agent.Academic_utils import AcademicState,  OpenAILLM
from subagents.Academic_Task_agent.system_prompt import *


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
llm = OpenAILLM(openai_key)
async def analyses_context(state: AcademicState)-> Dict:
    """
    Analysis the academic state context to inform coordination decision-making
    This function performs comprehensive context analysis by:
    1 Extract student profile information.
    2 Analysis calendar and task load
    3 Identifying relevant course context from the last messager
    4 Gather learning preferences and study patterns


    Args:
        state: current academic state include profile, calander, tasks
    Return:
        Structure analysis of the student's context for decision making
    """

    profile = state.get('profile', {})
    calendar = state.get('calendar', {})
    tasks = state.get('tasks', {})


    courses = profile.get('academic_info', {}).get('current_courses', [])
    current_course = None
    request = state['messages'][-1].content.lower()

    for course in courses:
        if course['name'].lower() in request:
            current_course = course
            break

    return {
        "student": {
            "major": profile.get("personal_info", {}).get("major", "Unknown"),
            "year": profile.get("personal_info", {}).get("academic_year"),
            "learning_style": profile.get("learning_preferences", {}).get("learning_style", {}),
        },
        "course": current_course,
        "upcoming_events": len(calendar.get("events", [])),  # Calendar load indicator
        "active_tasks": len(tasks.get("tasks", [])),        # Task load indicator
        "study_patterns": profile.get("learning_preferences", {}).get("study_patterns", {})
    }


def parse_coordinator_response(response: str)-> dict:
    """
    Parse coordinator response into structure analysis for agent execution.

    This function implement a robust parsing trategy:
    1 Start with safe default configuration
    2 Analysis ReAct on response
    3 Adjust agent requirement and priorities base on content
    4 Orginize the concurrent execution group
    

    Args: 
        response: Raw llm response text

    return: 
        dict: structure analysis containing
            requiered_agent : List of agent needed
            priority : priority level need for each agent
            concurrent_group: group of agent that can run together 
            reasoning: extracted reasoning for decisions 
    """

    try: 
        analysis = {
                "required_agents": ["PLANNER"],         # PLANNER is always required
                "priority": {"PLANNER": 1},             # Base priority structure
                "concurrent_groups": [["PLANNER"]],     # Default execution group
                "reasoning": "Default coordination" 
            }
        
        if "Thought" in response and "Decision" in response:
                # check NoteWriter requirements
                if "NoteWriter" in response or 'note' in response.lower():
                    analysis['required_agents'].append("NOTEWRITER")
                    analysis['priority']['NOTEWRITE'] = 2
                    analysis["concurrent_groups"] = [['PLANNER', 'NOTEWRITER']]
    
                # check advisor requirements
                if 'Advisor' in response or 'guidance' in response.lower():
                    analysis['required_agents'].append("ADVISOR")
                    analysis['priority']['ADVISOR'] = 3
    
                # extract and store reasoning from throught section
                though_section = response.split('Thought:')[1].split('Action:')[0].strip()
                analysis['reasoning'] = though_section
    
    
        return analysis

    except Exception as e:
        print(f"Parse error: {str(e)}")
        # Fallback to safe default configuration
        return {
            "required_agents": ["PLANNER"],
            "priority": {"PLANNER": 1},
            "concurrent_groups": [["PLANNER"]],
            "reasoning": "Fallback due to parse error"
        }


async def coordinator_agent(state: AcademicState)-> dict:
    """
    Primary coordinatory agent the orchestrate multiple academic support using ReAct framwork


    This agent implement a sophisticate coordinate strategy:

    1 Analisis academic context and student need
    2 Uses ReAct framework for structure decision making 
    3 Coordinatory parrallel agent execution
    4 Handel fallback scenarios

    Args:
        state: current academicstate
    Return:
        Coordination analysis include require agent, parallel and execution and execution group
    """
    try:
        context = await analyses_context(state)
        query = state["messages"][-1].content

        prompt = COORDINATOR_PROMPT
        response = await llm.agenerate([
            {"role": "system", "content": prompt.format(
                request = query,
                context = json.dumps(context, indent = 2)
            )}
        ])


        analysis = parse_coordinator_response(response) or {}

        return {
                "results": {
                    "coordinator_analysis": {
                        "required_agents": analysis.get("required_agents", ["PLANNER"]),
                        "priority": analysis.get("priority", {"PLANNER": 1}),
                        "concurrent_groups": analysis.get("concurrent_groups", [["PLANNER"]]),
                        "reasoning": response
                    }
                }
            }
    except Exception as e:
        print(f"Coordinator error: {e}")
        traceback.print_exc()
        # Fallback to basic planning configuration
        return {
            "results": {
                "coordinator_analysis": {
                    "required_agents": ["PLANNER"],
                    "priority": {"PLANNER": 1},
                    "concurrent_groups": [["PLANNER"]],
                    "reasoning": "Error in coordination. Falling back to planner."
                }
            }
        }


