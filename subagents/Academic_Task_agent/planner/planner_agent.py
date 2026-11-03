import json
from datetime import datetime, timedelta, timezone
from langgraph.graph import StateGraph, END, START

from subagents.Academic_Task_agent.Academic_utils import ReActAgent, AcademicState
from subagents.Academic_Task_agent.planner.planner_prompt import *


class PlannerAgent(ReActAgent):
    """
    Planner agent with own sudgraph fot the plan generate

    This agent will be analysis student calendar and analysis task determine,
    then combine with student profile to generate comprehensive study plan
    """

    def __init__(self, llm):
        """
        Initialize planner agent

        Attributes:
            self.llm ; llm use for planner agent
            self.few_short_examples: few example help llm understand and who to handle different situation
            self.workflow: workflow graph how to planner process request
        """
        super().__init__(llm)
        self.llm = llm
        self.few_short_examples =  self._initialize_fewshot()
        self.workflow = self.create_subgraph()


    def _initialize_fewshot(self):
        """
        Define example scenarior to help AI Understand how to handle different situation
        """
        return [
            {
                'input': 'Help with exam prep while managing ADHD and football',
                'thought': 'Need to check calendar conflicts and energy patterns',
                'action': 'search_calendar',
                'observation': 'Football match at 6PM exam tomorrow 9AM',
                'plan':"""ADHD-OPTIMIZED SCHEDULE:
                    PRE-FOOTBALL (2PM-5PM):
                    - 3x20min study sprints
                    - Movement breaks
                    - Quick rewards after each sprint

                    FOOTBALL MATCH (6PM-8PM):
                    - Use as dopamine reset
                    - Formula review during breaks

                    POST-MATCH (9PM-12AM):
                    - Environment: Café noise
                    - 15/5 study/break cycles
                    - Location changes hourly

                    EMERGENCY PROTOCOLS:
                    - Focus lost → jumping jacks
                    - Overwhelmed → room change
                    - Brain fog → cold shower"""
            },
            {
                "input": "Struggling with multiple deadlines",
                "thought": "Check task priorities and performance issues",
                "action": "analyze_tasks",
                "observation": "3 assignments due, lowest grade in Calculus",
                "plan": """PRIORITY SCHEDULE:
                    HIGH-FOCUS SLOTS:
                    - Morning: Calculus practice
                    - Post-workout: Assignments
                    - Night: Quick reviews

                    ADHD MANAGEMENT:
                    - Task timer challenges
                    - Reward system per completion
                    - Study buddy accountability"""
            }
        ]

    def create_subgraph(self) -> StateGraph:
        """
        Create workflow that defince how to planner agent  processes request

        calendar_analyzer -> tasks_analyzer -> plan_generate

        1 Analysis student calendar
        2 Analysis student tasks
        3 finally plan generate
        """
        subgraph = StateGraph(AcademicState)

        subgraph.set_entry_point("calendar_analyzer")
        subgraph.add_node("calendar_analyzer", self.calendar_analysis)
        subgraph.add_node("tasks_analyzer", self.task_analysis)
        subgraph.add_node("plan_generate", self.plan_generate)

        subgraph.add_edge("calendar_analyzer", "tasks_analyzer")
        subgraph.add_edge("tasks_analyzer", "plan_generate")
        subgraph.add_edge("plan_generate", END)


        return subgraph.compile()



 
    async def calendar_analysis(self, state: AcademicState)-> AcademicState:
        """
        Analysis calendar student to find
        + available study time
        + potential scheduling conflict
        + energy pattern thoughout they day
        """

        events = state["calendar"].get("events", [])
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=7)

        filtered_events =[event for event in events
                       if now <= datetime.fromisoformat(event['starts']['dateTime'])<future]

        prompt = CALENDAR_PROMPT.format(events = events)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(filtered_events)}
        ]   

        response = await self.llm.agenerate(messages)
   
        return {
            "results": {
                "calendar_analyzer": {
                    "analysis":response
                }
            }
        }

    async def task_analysis(self, state: AcademicState) -> AcademicState:
        """
        Analysis tasks a detemine:
        + Priority order
        + Time need to each task
        + Best approach to complete
        """

        tasks = state['tasks'].get('tasks', [])

        prompt = TASKS_PROMPT.format(tasks = tasks)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(tasks)}
        ]

        response = await self.llm.agenerate(messages)

        return {
            "results": {
                "task_analyzer": {
                    "analysis": response
                }
            }
        }

    async def plan_generate(self, state: AcademicState)-> AcademicState:
        """
        Create Comprehensize study plan by combine:
        + profile student (student learning style)
        + calendar of student (avalable time)
        + tasks of student (what time need to complete task)
        """

        profile = state["results"]['profile_analyzer']
        calendar_analysis = state["results"]["calendar_analyzer"]
        task_analysis = state["results"]["task_analyzer"]


        prompt = PLANT_PROMPT.format(
            profile_analysis = profile, 
            calendar_analysis = calendar_analysis, 
            task_analysis = task_analysis, 
            few_short_examples =json.dumps(self.few_short_examples, indent=2))
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': state["messages"][-1].content}
        ]

        response = await self.llm.agenerate(messages, temperature = 0.5)
        return {
                'results': {
                    'generated_notes':{
                        'notes': response
                    }
                }
            }

    async def __call__(self, state: AcademicState) -> AcademicState:
        """
        Main execution method that run the entire planning workflow
        1. calendar Analysis
        2. Tasks analysis
        3. plan generate
        """
        try:
            final_state = await self.workflow.ainvoke(state)

            return {"notes": final_state["results"]["generated_notes"]['notes']}

        except:
            return {'notes', 'Error generate notes. Please try again.'}

