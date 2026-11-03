import json
from datetime import datetime, timedelta, timezone
from langgraph.graph import StateGraph, END, START


from subagents.Academic_Task_agent.Academic_utils import ReActAgent, AcademicState

from subagents.Academic_Task_agent.notewrite.notewrite_prompt import *


class NoteWriteAgent(ReActAgent):
    """
    Notewrite agent with it own subgraph workflow for note generate.
    This agent specializes in creating personlized study materials by analyzing learning style and generate structure notes.
    """

    def __init__(self, llm):
        """Initilize the NoteWrite agent
        
        
        Attributes:
            self.llm ; llm use for writenote
            self.few_short_examples: few example help llm understand and who to handle different situation
            self.workflow: workflow graph how to planner process request
        """
        super().__init__(llm)
        self.llm = llm
        self.few_short_examples = self._initialize_examples()
        self.workflow = self.create_subgraph()


    def _initialize_examples(self):
        """
        Define example scenarior to help AI Understand how to handle different situation
        """
        return [
            {
                "input": "Need to cram Calculus III for tomorrow",
                "template": "Quick Review",
                "notes": """CALCULUS III ESSENTIALS:

                1. CORE CONCEPTS (80/20 Rule):
                   • Multiple Integrals → volume/area
                   • Vector Calculus → flow/force/rotation
                   • KEY FORMULAS:
                     - Triple integrals in cylindrical/spherical coords
                     - Curl, divergence, gradient relationships

                2. COMMON EXAM PATTERNS:
                   • Find critical points
                   • Calculate flux/work
                   • Optimize with constraints

                3. QUICKSTART GUIDE:
                   • Always draw 3D diagrams
                   • Check units match
                   • Use symmetry to simplify

                4. EMERGENCY TIPS:
                   • If stuck, try converting coordinates
                   • Check boundary conditions
                   • Look for special patterns"""
            }
        ]
    def create_subgraph(self) -> StateGraph:
        """
        nodewriter internal workflow as state machine
        The workflow include two main set:
        + Analysis learning style and content requirement
        + generate personalize notes
        """

        subgraph = StateGraph(AcademicState)
        subgraph.add_node("notewriter_analyzer", self.learning_style_analysis)
        subgraph.add_node("notewriter_generate", self.generate_notes)


        subgraph.add_edge(START, "notewriter_analyzer")
        subgraph.add_edge("notewriter_analyzer", "notewriter_generate")
        subgraph.add_edge("notewriter_generate", END)

        return subgraph.compile()
    async def learning_style_analysis(self, state: AcademicState) -> AcademicState:
        """
        Analysis student profile and request and determine student note structure
        Using LLM to analyse:
            - student's learning style of preference
            - specific content request
            - Time containts and requirement
        Args: 
            state: current acdemic state

        return 
            Update learning style from student profile
        """

        profile = state['profile']
        learning_style = profile["learning_preferences"]["learning_style"]

        prompt = LEARNING_STYLE_PROMPT.format(
       
            learning_style=json.dumps(learning_style, indent=2),
            request=state['messages'][-1].content
        
        )
        response = await self.llm.agenerate(
            [{'role':'user', 'content': prompt}]
        )
        return {
            "results":{
                "learning_analysis":{
                    "analysis": response
                }
            }
        }

    async def generate_notes(self, state: AcademicState) -> AcademicState:
        """
        Generate pertionalized study notes by on learning style

         Use the LLM  to create structure note that are:
            - Adapted the student's learning style
            - forcus on essitial concepts(80/20 principle)
            - Time optimize fot study period

        """
        analysis = state["results"].get('learning_analysis', {}).get('analysis', '')

        learning_style = state['profile']['learning_preferences']['learning_style']


        prompt = GENERATE_PROMPT.format(
            analysis=analysis,
            learning_style=json.dumps(learning_style, indent=2),
            request=state['messages'][-1].content
        )
        responses = await self.llm.agenerate(
            [{'role':'system', 'content': prompt}]
        )
        return {
            'results': {
                'generated_notes':{
                    'notes': responses
                }
            }
        }

    async def __call__(self, state: AcademicState)->dict:
        """
        The main execution method for the Notewriter agent.
        Executes the complete workflow:
        + analysis learning requirements
        + generate personalize notes
        """
        try:
            final_state = await self.workflow.ainvoke(state)
            # notes = final_state['results'].get('generated_notes', {})
            return {'notes': final_state['results']["generated_notes"]['notes']}
        except Exception as e:
            return{'notes': "Error generate note. Please try again"}