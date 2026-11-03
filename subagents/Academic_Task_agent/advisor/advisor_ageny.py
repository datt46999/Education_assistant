import json
from datetime import datetime, timedelta, timezone
from langgraph.graph import StateGraph, END, START


from subagents.Academic_Task_agent.Academic_utils import ReActAgent, AcademicState

from subagents.Academic_Task_agent.advisor.advisor_prompt import *



class AdvisorAgent(ReActAgent):
    """
    Academic advisor agent with subgraph workflow for personalized guidance.
    This agent specializes in analyse student situation and providing tailored academic advisor considering 
    learning style  and time constrains.
    """

    def __init__(self, llm):
        """
        Initialize the advisor agent with an LLM backend example instance
        """
        super().__init__(llm)
        self.llm = llm
        self.few_shot_examples = self._instance_fewshot()
        self.workflow = self.create_subgraph()


    def _instance_fewshot(self):
        """
        Define example scenarior the help AI can understand  how to handle different sistuation
        """
        return[{
                "request": "Managing multiple deadlines with limited time",
                "profile": {
                    "learning_style": "visual",
                    "workload": "heavy",
                    "time_constraints": ["2 hackathons", "project", "exam"]
                },
                "advice": """PRIORITY-BASED SCHEDULE:

                1. IMMEDIATE ACTIONS
                   • Create visual timeline of all deadlines
                   • Break each task into 45-min chunks
                   • Schedule high-focus work in mornings

                2. WORKLOAD MANAGEMENT
                   • Hackathons: Form team early, set clear roles
                   • Project: Daily 2-hour focused sessions
                   • Exam: Interleaved practice with breaks

                3. ENERGY OPTIMIZATION
                   • Use Pomodoro (25/5) for intensive tasks
                   • Physical activity between study blocks
                   • Regular progress tracking

                4. EMERGENCY PROTOCOLS
                   • If overwhelmed: Take 10min reset break
                   • If stuck: Switch tasks or environments
                   • If tired: Quick power nap, then review"""
            }]

    def create_subgraph(self)-> StateGraph:

        subgraph = StateGraph(AcademicState)

        # Add nodes for analysis and guidance - use consistent names
        subgraph.add_node("advisor_analyzer", self.analyze_situation)
        subgraph.add_node("advisor_generate", self.generate_guidance)

        # Connect workflow - use the new node names
        subgraph.add_edge(START, "advisor_analyzer")
        subgraph.add_edge("advisor_analyzer", "advisor_generate")
        subgraph.add_edge("advisor_generate", END)

        return subgraph.compile()


    async def analyze_situation(self, state: AcademicState)-> AcademicState:
        """
        Analyses student's current academic situation and needs,
         Evaluates:
        - Student profile and preferences
        - Current challenges and constraints
        - Learning style compatibility
        - Time and stress management needs

        Args:
            state: Current academic state with student profile and request

        Returns:
            Updated state with situation analysis results
        """

        profile = state["profile"]
        learning_prefs = profile.get("learning_preferences", {})

        prompt = ANALYSIS_SITUATION.format(
            profile=json.dumps(profile, indent=2),
            learning_prefs=json.dumps(learning_prefs, indent=2),
            request=state['messages'][-1].content
        )
        response = await self.llm.agenerate([
            {"role": "system", "content": prompt}
        ])

        return {
            "results": {
                "situation_analysis": {
                    "analysis": response
                }
            }
        }

    async def generate_guidance(self, state: AcademicState) -> AcademicState:
        """Generates personalized academic guidance based on situation analysis.

        Creates structured advice focusing on:
        - Immediate actionable steps
        - Schedule optimization
        - Energy and resource management
        - Support strategies
        - Contingency planning

        Args:
            state: Current academic state with situation analysis

        Returns:
            Updated state with generated guidance
        """

        analysis = state["results"].get("situation_analysis", "")
        prompt = GENERATE_GUIDANCE.format(
            analysis= analysis,
            few_short_examples=json.dumps(self.few_shot_examples, indent=2)
        )

        response = await self.llm.agenerate([
            {"role": "system", "content": prompt}
        ])


        return {
            "results": {
                "guidance": {
                    "advice": response
                }
            }
        }

    async def __call__(self, state: AcademicState) -> dict:
        """Main execution method for the Advisor agent.

        Executes the complete advisory workflow:
        1. Analyzes student situation
        2. Generates personalized guidance
        3. Returns formatted results with metadata

        Args:
            state: Initial academic state

        Returns:
            Dict containing guidance and metadata or error message

        Note:
            Includes metadata about guidance specificity and learning style consideration
        """

        try:
            final_state = await self.workflow.ainvoke(state)
            return {
                "advisor_output": {
                    "guidance": final_state["results"].get("guidance"),
                    "metadata": {
                        "course_specific": True,
                        "considers_learning_style": True
                    }
                }
            }
        except Exception as e:
            return {
                "advisor_output": {
                    "guidance": "Error generating guidance. Please try again."
                }
            }

