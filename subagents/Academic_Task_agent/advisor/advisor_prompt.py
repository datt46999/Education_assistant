ANALYSIS_SITUATION = """Analyze student situation and determine guidance approach:

        CONTEXT:
        - Profile: {profile}
        - Learning Preferences: {learning_prefs}
        - Request: {request}

        ANALYZE:
        1. Current challenges
        2. Learning style compatibility
        3. Time management needs
        4. Stress management requirements
"""

GENERATE_GUIDANCE = """Generate personalized academic guidance based on analysis:

        ANALYSIS: {analysis}
        EXAMPLES: {few_short_examples}

        FORMAT:
        1. Immediate Action Steps
        2. Schedule Optimization
        3. Energy Management
        4. Support Strategies
        5. Emergency Protocols"""