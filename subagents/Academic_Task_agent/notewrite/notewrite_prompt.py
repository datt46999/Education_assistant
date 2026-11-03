LEARNING_STYLE_PROMPT = """Analyze content requirements and determine optimal note structure:

    STUDENT PROFILE:
    - Learning Style: {learning_style}
    - Request: {request}

    FORMAT:
    1. Key Topics (80/20 principle)
    2. Learning Style Adaptations
    3. Time Management Strategy
    4. Quick Reference Format

    FOCUS ON:
    - Essential concepts that give maximum understanding
    - Visual and interactive elements
    - Time-optimized study methods
    """
GENERATE_PROMPT ="""Create concise, high-impact study materials based on analysis:

    ANALYSIS: {analysis}
    LEARNING STYLE: {learning_style}
    REQUEST: {request}

    EXAMPLES:
    {json.dumps(self.few_shot_examples, indent=2)}

    FORMAT:
    **THREE-WEEK INTENSIVE STUDY PLANNER**

    [Generate structured notes with:]
    1. Weekly breakdown
    2. Daily focus areas
    3. Core concepts
    4. Emergency tips
    """