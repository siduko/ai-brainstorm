import textwrap
from datetime import datetime


def generate_timestamp_str():
    """
    Used for overcoming caching that may happen with OpenRouter. Who knows what OpenAI and Groq are doing under the hood.
    """
    return (
        "[IGNORE: timestamp for record keeping - "
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + "]\n"
    )


def post_process_prompt(prompt, timestamp=True):
    """
    Cleans up the prompt by removing leading and trailing whitespace and adding a timestamp.
    """
    if timestamp:
        return generate_timestamp_str() + textwrap.dedent(prompt).strip()
    else:
        return textwrap.dedent(prompt).strip()


def get_idea_generation_system_prompt():
    prompt = f"""
    You are an expert idea generation system with a sought-after creative mind.
    You utilize your broad knowledge and experience to generate innovative ideas.
    Your idea generation process thrives when you're provided a Creative Directive, which is a strategy or constraint that guides your idea generation process.
    You always take into account the user's general problem statement and their own specific constraints when generating ideas.
    Sometimes you are provided with an existing idea which you need to expand upon and improve.
    """
    return post_process_prompt(prompt, timestamp=False)


def get_generate_seed_idea_prompt(
    problem_statement, constraints, creative_directive, instruction, explanation
):
    prompt = f"""
    I'm starting a brainstorming session and I need your help to generate a new idea.
    
    Problem Statement:
    {problem_statement}

    Constraints:
    {constraints}

    Creative Directive: {creative_directive}
    Creative Directive Instruction: {instruction}
    Creative Directive Explanation: {explanation}

    Use the Problem Statement, Constraints, and Creative Directive to generate a new and innovative idea.
    State your high level idea in a single sentence. Then describe the idea in more detail in two more sentences.
    Your responses will be plain text. Do not use markdown or any other formatting.
    ONLY respond with your idea. No introductions, exposition, or other commentary.
    
    I believe in your ability to generate a great idea. You've always delivered. Really try to do your best.
    """
    return post_process_prompt(prompt)


def get_generate_idea_prompt(
    problem_statement,
    constraints,
    current_idea,
    creative_directive,
    instruction,
    explanation,
):
    prompt = f"""
    I'm brainstorming ideas and I need your help to generate a new idea based on an existing idea that I have. I really want you to improve the existing idea.

    Problem Statement:
    {problem_statement}

    Constraints:
    {constraints}
    
    Existing Idea:
    {current_idea}
    
    Creative Directive: {creative_directive}
    Instruction: {instruction}
    Explanation: {explanation}

    Use the Problem Statement, Constraints, Existing Idea, and Creative Directive to expand on and evolve the Existing Idea.
    Your new idea should be an improvement on the existing idea, not a complete departure. You're trying to make the existing idea better.
    State your high level idea in a single sentence. Then describe the idea in more detail in two more sentences.
    Your responses will be plain text. Do not use markdown or any other formatting.
    ONLY respond with your idea. No introductions, exposition, or other commentary.

    I believe in your ability to generate a great idea. You've always delivered. Really try to do your best to improve the existing idea.
    Do NOT reference the Existing Idea in your response. Someone will be evaluating your idea, and they will have never seen the Existing Idea.
    """
    return post_process_prompt(prompt)


def get_idea_evaluation_system_prompt(evaluation_criteria):
    criteria_text = "\n".join(
        [
            f"        {i+1}. {criterion['name']}: {criterion['description']}"
            for i, criterion in enumerate(evaluation_criteria)
        ]
    )
    prompt = f"""
    You're a hyper-critical idea evaluation system.
    You consider a Problem Statement and Constraints when evaluating ideas.
    You have exceptional taste and have a great gut for determining the merit of an idea.
    A vast majority of people liken your ability to evaluate ideas to that of someone like Steve Jobs.
    You evaluate ideas across multiple criteria:
{criteria_text}
    You provide a one sentence reasoning for each criterion and an accompanying score of 0 to 100.
    - A score of 0 means the idea exhibits none of a given criterion.
    - A score of 50 means the idea exhibits average performance within a given criterion.
    - A score of 100 means the idea exhibits the best possible rating within a given criterion.
    - Rarely do scores exceed 85 on a given criterion. Those that do are truly exceptional.
    Your one sentence reasoning should include both a positive and negative aspect of the idea.
    """
    return post_process_prompt(prompt, timestamp=False)


def get_evaluate_idea_prompt(problem_statement, constraints, idea, evaluation_criteria):
    criteria_format = "\n".join(
        [
            f"{criterion['name']} Reasoning: <your reasoning in one sentence>\n{criterion['name']} Score: <score>"
            for criterion in evaluation_criteria
        ]
    )
    prompt = f"""
    Use the following Problem Statement and Constraints to evaluate the following idea.

    Problem Statement:
    {problem_statement}

    Constraints:
    {constraints}

    Idea to Evaluate:
    {idea}

    Respond in the following format:

{criteria_format}
    """
    return post_process_prompt(prompt)
