# -----------------------------------------------------------------------------
#
# AI-assisted Brainstorming Tool
#
# Follow the numbered comment blocks to customize the tool for your needs.
#
# -----------------------------------------------------------------------------
import os
import textwrap

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

from creative_directives import CREATIVE_DIRECTIVES
from mcts import MCTS
from node import Node
from utils import print_tree, save_tree_to_html

load_dotenv()

# Globals
root_node = None

# -----------------------------------------------------------------------------
#
# 1. Make sure your provider and API keys are set in the .env file.
#
#    You must provide at least one of the following:
#    OPENROUTER_API_KEY, OPENAI_API_KEY, GROQ_API_KEY
#
# -----------------------------------------------------------------------------
LLM_API_PROVIDER = os.getenv("LLM_API_PROVIDER", None)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

llm_client = None

# We need a flag here for anticipating hitting Groq rate limits since generation is so fast
# This will automatically be set to True if we're using Groq. No need to set it manually.
use_groq = False

# -----------------------------------------------------------------------------
#
# 2. Specify the LLM models to use.
#
#    Set the LLM models to use based on what your API provider makes available.
#    It's recommended to use smaller models for generation and evaluation in
#    order to keep costs down.
#
# -----------------------------------------------------------------------------
if LLM_API_PROVIDER == "openai":
    LLM_GENERATION_MODEL = "gpt-4o-mini"
    LLM_EVALUATION_MODEL = "gpt-4o-mini"
    if OPENAI_API_KEY:
        llm_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        print("OPENAI_API_KEY is not set. Please set it in the .env file.")
        exit()

elif LLM_API_PROVIDER == "groq":
    LLM_GENERATION_MODEL = "llama-3.1-8b-instant"
    LLM_EVALUATION_MODEL = "llama-3.1-8b-instant"
    chosen_api_key = GROQ_API_KEY
    use_groq = True
    if GROQ_API_KEY:
        llm_client = Groq(api_key=GROQ_API_KEY)
    else:
        print("GROQ_API_KEY is not set. Please set it in the .env file.")
        exit()

elif LLM_API_PROVIDER == "openrouter":
    LLM_GENERATION_MODEL = "microsoft/wizardlm-2-7b"
    LLM_EVALUATION_MODEL = "meta-llama/llama-3.1-8b-instruct"
    if OPENROUTER_API_KEY:
        llm_client = OpenAI(
            api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"
        )
    else:
        print("OPENROUTER_API_KEY is not set. Please set it in the .env file.")
        exit()

else:
    print(
        "Invalid LLM API provider. Make sure to set the LLM_API_PROVIDER environment variable and supply the correct API key."
    )
    exit()

# -----------------------------------------------------------------------------


def initialize_root_node(problem_statement, constraints):
    global root_node
    root_node = Node(problem_statement, is_problem_statement=True, constraints=constraints)
    return root_node


# -----------------------------------------------------------------------------


def get_best_ideas(node, n=5):
    """
    Retrieves the best ideas from the tree starting at the given node.

    This function traverses the entire tree rooted at the given node,
    collects all non-problem statement nodes, and returns the top n nodes
    based on their score-to-visits ratio.

    Args:
    node (Node): The root node to start the search from.
    n (int): The number of best ideas to return. Defaults to 5.

    Returns:
    list: A list of the n best Node objects, sorted by their score-to-visits ratio.
    """

    all_nodes = [node]
    for child in node.children:
        all_nodes.extend(get_best_ideas(child, n))
    # Remove the root node (problem statement) from the list
    all_nodes = [n for n in all_nodes if not n.is_problem_statement]
    return sorted(
        all_nodes, key=lambda n: n.score / n.visits if n.visits > 0 else 0, reverse=True
    )[:n]


def print_best_ideas(best_ideas):
    print("\033[95m\033[3mTOP " + str(num_best_ideas) + " IDEAS:\033[0m\033[0m\n")

    for i, idea in enumerate(best_ideas, 1):
        print(f"\033[95m\033[1mIdea {i}: ---------------------------------------\033[0m\n")
        print_tree(idea, include_full_content=True, single_node=True)


# -----------------------------------------------------------------------------


# Main execution
if __name__ == "__main__":
    import sys
    from utils import load_tree_from_json
    from datetime import datetime

    will_run_mcts = False

    # Check if a JSON file was passed as an argument
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        # Load the tree from the JSON file
        root_node = load_tree_from_json(sys.argv[1])
        print(f"Loaded tree from {sys.argv[1]}")

        # Check if there's an additional argument called "continue" so we can do more iterations.
        if len(sys.argv) > 2 and sys.argv[2] == "continue":
            print("Continuing from last run...")
            will_run_generation = True

    if root_node is None:
        # -----------------------------------------------------------------------------
        #
        # 3. Define the problem statement and constraints
        #
        # -----------------------------------------------------------------------------
        problem_statement = "I want to create a web app that will use generative AI large language models to assist with brainstorming. What would the UX for an interface for this app be like?"

        constraints = f"""
            - Your idea must be a web app.
            - Your idea must be achievable with today's technology.
            - Your idea must be accomplishable with only a computer with a web browser or a tablet. No additional hardware.
            - Current generative AI large language models are capable of natural language processing and generation and vision (for still images, but not precise charts or text).
            - Do not use any haptic, olfactory, taste, or audio feedback as part of the idea.
            - Your idea must be for a single user. It should NOT be a multi-user or collaborative app.
            - Your idea should focus on UTILITY, not entertainment.
            - Important: explain what a prototype MVP of your idea would be. The prototype should be completable by a single person in one week.
            """
        constraints = textwrap.dedent(constraints)

        root_node = initialize_root_node(problem_statement, constraints)
        will_run_mcts = True

    if will_run_mcts:
        # -----------------------------------------------------------------------------
        #
        # 4. Define the evaluation criteria.
        #
        #    These criteria should be rankable on a scale of 1-100.
        #    The higher the score, the better the idea.
        #    It cannot be used for boolean evaluation.
        #
        # -----------------------------------------------------------------------------
        EVALUATION_CRITERIA = [
            {
                "name": "Innovative",
                "description": "How innovative and compelling is the idea?",
            },
            {
                "name": "Usefulness",
                "description": "How useful is the idea?",
            },
            {
                "name": "Interestingness",
                "description": "How interesting or sticky would this experience be for a user?",
            },
            {
                "name": "Prototypability",
                "description": "Does this idea include a description of a prototype that is achievable with today's technology?",
            },
        ]

        # This is the main MCTS loop. It's what does all the things.
        # Be sure to look at the MCTS class for more details.
        mcts = MCTS(
            root_node,
            CREATIVE_DIRECTIVES,
            EVALUATION_CRITERIA,
            llm_client,
            LLM_GENERATION_MODEL,
            LLM_EVALUATION_MODEL,
            save_filename_prefix="brainstorm_output",
            use_groq=use_groq,
            iterations=100,
            c_param=MCTS.DEFAULT_C_PARAM,
        )
        mcts.run()

    # By default, the script will display the best ideas, whether or not we're continuing from a previous run.
    print("\n===============================================\n")

    num_best_ideas = 5
    best_ideas = get_best_ideas(root_node, num_best_ideas)
    print_best_ideas(best_ideas)

    if will_run_mcts:
        save_tree_to_html(root_node, mcts.save_filename_prefix + ".html")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_tree_to_html(root_node, f"brainstorm_output_{timestamp}.html")
