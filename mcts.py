import random
import time

from datetime import datetime
from node import Node
from prompts import (
    get_evaluate_idea_prompt,
    get_generate_idea_prompt,
    get_generate_seed_idea_prompt,
    get_idea_evaluation_system_prompt,
    get_idea_generation_system_prompt,
)
from utils import save_tree_to_json


class MCTS:
    # Defaults to sqrt(2)
    DEFAULT_C_PARAM = 1.414

    def __init__(
        self,
        root,
        creative_directives,
        evaluation_criteria,
        llm_client,
        llm_generation_model,
        llm_evaluation_model,
        save_filename_prefix="brainstorm",
        use_groq=False,
        iterations=250,
        c_param=DEFAULT_C_PARAM,
    ):
        """
        Initialize the Monte Carlo Tree Search (MCTS) object.

        Args:
            root (Node): The root node of the search tree.
            creative_directives (list): A list of creative directives to guide idea generation.
            evaluation_criteria (list): A list of criteria for evaluating generated ideas.
            llm_client: The client for interacting with the language model API. Should adhere to the OpenAI SDK.
            llm_generation_model (str): The name of the language model to use for idea generation.
            llm_evaluation_model (str): The name of the language model to use for idea evaluation.
            save_filename_prefix (str, optional): Prefix for the output file name. Defaults to "brainstorm". System will append a timestamp to the prefix.
            use_groq (bool, optional): Flag to indicate if using Groq API. Defaults to False.
            iterations (int, optional): Number of MCTS iterations to perform. Defaults to 250.
            c_param (float, optional): Exploration parameter for the UCT formula in MCTS. Defaults to sqrt(2). Increase to encourage exploration (approach 2), decrease to encourage exploitation (approach 1).
        """
        self.root = root
        self.creative_directives = creative_directives
        self.evaluation_criteria = evaluation_criteria
        self.iterations = iterations
        self.is_using_groq = use_groq
        self.llm_client = llm_client
        self.llm_generation_model = llm_generation_model
        self.llm_evaluation_model = llm_evaluation_model
        self.save_filename_prefix = save_filename_prefix
        self.save_filename = None
        self.c_param = c_param

    def run(self):
        """
        Performs Monte Carlo Tree Search (MCTS) to explore and evaluate ideas.

        This function iteratively selects, expands, simulates, and backpropagates through
        the tree of ideas, starting from the given root node. It uses creative directives
        to guide the expansion process and evaluates the quality of generated ideas.

        The function first ensures the root has initial children, then performs the
        specified number of iterations, printing progress and scores along the way.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_filename = f"{self.save_filename_prefix}_{timestamp}.json"

        request_count = 0
        start_time = time.time()
        NUM_REQUESTS_PER_MINUTE = 30
        NUM_REQUESTS_PER_ITERATION = 2
        SLEEP_TIME_BETWEEN_REQUESTS = 1.5

        # First, check if the root node has children
        if not self.root.children:
            for _ in range(Node.MAX_CHILDREN):
                node = self._expand(self.root)

                if self.is_using_groq:
                    time.sleep(SLEEP_TIME_BETWEEN_REQUESTS)

                if node is None:
                    continue

                score = self._simulate(node)
                print("\033[92m  Score: " + str(score) + "\033[0m" + "\n")
                self._backpropagate(node, score)

                if self.is_using_groq:
                    request_count += NUM_REQUESTS_PER_ITERATION
                    time.sleep(SLEEP_TIME_BETWEEN_REQUESTS)

                # We're going to save the tree to a JSON file after every iteration
                save_tree_to_json(self.root, self.save_filename)

        print("\033[93m\033[1mStarting MCTS iterations...\033[0m\n")
        for _ in range(self.iterations):
            print(
                "\033[93m\033[1mIteration "
                + str(_)
                + " of "
                + str(self.iterations)
                + "\033[0m\n"
            )

            # 1. Select the node
            node = self._select(self.root)

            if random.random() < 0.05:
                print("\033[91m\033[1mSelecting root node instead of `select`.\033[0m")
                node = self.root

            print("\033[93mSelecting node: \033[0m" + node.get_position_info())

            if not node.is_terminal():
                # Let's make sure we're not hitting the rate limit
                if self.is_using_groq:
                    request_count += NUM_REQUESTS_PER_ITERATION
                    if request_count >= NUM_REQUESTS_PER_MINUTE:
                        elapsed_time = time.time() - start_time
                        if elapsed_time < 60:
                            sleep_time = 60 - elapsed_time
                            print(
                                f"\033[93mRate limit approached. Sleeping for {sleep_time:.2f} seconds.\033[0m"
                            )
                            # Add a couple seconds just to make sure we don't hit the rate limit
                            time.sleep(sleep_time + 2)
                        start_time = time.time()
                        # Reset the request count accounting for the upcoming requests
                        # we're about to make after this sleep delay.
                        request_count = NUM_REQUESTS_PER_ITERATION
                    else:
                        time.sleep(SLEEP_TIME_BETWEEN_REQUESTS)

                # 2. Expand the node
                print("\033[93mExpanding node: \033[0m" + node.get_position_info())
                expanded_node = self._expand(node)
                if expanded_node is None:
                    print("\033[91mExpansion failed. Skipping this iteration.\033[0m")
                    continue

                if self.is_using_groq:
                    time.sleep(SLEEP_TIME_BETWEEN_REQUESTS)

                # 3. Simulate the node
                score = self._simulate(expanded_node)
                print("\033[92m  Score: " + str(score) + "\033[0m" + "\n")

                # 4. Backpropagate the score
                self._backpropagate(expanded_node, score)

                # We're going to save the tree to a JSON file after every iteration
                save_tree_to_json(self.root, self.save_filename)

    # -----------------------------------------------------------------------------

    def _select(self, node):
        """
        Selects a node for expansion in the Monte Carlo Tree Search.

        This function implements the selection phase of MCTS, traversing the tree
        from the root to a leaf node. It balances exploration and exploitation
        using the UCB1 algorithm, with an added element of randomness.

        The function continues to select child nodes until it reaches either:
        1. A terminal node (leaf)
        2. A node that is not fully expanded

        A 20% chance of random selection is introduced to encourage exploration
        of diverse paths in the tree.

        Args:
            node (Node): The starting node for selection, typically the root.

        Returns:
            Node: The selected node for expansion or simulation.
        """
        while not node.is_terminal():
            if not node.fully_expanded():
                return node
            else:
                # Introduce randomness in selection
                if random.random() < 0.2:
                    return random.choice(node.children)
                else:
                    node = node.best_child(c_param=self.c_param)
        return node

    # -----------------------------------------------------------------------------

    def _expand(self, node):
        """
        Expands the given node by creating a new child node with a generated idea.

        This function is responsible for the expansion phase of the Monte Carlo Tree Search.
        It selects an unused creative directive, generates a new idea based on the current
        node's content, and creates a new child node with this idea. The function handles
        both problem statement nodes and regular idea nodes differently.

        Args:
            node (Node): The node to be expanded.

        Returns:
            Node: The newly created child node, or None if expansion fails.
        """

        # Step 1: Extract the list of directive prefixes from the children of the node
        used_directives = []
        for child in node.children:
            used_directives.append(child.directive)

        # Step 2: Filter out the used directives from the creative_directives list
        available_directives = []
        for directive in list(self.creative_directives.keys()):
            if directive not in used_directives:
                available_directives.append(directive)

        # Step 3: Randomly choose one of the available directives
        # Check if there arent any available directives
        if not available_directives:
            print("\033[91m!!! Error: No available directives. Doing no expansion.\033[0m")
            return None
        else:
            directive = random.choice(available_directives)

        try:
            if node.is_problem_statement:
                new_idea = self._generate_idea_from_problem(
                    node.content,
                    node.constraints,
                    directive,
                )
            else:
                new_idea = self._generate_idea(
                    self.root.content, self.root.constraints, node.content, directive
                )

            child = Node(
                new_idea, is_problem_statement=False, parent=node, directive=directive
            )
            node.children.append(child)

            print(
                "\033[94m\033[1m\nNew idea generated \033[0m\033[94m"
                + child.get_position_info()
                + ":\033[0m"
            )
            print(new_idea + "\n")

            return child
        except Exception as e:
            print(f"\033[91mError in expand: {e}. Skipping this expansion.\033[0m")
            return None

    def _simulate(self, node):
        """
        Simulate the outcome of the current node.

        This function evaluates the quality of an idea represented by the node.
        For problem statement nodes, it returns a default score of 0.5 since there's nothing to evaluate.
        For idea nodes, it calls the evaluate_idea function to get a score.

        Args:
            node (Node): The node to simulate.

        Returns:
            float: The evaluation score of the node's idea.
        """
        if node.is_problem_statement:
            return 0.5
        else:
            return self._evaluate_idea(node.content)

    # -----------------------------------------------------------------------------

    def _backpropagate(self, node, score):
        """
        Backpropagate the evaluation score through the tree.

        This function updates the score and visit count of the current node
        and all its ancestors up to the root. It's a crucial part of the
        Monte Carlo Tree Search algorithm, allowing the tree to learn from
        the results of simulations.

        Args:
            node (Node): The starting node for backpropagation.
            score (float): The evaluation score to be propagated.
        """
        while node is not None:
            node.visits += 1
            node.score += score
            node = node.parent

    # -----------------------------------------------------------------------------

    def _generate_idea_from_problem(self, problem_statement, constraints, creative_directive):
        """
        Initial idea generation for problem statement root node.
        This uses a different prompt than the other generate_idea function.
        """
        instruction, explanation = self.creative_directives[creative_directive]
        prompt = get_generate_seed_idea_prompt(
            problem_statement,
            constraints,
            creative_directive,
            instruction,
            explanation,
        )
        system_prompt = get_idea_generation_system_prompt()
        max_retries = 2  # Adjust this value to change the number of retries
        for attempt in range(max_retries):
            try:
                return self._get_completion(
                    prompt,
                    system_prompt,
                    self.llm_generation_model,
                    max_tokens=600,
                    temperature=1,
                )
            except Exception:
                if attempt == max_retries - 1:  # If this is the last attempt
                    return False
        return False  # This line should never be reached, but it's here for completeness

    # -----------------------------------------------------------------------------

    def _generate_idea(
        self, problem_statement, constraints, current_idea, creative_directive
    ):
        """
        Generate an idea from a current idea.
        """
        instruction, explanation = self.creative_directives[creative_directive]
        prompt = get_generate_idea_prompt(
            problem_statement,
            constraints,
            current_idea,
            creative_directive,
            instruction,
            explanation,
        )
        system_prompt = get_idea_generation_system_prompt()
        max_retries = 2  # Adjust this value to change the number of retries
        for attempt in range(max_retries):
            try:
                return self._get_completion(
                    prompt,
                    system_prompt,
                    self.llm_generation_model,
                    max_tokens=600,
                    temperature=1,
                )
            except Exception:
                if attempt == max_retries - 1:  # If this is the last attempt
                    return False
        return False  # This line should never be reached, but it's here for completeness

    # -----------------------------------------------------------------------------

    def _evaluate_idea(self, idea):
        """
        Evaluate an idea using dynamic evaluation criteria.
        """

        DEFAULT_SCORE = 0.5

        prompt = get_evaluate_idea_prompt(
            self.root.content, self.root.constraints, idea, self.evaluation_criteria
        )
        system_prompt = get_idea_evaluation_system_prompt(self.evaluation_criteria)

        attempts = 0
        max_retries = 3  # Adjust this value to change the number of retries
        for _ in range(max_retries):
            try:
                response = self._get_completion(
                    prompt,
                    system_prompt,
                    self.llm_evaluation_model,
                    max_tokens=300,
                    temperature=0.6,
                )
                # response pre-processing, taking out any asterisks
                # because of Markdown formatting by the LLM
                response = response.replace("*", "").strip()

                import re

                # Check if all criterion labels exist with numeric scores using regex
                all_criteria_present = True
                for criterion in self.evaluation_criteria:
                    criterion_label = f"{criterion['name']} Score:"
                    pattern = re.escape(criterion_label) + r"\s+(\d+(\.\d+)?)"
                    if not re.search(pattern, response):
                        all_criteria_present = False
                        break

                if all_criteria_present:
                    break
                else:
                    print("\033[91mRetry Evaluation: Scores not found in response.\033[0m")
                    attempts += 1

            except Exception as e:
                print(f"\033[91mError getting response: {e}. Retrying...\033[0m")
                attempts += 1

            if self.is_using_groq:
                time.sleep(1)

        else:
            print(
                f"\033[91m!!! Error: Failed to evaluate idea after {max_retries} attempts. Defaulting to {DEFAULT_SCORE}.\033[0m"
            )
            return DEFAULT_SCORE

        print("\n\033[94m\033[1mIdea Evaluation:\033[0m")
        print(response + "\n")

        # Parse scores
        scores = {}
        for criterion in self.evaluation_criteria:
            criterion_label = f"{criterion['name']} Score:"
            pattern = re.escape(criterion_label) + r"\s*(\d+(\.\d+)?)"
            match = re.search(pattern, response)
            if match:
                scores[criterion["name"]] = float(match.group(1)) / 100
            else:
                print(
                    f"\033[91m!!! Error: Failed to parse {criterion['name']} score. Defaulting to {DEFAULT_SCORE}.\033[0m"
                )
                scores[criterion["name"]] = DEFAULT_SCORE

        # Calculate average score
        average_score = sum(scores.values()) / len(scores)
        return average_score

    # -----------------------------------------------------------------------------

    def _get_completion(
        self,
        prompt,
        system_prompt,
        model,
        max_tokens=250,
        temperature=0.8,
    ):
        """
        Get a completion from the LLM API.
        """
        try:
            print("\033[93m --> Getting completion...\033[0m")
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            print("\033[92m --> Completion received.\033[0m")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\033[91m\033[1m!!! Error in API completion call: {e}\033[0m")
            raise

    # -----------------------------------------------------------------------------

    def _prune_node_by_id(self, id) -> None:
        if Node.prune_node_by_id(self.root, id):
            print(f"Node with ID {id} has been pruned.")
        else:
            print(f"Node with ID {id} not found or is the root node.")
