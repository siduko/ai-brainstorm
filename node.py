import math
import random
import uuid


class Node:
    """
    A Node class representing an idea in a Monte Carlo Tree Search (MCTS) for brainstorming.

    This class encapsulates the structure and behavior of nodes in the search tree,
    including problem statements, generated ideas, and their evaluations.

    Attributes:
        QUALITY_THRESHOLD (float): The score threshold for considering a node as high-quality.
        C_PARAM (float): Exploration parameter for the UCT formula in MCTS.
        MAX_DEPTH (int): Maximum depth allowed for the search tree.
        MAX_CHILDREN (int): Maximum number of children allowed for each node.

    Each node contains:
    - Content of the idea
    - Constraints applicable to the idea
    - Reference to parent and child nodes
    - Visit count and score for MCTS
    - Depth in the tree and position among siblings
    - Unique identifier and parent's identifier
    - The creative directive used to generate this idea (if applicable)

    Methods are provided for score calculation, position information, and
    determining if a node is terminal or fully expanded.
    """

    QUALITY_THRESHOLD = 0.87
    C_PARAM = 1.414
    MAX_DEPTH = 5
    MAX_CHILDREN = 5

    def __init__(
        self,
        content,
        is_problem_statement=False,
        parent=None,
        directive=None,
        constraints=None,
        id=None,
    ):
        self.content = content
        self.constraints = constraints
        self.is_problem_statement = is_problem_statement
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0
        self.directive = directive
        self.depth = 0 if parent is None else parent.depth + 1
        self.child_number = len(parent.children) + 1 if parent else 0
        self.id = id if id else str(uuid.uuid4())
        self.parent_id = parent.id if parent else None

    def get_relative_score(self):
        if self.visits == 0:
            return self.score
        return self.score / self.visits

    def get_position_info(self):
        depth = self.depth
        child_number = self.child_number
        total_siblings = len(self.parent.children) if self.parent else 0
        return f"(Child {child_number}/{total_siblings} at depth {depth})"

    def is_terminal(self):
        if self.depth >= self.MAX_DEPTH:
            return True
        if self.visits == 0:
            return False
        return (self.score / self.visits) >= self.QUALITY_THRESHOLD

    def fully_expanded(self):
        """

        Somewhat arbitrary rationale for whether or not this node is fully expanded.
        Originally this was based on the number of children matching the number of creative directives.
        But this can impose additional inference cost without enough depth (based on global iterations).
        So we're just going to impose a fixed number of children for now.

        """
        return len(self.children) >= self.MAX_CHILDREN

    def best_child(self):
        if not self.children:
            return None

        if self.visits == 0:  # This is the root node on the first call
            # Use only the children's scores for selection
            best_score = float("-inf")
            best_children = []
            for child in self.children:
                if child.visits == 0:
                    continue
                if child.score > best_score:
                    best_score = child.score
                    best_children = [child]
                elif child.score == best_score:
                    best_children.append(child)

            if best_children:
                return random.choice(best_children)
            else:
                return random.choice(self.children)

        # Normal UCB1 selection for non-root nodes or root after first visit
        """
        
        The UCB1 (Upper Confidence Bound 1) selection in the `best_child` method is a key part of the MCTS algorithm.
        Here's a simple explanation of what it's doing:

        1. **Balancing exploration and exploitation**: UCB1 helps choose which child node to explore next, balancing between:
            - Exploitation: Choosing nodes that have performed well so far
            - Exploration: Giving a chance to less-visited nodes that might be promising

        2. **The formula**: For each child, it calculates a score using this formula:
            ```
            (child.score / child.visits) + C_PARAM * sqrt((2 * log(self.visits)) / child.visits)
            ```

        3. **Components of the formula**:
            - `child.score / child.visits`: This is the average score of the child (exploitation)
            - `C_PARAM * sqrt((2 * log(self.visits)) / child.visits)`: This is the exploration bonus
            - `C_PARAM` (usually √2) controls the balance between exploration and exploitation

        4. **Selection**: It calculates this score for all visited children and chooses the one with the highest score.

        5. **Handling edge cases**: 
            - If a child hasn't been visited, it's not included in the calculation
            - If no children have been visited, it randomly selects a child

        This method ensures that the algorithm doesn't just focus on the currently best-performing paths but also explores
        less-visited options that might turn out to be better in the long run.

        """
        choices_weights = [
            (child.score / child.visits)
            + self.C_PARAM * ((2 * math.log(self.visits) / child.visits) ** 0.5)
            for child in self.children
            if child.visits > 0
        ]

        if not choices_weights:
            return random.choice(self.children)

        return self.children[choices_weights.index(max(choices_weights))]

    @staticmethod
    def prune_node_by_id(root: "Node", id: str) -> bool:
        """
        Prunes a node with the given id from the tree.

        Args:
            root (Node): The root node of the tree.
            id (str): The id of the node to prune.

        Returns:
            bool: True if a node was pruned, False otherwise.
        """
        if root.id == id:
            # Can't prune the root
            return False

        return root._prune_child(id)

    def _prune_child(self, id: str) -> bool:
        for i, child in enumerate(self.children):
            if child.id == id:
                del self.children[i]
                return True
            if child._prune_child(id):
                return True
        return False

    def to_dict(self):
        node_dict = {
            "id": self.id,
            "parent_id": self.parent_id,
            "content": self.content,
            "is_problem_statement": self.is_problem_statement,
            "constraints": self.constraints,
            "directive": self.directive,
            "visits": self.visits,
            "score": self.score,
            "depth": self.depth,
            "child_number": self.child_number,
        }
        return node_dict
