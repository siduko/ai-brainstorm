import json
import os
from pathlib import Path

from node import Node


def print_tree(node, indent="", include_full_content=False, single_node=False):
    if include_full_content:
        print(f"{indent}Content:\n{node.content}")
    else:
        print(f"{indent}Content:\n{node.content[:50]}...")  # Print first 50 chars of content
    print("")
    print(f"{indent}\033[1m\033[92mRelative Score: {node.get_relative_score()}\033[0m")
    print(f"{indent}Depth: {node.depth}")
    print(f"{indent}Visits: {node.visits}")
    print(f"{indent}Directive: {node.directive}")
    print(f"{indent}Num Children: {len(node.children)}")
    print(f"{indent}ID: {node.id}")
    print()

    if single_node:
        return
    for child in node.children:
        print_tree(child, indent + "  ")


def save_tree_to_json(root_node, filename):

    def node_to_dict(node):
        node_dict = node.to_dict()
        node_dict["children"] = [node_to_dict(child) for child in node.children]
        return node_dict

    tree_dict = node_to_dict(root_node)

    with open(os.path.join("outputs", filename), "w", encoding="utf-8") as f:
        json.dump(tree_dict, f, ensure_ascii=False, indent=2)


def load_tree_from_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        tree_dict = json.load(f)

    def dict_to_node(node_dict, parent=None):
        node = Node(
            content=node_dict["content"],
            is_problem_statement=node_dict["is_problem_statement"],
            parent=parent,
            directive=node_dict["directive"],
            constraints=node_dict["constraints"],
            id=node_dict["id"],
        )
        node.depth = node_dict["depth"]
        node.visits = node_dict["visits"]
        node.score = node_dict["score"]

        for child_dict in node_dict["children"]:
            child_node = dict_to_node(child_dict, parent=node)
            node.children.append(child_node)

        return node

    root_node = dict_to_node(tree_dict)
    return root_node


def save_tree_to_html(root_node, filename):
    def node_to_dict(node):
        return {
            "content": node.content,
            "relative_score": node.get_relative_score(),
            "children": [node_to_dict(child) for child in node.children],
        }

    tree_data = node_to_dict(root_node)

    template_path = Path(__file__).parent / "tree_template.html"
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    html_content = template.replace("{{tree_data}}", json.dumps(tree_data))

    output_path = Path("outputs") / filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML tree saved to {output_path}")
