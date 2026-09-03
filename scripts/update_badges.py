"""Update test and documentation coverage badges in the README.md file.

This script retrieves test coverage by running the coverage tool and generating a JSON
report, gets documentation coverage, formats these values, determines appropriate badge
colors, and updates the corresponding shields.io badge URLs in the README.md file.
"""

import json
import os
import re
import subprocess
import sys
from typing import Optional


def get_color(pct: float) -> str:
    """Determine the badge color based on the coverage percentage.

    Args:
        pct (float | int): The coverage percentage.

    Returns:
        str: The color name corresponding to the coverage range.
    """
    threshold_brightgreen: float = 100.0
    threshold_green: float = 90.0
    threshold_yellowgreen: float = 80.0
    threshold_yellow: float = 70.0
    threshold_orange: float = 60.0

    if pct >= threshold_brightgreen:
        return "brightgreen"
    if pct >= threshold_green:
        return "green"
    if pct >= threshold_yellowgreen:
        return "yellowgreen"
    if pct >= threshold_yellow:
        return "yellow"
    if pct >= threshold_orange:
        return "orange"
    return "red"


def format_cov(cov: float) -> str:
    """Format a coverage percentage value into a string.

    If the coverage is a whole number, it is formatted as an integer. Otherwise, it is
    formatted as a float with one decimal place.

    Args:
        cov (float | int): The coverage percentage to format.

    Returns:
        str: The formatted coverage percentage.
    """
    if int(cov) == cov:
        return str(int(cov))
    return f"{cov:.1f}"


def get_test_coverage() -> Optional[float]:
    """Retrieve the total test coverage percentage from the coverage tool.

    Runs the `coverage json` command to generate a report, parses the resulting JSON
    file, and extracts the total percentage covered. If an error occurs, it defaults
    to None to avoid overriding existing coverage with 0.0 improperly.

    Returns:
        float | None: The total test coverage percentage or None if no data is available.
    """
    try:
        # Run coverage json, check=True will raise an exception if it fails (e.g. no .coverage file)
        subprocess.run(
            [sys.executable, "-m", "coverage", "json", "-o", "coverage.json"],
            check=True,
            capture_output=True,
        )
        with open("coverage.json") as f:
            data = json.load(f)

        if os.path.exists("coverage.json"):
            os.remove("coverage.json")

        return float(data["totals"]["percent_covered"])
    except Exception as e:
        print(f"Warning: Could not parse test coverage: {e}")
        return None


def get_doc_coverage() -> float:
    """Retrieve the documentation coverage percentage using an AST linter.

    Parses the Python files in the source directory to check for the presence of
    docstrings on modules, classes, and public functions/methods.

    Returns:
        float: The documentation coverage percentage.
    """
    import ast

    class DocVisitor(ast.NodeVisitor):
        """Visitor to count AST nodes and their docstrings."""

        def __init__(self) -> None:
            """Initialize the visitor.

            Returns:
                None
            """
            self.total_nodes: int = 0
            self.nodes_with_docstrings: int = 0

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            """Visit a class definition node.

            Args:
                node (ast.ClassDef): The class definition node.

            Returns:
                None
            """
            if not node.name.startswith("_"):
                self.total_nodes += 1
                if ast.get_docstring(node):
                    self.nodes_with_docstrings += 1
                # Visit methods
                self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            """Visit a function definition node.

            Args:
                node (ast.FunctionDef): The function definition node.

            Returns:
                None
            """
            if not node.name.startswith("_"):
                self.total_nodes += 1
                if ast.get_docstring(node):
                    self.nodes_with_docstrings += 1
            # Do not visit inner functions/classes

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            """Visit an async function definition node.

            Args:
                node (ast.AsyncFunctionDef): The async function definition node.

            Returns:
                None
            """
            if not node.name.startswith("_"):
                self.total_nodes += 1
                if ast.get_docstring(node):
                    self.nodes_with_docstrings += 1
            # Do not visit inner functions/classes

    total_nodes: int = 0
    nodes_with_docstrings: int = 0

    # We can walk the source directory
    src_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "ml_switcheroo_compiler")

    if not os.path.exists(src_dir):
        return 0.0

    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                # Similar to interrogate -i, ignore module docstrings for __init__.py
                if file == "__init__.py":
                    continue

                path: str = os.path.join(root, file)
                with open(path, encoding="utf-8") as f:
                    try:
                        content: str = f.read()
                        tree: ast.Module = ast.parse(content, filename=path)

                        # Check module docstring
                        total_nodes += 1
                        if ast.get_docstring(tree):
                            nodes_with_docstrings += 1

                        visitor: DocVisitor = DocVisitor()
                        visitor.visit(tree)
                        total_nodes += visitor.total_nodes
                        nodes_with_docstrings += visitor.nodes_with_docstrings

                    except Exception:
                        pass

    if total_nodes == 0:
        return 0.0
    return (nodes_with_docstrings / total_nodes) * 100.0


def update_readme() -> None:
    """Update the coverage badges in the README.md file.

    Reads the README.md file, retrieves the current test and documentation coverage
    percentages, formats them, and replaces the existing shields.io badge markdown
    links with updated values and colors.

    Returns:
        None
    """
    if not os.path.exists("README.md"):
        return

    test_cov: Optional[float] = get_test_coverage()
    doc_cov: float = get_doc_coverage()

    with open("README.md") as f:
        content: str = f.read()

    if test_cov is not None:
        test_str: str = format_cov(test_cov)
        test_color: str = get_color(test_cov)
        test_re: re.Pattern[str] = re.compile(
            r"\[?\!\[Test Coverage\]\(https://img\.shields\.io/badge/(?:[tT]est_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?",
        )
        content = test_re.sub(
            f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](#)",
            content,
        )

    if doc_cov is not None:
        doc_str: str = format_cov(doc_cov)
        doc_color: str = get_color(doc_cov)
        doc_re: re.Pattern[str] = re.compile(
            r"\[?\!\[Doc Coverage\]\(https://img\.shields\.io/badge/(?:[dD]oc_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?",
        )
        content = doc_re.sub(
            f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)](#)",
            content,
        )

    with open("README.md", "w") as f:
        f.write(content)


if __name__ == "__main__":
    update_readme()
