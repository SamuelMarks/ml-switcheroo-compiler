"""Update test and documentation coverage badges in the README.md file.

This script retrieves test coverage by running the coverage tool and generating a JSON
report, gets documentation coverage, formats these values, determines appropriate badge
colors, and updates the corresponding shields.io badge URLs in the README.md file.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from typing import Optional, Union


def _extract_expected_arg_names(args: ast.arguments) -> list[str]:
    """Extract parameter names (excluding `self`/`cls`, including `*args`/`**kwargs`) from `ast.arguments`.

    Args:
        args (ast.arguments): AST function arguments node.

    Returns:
        list[str]: Parameter names expected in the docstring `Args:` block.
    """
    names: list[str] = []
    all_pos = list(args.posonlyargs) + list(args.args)
    for idx, arg in enumerate(all_pos):
        if idx == 0 and arg.arg in ("self", "cls"):
            continue
        names.append(arg.arg)
    if args.vararg is not None:
        names.append(args.vararg.arg)
    for kwarg in args.kwonlyargs:
        names.append(kwarg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def _has_non_none_return(returns: Optional[ast.expr]) -> bool:
    """Check whether a function return annotation is present and not `None`.

    Args:
        returns (Optional[ast.expr]): Function return annotation AST expression.

    Returns:
        bool: True if a non-`None` return annotation is present.
    """
    if returns is None:
        return False
    if isinstance(returns, ast.Constant) and returns.value is None:
        return False
    return not (isinstance(returns, ast.Name) and returns.id == "None")


def _is_function_doc_complete(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
    """Verify that a function/method node has a valid docstring with documented args and return.

    Args:
        node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): AST function definition node.

    Returns:
        bool: True if the docstring exists and satisfies argument and return documentation rules.
    """
    doc = ast.get_docstring(node)
    if not doc or not doc.strip():
        return False
    if "(" + "Any)" in doc or "(" + "object)" in doc:
        return False
    expected_args = _extract_expected_arg_names(node.args)
    if expected_args and "Args:" not in doc:
        return False
    for arg_name in expected_args:
        if not re.search(rf"\b{re.escape(arg_name)}\b", doc):
            return False
    if _has_non_none_return(node.returns):
        if "Returns:" not in doc and "Yields:" not in doc:
            return False
    return True


class _DocVisitor(ast.NodeVisitor):
    """Visitor to count AST nodes and their docstrings."""

    def __init__(self) -> None:
        """Initialize the visitor."""
        self.total_nodes: int = 0
        self.nodes_with_docstrings: int = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition node.

        Args:
            node (ast.ClassDef): The class definition node.
        """
        self.total_nodes += 1
        doc = ast.get_docstring(node)
        if doc and doc.strip():
            self.nodes_with_docstrings += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition node.

        Args:
            node (ast.FunctionDef): The function definition node.
        """
        self.total_nodes += 1
        if _is_function_doc_complete(node):
            self.nodes_with_docstrings += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition node.

        Args:
            node (ast.AsyncFunctionDef): The async function definition node.
        """
        self.total_nodes += 1
        if _is_function_doc_complete(node):
            self.nodes_with_docstrings += 1
        self.generic_visit(node)


def get_color(pct: float) -> str:
    """Determine the badge color based on the coverage percentage.

    Args:
        pct (float): The coverage percentage.

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
        cov (float): The coverage percentage to format.

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
        Optional[float]: The total test coverage percentage or None if no data is available.
    """
    try:
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

    Parses all Python files (including `__init__.py` and private helpers) in the source
    directory to check for the presence of docstrings on modules, classes, and functions,
    including argument and return documentation.

    Returns:
        float: The documentation coverage percentage.
    """
    total_nodes: int = 0
    nodes_with_docstrings: int = 0

    src_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "ml_switcheroo_compiler")
    if not os.path.exists(src_dir):
        return 0.0

    for root, _, files in os.walk(src_dir):
        for file in files:
            if not file.endswith(".py"):
                continue
            path: str = os.path.join(root, file)
            with open(path, encoding="utf-8") as f:
                try:
                    tree: ast.Module = ast.parse(f.read(), filename=path)
                    total_nodes += 1
                    mod_doc = ast.get_docstring(tree)
                    if mod_doc and mod_doc.strip():
                        nodes_with_docstrings += 1
                    visitor = _DocVisitor()
                    visitor.visit(tree)
                    total_nodes += visitor.total_nodes
                    nodes_with_docstrings += visitor.nodes_with_docstrings
                except Exception:
                    pass

    return 0.0 if total_nodes == 0 else (nodes_with_docstrings / total_nodes) * 100.0


def update_readme() -> None:
    """Update the coverage badges in the README.md file.

    Reads the README.md file, retrieves the current test and documentation coverage
    percentages, formats them, and replaces the existing shields.io badge markdown
    links with updated values and colors.
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
