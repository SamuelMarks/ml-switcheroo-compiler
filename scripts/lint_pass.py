"""Linting script to enforce lack of empty passes."""

import ast
import glob
import sys


def check_for_pass_in_functions(directory: str) -> None:
    """Check for empty pass statements in functions within a given directory."""
    violations = []

    for filepath in glob.glob(directory + "/**/*.py", recursive=True):
        with open(filepath) as f:
            try:
                tree = ast.parse(f.read(), filename=filepath)
            except SyntaxError:
                continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in node.body:
                    if isinstance(child, ast.Pass):
                        violations.append(filepath + ":" + str(child.lineno))

    if violations:
        print("Linting failed.")
        for v in violations:
            print(v)
        sys.exit(1)
    else:
        print("Linting passed.")
        sys.exit(0)


if __name__ == "__main__":
    check_for_pass_in_functions("src/ml_switcheroo_compiler/ops")
