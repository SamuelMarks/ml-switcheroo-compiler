"""Enforce backend isolation rules."""

import ast
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from typing import get_args

from ml_switcheroo_compiler.backends.registry import BackendName

ALL_BACKENDS = set(get_args(BackendName))

FORBIDDEN_IMPORTS = {}
for backend in ALL_BACKENDS:
    if backend == "torch":
        file_path = "src/ml_switcheroo_compiler/backends/pytorch.py"
    else:
        file_path = f"src/ml_switcheroo_compiler/backends/{backend}.py"

    FORBIDDEN_IMPORTS[file_path] = ALL_BACKENDS - {backend}


def _check_imports(node: ast.AST, forbidden: set[str], file_path: str) -> bool:
    has_error = False
    if isinstance(node, ast.Import):
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module in forbidden:
                print(f"[{file_path}:{node.lineno}] Forbidden import detected: {alias.name}")
                has_error = True
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module in forbidden:
                msg = f"[{file_path}:{node.lineno}] Forbidden import detected: "
                msg += f"from {node.module} import ..."
                print(msg)
                has_error = True
    return has_error


def check_file(file_path: str) -> bool:
    """Check a file for forbidden imports.

    Args:
        file_path (str): The file path to check.

    Returns:
        bool: True if file is valid, False otherwise.
    """
    if not os.path.exists(file_path):
        return True

    # Convert to standard format
    file_path = os.path.normpath(file_path)

    forbidden = FORBIDDEN_IMPORTS.get(file_path, set())
    if (
        "src/ml_switcheroo_compiler/core/" in file_path
        or "src/ml_switcheroo_compiler/ops/" in file_path
    ):
        forbidden = ALL_BACKENDS.copy()

    if not forbidden:
        return True

    with open(file_path, encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return False

    has_error = False
    for node in ast.walk(tree):
        if _check_imports(node, forbidden, file_path):
            has_error = True

    return not has_error


if __name__ == "__main__":
    files_to_check = sys.argv[1:]
    success = True
    for file_path in files_to_check:
        if not check_file(file_path):
            success = False
    if not success:
        sys.exit(1)
