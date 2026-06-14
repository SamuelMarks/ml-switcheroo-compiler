#!/usr/bin/env python3
"""Enforces import restrictions across different modules in the ml_switcheroo_compiler package.

This script scans the codebase and uses AST parsing to ensure that backend-specific
libraries (like PyTorch, JAX, TensorFlow) are not imported in backend-agnostic areas
such as the intermediate representation (IR) or core transforms.
"""

import ast
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from typing import get_args
from ml_switcheroo_compiler.backends.registry import BackendName

ALL_BACKENDS = list(get_args(BackendName))

FORBIDDEN_IMPORTS = {
    "ir": ALL_BACKENDS,
    "core": ALL_BACKENDS,
    "ops": ALL_BACKENDS,
    "transforms": ALL_BACKENDS,
    "backends/base_generator.py": ALL_BACKENDS,
}

# For each backend file, forbid importing all OTHER backends
# E.g. backends/jax.py cannot import torch, mlx, keras, tensorflow, numpy, cupy, dask
for backend in ALL_BACKENDS:
    if backend == "torch":
        file_path = "backends/pytorch.py"
    else:
        file_path = f"backends/{backend}.py"

    FORBIDDEN_IMPORTS[file_path] = [b for b in ALL_BACKENDS if b != backend]


class ImportVisitor(ast.NodeVisitor):
    """An AST visitor that checks Python files for forbidden imports.

    Attributes:
    filename (object): The path to the file being analyzed.
    violations (list): A list of violation messages found during analysis.
    forbidden (list): A list of forbidden module names for the current file.
    """

    def __init__(self, filename: object) -> None:
        """Visit the node to check for forbidden imports."""
        self.filename = filename
        self.violations = []
        rel_path = os.path.relpath(filename, "src/ml_switcheroo_compiler")
        self.forbidden = []
        for path_prefix, forbidden_list in FORBIDDEN_IMPORTS.items():
            if rel_path.startswith(path_prefix):
                self.forbidden.extend(forbidden_list)

    def check_module(self, module_name: object, lineno: object) -> object:
        """Visit the node to check for forbidden imports."""
        if module_name is None:
            return
        base_module = module_name.split(".")[0]
        if base_module in self.forbidden:
            self.violations.append(
                f"{self.filename}:{lineno} - Forbidden import '{module_name}'",
            )

    def visit_Import(self, node: object) -> object:
        """Visit the node to check for forbidden imports."""
        for alias in node.names:
            self.check_module(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: object) -> object:
        """Visit the node to check for forbidden imports."""
        self.check_module(node.module, node.lineno)
        self.generic_visit(node)


def main() -> object:
    """Scans the codebase for forbidden imports and exits with an appropriate status code.

    Returns:
    object: None.
    """
    has_violations = False
    for root, _, files in os.walk("src/ml_switcheroo_compiler"):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=path)
                    visitor = ImportVisitor(path)
                    visitor.visit(tree)
                    if visitor.violations:
                        has_violations = True
                        for v in visitor.violations:
                            print(v)
                except Exception:
                    pass
    if has_violations:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
