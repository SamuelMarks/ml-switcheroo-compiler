#!/usr/bin/env python3
"""Docstring."""

import ast
import os
import sys

FORBIDDEN_IMPORTS = {
    "ir": ["jax", "torch", "mlx", "keras", "tensorflow", "numpy"],
    "ops": ["jax", "torch", "mlx", "keras", "tensorflow"],
    "transforms": ["jax", "torch", "mlx", "keras", "tensorflow"],
    "backends/jax.py": ["torch", "mlx", "keras", "tensorflow", "numpy"],
    "backends/pytorch.py": ["jax", "mlx", "keras", "tensorflow", "numpy"],
    "backends/mlx.py": ["jax", "torch", "keras", "tensorflow", "numpy"],
    "backends/keras.py": ["jax", "torch", "mlx", "tensorflow", "numpy"],
    "backends/tensorflow.py": ["jax", "torch", "mlx", "keras", "numpy"],
    "backends/base_generator.py": [
        "jax",
        "torch",
        "mlx",
        "keras",
        "tensorflow",
        "numpy",
    ],
}


class ImportVisitor(ast.NodeVisitor):
    """Docstring."""

    def __init__(self, filename: object) -> None:
        """Docstring."""
        self.filename = filename
        self.violations = []
        rel_path = os.path.relpath(filename, "src/ml_switcheroo")
        self.forbidden = []
        for path_prefix, forbidden_list in FORBIDDEN_IMPORTS.items():
            if rel_path.startswith(path_prefix):
                self.forbidden.extend(forbidden_list)

    def check_module(self, module_name: object, lineno: object) -> object:
        """Docstring."""
        if module_name is None:
            return
        base_module = module_name.split(".")[0]
        if base_module in self.forbidden:
            self.violations.append(
                f"{self.filename}:{lineno} - Forbidden import '{module_name}'"
            )

    def visit_Import(self, node: object) -> object:
        """Docstring."""
        for alias in node.names:
            self.check_module(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: object) -> object:
        """Docstring."""
        self.check_module(node.module, node.lineno)
        self.generic_visit(node)


def main() -> object:
    """Docstring."""
    has_violations = False
    for root, _, files in os.walk("src/ml_switcheroo"):
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
        print("Dependency leaks detected!")
        sys.exit(1)
    else:
        print("No dependency leaks detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
