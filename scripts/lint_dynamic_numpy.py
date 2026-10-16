"""Pre-commit hook guarding against dynamic or indirect imports of numpy outside backends/numpy."""

from __future__ import annotations

import ast
import os
import sys

ALLOWED_MODULE_PATTERNS: tuple[str, ...] = (os.path.join("backends", "numpy"),)


def _is_prohibited_dynamic_numpy_call(node: ast.AST) -> str | None:
    """Check if an AST node is a dynamic import call for numpy.

    Args:
        node (ast.AST): AST node to inspect.

    Returns:
        str | None: Prohibited module string if matched, else None.
    """
    if not isinstance(node, ast.Call) or not node.args:
        return None
    is_dynamic_import = (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module") or (isinstance(node.func, ast.Name) and node.func.id in ("import_module", "__import__"))
    if not is_dynamic_import:
        return None
    first_arg = node.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        if first_arg.value == "numpy" or first_arg.value.startswith("numpy."):
            return first_arg.value
    return None


def check_for_prohibited_numpy_imports(directory: str) -> list[str]:
    """Scan Python files for dynamic numpy import calls like importlib.import_module('numpy').

    Args:
        directory (str): The root source directory to scan.

    Returns:
        list[str]: Violations list formatted as file:line.
    """
    violations: list[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".py"):
                continue
            file_path: str = os.path.join(root, file)
            if any(pattern in file_path for pattern in ALLOWED_MODULE_PATTERNS):
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    tree: ast.AST = ast.parse(f.read(), filename=file_path)
            except Exception:
                continue

            for node in ast.walk(tree):
                mod_val = _is_prohibited_dynamic_numpy_call(node)
                if mod_val is not None and hasattr(node, "lineno"):
                    violations.append(f"{file_path}:{node.lineno}: Prohibited dynamic numpy import '{mod_val}' outside ml-switcheroo-compiler numpy/scipy module.")
    return violations


def main() -> int:
    """Run dynamic numpy import guard.

    Returns:
        int: 0 if passed, 1 if violations found.
    """
    directory: str = "src/ml_switcheroo_compiler"
    violations: list[str] = check_for_prohibited_numpy_imports(directory)
    if violations:
        print("Prohibited dynamic numpy imports detected:")
        for v in violations:
            print(f"  {v}")
        return 1
    print("Dynamic numpy import guard passed: No unauthorized dynamic numpy imports found.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
