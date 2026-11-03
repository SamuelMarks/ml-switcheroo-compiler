"""AST-based static typing and anti-reflection linter for ml-switcheroo-compiler."""

from __future__ import annotations

import ast
import os
import sys
from collections.abc import Sequence

DEFAULT_TARGET_DIRS: tuple[str, ...] = (
    "src/ml_switcheroo_compiler",
    "scripts",
    "tests",
)

_FORBIDDEN_BUILTIN_CALLS: frozenset[str] = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "vars",
        "globals",
        "locals",
        "cast",
    }
)

_FORBIDDEN_DUNDER_ATTRS: frozenset[str] = frozenset(
    {
        "__dict__",
        "__mro__",
        "__subclasses__",
        "__setattr__",
        "__getattribute__",
        "__delattr__",
        "__class__",
    }
)

_FORBIDDEN_DUNDER_CALLS: frozenset[str] = frozenset(
    {
        "__call__",
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__matmul__",
        "__rmatmul__",
        "__truediv__",
        "__rtruediv__",
        "__floordiv__",
        "__rfloordiv__",
        "__mod__",
        "__rmod__",
        "__pow__",
        "__rpow__",
        "__neg__",
        "__pos__",
        "__abs__",
        "__invert__",
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__setattr__",
        "__getattribute__",
        "__delattr__",
        "__array__",
        "__index__",
        "__await__",
    }
)


def _scan_strict_ast(file_path: str, tree: ast.Module) -> list[str]:
    """Scan an AST module for prohibited reflection, cast, and direct dunder calls.

    Args:
        file_path (str): File path for error reporting.
        tree (ast.Module): Parsed Python AST module.

    Returns:
        list[str]: List of violation messages found in the AST.
    """
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_BUILTIN_CALLS:
                violations.append(f"{file_path}:{node.lineno}: prohibited reflection/cast call '{node.func.id}()'")
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id in ("typing", "typing_extensions") and node.func.attr == "cast":
                    violations.append(f"{file_path}:{node.lineno}: prohibited 'typing.cast()' call")
                elif node.func.attr in _FORBIDDEN_DUNDER_CALLS:
                    is_super = isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Name) and node.func.value.func.id == "super"
                    if not is_super:
                        violations.append(f"{file_path}:{node.lineno}: prohibited direct dunder call '.{node.func.attr}()'")
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_DUNDER_ATTRS:
            violations.append(f"{file_path}:{node.lineno}: prohibited dunder attribute '.{node.attr}'")
    return violations


def scan_file(file_path: str, force_strict_reflection: bool = True) -> list[str]:
    """Scan a single Python source file for typing and reflection violations.

    Args:
        file_path (str): Path to the Python file to scan.
        force_strict_reflection (bool): Retained for CLI compatibility; all files are strictly scanned.

    Returns:
        list[str]: Violation messages discovered in the file.
    """
    del force_strict_reflection
    with open(file_path, encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return []

    return _scan_strict_ast(file_path, tree)


def scan_directory(target_dir: str) -> list[str]:
    """Recursively scan a directory or single file for typing and reflection violations.

    Args:
        target_dir (str): Directory or file path to inspect.

    Returns:
        list[str]: Aggregated violation messages across all inspected Python files.
    """
    if not os.path.exists(target_dir):
        return []
    if os.path.isfile(target_dir):
        return scan_file(target_dir) if target_dir.endswith(".py") else []

    violations: list[str] = []
    for root, _, files in sorted(os.walk(target_dir)):
        for fname in sorted(files):
            if fname.endswith(".py"):
                violations.extend(scan_file(os.path.join(root, fname)))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the typing and anti-reflection linter CLI across target paths.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments; defaults to `DEFAULT_TARGET_DIRS`.

    Returns:
        int: Exit code (0 if clean, 1 if violations are found).
    """
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    targets: Sequence[str] = raw_args if raw_args else DEFAULT_TARGET_DIRS

    all_violations: list[str] = []
    for target in targets:
        all_violations.extend(scan_directory(str(target)))

    if all_violations:
        for violation in all_violations:
            print(violation, file=sys.stderr)
        return 1
    print("Strict typing and anti-reflection check passed (0 violations).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
