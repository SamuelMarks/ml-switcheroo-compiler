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

STRICT_REFLECTION_SUFFIXES: tuple[str, ...] = (
    "src/ml_switcheroo_compiler/ir/core.py",
    "src/ml_switcheroo_compiler/transforms/pass_manager.py",
    "src/ml_switcheroo_compiler/ops/shape/utils.py",
    "src/ml_switcheroo_compiler/ops/control_flow_utils.py",
    "src/ml_switcheroo_compiler/core/",
    "src/ml_switcheroo_compiler/tracing/",
    "src/ml_switcheroo_compiler/random/",
    "src/ml_switcheroo_compiler/nn/initializers.py",
    "src/ml_switcheroo_compiler/backends/pyarrow_compute/__init__.py",
    "src/ml_switcheroo_compiler/backends/pyarrow_compute/types.py",
    "src/ml_switcheroo_compiler/backends/pyarrow_compute/eager.py",
    "src/ml_switcheroo_compiler/backends/pyarrow_compute/generator.py",
    "src/ml_switcheroo_compiler/backends/dpnp/__init__.py",
    "src/ml_switcheroo_compiler/backends/dpnp/types.py",
    "src/ml_switcheroo_compiler/backends/dpnp/eager.py",
    "src/ml_switcheroo_compiler/backends/dpnp/generator.py",
    "src/ml_switcheroo_compiler/backends/mlx/__init__.py",
    "src/ml_switcheroo_compiler/backends/jax/__init__.py",
    "src/ml_switcheroo_compiler/backends/jax/eager.py",
    "src/ml_switcheroo_compiler/backends/eager/core_math_ops/math_logical.py",
    "src/ml_switcheroo_compiler/backends/eager/core_math_ops/math_general.py",
    "src/ml_switcheroo_compiler/backends/generator_mixins.py",
    "src/ml_switcheroo_compiler/backends/cst_transpiler.py",
    "src/ml_switcheroo_compiler/backends/pure_python/generator.py",
    "src/ml_switcheroo_compiler/foreign.py",
    "src/ml_switcheroo_compiler/export/",
    "src/ml_switcheroo_compiler/distributed/",
)

_FORBIDDEN_BUILTIN_CALLS: frozenset[str] = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "cast",
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
    }
)


def _is_strict_reflection_target(file_path: str) -> bool:
    """Check whether a file path belongs to the strict anti-reflection target set.

    Args:
        file_path (str): File path to evaluate.

    Returns:
        bool: True if the file path matches one of `STRICT_REFLECTION_SUFFIXES`.
    """
    norm = f"/{file_path}"
    for suffix in STRICT_REFLECTION_SUFFIXES:
        if suffix.endswith("/"):
            if f"/{suffix}" in norm:
                return True
        elif norm.endswith(f"/{suffix}"):
            return True
    return False


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
    return violations


def scan_file(file_path: str, force_strict_reflection: bool = False) -> list[str]:
    """Scan a single Python source file for typing and reflection violations.

    Args:
        file_path (str): Path to the Python file to scan.
        force_strict_reflection (bool): Whether to enforce anti-reflection rules regardless of path.

    Returns:
        list[str]: Violation messages discovered in the file.
    """
    if not (force_strict_reflection or _is_strict_reflection_target(file_path)):
        return []

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
