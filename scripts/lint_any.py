"""AST-based linter to detect Any, object usages, disguised Any aliases, stringified Any/object casts, and type-ignore comments across src/, scripts/, and tests/."""

from __future__ import annotations

import argparse
import ast
import os
import sys
import tokenize
import typing
from io import StringIO

DEFAULT_TARGET_DIRS: tuple[str, ...] = (
    "src/ml_switcheroo_compiler",
    "scripts",
    "tests",
)


def _annotation_contains_any_or_object(node: ast.AST | None) -> list[str]:
    """Inspect a type annotation AST subtree for Any or object usages.

    Args:
        node (ast.AST | None): Annotation AST node to inspect.

    Returns:
        list[str]: List of detected prohibited type names ('Any', 'typing.Any', 'object').
    """
    if node is None:
        return []
    hits: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in ("Any", "object", "AnyType"):
            hits.append(sub.id)
        elif isinstance(sub, ast.Attribute) and sub.attr in ("Any", "object"):
            hits.append(f"typing.{sub.attr}" if sub.attr == "Any" else "builtins.object")
    return hits


def _check_function_annotations(file_path: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Inspect a function or async function AST node for prohibited Any/object annotations.

    Args:
        file_path (str): Source file path.
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function AST node.

    Returns:
        list[str]: Violations found on the function signature.
    """
    violations: list[str] = []
    for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        for hit in _annotation_contains_any_or_object(arg.annotation):
            violations.append(f"{file_path}:{arg.lineno}: prohibited parameter annotation '{hit}'")
    if node.args.vararg is not None:
        for hit in _annotation_contains_any_or_object(node.args.vararg.annotation):
            violations.append(f"{file_path}:{node.args.vararg.lineno}: prohibited *args annotation '{hit}'")
    if node.args.kwarg is not None:
        for hit in _annotation_contains_any_or_object(node.args.kwarg.annotation):
            violations.append(f"{file_path}:{node.args.kwarg.lineno}: prohibited **kwargs annotation '{hit}'")
    for hit in _annotation_contains_any_or_object(node.returns):
        violations.append(f"{file_path}:{node.lineno}: prohibited return annotation '{hit}'")
    return violations


def _check_cast_call(file_path: str, node: ast.Call) -> list[str]:
    """Inspect a typing.cast(...) call for prohibited Any/object string-literal casts or `object()` calls.

    Args:
        file_path (str): Source file path.
        node (ast.Call): Call AST node.

    Returns:
        list[str]: Violations found in cast target type or `object()` call.
    """
    if (isinstance(node.func, ast.Name) and node.func.id == "object") or (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "builtins" and node.func.attr == "object"):
        return [f"{file_path}:{node.lineno}: prohibited 'object()' instantiation"]
    is_cast = (isinstance(node.func, ast.Name) and node.func.id == "cast") or (isinstance(node.func, ast.Attribute) and node.func.attr == "cast")
    if not is_cast or not node.args:
        return []
    first_arg = node.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        try:
            parsed = ast.parse(first_arg.value, mode="eval")
        except SyntaxError:
            return []
        return [f"{file_path}:{node.lineno}: prohibited string-literal cast '{hit}'" for hit in _annotation_contains_any_or_object(parsed)]
    return [f"{file_path}:{node.lineno}: prohibited cast '{hit}'" for hit in _annotation_contains_any_or_object(first_arg)]


def _scan_type_ignore_comments(file_path: str, source: str) -> list[str]:
    """Scan Python source tokens for prohibited `# type: ignore` comments.

    Args:
        file_path (str): Path to the Python file.
        source (str): Raw Python source text.

    Returns:
        list[str]: Violation strings for any `# type: ignore` comments.
    """
    violations: list[str] = []
    try:
        for tok in tokenize.generate_tokens(StringIO(source).readline):
            if tok.type == tokenize.COMMENT and "type: ignore" in tok.string:
                violations.append(f"{file_path}:{tok.start[0]}: prohibited '# type: ignore' comment")
    except tokenize.TokenError:
        return violations
    return violations


def _check_import_or_assign_node(file_path: str, node: ast.AST, is_outside_backends: bool) -> list[str]:
    """Check an import or assignment AST node for prohibited `Any`, `EagerValue`, or `AnyType`.

    Args:
        file_path (str): File path for error reporting.
        node (ast.AST): AST node to inspect.
        is_outside_backends (bool): True when scanning src/ outside of backends/.

    Returns:
        list[str]: Violations found on `node`.
    """
    if isinstance(node, ast.ImportFrom):
        res = [f"{file_path}:{node.lineno}: prohibited import 'Any'" for alias in node.names if alias.name == "Any"]
        if is_outside_backends:
            res.extend(f"{file_path}:{node.lineno}: prohibited import 'EagerValue' outside backends/" for alias in node.names if alias.name == "EagerValue")
        return res
    if isinstance(node, ast.Assign):
        return [f"{file_path}:{node.lineno}: disguised Any alias 'AnyType'" for target in node.targets if isinstance(target, ast.Name) and target.id == "AnyType"]
    return []


def _check_node_for_any_or_object(
    file_path: str,
    node: ast.AST,
    is_outside_backends: bool,
) -> list[str]:
    """Check a single AST node for Any, object, AnyType, or EagerValue violations.

    Args:
        file_path (str): File path for error reporting.
        node (ast.AST): AST node to inspect.
        is_outside_backends (bool): True when scanning src/ outside of backends/.

    Returns:
        list[str]: Violations found on `node`.
    """
    violations: list[str] = []
    if isinstance(node, (ast.ImportFrom, ast.Assign)):
        violations.extend(_check_import_or_assign_node(file_path, node, is_outside_backends))
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        violations.extend(_check_function_annotations(file_path, node))
    elif isinstance(node, ast.AnnAssign):
        violations.extend(f"{file_path}:{node.lineno}: prohibited variable annotation '{hit}'" for hit in _annotation_contains_any_or_object(node.annotation))
    elif isinstance(node, ast.Call):
        violations.extend(_check_cast_call(file_path, node))
    elif isinstance(node, ast.ClassDef):
        violations.extend(f"{file_path}:{node.lineno}: prohibited 'object' base class" for base in node.bases if (isinstance(base, ast.Name) and base.id == "object") or (isinstance(base, ast.Attribute) and base.attr == "object"))
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "typing" and node.attr == "Any":
        violations.append(f"{file_path}:{node.lineno}: prohibited usage 'typing.Any'")
    elif isinstance(node, ast.Name) and node.id == "object":
        violations.append(f"{file_path}:{node.lineno}: prohibited usage 'object'")
    return violations


def scan_file_for_any(file_path: str) -> list[str]:
    """Scan a single Python source file for Any imports, annotations, object usages, or type-ignore comments.

    Args:
        file_path (str): Path to the Python file to scan.

    Returns:
        list[str]: List of human-readable violation strings with line numbers.
    """
    with open(file_path, encoding="utf-8") as f:
        source: str = f.read()

    try:
        tree: ast.Module = ast.parse(source, filename=file_path)
    except SyntaxError:
        return []

    violations: list[str] = _scan_type_ignore_comments(file_path, source)
    norm_parts = os.path.normpath(file_path).split(os.sep)
    is_outside_backends = "ml_switcheroo_compiler" in norm_parts and "backends" not in norm_parts and "tests" not in norm_parts

    for node in ast.walk(tree):
        violations.extend(_check_node_for_any_or_object(file_path, node, is_outside_backends))

    return violations


def scan_directory_for_any(target_dir: str = "src/ml_switcheroo_compiler") -> list[str]:
    """Recursively scan a directory for Any and object type annotation violations.

    Args:
        target_dir (str): Root directory to scan.

    Returns:
        list[str]: Aggregated list of violations across all Python files.
    """
    all_violations: list[str] = []
    if not os.path.exists(target_dir):
        return all_violations

    for root, _, files in sorted(os.walk(target_dir)):
        for fname in sorted(files):
            if fname.endswith(".py"):
                fpath: str = os.path.join(root, fname)
                all_violations.extend(scan_file_for_any(fpath))
    return all_violations


def main(argv: typing.Sequence[str] | None = None) -> int:
    """Execute the AST-based Any/object linter CLI across src/, scripts/, and tests/.

    Args:
        argv (typing.Sequence[str] | None): Optional CLI arguments. Defaults to sys.argv[1:].

    Returns:
        int: Process exit code (0 on pass, 1 when violations exceed max-count).
    """
    parser = argparse.ArgumentParser(description="Lint Python ASTs for Any and object annotations.")
    parser.add_argument("--max-count", type=int, default=0, help="Maximum allowed Any/object occurrences.")
    parser.add_argument("--target-dir", type=str, default=None, help="Optional single target directory to scan.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    dirs_to_scan: tuple[str, ...] = (args.target_dir,) if args.target_dir is not None else DEFAULT_TARGET_DIRS
    violations: list[str] = []
    for target_dir in dirs_to_scan:
        violations.extend(scan_directory_for_any(target_dir))
    count: int = len(violations)

    if count > args.max_count:
        print(f"Found {count} usages of Any/object. Maximum allowed is {args.max_count}.")
        sys.exit(1)

    print("typing.Any check passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
