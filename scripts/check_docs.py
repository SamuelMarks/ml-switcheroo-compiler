"""Granular AST docstring coverage and type-parity verifier for modules, classes, functions, and parameters."""

from __future__ import annotations

import ast
import os
import re
import sys
from collections.abc import Sequence

_BUILTIN_GENERICS: dict[str, str] = {
    "List": "list",
    "Dict": "dict",
    "Tuple": "tuple",
    "Set": "set",
    "FrozenSet": "frozenset",
    "Type": "type",
}


def _collect_union_parts(expr: ast.expr) -> list[str]:
    """Flatten a bitwise-or (`|`) union AST expression into canonicalized member strings.

    Args:
        expr (ast.expr): AST type expression.

    Returns:
        list[str]: Flattened canonical union member strings.
    """
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
        return _collect_union_parts(expr.left) + _collect_union_parts(expr.right)
    return _canonicalize_type_expr(expr).split(" | ")


def _canonicalize_constant(expr: ast.Constant) -> str:
    """Normalize an `ast.Constant` type annotation expression.

    Args:
        expr (ast.Constant): Constant AST expression (`str`, `None`, `...`, or literal).

    Returns:
        str: Canonical string representation.
    """
    if isinstance(expr.value, str):
        try:
            parsed_inner = ast.parse(expr.value, mode="eval").body
        except SyntaxError:
            return expr.value
        return _canonicalize_type_expr(parsed_inner)
    if expr.value is None:
        return "None"
    if expr.value is Ellipsis:
        return "..."
    return ast.unparse(expr)


def _canonicalize_subscript(expr: ast.Subscript) -> str:
    """Normalize an `ast.Subscript` generic type annotation (`Optional`, `Union`, `Annotated`, etc.).

    Args:
        expr (ast.Subscript): Subscript AST expression.

    Returns:
        str: Canonical normalized generic type string.
    """
    base = _canonicalize_type_expr(expr.value)
    elts = list(expr.slice.elts) if isinstance(expr.slice, ast.Tuple) else [expr.slice]
    if base == "Optional" and len(elts) == 1:
        return " | ".join(sorted({_canonicalize_type_expr(elts[0]), "None"}))
    if base == "Union":
        parts: list[str] = []
        for item in elts:
            parts.extend(_canonicalize_type_expr(item).split(" | "))
        return " | ".join(sorted(set(parts)))
    if base == "Annotated" and elts:
        return _canonicalize_type_expr(elts[0])
    inner = ", ".join(_canonicalize_type_expr(item) for item in elts)
    return f"{base}[{inner}]"


def _canonicalize_type_expr(expr: ast.expr) -> str:
    """Normalize an AST type annotation expression for order-independent comparison.

    Args:
        expr (ast.expr): AST expression representing a Python type annotation.

    Returns:
        str: Canonical normalized string representation of the type.
    """
    if isinstance(expr, ast.Constant):
        return _canonicalize_constant(expr)
    if isinstance(expr, (ast.Name, ast.Attribute)):
        ident = expr.id if isinstance(expr, ast.Name) else expr.attr
        return _BUILTIN_GENERICS.get(ident, ident)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
        return " | ".join(sorted(set(_collect_union_parts(expr))))
    if isinstance(expr, ast.Subscript):
        return _canonicalize_subscript(expr)
    return ast.unparse(expr)


def _extract_expected_args(args: ast.arguments) -> list[tuple[str, ast.expr | None, str]]:
    """Extract `(name, annotation, kind)` tuples (excluding `self`/`cls`) from `ast.arguments`.

    Args:
        args (ast.arguments): AST function arguments node.

    Returns:
        list[tuple[str, ast.expr | None, str]]: Parameter tuples expected in the docstring `Args:` block.
    """
    items: list[tuple[str, ast.expr | None, str]] = []
    all_pos = list(args.posonlyargs) + list(args.args)
    for idx, arg in enumerate(all_pos):
        if idx == 0 and arg.arg in ("self", "cls"):
            continue
        items.append((arg.arg, arg.annotation, "regular"))
    if args.vararg is not None:
        items.append((args.vararg.arg, args.vararg.annotation, "vararg"))
    for kwarg in args.kwonlyargs:
        items.append((kwarg.arg, kwarg.annotation, "regular"))
    if args.kwarg is not None:
        items.append((args.kwarg.arg, args.kwarg.annotation, "kwarg"))
    return items


def _has_non_none_return(returns: ast.expr | None) -> bool:
    """Check whether a function return annotation is present and not `None` or `NoReturn` / `Never`.

    Args:
        returns (ast.expr | None): Function return annotation AST expression.

    Returns:
        bool: True if a non-`None` value return annotation is present.
    """
    if returns is None:
        return False
    if isinstance(returns, ast.Constant) and returns.value is None:
        return False
    if isinstance(returns, ast.Name) and returns.id in ("None", "NoReturn", "Never"):
        return False
    return not (isinstance(returns, ast.Attribute) and returns.attr in ("NoReturn", "Never"))


def _scan_body_flow(body: Sequence[ast.stmt]) -> tuple[bool, bool, bool]:
    """Scan a function body (excluding nested scopes) for value returns, yields, and explicit raises.

    Args:
        body (Sequence[ast.stmt]): Top-level statements of a function body.

    Returns:
        tuple[bool, bool, bool]: `(has_value_return, has_yield, has_explicit_raise)`.
    """
    has_val_return = False
    has_yield = False
    has_explicit_raise = False
    stack: list[ast.AST] = list(body)
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(cur, ast.Return) and cur.value is not None:
            if not (isinstance(cur.value, ast.Constant) and cur.value.value is None):
                has_val_return = True
        elif isinstance(cur, (ast.Yield, ast.YieldFrom)):
            has_yield = True
        elif isinstance(cur, ast.Raise) and cur.exc is not None:
            has_explicit_raise = True
        stack.extend(ast.iter_child_nodes(cur))
    return has_val_return, has_yield, has_explicit_raise


def _is_overload_decorated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the function has an `@overload` or `@typing.overload` decorator.

    Args:
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function AST node.

    Returns:
        bool: True if decorated with `@overload`.
    """
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "overload":
            return True
    return False


def _extract_args_section(doc: str) -> str:
    """Extract the `Args:` section body from a Google-style docstring.

    Args:
        doc (str): Full normalized docstring text.

    Returns:
        str: Text belonging to the `Args:` section, or empty string if absent.
    """
    section_headers = {"Returns:", "Yields:", "Raises:", "Attributes:", "Examples:", "Note:", "Notes:"}
    lines = doc.splitlines()
    collecting = False
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "Args:":
            collecting = True
            continue
        if collecting:
            if stripped in section_headers:
                break
            collected.append(line)
    return "\x0a".join(collected)


def _check_arg_type_parity(
    path: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    args_block: str,
    arg_name: str,
    ann: ast.expr | None,
    kind: str,
) -> list[str]:
    """Validate that a documented parameter type in `Args:` matches its AST type annotation.

    Args:
        path (str): File path being inspected.
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function AST node.
        args_block (str): Extracted `Args:` block text.
        arg_name (str): Parameter name.
        ann (ast.expr | None): AST type annotation expression, if annotated.
        kind (str): Parameter kind (`'regular'`, `'vararg'`, or `'kwarg'`).

    Returns:
        list[str]: Violation messages for the parameter's documented type.
    """
    if ann is None or not args_block:
        return []
    raw_doc_type: str | None = None
    for line in args_block.splitlines():
        m = re.match(rf"^\s*\**{re.escape(arg_name)}\s*\(([^)]+)\)\s*:", line)
        if m is not None:
            raw_doc_type = re.sub(r",\s*optional$", "", m.group(1).strip())
            break
    if raw_doc_type is None:
        return []
    forbidden_pat = r"\b(?:" + "Any" + "|" + "object" + r")\b"
    if re.search(forbidden_pat, raw_doc_type):
        return [f"{path}:{node.lineno}: prohibited Any/object docstring type on '{arg_name}' in '{node.name}'"]
    try:
        doc_expr = ast.parse(raw_doc_type, mode="eval").body
    except SyntaxError:
        return [f"{path}:{node.lineno}: invalid documented type '{raw_doc_type}' for arg '{arg_name}' in '{node.name}'"]
    canon_doc = _canonicalize_type_expr(doc_expr)
    canon_ann = _canonicalize_type_expr(ann)
    allowed = {canon_ann}
    if kind == "vararg":
        allowed.add(f"tuple[{canon_ann}, ...]")
    elif kind == "kwarg":
        allowed.add(f"dict[str, {canon_ann}]")
    if canon_doc not in allowed:
        return [
            f"{path}:{node.lineno}: arg '{arg_name}' documented type '{raw_doc_type}' does not match "
            f"annotation '{ast.unparse(ann)}' in '{node.name}'"
        ]
    return []


def _check_function_node(path: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Check that a function or method AST node has a complete, type-consistent docstring.

    Args:
        path (str): File path being inspected.
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function or async function AST node.

    Returns:
        list[str]: List of violation messages for the function.
    """
    if _is_overload_decorated(node):
        return []
    doc = ast.get_docstring(node)
    if not doc or not doc.strip():
        return [f"{path}:{node.lineno}: missing docstring on '{node.name}'"]
    violations: list[str] = []
    if "(" + "Any)" in doc or "(" + "object)" in doc:
        violations.append(f"{path}:{node.lineno}: prohibited Any/object docstring type in '{node.name}'")

    expected_args = _extract_expected_args(node.args)
    if expected_args and "Args:" not in doc:
        violations.append(f"{path}:{node.lineno}: missing 'Args:' section in '{node.name}'")

    args_block = _extract_args_section(doc)
    for arg_name, ann, kind in expected_args:
        if not re.search(rf"\b{re.escape(arg_name)}\b", doc):
            violations.append(f"{path}:{node.lineno}: missing arg '{arg_name}' in '{node.name}'")
        else:
            violations.extend(_check_arg_type_parity(path, node, args_block, arg_name, ann, kind))

    has_val_ret, has_yield, has_explicit_raise = _scan_body_flow(node.body)
    if has_yield and "Yields:" not in doc:
        violations.append(f"{path}:{node.lineno}: missing 'Yields:' section in '{node.name}'")
    if (_has_non_none_return(node.returns) or has_val_ret) and "Returns:" not in doc and "Yields:" not in doc:
        violations.append(f"{path}:{node.lineno}: missing 'Returns:'/'Yields:' section in '{node.name}'")
    if has_explicit_raise and "Raises:" not in doc:
        violations.append(f"{path}:{node.lineno}: missing 'Raises:' section in '{node.name}'")
    return violations


def check_file(path: str) -> list[str]:
    """Check a single Python file for module, class, function, and argument docstring coverage.

    Args:
        path (str): Path to the Python file to inspect.

    Returns:
        list[str]: Violation messages discovered in the file.
    """
    with open(path, encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=path)
    violations: list[str] = []
    mod_doc = ast.get_docstring(tree)
    if not mod_doc or not mod_doc.strip():
        violations.append(f"{path}:1: missing module docstring")

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            cls_doc = ast.get_docstring(node)
            if not cls_doc or not cls_doc.strip():
                violations.append(f"{path}:{node.lineno}: missing class docstring on '{node.name}'")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_check_function_node(path, node))

    return violations


def check_directory(target_dir: str) -> list[str]:
    """Recursively scan a directory or file for docstring violations.

    Args:
        target_dir (str): Directory or file path to inspect.

    Returns:
        list[str]: Aggregated violation messages.
    """
    if not os.path.exists(target_dir):
        return []
    if os.path.isfile(target_dir):
        return check_file(target_dir) if target_dir.endswith(".py") else []

    violations: list[str] = []
    for root, _, files in sorted(os.walk(target_dir)):
        for fname in sorted(files):
            if fname.endswith(".py"):
                violations.extend(check_file(os.path.join(root, fname)))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the documentation coverage check CLI.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments; defaults to `("src/ml_switcheroo_compiler", "scripts",
            "tests")`.

    Returns:
        int: Process exit code (0 when 100% compliant, 1 otherwise).
    """
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    targets: Sequence[str] = raw_args if raw_args else ("src/ml_switcheroo_compiler", "scripts", "tests")
    all_violations: list[str] = []
    for target in targets:
        all_violations.extend(check_directory(str(target)))

    if all_violations:
        for v in all_violations:
            print(v, file=sys.stderr)
        return 1
    print("Documentation coverage check passed (0 violations).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
