"""AST-based linting pass for tautological assertions, swallowed exceptions, and Any annotations."""

from __future__ import annotations

import argparse
import ast
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.lint_any import scan_directory_for_any


def _is_verified_helper_call(
    name: str,
    local_funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None,
    visited_helpers: set[str] | None,
) -> bool:
    """Check whether a function call name invokes a verification helper.

    Args:
        name (str): Called function identifier.
        local_funcs (dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None): Module-local function table.
        visited_helpers (set[str] | None): Visited helper names to prevent recursion cycles.

    Returns:
        bool: True if the called function performs verification.
    """
    if name.startswith(("assert", "check_", "_check", "verify")):
        return True
    if not name.startswith(("_test", "_run")):
        return False
    if local_funcs is None or name not in local_funcs:
        return True
    seen = set() if visited_helpers is None else set(visited_helpers)
    if name in seen:
        return False
    seen.add(name)
    return _has_assertion_or_verification(local_funcs[name], local_funcs, seen)


def _node_is_verification(
    sub: ast.AST,
    local_funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
    visited_helpers: set[str] | None = None,
) -> bool:
    """Return True if an AST node is an assertion, raise, or verified helper call.

    Args:
        sub (ast.AST): AST node to inspect.
        local_funcs (dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None): Module-local function table.
        visited_helpers (set[str] | None): Visited helper names to prevent recursion cycles.

    Returns:
        bool: True if the node performs test verification.
    """
    if isinstance(sub, (ast.Assert, ast.Raise)):
        return True
    if not isinstance(sub, ast.Call):
        return False
    target: ast.expr = sub.func.args[-1] if isinstance(sub.func, ast.Call) and sub.func.args else sub.func
    if isinstance(target, ast.Name):
        return _is_verified_helper_call(target.id, local_funcs, visited_helpers)
    return isinstance(target, ast.Attribute) and (target.attr.startswith(("assert", "check", "verify")) or target.attr in ("raises", "warns", "fail", "approx"))


def _has_assertion_or_verification(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    local_funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
    visited_helpers: set[str] | None = None,
) -> bool:
    """Check whether a function AST contains at least one assertion or verification call outside except blocks.

    Args:
        func_node (ast.FunctionDef | ast.AsyncFunctionDef): Function AST node to inspect.
        local_funcs (dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None): Module-local function table.
        visited_helpers (set[str] | None): Visited helper names to prevent recursion cycles.

    Returns:
        bool: True if an assertion, raises context, or verification call is present outside except blocks.
    """
    except_nodes: set[int] = set()
    for sub in ast.walk(func_node):
        if isinstance(sub, ast.ExceptHandler):
            for child in ast.walk(sub):
                except_nodes.add(id(child))

    for sub in ast.walk(func_node):
        if id(sub) in except_nodes:
            continue
        if _node_is_verification(sub, local_funcs, visited_helpers):
            return True
    return False


def _is_excluded_middle_boolop(expr: ast.BoolOp) -> bool:
    """Return True if a BoolOp is an `or` tautology such as `x is not None or x is None` or `... or True`.

    Args:
        expr (ast.BoolOp): BoolOp AST node to inspect.

    Returns:
        bool: True if the BoolOp is a tautological disjunction.
    """
    if not isinstance(expr.op, ast.Or):
        return False
    for val in expr.values:
        if isinstance(val, ast.Constant) and val.value is True:
            return True
    if len(expr.values) == 2:
        lhs, rhs = expr.values[0], expr.values[1]
        if (
            isinstance(lhs, ast.Compare)
            and isinstance(rhs, ast.Compare)
            and len(lhs.ops) == 1
            and len(rhs.ops) == 1
            and ast.dump(lhs.left) == ast.dump(rhs.left)
            and ast.dump(lhs.comparators[0]) == ast.dump(rhs.comparators[0])
            and {type(lhs.ops[0]), type(rhs.ops[0])} in ({ast.Is, ast.IsNot}, {ast.Eq, ast.NotEq})
        ):
            return True
    return False


def _is_len_call(expr: ast.expr) -> bool:
    """Return True if `expr` is a `len(...)` builtin call.

    Args:
        expr (ast.expr): Expression AST node.

    Returns:
        bool: True if `expr` calls `len(...)`.
    """
    return isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "len"


def _is_nonneg_len_compare(expr: ast.Compare) -> bool:
    """Return True if `expr` is `len(x) >= 0` or `0 <= len(x)`.

    Args:
        expr (ast.Compare): Compare AST node.

    Returns:
        bool: True if `expr` is a non-negative length tautology.
    """
    if len(expr.ops) != 1:
        return False
    op = expr.ops[0]
    rhs = expr.comparators[0]
    if isinstance(op, ast.GtE) and _is_len_call(expr.left) and isinstance(rhs, ast.Constant) and rhs.value == 0 and not isinstance(rhs.value, bool):
        return True
    return isinstance(op, ast.LtE) and isinstance(expr.left, ast.Constant) and expr.left.value == 0 and not isinstance(expr.left.value, bool) and _is_len_call(rhs)


def _is_ellipsis_expr(expr: ast.expr) -> bool:
    """Return True if `expr` represents `Ellipsis` or `...`.

    Args:
        expr (ast.expr): Expression AST node.

    Returns:
        bool: True if `expr` is `Ellipsis` or `...`.
    """
    if isinstance(expr, ast.Constant) and expr.value is Ellipsis:
        return True
    return isinstance(expr, ast.Name) and expr.id == "Ellipsis"


def _check_compare_assert(path: str, lineno: int, expr: ast.Compare) -> str | None:
    """Validate an `ast.Compare` assertion expression against tautological comparisons.

    Args:
        path (str): File path of the test module.
        lineno (int): Source line number of the assertion.
        expr (ast.Compare): Compare AST node.

    Returns:
        str | None: Error message if tautological, else None.
    """
    if len(expr.ops) == 1:
        op, rhs = expr.ops[0], expr.comparators[0]
        if isinstance(op, (ast.Eq, ast.Is)) and ast.dump(expr.left) == ast.dump(rhs):
            return f"FAILED: Tautological comparison found in {path}:{lineno}"
        if isinstance(op, ast.IsNot) and isinstance(expr.left, ast.Name) and expr.left.id == "_exc":
            return f"FAILED: Tautological 'assert _exc is not None' found in {path}:{lineno}"
        if isinstance(op, ast.IsNot) and _is_ellipsis_expr(rhs):
            return f"FAILED: Tautological 'assert x is not Ellipsis' found in {path}:{lineno}"
    if _is_nonneg_len_compare(expr):
        return f"FAILED: Tautological 'len(...) >= 0' found in {path}:{lineno}"
    return None


def _check_call_assert(path: str, lineno: int, expr: ast.Call) -> str | None:
    """Validate an `ast.Call` assertion expression against self-referential or vacuous checks.

    Args:
        path (str): File path of the test module.
        lineno (int): Source line number of the assertion.
        expr (ast.Call): Call AST node.

    Returns:
        str | None: Error message if vacuous, else None.
    """
    if not isinstance(expr.func, ast.Name):
        return None
    if expr.func.id == "callable" and len(expr.args) == 1 and isinstance(expr.args[0], ast.Name) and expr.args[0].id.startswith("test_"):
        return f"FAILED: Self-referential 'assert callable({expr.args[0].id})' found in {path}:{lineno}"
    if expr.func.id == "isinstance" and len(expr.args) == 2:
        if isinstance(expr.args[0], ast.Name) and expr.args[0].id == "_state":
            return f"FAILED: Vacuous 'assert isinstance(_state, ...)' found in {path}:{lineno}"
        if isinstance(expr.args[1], ast.Name) and expr.args[1].id == "object":
            return f"FAILED: Tautological 'assert isinstance(..., object)' found in {path}:{lineno}"
    return None


def _check_assert_node(path: str, node: ast.Assert) -> str | None:
    """Check an ast.Assert node for vacuous boolean constants or tautological comparisons.

    Args:
        path (str): File path of the test module.
        node (ast.Assert): Assert AST node.

    Returns:
        str | None: Error message if invalid, else None.
    """
    test_expr: ast.expr = node.test
    if isinstance(test_expr, ast.Constant) and isinstance(test_expr.value, bool):
        return f"FAILED: Vacuous 'assert {test_expr.value}' found in {path}:{node.lineno}"
    if isinstance(test_expr, ast.BoolOp) and _is_excluded_middle_boolop(test_expr):
        return f"FAILED: Tautological disjunction found in {path}:{node.lineno}"
    if isinstance(test_expr, ast.Compare):
        return _check_compare_assert(path, node.lineno, test_expr)
    if isinstance(test_expr, ast.Call):
        return _check_call_assert(path, node.lineno, test_expr)
    return None


def _is_dummy_swallow_assign(stmt: ast.stmt) -> bool:
    """Return True if `stmt` is a dummy assignment (`_ = ...`, `_caught_err = ...`) used to swallow exceptions.

    Args:
        stmt (ast.stmt): Statement inside an `ExceptHandler` body.

    Returns:
        bool: True if `stmt` is a dummy exception-swallowing assignment.
    """
    if not isinstance(stmt, ast.Assign):
        return False
    for target in stmt.targets:
        if isinstance(target, ast.Name) and (target.id == "_" or target.id.startswith(("_caught", "_ignore", "_unused"))):
            return True
    return False


def _check_except_handler(path: str, node: ast.ExceptHandler) -> list[str]:
    """Check an ast.ExceptHandler for swallowed exceptions, AssertionError catches, and _exc tautologies.

    Args:
        path (str): File path of the Python module.
        node (ast.ExceptHandler): ExceptHandler AST node.

    Returns:
        list[str]: List of violation messages.
    """
    errs: list[str] = []
    if node.type is None:
        errs.append(f"FAILED: Vacuous 'except: pass' found in {path}:{node.lineno}")
    else:
        for sub in ast.walk(node.type):
            if isinstance(sub, ast.Name) and sub.id == "AssertionError":
                errs.append(f"FAILED: Catching AssertionError in except handler at {path}:{node.lineno}")
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            errs.append(f"FAILED: Vacuous 'except Exception: pass' found in {path}:{node.lineno}")
        elif len(node.body) == 1 and _is_dummy_swallow_assign(node.body[0]):
            errs.append(f"FAILED: Swallowed exception via dummy assignment found in {path}:{node.lineno}")
    if node.name == "_exc":
        errs.append(f"FAILED: Tautological 'as _exc' handler found in {path}:{node.lineno}")
    return errs


def _is_suppress_call(node: ast.Call) -> bool:
    """Check whether a Call AST node invokes `contextlib.suppress` or `suppress`.

    Args:
        node (ast.Call): Call AST node.

    Returns:
        bool: True if it calls `suppress`.
    """
    if isinstance(node.func, ast.Name) and node.func.id == "suppress":
        return True
    return isinstance(node.func, ast.Attribute) and node.func.attr == "suppress"


def _check_try_node(path: str, node: ast.Try) -> list[str]:
    """Check an ast.Try node for stray string literals inside its body.

    Args:
        path (str): File path of the test module.
        node (ast.Try): Try AST node.

    Returns:
        list[str]: List of violation messages.
    """
    return [f"FAILED: Stray string literal inside try block at {path}:{s.lineno}" for s in node.body if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)]


def check_test_file_ast(
    path: str,
    check_zero_assertions: bool = True,
    check_swallowed_exceptions: bool = True,
) -> list[str]:
    """Inspect a test file's AST for vacuous assertions, self-comparisons, and swallowed exceptions.

    Args:
        path (str): File path of the Python test module.
        check_zero_assertions (bool): Whether to flag test_* functions with zero assertions.
        check_swallowed_exceptions (bool): Whether to flag try...except: pass blocks.

    Returns:
        list[str]: List of violation messages discovered in the file.
    """
    with open(path, encoding="utf-8") as file:
        content: str = file.read()

    try:
        tree: ast.Module = ast.parse(content, filename=path)
    except SyntaxError:
        return []

    local_funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {stmt.name: stmt for stmt in tree.body if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))}

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            err = _check_assert_node(path, node)
            if err is not None:
                errors.append(err)
        elif isinstance(node, ast.Try):
            errors.extend(_check_try_node(path, node))
        elif isinstance(node, ast.Call) and _is_suppress_call(node):
            errors.append(f"FAILED: Prohibited contextlib.suppress usage in {path}:{node.lineno}")
        elif check_swallowed_exceptions and isinstance(node, ast.ExceptHandler):
            errors.extend(_check_except_handler(path, node))
        elif check_zero_assertions and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") and not _has_assertion_or_verification(node, local_funcs):
                errors.append(f"FAILED: Zero assertions in test function '{node.name}' at {path}:{node.lineno}")

    return errors


def _scan_dummy_swallows_in_dir(target_dir: str) -> list[str]:
    """Scan Python modules in `target_dir` for dummy exception-swallowing assignments (`_ = None`, `_caught_err = True`).

    Args:
        target_dir (str): Directory root to scan recursively.

    Returns:
        list[str]: Discovered violation messages.
    """
    violations: list[str] = []
    for root, _, files in sorted(os.walk(target_dir)):
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            try:
                tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and len(node.body) == 1 and _is_dummy_swallow_assign(node.body[0]):
                    violations.append(f"FAILED: Swallowed exception via dummy assignment found in {path}:{node.lineno}")
    return violations


def main(argv: list[str] | None = None) -> None:
    """Run AST-based lint pass across tests and optionally source directories.

    Args:
        argv (list[str] | None): Optional CLI arguments. Defaults to None.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Run AST-based test and typing lint checks.")
    parser.add_argument("--tests-dir", type=str, default="tests", help="Tests directory to scan.")
    parser.add_argument("--src-dir", type=str, default="src/ml_switcheroo_compiler", help="Source directory to scan.")
    parser.add_argument("--check-zero-assertions", action="store_true", default=True, help="Flag test functions with zero assertions.")
    parser.add_argument("--check-any", action="store_true", default=True, help="Also fail if Any/object annotations are found in src.")
    args, _ = parser.parse_known_args(argv if argv is not None else [])

    all_errors: list[str] = []
    if os.path.exists(args.tests_dir):
        for root, _, files in sorted(os.walk(args.tests_dir)):
            for f in sorted(files):
                if f.endswith(".py"):
                    path: str = os.path.join(root, f)
                    all_errors.extend(
                        check_test_file_ast(
                            path,
                            check_zero_assertions=args.check_zero_assertions,
                        )
                    )

    if os.path.exists(args.src_dir):
        all_errors.extend(_scan_dummy_swallows_in_dir(args.src_dir))
        any_violations: list[str] = scan_directory_for_any(args.src_dir)
        if any_violations:
            all_errors.append(f"FAILED: Found {len(any_violations)} usages of Any/object")

    if all_errors:
        for err in all_errors:
            print(err)
        sys.exit(1)

    print("Linting passed.")
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
