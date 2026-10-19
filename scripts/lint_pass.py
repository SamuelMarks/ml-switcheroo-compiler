"""AST-based linting pass for tautological assertions, swallowed exceptions, and Any annotations."""

from __future__ import annotations

import argparse
import ast
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.lint_any import scan_directory_for_any


def _node_is_verification(sub: ast.AST) -> bool:
    """Return True if an AST node is an assertion, raise, or verification call.

    Args:
        sub (ast.AST): AST node to inspect.

    Returns:
        bool: True if the node performs test verification.
    """
    if isinstance(sub, (ast.Assert, ast.Raise)):
        return True
    if isinstance(sub, ast.Call):
        target: ast.expr = sub.func
        if isinstance(target, ast.Call) and isinstance(target.func, ast.Name) and target.func.id == "cast" and len(target.args) == 2:
            target = target.args[1]
        if isinstance(target, ast.Name) and (target.id.startswith("assert") or target.id.startswith("_check") or target.id.startswith("verify") or target.id.startswith("_test") or target.id.startswith("_run")):
            return True
        if isinstance(target, ast.Attribute) and (target.attr.startswith("assert") or target.attr in ("raises", "warns", "fail", "approx") or target.attr.startswith("check") or target.attr.startswith("verify")):
            return True
    return False


def _has_assertion_or_verification(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether a test function AST contains at least one assertion or verification call outside except blocks.

    Args:
        func_node (ast.FunctionDef | ast.AsyncFunctionDef): Function AST node to inspect.

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
        if _node_is_verification(sub):
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
    if isinstance(test_expr, ast.Compare) and len(test_expr.ops) == 1 and isinstance(test_expr.ops[0], (ast.Eq, ast.Is)) and ast.dump(test_expr.left) == ast.dump(test_expr.comparators[0]):
        return f"FAILED: Tautological comparison found in {path}:{node.lineno}"
    if isinstance(test_expr, ast.Compare) and len(test_expr.ops) == 1 and isinstance(test_expr.ops[0], ast.IsNot) and isinstance(test_expr.left, ast.Name) and test_expr.left.id == "_exc":
        return f"FAILED: Tautological 'assert _exc is not None' found in {path}:{node.lineno}"
    if isinstance(test_expr, ast.Call) and isinstance(test_expr.func, ast.Name) and test_expr.func.id == "isinstance" and len(test_expr.args) == 2 and isinstance(test_expr.args[0], ast.Name) and test_expr.args[0].id == "_state":
        return f"FAILED: Vacuous 'assert isinstance(_state, ...)' found in {path}:{node.lineno}"
    return None


def _check_except_handler(path: str, node: ast.ExceptHandler) -> list[str]:
    """Check an ast.ExceptHandler for swallowed exceptions, AssertionError catches, and _exc tautologies.

    Args:
        path (str): File path of the test module.
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
            if node.name.startswith("test_") and not _has_assertion_or_verification(node):
                errors.append(f"FAILED: Zero assertions in test function '{node.name}' at {path}:{node.lineno}")

    return errors


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

    if args.check_any and os.path.exists(args.src_dir):
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
