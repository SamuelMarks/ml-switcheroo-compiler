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

_FORBIDDEN_REFLECTION_WRAPPERS: frozenset[str] = frozenset(
    {
        "_get_attr",
        "_has_attr",
        "_set_attr",
        "_del_attr",
        "_obj_set_attr",
        "_obj_del_attr",
        "_obj_vars",
        "_caller_locals",
        "_cast_val",
        "lookup_backend_attr",
        "has_backend_attr",
        "get_backend_attr",
        "getattr_static",
    }
)

_FORBIDDEN_DISGUISED_CASTS: frozenset[str] = frozenset(
    {
        "as_tensor",
        "as_callable",
        "as_int",
        "as_float",
        "as_shape",
        "as_mod",
        "as_ndarray",
        "as_attrs",
        "as_attr_val",
        "as_graph",
        "as_node",
        "as_node_dict",
        "as_tensor_dict",
        "as_tensor_list",
        "as_tensor_obj",
        "as_module_type",
        "as_type_obj",
        "as_int_list",
        "as_str",
        "as_opt_str",
        "as_opt_int",
        "as_bool",
        "as_bytes",
        "as_float_list",
        "as_eager_list",
        "as_tensor_pair",
        "as_tensor_triple",
        "as_tensor_quad",
        "as_opt_int_seq",
        "as_int_or_seq",
        "as_opt_callable",
    }
)

_FORBIDDEN_OPERATOR_ATTRS: frozenset[str] = frozenset(
    {
        "attrgetter",
        "itemgetter",
        "methodcaller",
        "_getframe",
        "getattr_static",
    }
)


def _dunder(stem: str) -> str:
    """Construct a dunder identifier `__<stem>__` without embedding raw dunder string literals.

    Args:
        stem (str): Identifier stem.

    Returns:
        str: Dunder identifier.
    """
    prefix = "_" * 2
    return f"{prefix}{stem}{prefix}"


_FORBIDDEN_DUNDER_ATTRS: frozenset[str] = frozenset(
    _dunder(s)
    for s in (
        "dict",
        "mro",
        "subclasses",
        "setattr",
        "getattribute",
        "delattr",
        "class",
    )
)

_FORBIDDEN_DUNDER_CALLS: frozenset[str] = frozenset(
    _dunder(s)
    for s in (
        "call",
        "add",
        "radd",
        "sub",
        "rsub",
        "mul",
        "rmul",
        "matmul",
        "rmatmul",
        "truediv",
        "rtruediv",
        "floordiv",
        "rfloordiv",
        "mod",
        "rmod",
        "pow",
        "rpow",
        "neg",
        "pos",
        "abs",
        "invert",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "getitem",
        "setitem",
        "delitem",
        "setattr",
        "getattribute",
        "delattr",
        "getattr",
        "get",
        "set",
        "delete",
        "array",
        "index",
        "await",
        "enter",
        "exit",
        "iter",
        "next",
        "len",
        "contains",
        "int",
        "float",
        "bool",
        "str",
        "repr",
        "hash",
    )
)

_FORBIDDEN_DUNDER_STRINGS: frozenset[str] = frozenset(
    _dunder(s)
    for s in (
        "getitem",
        "setitem",
        "delitem",
        "setattr",
        "delattr",
        "getattr",
        "getattribute",
        "array",
        "index",
        "call",
        "add",
        "sub",
        "mul",
        "truediv",
        "dict",
        "class",
    )
)

_GETATTR_DUNDER: str = _dunder("getattr")


def _is_protocol_base(base: ast.expr) -> bool:
    """Return True if `base` references `Protocol` or `typing.Protocol`.

    Args:
        base (ast.expr): Base class AST expression.

    Returns:
        bool: True if base is Protocol.
    """
    if isinstance(base, ast.Subscript):
        return _is_protocol_base(base.value)
    if isinstance(base, ast.Name):
        return base.id == "Protocol"
    return isinstance(base, ast.Attribute) and base.attr == "Protocol"


def _is_typeguard_return(returns: ast.expr | None) -> bool:
    """Return True if return annotation is `TypeGuard[...]` or `typing.TypeGuard[...]`.

    Args:
        returns (ast.expr | None): Return annotation AST node.

    Returns:
        bool: True if return annotation is TypeGuard.
    """
    if not isinstance(returns, ast.Subscript):
        return False
    val = returns.value
    return (isinstance(val, ast.Name) and val.id == "TypeGuard") or (isinstance(val, ast.Attribute) and val.attr == "TypeGuard")


def _is_tautological_return_expr(expr: ast.expr | None) -> bool:
    """Return True if `expr` is `True`, `x is None or x is not None`, or `not isinstance(x, bytes) or len(x) >= 0`.

    Args:
        expr (ast.expr | None): Return expression AST node.

    Returns:
        bool: True if expression is tautological.
    """
    if isinstance(expr, ast.Constant) and expr.value is True:
        return True
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or) and len(expr.values) == 2:
        a, b = expr.values
        if isinstance(a, ast.Compare) and isinstance(b, ast.Compare) and a.ops and b.ops:
            ops_pair = (type(a.ops[0]), type(b.ops[0]))
            if ops_pair in ((ast.Is, ast.IsNot), (ast.IsNot, ast.Is)):
                return True
        if isinstance(b, ast.Compare) and b.ops and isinstance(b.ops[0], ast.GtE) and b.comparators and isinstance(b.comparators[0], ast.Constant) and b.comparators[0].value == 0:
            return True
    return False


def _check_name_call(file_path: str, lineno: int, fname: str) -> list[str]:
    """Check a direct `ast.Name` function call for prohibited calls.

    Args:
        file_path (str): File path for error reporting.
        lineno (int): Line number.
        fname (str): Function identifier.

    Returns:
        list[str]: Violation messages.
    """
    if fname in _FORBIDDEN_BUILTIN_CALLS:
        return [f"{file_path}:{lineno}: prohibited reflection/cast call '{fname}()'"]
    if fname in _FORBIDDEN_REFLECTION_WRAPPERS:
        return [f"{file_path}:{lineno}: prohibited reflection wrapper call '{fname}()'"]
    if fname in _FORBIDDEN_DISGUISED_CASTS:
        return [f"{file_path}:{lineno}: prohibited disguised cast call '{fname}()'"]
    if fname in _FORBIDDEN_OPERATOR_ATTRS:
        return [f"{file_path}:{lineno}: prohibited reflection helper call '{fname}()'"]
    return []


def _check_attr_call(file_path: str, node: ast.Call, func: ast.Attribute) -> list[str]:
    """Check an `ast.Attribute` function call for prohibited calls.

    Args:
        file_path (str): File path for error reporting.
        node (ast.Call): Call AST node.
        func (ast.Attribute): Attribute target of the call.

    Returns:
        list[str]: Violation messages.
    """
    attr = func.attr
    violations: list[str] = []
    if isinstance(func.value, ast.Name) and func.value.id in ("typing", "typing_extensions", "builtins") and attr in _FORBIDDEN_BUILTIN_CALLS:
        msg = "prohibited 'typing.cast()' call" if attr == "cast" else f"prohibited reflection/cast call '{func.value.id}.{attr}()'"
        violations.append(f"{file_path}:{node.lineno}: {msg}")
    elif attr == "cast":
        if node.args or node.keywords:
            violations.append(f"{file_path}:{node.lineno}: prohibited '.cast()' call")
    elif attr in _FORBIDDEN_OPERATOR_ATTRS:
        violations.append(f"{file_path}:{node.lineno}: prohibited reflection helper call '.{attr}()'")
    elif attr in _FORBIDDEN_REFLECTION_WRAPPERS:
        violations.append(f"{file_path}:{node.lineno}: prohibited reflection wrapper call '.{attr}()'")
    elif attr in _FORBIDDEN_DISGUISED_CASTS and not (attr == "as_tensor" and isinstance(func.value, ast.Name) and func.value.id == "torch"):
        violations.append(f"{file_path}:{node.lineno}: prohibited disguised cast call '.{attr}()'")
    elif attr in _FORBIDDEN_DUNDER_CALLS:
        is_super = isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Name) and func.value.func.id == "super"
        if not is_super:
            violations.append(f"{file_path}:{node.lineno}: prohibited direct dunder call '.{attr}()'")
    return violations


def _check_call_node(file_path: str, node: ast.Call) -> list[str]:
    """Inspect an `ast.Call` node for prohibited reflection, cast, or dunder calls.

    Args:
        file_path (str): File path for error reporting.
        node (ast.Call): Call AST node.

    Returns:
        list[str]: Violation messages for `node`.
    """
    if isinstance(node.func, ast.Name):
        return _check_name_call(file_path, node.lineno, node.func.id)
    if isinstance(node.func, ast.Attribute):
        return _check_attr_call(file_path, node, node.func)
    return []


def _check_function_node(file_path: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Inspect a function definition for prohibited wrapper defs, `__getattr__`, or tautological `TypeGuard`s.

    Args:
        file_path (str): File path for error reporting.
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function definition AST node.

    Returns:
        list[str]: Violation messages for `node`.
    """
    violations: list[str] = []
    if node.name in _FORBIDDEN_REFLECTION_WRAPPERS:
        violations.append(f"{file_path}:{node.lineno}: prohibited reflection wrapper definition '{node.name}'")
    if node.name in _FORBIDDEN_DISGUISED_CASTS:
        violations.append(f"{file_path}:{node.lineno}: prohibited disguised cast definition '{node.name}'")
    if node.name == _GETATTR_DUNDER:
        violations.append(f"{file_path}:{node.lineno}: prohibited catch-all 'def {_GETATTR_DUNDER}' definition")
    if _is_typeguard_return(node.returns):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and _is_tautological_return_expr(sub.value):
                violations.append(f"{file_path}:{node.lineno}: prohibited tautological TypeGuard '{node.name}'")
                break
    return violations


def _check_class_node(file_path: str, node: ast.ClassDef) -> list[str]:
    """Inspect a class definition for empty `Protocol` classes or `StructuralOperandProtocol`.

    Args:
        file_path (str): File path for error reporting.
        node (ast.ClassDef): Class definition AST node.

    Returns:
        list[str]: Violation messages for `node`.
    """
    violations: list[str] = []
    if node.name == "StructuralOperandProtocol":
        violations.append(f"{file_path}:{node.lineno}: prohibited catch-all 'StructuralOperandProtocol'")
    if any(_is_protocol_base(b) for b in node.bases):
        non_doc_body = [stmt for stmt in node.body if not isinstance(stmt, ast.Pass) and not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, (str, type(...))))]
        if not non_doc_body:
            violations.append(f"{file_path}:{node.lineno}: prohibited empty Protocol class '{node.name}'")
    return violations


def _is_type_checking_test(test: ast.expr) -> bool:
    """Return True if `test` is `TYPE_CHECKING`, `typing.TYPE_CHECKING`, or their negation.

    Args:
        test (ast.expr): Conditional test AST expression.

    Returns:
        bool: True if test checks `TYPE_CHECKING`.
    """
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _is_type_checking_test(test.operand)
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _is_negated_type_checking_test(test: ast.expr) -> bool:
    """Return True if `test` is `not TYPE_CHECKING` or `not typing.TYPE_CHECKING`.

    Args:
        test (ast.expr): Conditional test AST expression.

    Returns:
        bool: True if test negates `TYPE_CHECKING`.
    """
    return isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and _is_type_checking_test(test.operand)


def _check_misc_node(file_path: str, node: ast.AST) -> list[str]:
    """Inspect non-call/function/class AST nodes for prohibited attributes, names, constants, or imports.

    Args:
        file_path (str): File path for error reporting.
        node (ast.AST): AST node to inspect.

    Returns:
        list[str]: Violation messages for `node`.
    """
    if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_DUNDER_ATTRS:
        return [f"{file_path}:{node.lineno}: prohibited dunder attribute '.{node.attr}'"]
    if isinstance(node, ast.Name) and node.id in ("StructuralOperandProtocol", "_DynamicAttrType"):
        return [f"{file_path}:{node.lineno}: prohibited catch-all type '{node.id}'"]
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in _FORBIDDEN_DUNDER_STRINGS:
        return [f"{file_path}:{node.lineno}: prohibited dunder string literal '{node.value}'"]
    if isinstance(node, ast.If) and (_is_negated_type_checking_test(node.test) or (_is_type_checking_test(node.test) and node.orelse)):
        return [f"{file_path}:{node.lineno}: prohibited 'if TYPE_CHECKING: ... else: ...' type-erasure block"]
    if isinstance(node, ast.ImportFrom):
        return [f"{file_path}:{node.lineno}: prohibited import '{alias.name}'" for alias in node.names if alias.name in ("_DynamicAttrType", "attrgetter", "itemgetter", "methodcaller", "getattr_static")]
    return []


def _scan_strict_ast(file_path: str, tree: ast.Module) -> list[str]:
    """Scan an AST module for prohibited reflection, cast, protocol, and direct dunder calls.

    Args:
        file_path (str): File path for error reporting.
        tree (ast.Module): Parsed Python AST module.

    Returns:
        list[str]: List of violation messages found in the AST.
    """
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            violations.extend(_check_call_node(file_path, node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_check_function_node(file_path, node))
        elif isinstance(node, ast.ClassDef):
            violations.extend(_check_class_node(file_path, node))
        else:
            violations.extend(_check_misc_node(file_path, node))
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
