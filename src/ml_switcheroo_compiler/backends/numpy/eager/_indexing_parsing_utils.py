"""Utilities for parsing stringified indexing keys in NumPy backend."""

from __future__ import annotations

import ast
import typing
from collections.abc import Callable

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.core.shape import SliceType


def _as_slice_arg(val: EagerValue) -> int | float | str | None:
    """Narrow slice argument value to `int | float | str | None`.

    Args:
        val (EagerValue): Evaluated AST argument.

    Returns:
        int | float | str | None: Narrowed slice bound or step.
    """

    def _guard(cand: EagerValue) -> typing.TypeGuard[int | float | str | None]:
        """Type guard narrowing `cand` to `int | float | str | None`.

        Args:
            cand (EagerValue): Candidate slice argument.

        Returns:
            bool: True for slice arguments.
        """
        return not isinstance(cand, type(Ellipsis))

    assert _guard(val)
    return val


def _eval_constant(node: ast.AST) -> EagerValue:
    """Execute the eager eval constant operation on the active backend array.

    Args:
        node (ast.AST): Input `node` argument (node).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    val = node.value if isinstance(node, ast.Constant) else None
    if val is ...:
        return ...
    return val if isinstance(val, (int, float, complex, str, bool)) else None


def _eval_slice_call(node: ast.AST, _eval_fn: Callable[[ast.AST], EagerValue]) -> SliceType:
    """Execute the eager eval slice call operation on the active backend array.

    Args:
        node (ast.AST): Input `node` argument (node).
        _eval_fn (Callable[[ast.AST], EagerValue]): Input `_eval_fn` argument ( eval fn).

    Returns:
        SliceType: Computed `SliceType` output.
    """
    args_seq = node.args if isinstance(node, ast.Call) else []
    return slice(*[_as_slice_arg(_eval_fn(a)) for a in args_seq])


def _eval_array_call(node: ast.AST, _eval_fn: Callable[[ast.AST], EagerValue]) -> EagerValue:
    """Execute the eager eval array call operation on the active backend array.

    Args:
        node (ast.AST): Input `node` argument (node).
        _eval_fn (Callable[[ast.AST], EagerValue]): Input `_eval_fn` argument ( eval fn).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    args_seq = node.args if isinstance(node, ast.Call) else []
    return np.array([_eval_fn(a) for a in args_seq])


def _is_np_array_call(node: ast.AST) -> bool:
    """Check if the AST Call node is `np.array`.

    Args:
        node: The AST Call node.

    Returns:
        True if the node represents a call to `np.array`, False otherwise.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return isinstance(node.func.value, ast.Name) and node.func.value.id == "np" and node.func.attr == "array"
    return False


def _eval_call(node: ast.AST, _eval_fn: Callable[[ast.AST], EagerValue]) -> EagerValue:
    """Execute the eager eval call operation on the active backend array.

    Args:
        node (ast.AST): Input `node` argument (node).
        _eval_fn (Callable[[ast.AST], EagerValue]): Input `_eval_fn` argument ( eval fn).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "slice":
            return _eval_slice_call(node, _eval_fn)
        if node.func.id == "array":
            return _eval_array_call(node, _eval_fn)
    if _is_np_array_call(node):
        return _eval_array_call(node, _eval_fn)
    raise ValueError("Unsupported Call")


def _eval_name(node: ast.AST) -> EagerValue:
    """Execute the eager eval name operation on the active backend array.

    Args:
        node (ast.AST): Input `node` argument (node).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    node_id = node.id if isinstance(node, ast.Name) else ""
    if node_id == "Ellipsis":
        return Ellipsis
    if node_id == "None":
        return None
    raise ValueError("Unsupported Name")


def _eval_unary_op(node: ast.AST, _eval_fn: Callable[[ast.AST], EagerValue]) -> EagerValue:
    """Execute the eager eval unary op operation on the active backend array.

    Args:
        node (ast.AST): Input `node` argument (node).
        _eval_fn (Callable[[ast.AST], EagerValue]): Input `_eval_fn` argument ( eval fn).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _eval_fn(node.operand)
        if isinstance(val, (int, float)):
            return -val
    raise ValueError("Unsupported UnaryOp")


def _get_node_evaluators() -> dict[type[ast.AST], Callable[[ast.AST, Callable[[ast.AST], EagerValue]], EagerValue]]:
    """Get the dictionary mapping AST node types to evaluation functions.

    Returns:
        A dictionary mapping AST node types to their corresponding evaluation functions.
    """
    return {
        ast.Constant: lambda n, e: _eval_constant(n),
        ast.Tuple: lambda n, e: tuple(e(elt) for elt in (n.elts if isinstance(n, ast.Tuple) else [])),
        ast.List: lambda n, e: [e(elt) for elt in (n.elts if isinstance(n, ast.List) else [])],
        ast.Call: _eval_call,
        ast.Name: lambda n, e: _eval_name(n),
        ast.UnaryOp: _eval_unary_op,
    }


def _safe_parse_key(key_str: str) -> EagerValue:
    """Safely parse a stringified indexing key.

    Args:
        key_str (str): Input `key_str` argument (key str).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tree = ast.parse(key_str, mode="eval").body
    evaluators = _get_node_evaluators()

    def _eval(node: ast.AST) -> EagerValue:
        """Execute the eager eval operation on the active backend array.

        Args:
            node (ast.AST): Input `node` argument (node).

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        node_type = type(node)
        if node_type in evaluators:
            eval_fn = evaluators[node_type]
            return eval_fn(node, _eval)
        raise ValueError(f"Unsupported AST node: {node_type}")

    return _eval(tree)
