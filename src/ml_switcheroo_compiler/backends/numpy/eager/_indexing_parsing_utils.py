"""Utilities for parsing stringified indexing keys in NumPy backend."""

import ast
from typing import Callable


def _eval_constant(node: ast.AST) -> object:
    """Evaluate an AST Constant node.

    Args:
        node: The AST node to evaluate.

    Returns:
        The constant value.
    """
    if getattr(node, "value", None) is Ellipsis:
        return Ellipsis
    return getattr(node, "value", None)


def _eval_slice_call(node: ast.AST, _eval_fn: Callable[[ast.AST], object]) -> slice:
    """Evaluate an AST Call node that represents a slice.

    Args:
        node: The AST Call node.
        _eval_fn: The evaluation function to recursively parse arguments.

    Returns:
        A slice object.
    """
    return slice(*[_eval_fn(a) for a in getattr(node, "args", [])])


def _eval_array_call(node: ast.AST, _eval_fn: Callable[[ast.AST], object]) -> object:
    """Evaluate an AST Call node that represents a numpy array.

    Args:
        node: The AST Call node.
        _eval_fn: The evaluation function to recursively parse arguments.

    Returns:
        A numpy array.
    """
    import numpy as np

    return np.array([_eval_fn(a) for a in getattr(node, "args", [])])


def _is_np_array_call(node: ast.AST) -> bool:
    """Check if the AST Call node is `np.array`.

    Args:
        node: The AST Call node.

    Returns:
        True if the node represents a call to `np.array`, False otherwise.
    """
    func = getattr(node, "func", None)
    if isinstance(func, ast.Attribute):
        return getattr(getattr(func, "value", None), "id", "") == "np" and getattr(func, "attr", "") == "array"
    return False


def _eval_call(node: ast.AST, _eval_fn: Callable[[ast.AST], object]) -> object:
    """Evaluate an AST Call node.

    Args:
        node: The AST Call node.
        _eval_fn: The evaluation function to recursively parse arguments.

    Returns:
        The result of the call evaluation.

    Raises:
        ValueError: If the call is not supported.
    """
    func = getattr(node, "func", None)
    if isinstance(func, ast.Name):
        if getattr(func, "id", "") == "slice":
            return _eval_slice_call(node, _eval_fn)
        if getattr(func, "id", "") == "array":
            return _eval_array_call(node, _eval_fn)
    if _is_np_array_call(node):
        return _eval_array_call(node, _eval_fn)
    raise ValueError("Unsupported Call")


def _eval_name(node: ast.AST) -> object:
    """Evaluate an AST Name node.

    Args:
        node: The AST Name node.

    Returns:
        The value represented by the name, such as Ellipsis or None.

    Raises:
        ValueError: If the name is not supported.
    """
    node_id = getattr(node, "id", "")
    if node_id == "Ellipsis":
        return Ellipsis
    if node_id == "None":
        return None
    raise ValueError("Unsupported Name")


def _eval_unary_op(node: ast.AST, _eval_fn: Callable[[ast.AST], object]) -> object:
    """Evaluate an AST UnaryOp node.

    Args:
        node: The AST UnaryOp node.
        _eval_fn: The evaluation function to recursively parse arguments.

    Returns:
        The evaluated value.

    Raises:
        ValueError: If the unary operation is not supported.
    """
    if isinstance(getattr(node, "op", None), ast.USub):
        val = _eval_fn(getattr(node, "operand", None))
        if isinstance(val, (int, float)):
            return -val
    raise ValueError("Unsupported UnaryOp")


def _get_node_evaluators() -> dict[type, Callable[..., object]]:
    """Get the dictionary mapping AST node types to evaluation functions.

    Returns:
        A dictionary mapping AST node types to their corresponding evaluation functions.
    """
    return {
        ast.Constant: lambda n, e: _eval_constant(n),
        ast.Tuple: lambda n, e: tuple(e(elt) for elt in getattr(n, "elts", [])),
        ast.List: lambda n, e: list(e(elt) for elt in getattr(n, "elts", [])),
        ast.Call: _eval_call,
        ast.Name: lambda n, e: _eval_name(n),
        ast.UnaryOp: _eval_unary_op,
    }


def _safe_parse_key(key_str: str) -> object:
    """Safely parse a stringified indexing key.

    Args:
        key_str (str): The stringified indexing key to parse.

    Returns:
        The parsed key, which can be an integer, slice, tuple, list, array, or Ellipsis.

    Raises:
        ValueError: If an unsupported AST node is encountered.
    """
    tree = ast.parse(key_str, mode="eval").body
    evaluators = _get_node_evaluators()

    def _eval(node: ast.AST) -> object:
        node_type = type(node)
        if node_type in evaluators:
            eval_fn = evaluators[node_type]
            return eval_fn(node, _eval)
        raise ValueError(f"Unsupported AST node: {node_type}")

    return _eval(tree)
