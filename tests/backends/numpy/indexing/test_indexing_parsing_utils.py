"""Tests for numpy eager indexing parsing utils."""

import ast

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager._indexing_parsing_utils import (
    _eval_array_call,
    _eval_call,
    _eval_constant,
    _eval_name,
    _eval_slice_call,
    _eval_unary_op,
    _is_np_array_call,
    _safe_parse_key,
)


def test_eval_constant() -> None:
    """Test eval constant.

    Returns:
        None
    """
    node = ast.Constant(value=5)
    assert _eval_constant(node) == 5

    node2 = ast.Constant(value=Ellipsis)
    assert _eval_constant(node2) is Ellipsis


def test_eval_slice_call() -> None:
    """Test eval slice call.

    Returns:
        None
    """
    node = ast.Call(
        func=ast.Name(id="slice", ctx=ast.Load()),
        args=[ast.Constant(value=1), ast.Constant(value=5), ast.Constant(value=2)],
        keywords=[],
    )

    def dummy_eval(n: ast.AST):
        return getattr(n, "value", None)

    s = _eval_slice_call(node, dummy_eval)
    assert isinstance(s, slice)
    assert s.start == 1
    assert s.stop == 5
    assert s.step == 2


def test_eval_array_call() -> None:
    """Test eval array call.

    Returns:
        None
    """
    node = ast.Call(
        func=ast.Name(id="array", ctx=ast.Load()),
        args=[ast.Constant(value=1), ast.Constant(value=2)],
        keywords=[],
    )

    def dummy_eval(n: ast.AST):
        return getattr(n, "value", None)

    arr = _eval_array_call(node, dummy_eval)
    assert isinstance(arr, np.ndarray)
    assert arr.tolist() == [1, 2]


def test_is_np_array_call() -> None:
    """Test is np array call.

    Returns:
        None
    """
    node = ast.Call(
        func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="array", ctx=ast.Load()),
        args=[],
        keywords=[],
    )
    assert _is_np_array_call(node)

    node_f1 = ast.Call(func=ast.Name(id="array", ctx=ast.Load()), args=[], keywords=[])
    assert not _is_np_array_call(node_f1)


def test_eval_call() -> None:
    """Test eval call.

    Returns:
        None
    """

    def dummy_eval(n: ast.AST):
        return getattr(n, "value", None)

    node_slice = ast.Call(func=ast.Name(id="slice", ctx=ast.Load()), args=[ast.Constant(value=1)], keywords=[])
    assert isinstance(_eval_call(node_slice, dummy_eval), slice)

    node_array = ast.Call(func=ast.Name(id="array", ctx=ast.Load()), args=[ast.Constant(value=1)], keywords=[])
    assert isinstance(_eval_call(node_array, dummy_eval), np.ndarray)

    node_np_array = ast.Call(
        func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="array", ctx=ast.Load()),
        args=[ast.Constant(value=1)],
        keywords=[],
    )
    assert isinstance(_eval_call(node_np_array, dummy_eval), np.ndarray)

    node_unsupported = ast.Call(func=ast.Name(id="foo", ctx=ast.Load()), args=[], keywords=[])
    with pytest.raises(ValueError, match="Unsupported Call"):
        _eval_call(node_unsupported, dummy_eval)


def test_eval_name() -> None:
    """Test eval name.

    Returns:
        None
    """
    node = ast.Name(id="Ellipsis", ctx=ast.Load())
    assert _eval_name(node) is Ellipsis

    node_none = ast.Name(id="None", ctx=ast.Load())
    assert _eval_name(node_none) is None

    node_unsupp = ast.Name(id="foo", ctx=ast.Load())
    with pytest.raises(ValueError, match="Unsupported Name"):
        _eval_name(node_unsupp)


def test_eval_unary_op() -> None:
    """Test eval unary op.

    Returns:
        None
    """

    def dummy_eval(n: ast.AST):
        return getattr(n, "value", None)

    node = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=5))
    assert _eval_unary_op(node, dummy_eval) == -5

    node_unsupp_op = ast.UnaryOp(op=ast.UAdd(), operand=ast.Constant(value=5))
    with pytest.raises(ValueError, match="Unsupported UnaryOp"):
        _eval_unary_op(node_unsupp_op, dummy_eval)

    node_unsupp_val = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value="foo"))
    with pytest.raises(ValueError, match="Unsupported UnaryOp"):
        _eval_unary_op(node_unsupp_val, dummy_eval)


def test_safe_parse_key() -> None:
    """Test safe parse key.

    Returns:
        None
    """
    assert _safe_parse_key("5") == 5

    assert _safe_parse_key("(1, 2)") == (1, 2)

    assert _safe_parse_key("[1, 2]") == [1, 2]

    s = _safe_parse_key("slice(1, 5)")
    assert isinstance(s, slice)
    assert s.start == 1
    assert s.stop == 5
    assert s.step is None

    arr = _safe_parse_key("np.array([1, 2])")
    assert isinstance(arr, np.ndarray)
    assert arr.tolist() == [[1, 2]]

    assert _safe_parse_key("-5") == -5

    with pytest.raises(ValueError, match="Unsupported AST node"):
        _safe_parse_key("1 + 1")
