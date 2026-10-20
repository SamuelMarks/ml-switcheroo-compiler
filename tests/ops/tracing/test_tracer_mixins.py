"""Tests for test_tracer_mixins."""

from __future__ import annotations

import typing

# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.ir.core import LogicalGraph
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer_mixins import ProxyMathOverloadsMixin

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class MockProxyTensor(ProxyMathOverloadsMixin):
    """Test class MockProxyTensor."""

    def __init__(self, id: EagerValue, shape: EagerValue, dtype: EagerValue) -> None:
        """Test function __init__.

        Args:
            id: Test parameter `id`.
            shape: Test parameter `shape`.
            dtype: Test parameter `dtype`.
        """
        setattr(self, "id", id)
        setattr(self, "shape", shape)
        setattr(self, "dtype", dtype)


def test_tracer_mixins(mocker: EagerTensorProtocol) -> None:
    """Test function test_tracer_mixins.

    Args:
        mocker: Test parameter `mocker`.
    """
    global_tracing_state.is_tracing = False
    proxy = MockProxyTensor("1", (2, 3), "float32")
    from ml_switcheroo_compiler.core.errors import TracingError

    with pytest.raises(TracingError):
        proxy._binary_op(1, "Add")
    with pytest.raises(TracingError):
        proxy._unary_op("Neg")
    with pytest.raises(TracingError):
        proxy[0]
    with pytest.raises(TracingError):
        proxy @ proxy
    with pytest.raises(TracingError):
        typing.cast("EagerCallableProtocol", proxy.assign)(proxy)
    global_tracing_state.is_tracing = True
    global_tracing_state.active_graph = LogicalGraph("Test")
    mocker.patch("ml_switcheroo_compiler.ir.shape_system.broadcast_shapes", return_value=(2, 3))
    res1 = proxy._binary_op(42, "Add")
    assert res1.shape == (2, 3)
    res2 = proxy._binary_op(proxy, "Mul")
    assert res2.shape == (2, 3)
    res3 = proxy._unary_op("Neg")
    assert res3.shape == (2, 3)
    res4 = proxy[0]
    assert res4.shape == (2, 3)
    with pytest.raises(ValueError):
        proxy @ 42
    mocker.patch("ml_switcheroo_compiler.ir.shape_system.matmul_shape", return_value=(2, 3))
    res5 = proxy @ proxy
    assert res5.shape == (2, 3)
    from ml_switcheroo_compiler.ir.core import IRNode

    node = IRNode("1", "ReadVariable", inputs=[], attributes={"variable_name": "var1"}, shape_metadata=(2, 3))
    typing.cast("EagerCallableProtocol", global_tracing_state.add_node)(node)
    res6 = typing.cast("EagerCallableProtocol", proxy.assign)(42)
    assert res6.shape == ()
    res7 = typing.cast("EagerCallableProtocol", proxy.assign)(proxy)
    assert res7.shape == (2, 3)
    setattr(MockProxyTensor, "__add__", lambda self, other: typing.cast("EagerCallableProtocol", self._binary_op)(other, "Add"))
    setattr(MockProxyTensor, "__sub__", lambda self, other: typing.cast("EagerCallableProtocol", self._binary_op)(other, "Sub"))
    res8 = typing.cast("EagerCallableProtocol", proxy.assign_add)(proxy)
    assert res8.shape == (2, 3)
    res9 = typing.cast("EagerCallableProtocol", proxy.assign_sub)(proxy)
    assert res9.shape == (2, 3)
    proxy2 = MockProxyTensor("2", (2, 3), "float32")
    node2 = IRNode("2", "Add", inputs=["1", "1"], shape_metadata=(2, 3))
    typing.cast("EagerCallableProtocol", global_tracing_state.add_node)(node2)
    with pytest.raises(ValueError):
        typing.cast("EagerCallableProtocol", proxy2.assign)(proxy)
    global_tracing_state.is_tracing = False
