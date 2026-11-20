"""Tests for test_tracer_mixins."""

from __future__ import annotations

import typing

import pytest
from ml_switcheroo_ir import LogicalGraph
from pytest_mock import MockerFixture

from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from ml_switcheroo_compiler.tracing.tracer_mixins import ProxyMathOverloadsMixin

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockProxyTensor(ProxyMathOverloadsMixin):
    """Test class MockProxyTensor."""

    def __init__(self, id: BackendArg, shape: BackendArg, dtype: BackendArg) -> None:
        """Test function __init__.

        Args:
            id (BackendArg): Test parameter `id`.
            shape (BackendArg): Test parameter `shape`.
            dtype (BackendArg): Test parameter `dtype`.
        """
        pytest.MonkeyPatch().setattr(self, "id", id, raising=False)
        pytest.MonkeyPatch().setattr(self, "shape", shape, raising=False)
        pytest.MonkeyPatch().setattr(self, "dtype", dtype, raising=False)


def test_tracer_mixins(mocker: MockerFixture) -> None:
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
        proxy.assign(proxy)
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
    from ml_switcheroo_ir import IRNode

    node = IRNode("1", "ReadVariable", inputs=[], attributes={"variable_name": "var1"}, shape_metadata=(2, 3))
    global_tracing_state.add_node(node)
    res6 = proxy.assign(42)
    assert res6.shape == ()
    res7 = proxy.assign(proxy)
    assert res7.shape == (2, 3)

    class AddSubMockProxyTensor(ProxyTensor):
        """Mock proxy tensor subclassing `ProxyTensor` for `assign_add` and `assign_sub`."""

    proxy_addsub = AddSubMockProxyTensor("1", (2, 3), "float32")
    res8 = proxy_addsub.assign_add(proxy_addsub)
    assert res8.shape == (2, 3)
    res9 = proxy_addsub.assign_sub(proxy_addsub)
    assert res9.shape == (2, 3)
    proxy2 = MockProxyTensor("2", (2, 3), "float32")
    node2 = IRNode("2", "Add", inputs=["1", "1"], shape_metadata=(2, 3))
    global_tracing_state.add_node(node2)
    with pytest.raises(ValueError):
        proxy2.assign(proxy)
    global_tracing_state.is_tracing = False
