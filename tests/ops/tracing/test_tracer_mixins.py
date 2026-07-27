# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.ir.core import LogicalGraph
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer_mixins import ProxyMathOverloadsMixin


class MockProxyTensor(ProxyMathOverloadsMixin):
    def __init__(self, id, shape, dtype):
        self.id = id
        self.shape = shape
        self.dtype = dtype


def test_tracer_mixins(mocker):
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
    from ml_switcheroo_compiler.ir.core import IRNode

    node = IRNode("1", "ReadVariable", [], attributes={"variable_name": "var1"}, shape_metadata=(2, 3))
    global_tracing_state.add_node(node)
    res6 = proxy.assign(42)
    assert res6.shape == ()
    res7 = proxy.assign(proxy)
    assert res7.shape == (2, 3)
    MockProxyTensor.__add__ = lambda self, other: self._binary_op(other, "Add")
    MockProxyTensor.__sub__ = lambda self, other: self._binary_op(other, "Sub")
    res8 = proxy.assign_add(proxy)
    assert res8.shape == (2, 3)
    res9 = proxy.assign_sub(proxy)
    assert res9.shape == (2, 3)
    proxy2 = MockProxyTensor("2", (2, 3), "float32")
    node2 = IRNode("2", "Add", ["1", "1"], shape_metadata=(2, 3))
    global_tracing_state.add_node(node2)
    with pytest.raises(ValueError):
        proxy2.assign(proxy)
    global_tracing_state.is_tracing = False
