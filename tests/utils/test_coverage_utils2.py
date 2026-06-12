"""Docstring module."""

import pytest
from ml_switcheroo.core import ConfigContext, DType, Tensor, Device, DeviceType
from ml_switcheroo.tracing import ProxyTensor
from ml_switcheroo.ops.binary.frontend import _emit_binary_node
from ml_switcheroo.ops.unary.frontend import _emit_unary_node


def test_emit_helpers_coverage() -> None:
    """Docstring."""
    t1 = Tensor(
        ProxyTensor("a", (2,), DType.Float32.value),
        (2,),
        DType.Float32,
        Device(DeviceType.CPU),
    )  # noqa: E501
    t2 = Tensor(
        ProxyTensor("b", (2,), DType.Float32.value),
        (2,),
        DType.Float32,
        Device(DeviceType.CPU),
    )  # noqa: E501

    with ConfigContext(eager_mode=True):
        with pytest.raises(RuntimeError, match="Cannot emit node in eager mode"):
            _emit_binary_node(t1, t2, "Add")

        with pytest.raises(RuntimeError, match="Cannot emit node in eager mode"):
            _emit_unary_node(t1, "Abs")

    with ConfigContext(eager_mode=False):
        # We need to be in tracing mode to not raise another error.
        from ml_switcheroo.tracing import _tracer

        _tracer.start_tracing()
        try:
            res1 = _emit_binary_node(t1, t2, "Add", out_dtype=None)
            assert res1.dtype == DType.Float32
            res2 = _emit_unary_node(t1, "Abs", out_dtype=None)
            assert res2.dtype == DType.Float32
        finally:
            _tracer.stop_tracing()


def test_proxy_magic_methods() -> None:
    """Docstring."""
    from ml_switcheroo.tracing import _tracer

    t1 = ProxyTensor("a", (2,), DType.Float32.value)
    t2 = ProxyTensor("b", (2,), DType.Float32.value)

    with pytest.raises(RuntimeError):
        t1._unary_op("Neg")
    with pytest.raises(RuntimeError):
        t1[0]

    _tracer.start_tracing()
    try:
        # Binary
        _ = t1 // t2
        _ = t2.__rfloordiv__(t1)
        _ = t1 % t2
        _ = t2.__rmod__(t1)
        _ = t1 & t2
        _ = t2.__rand__(t1)
        _ = t1 | t2
        _ = t2.__ror__(t1)
        _ = t1 ^ t2
        _ = t2.__rxor__(t1)
        _ = t1 << t2
        _ = t2.__rlshift__(t1)
        _ = t1 >> t2
        _ = t2.__rrshift__(t1)

        # Unary
        _ = -t1
        _ = +t1
        _ = abs(t1)
        _ = ~t1

        # Getitem
        _ = t1[0]

    finally:
        _tracer.stop_tracing()
