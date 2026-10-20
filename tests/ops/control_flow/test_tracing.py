"""Tests for test_tracing."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import TracingError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow.tracing import _flatten_inputs, assert_value_tracing, cond_tracing, map_fn_tracing, pmap_tracing, scan_tracing, stop_gradient_tracing, while_loop_tracing
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class MockData:
    """Test class MockData."""

    def __init__(self, id_val: EagerValue = "mock") -> None:
        """Test function __init__.

        Args:
            id_val: Test parameter `id_val`.
        """
        self.id = id_val


@pytest.fixture(autouse=True)
def enable_tracing_mode() -> Iterator[None]:
    """Test function enable_tracing_mode."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    yield
    global_tracing_state.stop_tracing()


def test_cond_tracing() -> None:
    """Test function test_cond_tracing."""
    pred = typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("pred"), typing.cast("EagerCallableProtocol", TensorConfig)((), DType.Bool, typing.cast("EagerCallableProtocol", Device)("cpu")))

    def true_fn() -> EagerValue:
        """Test function true_fn."""
        return typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("t"), typing.cast("EagerCallableProtocol", TensorConfig)((), DType.Float32, typing.cast("EagerCallableProtocol", Device)("cpu")))

    def false_fn() -> EagerValue:
        """Test function false_fn."""
        return typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("f"), typing.cast("EagerCallableProtocol", TensorConfig)((), DType.Float32, typing.cast("EagerCallableProtocol", Device)("cpu")))

    res = typing.cast("EagerCallableProtocol", cond_tracing)(pred, true_fn, false_fn)
    assert typing.cast("EagerTensorProtocol", getattr(res, "config")).dtype == DType.Float32


def test_while_loop_tracing() -> None:
    """Test function test_while_loop_tracing."""

    def cond_fn(x: EagerValue) -> EagerValue:
        """Test function cond_fn.

        Args:
            x: Test parameter `x`.
        """
        return typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("c"), typing.cast("EagerCallableProtocol", TensorConfig)((), DType.Bool, typing.cast("EagerCallableProtocol", Device)("cpu")))

    def body_fn(x: EagerValue) -> EagerValue:
        """Test function body_fn.

        Args:
            x: Test parameter `x`.
        """
        return [typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("b"), typing.cast("EagerCallableProtocol", TensorConfig)((), DType.Float32, typing.cast("EagerCallableProtocol", Device)("cpu")))]

    init_val = typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("init"), typing.cast("EagerCallableProtocol", TensorConfig)((), DType.Float32, typing.cast("EagerCallableProtocol", Device)("cpu")))
    res = typing.cast("EagerCallableProtocol", while_loop_tracing)(cond_fn, body_fn, [init_val])
    assert res[0].config.dtype == DType.Float32


def test_scan_tracing() -> None:
    """Test function test_scan_tracing."""

    def scan_fn(carry: EagerValue, x: EagerValue) -> EagerValue:
        """Test function scan_fn.

        Args:
            carry: Test parameter `carry`.
            x: Test parameter `x`.
        """
        new_carry = typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("nc"), typing.cast("EagerCallableProtocol", TensorConfig)((), DType.Float32, typing.cast("EagerCallableProtocol", Device)("cpu")))
        return [new_carry], [new_carry]

    init = [typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("init"), typing.cast("EagerCallableProtocol", TensorConfig)((), DType.Float32, typing.cast("EagerCallableProtocol", Device)("cpu")))]
    xs = typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("xs"), typing.cast("EagerCallableProtocol", TensorConfig)((3,), DType.Float32, typing.cast("EagerCallableProtocol", Device)("cpu")))

    res_carry, res_ys = typing.cast("EagerCallableProtocol", scan_tracing)(scan_fn, init, xs)
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerTensorProtocol", getattr(res_carry, "__getitem__")(0)), "config")).dtype == DType.Float32
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerTensorProtocol", getattr(res_ys, "__getitem__")(0)), "config")).dtype == DType.Float32


def test_map_fn_tracing() -> None:
    """Test function test_map_fn_tracing."""

    def fn(x: EagerValue) -> EagerValue:
        """Test function fn.

        Args:
            x: Test parameter `x`.
        """
        return typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("m"), typing.cast("EagerCallableProtocol", TensorConfig)((), DType.Float32, typing.cast("EagerCallableProtocol", Device)("cpu")))

    elems = typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("elems"), typing.cast("EagerCallableProtocol", TensorConfig)((3,), DType.Float32, typing.cast("EagerCallableProtocol", Device)("cpu")))

    res = typing.cast("EagerCallableProtocol", map_fn_tracing)(fn, elems)
    assert typing.cast("EagerTensorProtocol", getattr(res, "config")).dtype == DType.Float32


def test_pmap_tracing() -> None:
    """Test function test_pmap_tracing."""
    from ml_switcheroo_compiler.ops.control_flow.tracing import pmap_tracing

    @pmap_tracing
    def fn(x: EagerValue) -> EagerValue:
        """Test function fn.

        Args:
            x: Test parameter `x`.
        """
        return typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("p"), typing.cast("EagerCallableProtocol", TensorConfig)((), DType.Float32, typing.cast("EagerCallableProtocol", Device)("cpu")))

    x = typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("x"), typing.cast("EagerCallableProtocol", TensorConfig)((), DType.Float32, typing.cast("EagerCallableProtocol", Device)("cpu")))
    res = typing.cast("EagerCallableProtocol", fn)(x)
    assert typing.cast("EagerTensorProtocol", getattr(res, "config")).dtype == DType.Float32


def test_stop_gradient_tracing() -> None:
    """Test function test_stop_gradient_tracing."""
    from ml_switcheroo_compiler.ops.control_flow.tracing import stop_gradient_tracing

    x = typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("x"), typing.cast("EagerCallableProtocol", TensorConfig)((), DType.Float32, typing.cast("EagerCallableProtocol", Device)("cpu")))
    res = stop_gradient_tracing(x)
    assert res.config.dtype == DType.Float32
    assert bool(res.config.requires_grad) is False


def test_assert_value_tracing() -> None:
    """Test function test_assert_value_tracing."""
    from ml_switcheroo_compiler.ops.control_flow.tracing import assert_value_tracing

    x = typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockData)("x"), typing.cast("EagerCallableProtocol", TensorConfig)((), DType.Bool, typing.cast("EagerCallableProtocol", Device)("cpu")))
    assert_value_tracing(x, "test msg")


def test_tracing_error_outside_context() -> None:
    """Test function test_tracing_error_outside_context."""
    global_tracing_state.is_tracing = False
    with pytest.raises(TracingError):
        cond_tracing(MagicMock(), MagicMock(), MagicMock())
    with pytest.raises(TracingError):
        while_loop_tracing(MagicMock(), MagicMock(), MagicMock())
    with pytest.raises(TracingError):
        scan_tracing(MagicMock(), MagicMock(), MagicMock())
    with pytest.raises(TracingError):
        map_fn_tracing(MagicMock(), MagicMock())
    with pytest.raises(TracingError):
        typing.cast("EagerCallableProtocol", pmap_tracing)(typing.cast("EagerCallableProtocol", MagicMock)())(1)

    assert stop_gradient_tracing(1) == 1

    with patch("ml_switcheroo_compiler.ops.control_flow.tracing.record_assertion") as mock_assert:
        assert_value_tracing(MagicMock(), "msg")
        mock_assert.assert_called_once()


def test_while_loop_tensor_init() -> None:
    """Test function test_while_loop_tensor_init."""

    def c(x: EagerValue) -> EagerValue:
        """Test function c.

        Args:
            x: Test parameter `x`.
        """
        return x

    def b(x: EagerValue) -> EagerValue:
        """Test function b.

        Args:
            x: Test parameter `x`.
        """
        return x

    t = Tensor(ProxyTensor("t", (), DType.Int32.value), TensorConfig((), DType.Int32, None))

    global_tracing_state.start_tracing()
    res = typing.cast("EagerCallableProtocol", while_loop_tracing)(c, b, t)
    global_tracing_state.stop_tracing()
    assert isinstance(typing.cast("EagerValue", res), Tensor)


def test_flatten_inputs_other() -> None:
    """Test function test_flatten_inputs_other."""
    assert _flatten_inputs(1) == []


def test_pmap_tracing_other_args() -> None:
    """Test function test_pmap_tracing_other_args."""

    def f(x: EagerValue, y: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.
        """
        return x

    t = Tensor(ProxyTensor("t", (), DType.Int32.value), TensorConfig((), DType.Int32, None))
    global_tracing_state.start_tracing()
    res = typing.cast("EagerCallableProtocol", pmap_tracing)(f)(t, 1)
    global_tracing_state.stop_tracing()
    assert res is not None


def test_stop_gradient_tracing_traced() -> None:
    """Test function test_stop_gradient_tracing_traced."""
    t = Tensor(ProxyTensor("t", (), DType.Int32.value), TensorConfig((), DType.Int32, None))
    p = ProxyTensor("p", (), DType.Int32.value)

    global_tracing_state.start_tracing()
    res1 = stop_gradient_tracing(t)
    assert isinstance(res1, Tensor)

    res2 = stop_gradient_tracing(p)
    assert isinstance(res2, ProxyTensor)
    global_tracing_state.stop_tracing()
