"""Tests for test_tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import TracingError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow.tracing import _flatten_inputs, assert_value_tracing, cond_tracing, map_fn_tracing, pmap_tracing, scan_tracing, stop_gradient_tracing, while_loop_tracing
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockData:
    """Test class MockData."""

    def __init__(self, id_val: EagerValue = "mock") -> None:
        """Test function __init__."""
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
    pred = cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("pred"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Bool, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))

    def true_fn() -> EagerValue:
        """Test function true_fn."""
        return cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("t"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))

    def false_fn() -> EagerValue:
        """Test function false_fn."""
        return cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("f"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))

    res = cast("Callable[..., EagerTensorProtocol]", cond_tracing)(pred, true_fn, false_fn)
    assert cast("EagerTensorProtocol", getattr(res, "config")).dtype == DType.Float32


def test_while_loop_tracing() -> None:
    """Test function test_while_loop_tracing."""

    def cond_fn(x: EagerValue) -> EagerValue:
        """Test function cond_fn."""
        return cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("c"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Bool, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))

    def body_fn(x: EagerValue) -> EagerValue:
        """Test function body_fn."""
        return [cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("b"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))]

    init_val = cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("init"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))
    res = cast("Callable[..., EagerTensorProtocol]", while_loop_tracing)(cond_fn, body_fn, [init_val])
    assert res[0].config.dtype == DType.Float32


def test_scan_tracing() -> None:
    """Test function test_scan_tracing."""

    def scan_fn(carry: EagerValue, x: EagerValue) -> EagerValue:
        """Test function scan_fn."""
        new_carry = cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("nc"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))
        return [new_carry], [new_carry]

    init = [cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("init"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))]
    xs = cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("xs"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((3,), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))

    res_carry, res_ys = cast("Callable[..., EagerTensorProtocol]", scan_tracing)(scan_fn, init, xs)
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(res_carry, "__getitem__")(0)), "config")).dtype == DType.Float32
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(res_ys, "__getitem__")(0)), "config")).dtype == DType.Float32


def test_map_fn_tracing() -> None:
    """Test function test_map_fn_tracing."""

    def fn(x: EagerValue) -> EagerValue:
        """Test function fn."""
        return cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("m"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))

    elems = cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("elems"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((3,), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))

    res = cast("Callable[..., EagerTensorProtocol]", map_fn_tracing)(fn, elems)
    assert cast("EagerTensorProtocol", getattr(res, "config")).dtype == DType.Float32


def test_pmap_tracing() -> None:
    """Test function test_pmap_tracing."""
    from ml_switcheroo_compiler.ops.control_flow.tracing import pmap_tracing

    @pmap_tracing
    def fn(x: EagerValue) -> EagerValue:
        """Test function fn."""
        return cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("p"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))

    x = cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("x"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))
    res = cast("Callable[..., EagerTensorProtocol]", fn)(x)
    assert cast("EagerTensorProtocol", getattr(res, "config")).dtype == DType.Float32


def test_stop_gradient_tracing() -> None:
    """Test function test_stop_gradient_tracing."""
    from ml_switcheroo_compiler.ops.control_flow.tracing import stop_gradient_tracing

    x = cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("x"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))
    res = stop_gradient_tracing(x)
    assert res.config.dtype == DType.Float32
    assert bool(res.config.requires_grad) is False


def test_assert_value_tracing() -> None:
    """Test function test_assert_value_tracing."""
    from ml_switcheroo_compiler.ops.control_flow.tracing import assert_value_tracing

    x = cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockData)("x"), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Bool, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))
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
        cast("Callable[..., EagerTensorProtocol]", pmap_tracing)(cast("Callable[..., EagerTensorProtocol]", MagicMock)())(1)

    assert stop_gradient_tracing(1) == 1

    with patch("ml_switcheroo_compiler.ops.control_flow.tracing.record_assertion") as mock_assert:
        assert_value_tracing(MagicMock(), "msg")
        mock_assert.assert_called_once()


def test_while_loop_tensor_init() -> None:
    """Test function test_while_loop_tensor_init."""

    def c(x: EagerValue) -> EagerValue:
        """Test function c."""
        return x

    def b(x: EagerValue) -> EagerValue:
        """Test function b."""
        return x

    t = Tensor(ProxyTensor("t", (), DType.Int32.value), TensorConfig((), DType.Int32, None))

    global_tracing_state.start_tracing()
    res = cast("Callable[..., EagerTensorProtocol]", while_loop_tracing)(c, b, t)
    global_tracing_state.stop_tracing()
    assert isinstance(res, Tensor)


def test_flatten_inputs_other() -> None:
    """Test function test_flatten_inputs_other."""
    assert _flatten_inputs(1) == []


def test_pmap_tracing_other_args() -> None:
    """Test function test_pmap_tracing_other_args."""

    def f(x: EagerValue, y: EagerValue) -> EagerValue:
        """Test function f."""
        return x

    t = Tensor(ProxyTensor("t", (), DType.Int32.value), TensorConfig((), DType.Int32, None))
    global_tracing_state.start_tracing()
    res = cast("Callable[..., EagerTensorProtocol]", pmap_tracing)(f)(t, 1)
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
