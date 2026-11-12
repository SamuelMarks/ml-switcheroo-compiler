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
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


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
    """Test function enable_tracing_mode.

    Yields:
        None: Yielded test context or value from `enable_tracing_mode`.
    """
    config.eager_mode = False
    global_tracing_state.start_tracing()
    yield
    global_tracing_state.stop_tracing()


def test_cond_tracing() -> None:
    """Test function test_cond_tracing."""
    pred = ensure_callable(Tensor)(ensure_callable(MockData)("pred"), ensure_callable(TensorConfig)((), DType.Bool, ensure_callable(Device)("cpu")))

    def true_fn() -> EagerValue:
        """Test function true_fn.

        Returns:
            EagerValue: Computed output value from `true_fn`.
        """
        return ensure_callable(Tensor)(ensure_callable(MockData)("t"), ensure_callable(TensorConfig)((), DType.Float32, ensure_callable(Device)("cpu")))

    def false_fn() -> EagerValue:
        """Test function false_fn.

        Returns:
            EagerValue: Computed output value from `false_fn`.
        """
        return ensure_callable(Tensor)(ensure_callable(MockData)("f"), ensure_callable(TensorConfig)((), DType.Float32, ensure_callable(Device)("cpu")))

    res = ensure_callable(cond_tracing)(pred, true_fn, false_fn)
    assert ensure_eager_tensor(ensure_eager_tensor(res).config).dtype == DType.Float32


def test_while_loop_tracing() -> None:
    """Test function test_while_loop_tracing."""

    def cond_fn(x: EagerValue) -> EagerValue:
        """Test function cond_fn.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `cond_fn`.
        """
        return ensure_callable(Tensor)(ensure_callable(MockData)("c"), ensure_callable(TensorConfig)((), DType.Bool, ensure_callable(Device)("cpu")))

    def body_fn(x: EagerValue) -> EagerValue:
        """Test function body_fn.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `body_fn`.
        """
        return [ensure_callable(Tensor)(ensure_callable(MockData)("b"), ensure_callable(TensorConfig)((), DType.Float32, ensure_callable(Device)("cpu")))]

    init_val = ensure_callable(Tensor)(ensure_callable(MockData)("init"), ensure_callable(TensorConfig)((), DType.Float32, ensure_callable(Device)("cpu")))
    res = ensure_callable(while_loop_tracing)(cond_fn, body_fn, [init_val])
    assert res[0].config.dtype == DType.Float32


def test_scan_tracing() -> None:
    """Test function test_scan_tracing."""

    def scan_fn(carry: EagerValue, x: EagerValue) -> EagerValue:
        """Test function scan_fn.

        Args:
            carry: Test parameter `carry`.
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `scan_fn`.
        """
        new_carry = ensure_callable(Tensor)(ensure_callable(MockData)("nc"), ensure_callable(TensorConfig)((), DType.Float32, ensure_callable(Device)("cpu")))
        return [new_carry], [new_carry]

    init = [ensure_callable(Tensor)(ensure_callable(MockData)("init"), ensure_callable(TensorConfig)((), DType.Float32, ensure_callable(Device)("cpu")))]
    xs = ensure_callable(Tensor)(ensure_callable(MockData)("xs"), ensure_callable(TensorConfig)((3,), DType.Float32, ensure_callable(Device)("cpu")))

    res_carry, res_ys = ensure_callable(scan_tracing)(scan_fn, init, xs)
    assert ensure_eager_tensor(ensure_eager_tensor(res_carry)[0].config).dtype == DType.Float32
    assert ensure_eager_tensor(ensure_eager_tensor(res_ys)[0].config).dtype == DType.Float32


def test_map_fn_tracing() -> None:
    """Test function test_map_fn_tracing."""

    def fn(x: EagerValue) -> EagerValue:
        """Test function fn.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `fn`.
        """
        return ensure_callable(Tensor)(ensure_callable(MockData)("m"), ensure_callable(TensorConfig)((), DType.Float32, ensure_callable(Device)("cpu")))

    elems = ensure_callable(Tensor)(ensure_callable(MockData)("elems"), ensure_callable(TensorConfig)((3,), DType.Float32, ensure_callable(Device)("cpu")))

    res = ensure_callable(map_fn_tracing)(fn, elems)
    assert ensure_eager_tensor(ensure_eager_tensor(res).config).dtype == DType.Float32


def test_pmap_tracing() -> None:
    """Test function test_pmap_tracing."""
    from ml_switcheroo_compiler.ops.control_flow.tracing import pmap_tracing

    @pmap_tracing
    def fn(x: EagerValue) -> EagerValue:
        """Test function fn.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `fn`.
        """
        return ensure_callable(Tensor)(ensure_callable(MockData)("p"), ensure_callable(TensorConfig)((), DType.Float32, ensure_callable(Device)("cpu")))

    x = ensure_callable(Tensor)(ensure_callable(MockData)("x"), ensure_callable(TensorConfig)((), DType.Float32, ensure_callable(Device)("cpu")))
    res = ensure_callable(fn)(x)
    assert ensure_eager_tensor(ensure_eager_tensor(res).config).dtype == DType.Float32


def test_stop_gradient_tracing() -> None:
    """Test function test_stop_gradient_tracing."""
    from ml_switcheroo_compiler.ops.control_flow.tracing import stop_gradient_tracing

    x = ensure_callable(Tensor)(ensure_callable(MockData)("x"), ensure_callable(TensorConfig)((), DType.Float32, ensure_callable(Device)("cpu")))
    res = stop_gradient_tracing(x)
    assert res.config.dtype == DType.Float32
    assert bool(res.config.requires_grad) is False


def test_assert_value_tracing() -> None:
    """Test function test_assert_value_tracing."""
    from ml_switcheroo_compiler.ops.control_flow.tracing import assert_value_tracing

    x = ensure_callable(Tensor)(ensure_callable(MockData)("x"), ensure_callable(TensorConfig)((), DType.Bool, ensure_callable(Device)("cpu")))
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
        ensure_callable(pmap_tracing)(ensure_callable(MagicMock)())(1)

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

        Returns:
            EagerValue: Computed output value from `c`.
        """
        return x

    def b(x: EagerValue) -> EagerValue:
        """Test function b.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `b`.
        """
        return x

    t = Tensor(ProxyTensor("t", (), DType.Int32.value), TensorConfig((), DType.Int32, None))

    global_tracing_state.start_tracing()
    res = ensure_callable(while_loop_tracing)(c, b, t)
    global_tracing_state.stop_tracing()
    assert isinstance(ensure_eager_tensor(res), Tensor)


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

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return x

    t = Tensor(ProxyTensor("t", (), DType.Int32.value), TensorConfig((), DType.Int32, None))
    global_tracing_state.start_tracing()
    res = ensure_callable(pmap_tracing)(f)(t, 1)
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
