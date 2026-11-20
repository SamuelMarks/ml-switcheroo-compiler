"""Tests for test_tracing."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import TracingError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow.tracing import (
    _flatten_inputs,
    assert_value_tracing,
    cond_tracing,
    map_fn_tracing,
    pmap_tracing,
    scan_tracing,
    stop_gradient_tracing,
    while_loop_tracing,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from tests.testing_utils import (
    MockTestStub,
    coerce_operand_list,
    ensure_tensor,
    extract_test_shape,
    require_callable,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockData(MockTestStub):
    """Test class MockData."""

    def __init__(self, id_val: BackendArg = "mock") -> None:
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
    pred = Tensor(MockData("pred"), TensorConfig((), DType.Bool, Device("cpu")))

    def true_fn() -> BackendArg:
        """Test function true_fn.

        Returns:
            BackendArg: Computed output value from `true_fn`.
        """
        return Tensor(
            MockData("t"),
            TensorConfig((), DType.Float32, Device("cpu")),
        )

    def false_fn() -> BackendArg:
        """Test function false_fn.

        Returns:
            BackendArg: Computed output value from `false_fn`.
        """
        return Tensor(
            MockData("f"),
            TensorConfig((), DType.Float32, Device("cpu")),
        )

    res = cond_tracing(pred, true_fn, false_fn)
    assert res.config.dtype == DType.Float32


def test_while_loop_tracing() -> None:
    """Test function test_while_loop_tracing."""

    def cond_fn(x: BackendArg) -> BackendArg:
        """Test function cond_fn.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `cond_fn`.
        """
        return Tensor(
            MockData("c"),
            TensorConfig((), DType.Bool, Device("cpu")),
        )

    def body_fn(x: list[Tensor]) -> list[Tensor]:
        """Test function body_fn.

        Args:
            x: Test parameter `x`.

        Returns:
            list[Tensor]: Computed output value from `body_fn`.
        """
        del x
        return [
            Tensor(
                MockData("b"),
                TensorConfig((), DType.Float32, Device("cpu")),
            )
        ]

    init_val = Tensor(
        MockData("init"),
        TensorConfig((), DType.Float32, Device("cpu")),
    )
    res = while_loop_tracing(cond_fn, body_fn, [init_val])
    assert res[0].config.dtype == DType.Float32


def test_scan_tracing() -> None:
    """Test function test_scan_tracing."""

    def scan_fn(carry: BackendArg, x: BackendArg) -> BackendArg:
        """Test function scan_fn.

        Args:
            carry: Test parameter `carry`.
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `scan_fn`.
        """
        new_carry = Tensor(
            MockData("nc"),
            TensorConfig((), DType.Float32, Device("cpu")),
        )
        return [new_carry], [new_carry]

    init = [
        Tensor(
            MockData("init"),
            TensorConfig((), DType.Float32, Device("cpu")),
        )
    ]
    xs = Tensor(
        MockData("xs"),
        TensorConfig((3,), DType.Float32, Device("cpu")),
    )

    res_carry, res_ys = scan_tracing(scan_fn, init, xs)
    assert ensure_tensor(coerce_operand_list(res_carry)[0]).config.dtype == DType.Float32
    assert res_ys[0].config.dtype == DType.Float32


def test_map_fn_tracing() -> None:
    """Test function test_map_fn_tracing."""

    def fn(x: BackendArg) -> BackendArg:
        """Test function fn.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `fn`.
        """
        return Tensor(
            MockData("m"),
            TensorConfig((), DType.Float32, Device("cpu")),
        )

    elems = Tensor(
        MockData("elems"),
        TensorConfig((3,), DType.Float32, Device("cpu")),
    )

    res = map_fn_tracing(fn, elems)
    assert res.config.dtype == DType.Float32


def test_pmap_tracing() -> None:
    """Test function test_pmap_tracing."""
    from ml_switcheroo_compiler.ops.control_flow.tracing import pmap_tracing

    @pmap_tracing
    def fn(x: BackendArg) -> BackendArg:
        """Test function fn.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `fn`.
        """
        return Tensor(
            MockData("p"),
            TensorConfig((), DType.Float32, Device("cpu")),
        )

    x = Tensor(MockData("x"), TensorConfig((), DType.Float32, Device("cpu")))
    res = require_callable(fn)(x)
    assert res.config.dtype == DType.Float32


def test_stop_gradient_tracing() -> None:
    """Test function test_stop_gradient_tracing."""
    from ml_switcheroo_compiler.ops.control_flow.tracing import stop_gradient_tracing

    x = Tensor(MockData("x"), TensorConfig((), DType.Float32, Device("cpu")))
    res = stop_gradient_tracing(x)
    assert res.config.dtype == DType.Float32
    assert bool(res.config.requires_grad) is False


def test_assert_value_tracing() -> None:
    """Test function test_assert_value_tracing."""
    from ml_switcheroo_compiler.ops.control_flow.tracing import assert_value_tracing

    x = Tensor(MockData("x"), TensorConfig((), DType.Bool, Device("cpu")))
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
        require_callable(pmap_tracing)(require_callable(MagicMock)())(1)

    assert stop_gradient_tracing(1) == 1

    with patch("ml_switcheroo_compiler.ops.control_flow.tracing.record_assertion") as mock_assert:
        assert_value_tracing(MagicMock(), "msg")
        mock_assert.assert_called_once()


def test_while_loop_tensor_init() -> None:
    """Test function test_while_loop_tensor_init."""

    def c(x: Tensor) -> Tensor:
        """Test function c.

        Args:
            x (Tensor): Test parameter `x`.

        Returns:
            Tensor: Computed output value from `c`.
        """
        return x

    def b(x: Tensor) -> Tensor:
        """Test function b.

        Args:
            x (Tensor): Test parameter `x`.

        Returns:
            Tensor: Computed output value from `b`.
        """
        return x

    t = Tensor(ProxyTensor("t", (), DType.Int32.value), TensorConfig((), DType.Int32, None))

    global_tracing_state.start_tracing()
    res = while_loop_tracing(c, b, t)
    global_tracing_state.stop_tracing()
    assert isinstance(res, Tensor)


def test_flatten_inputs_other() -> None:
    """Test function test_flatten_inputs_other."""
    assert _flatten_inputs(1) == []


def test_pmap_tracing_other_args() -> None:
    """Test function test_pmap_tracing_other_args."""

    def f(x: BackendArg, y: BackendArg) -> BackendArg:
        """Test function f.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.

        Returns:
            BackendArg: Computed output value from `f`.
        """
        return x

    t = Tensor(ProxyTensor("t", (), DType.Int32.value), TensorConfig((), DType.Int32, None))
    global_tracing_state.start_tracing()
    res = require_callable(pmap_tracing)(f)(t, 1)
    global_tracing_state.stop_tracing()
    assert tuple(extract_test_shape(res)) == ()


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


def test_control_flow_invalid_traced_and_indexable_inputs() -> None:
    """Verify _extract_traced_id and _index_cf_data raise clear errors on invalid payloads."""
    from ml_switcheroo_compiler.ops.control_flow.eager import _index_cf_data
    from ml_switcheroo_compiler.ops.control_flow.tracing import _extract_traced_id

    with pytest.raises(AttributeError, match="has no attribute 'id'"):
        _extract_traced_id(123)
    with pytest.raises(TypeError, match="not subscriptable"):
        _index_cf_data(123, 0)
