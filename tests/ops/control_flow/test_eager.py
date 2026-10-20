"""Tests for test_eager."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow.eager import cond_eager, map_fn_eager, scan_eager, while_loop_eager

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


@pytest.fixture(autouse=True)
def enable_eager_mode() -> Iterator[None]:
    """Test function enable_eager_mode."""
    config.eager_mode = True
    config.backend = "numpy"
    yield
    config.eager_mode = False
    setattr(config, "backend", None)


def test_cond_eager_true() -> None:
    """Test function test_cond_eager_true."""
    pred = Tensor(np.array(True), TensorConfig((), DType.Bool, Device("cpu")))

    def true_fn() -> EagerValue:
        """Test function true_fn."""
        return Tensor(np.array(1.0), TensorConfig((), DType.Float32, Device("cpu")))

    def false_fn() -> EagerValue:
        """Test function false_fn."""
        return Tensor(np.array(0.0), TensorConfig((), DType.Float32, Device("cpu")))

    res = cond_eager(pred, true_fn, false_fn)
    assert typing.cast("EagerTensorProtocol", getattr(res, "numpy"))() == 1.0


def test_cond_eager_false() -> None:
    """Test function test_cond_eager_false."""
    pred = Tensor(np.array(False), TensorConfig((), DType.Bool, Device("cpu")))

    def true_fn() -> EagerValue:
        """Test function true_fn."""
        return Tensor(np.array(1.0), TensorConfig((), DType.Float32, Device("cpu")))

    def false_fn() -> EagerValue:
        """Test function false_fn."""
        return Tensor(np.array(0.0), TensorConfig((), DType.Float32, Device("cpu")))

    res = cond_eager(pred, true_fn, false_fn)
    assert typing.cast("EagerTensorProtocol", getattr(res, "numpy"))() == 0.0


def test_while_loop_eager() -> None:
    """Test function test_while_loop_eager."""

    def cond_fn(x: EagerValue) -> EagerValue:
        """Test function cond_fn.

        Args:
            x: Test parameter `x`.
        """
        return Tensor(typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerTensorProtocol", getattr(x, "__getitem__")(0)), "numpy"))() < 5, TensorConfig((), DType.Bool, Device("cpu")))

    def body_fn(x: EagerValue) -> EagerValue:
        """Test function body_fn.

        Args:
            x: Test parameter `x`.
        """
        return [Tensor(typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerTensorProtocol", getattr(x, "__getitem__")(0)), "numpy"))() + 1, TensorConfig((), DType.Float32, Device("cpu")))]

    init_val = Tensor(np.array(0.0), TensorConfig((), DType.Float32, Device("cpu")))
    res = typing.cast("EagerCallableProtocol", while_loop_eager)(cond_fn, body_fn, [init_val])
    assert res[0].numpy() == 5.0


def test_scan_eager() -> None:
    """Test function test_scan_eager."""
    from ml_switcheroo_compiler.ops.control_flow.eager import scan_eager

    def scan_fn(carry: EagerValue, x: EagerValue) -> EagerValue:
        """Test function scan_fn.

        Args:
            carry: Test parameter `carry`.
            x: Test parameter `x`.
        """
        new_carry = Tensor(typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerTensorProtocol", getattr(carry, "__getitem__")(0)), "numpy"))() + typing.cast("EagerTensorProtocol", getattr(x, "numpy"))(), TensorConfig((), DType.Float32, Device("cpu")))
        return [new_carry], [new_carry]

    init = [Tensor(np.array(0.0), TensorConfig((), DType.Float32, Device("cpu")))]
    xs = Tensor(np.array([1.0, 2.0, 3.0]), TensorConfig((3,), DType.Float32, Device("cpu")))

    res_carry, res_ys = typing.cast("EagerCallableProtocol", scan_eager)(scan_fn, init, xs)
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerTensorProtocol", getattr(res_carry, "__getitem__")(0)), "numpy"))() == 6.0


def test_map_fn_eager() -> None:
    """Test function test_map_fn_eager."""
    from ml_switcheroo_compiler.ops.control_flow.eager import map_fn_eager

    def fn(x: EagerValue) -> EagerValue:
        """Test function fn.

        Args:
            x: Test parameter `x`.
        """
        return Tensor(getattr(x, "numpy")() * 2, TensorConfig((), DType.Float32, Device("cpu")))

    elems = Tensor(np.array([1.0, 2.0, 3.0]), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = map_fn_eager(fn, elems)
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", typing.cast("EagerTensorProtocol", getattr(res, "numpy")))(), typing.cast("EagerCallableProtocol", np.array)([2.0, 4.0, 6.0]))


def test_pmap_eager() -> None:
    """Test function test_pmap_eager."""
    from ml_switcheroo_compiler.ops.control_flow.eager import pmap_eager

    @pmap_eager
    def fn(x: EagerValue) -> EagerValue:
        """Test function fn.

        Args:
            x: Test parameter `x`.
        """
        return Tensor(getattr(x, "numpy")() * 2, TensorConfig((), DType.Float32, Device("cpu")))

    x = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, Device("cpu")))
    res = typing.cast("EagerCallableProtocol", fn)(x)
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", typing.cast("EagerTensorProtocol", getattr(res, "numpy")))(), typing.cast("EagerCallableProtocol", np.array)([2.0, 4.0]))


def test_stop_gradient_eager() -> None:
    """Test function test_stop_gradient_eager."""
    from ml_switcheroo_compiler.ops.control_flow.eager import stop_gradient_eager

    x = Tensor(np.array(1.0), TensorConfig((), DType.Float32, Device("cpu")))
    res = stop_gradient_eager(x)
    assert res.config.requires_grad is False
    assert res.numpy() == 1.0


def test_assert_value_eager() -> None:
    """Test function test_assert_value_eager."""
    from ml_switcheroo_compiler.ops.control_flow.eager import assert_value_eager

    x = Tensor(np.array(True), TensorConfig((), DType.Bool, Device("cpu")))
    assert_value_eager(x, "test msg")


def test_while_loop_eager_non_tuple() -> None:
    """Test function test_while_loop_eager_non_tuple."""

    def cond_fn(x: EagerValue) -> EagerValue:
        """Test function cond_fn.

        Args:
            x: Test parameter `x`.
        """
        return Tensor(getattr(x, "numpy")() < 5, TensorConfig((), DType.Bool, Device("cpu")))

    def body_fn(x: EagerValue) -> EagerValue:
        """Test function body_fn.

        Args:
            x: Test parameter `x`.
        """
        return Tensor(getattr(x, "numpy")() + 1, TensorConfig((), DType.Float32, Device("cpu")))

    init_val = Tensor(np.array(0.0), TensorConfig((), DType.Float32, Device("cpu")))
    res = typing.cast("EagerCallableProtocol", while_loop_eager)(cond_fn, body_fn, init_val)
    assert res.numpy() == 5.0


def test_scan_tuple_return() -> None:
    # test scan returning a tuple for y
    """Test function test_scan_tuple_return."""

    def f(carry: EagerValue, x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            carry: Test parameter `carry`.
            x: Test parameter `x`.
        """
        return carry, (x, x)

    init = Tensor(np.array(0), TensorConfig((), DType.Int32, None))
    xs = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, None))

    with patch("ml_switcheroo_compiler.ops.control_flow.eager.get_active_backend") as mock_backend:
        backend = MagicMock()
        backend.execute_op.return_value = MagicMock(shape=(2, 2))
        mock_backend.return_value = backend

        carry, y = typing.cast("EagerCallableProtocol", scan_eager)(f, init, xs)
        assert y is not None


def test_map_fn_tuple_return() -> None:
    """Test function test_map_fn_tuple_return."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return (x, x)

    xs = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, None))
    with patch("ml_switcheroo_compiler.ops.control_flow.eager.get_active_backend") as mock_backend:
        backend = MagicMock()
        backend.execute_op.return_value = MagicMock(shape=(2, 2))
        mock_backend.return_value = backend

        y = map_fn_eager(f, xs)
        assert y is not None
