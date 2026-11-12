"""Tests for test_ad_shape_inference."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.jvp_vjp import vjp
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def make_tensor(val: EagerValue, shape: EagerValue = None) -> EagerValue:
    """Test function make_tensor.

    Args:
        val: Test parameter `val`.
        shape: Test parameter `shape`.

    Returns:
        EagerValue: Computed output value from `make_tensor`.
    """
    val = np.array(val, dtype=np.float32)
    if shape is None:
        shape = val.shape
    return ensure_callable(Tensor)(val, ensure_callable(TensorConfig)(shape, DType.Float32, ensure_callable(Device)("cpu")))


def test_vjp_shape_dtype_inference() -> None:
    """Test function test_vjp_shape_dtype_inference."""

    def f(x: EagerValue, y: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return dispatch_op("Multiply", x, y)

    x = make_tensor(np.ones((2, 3), dtype=np.float32))
    y = make_tensor(np.ones((2, 3), dtype=np.float32))

    out, vjp_fn = ensure_callable(vjp)(f, x, y)

    assert ensure_eager_tensor(out).shape == (2, 3)
    assert ensure_eager_tensor(ensure_eager_tensor(out).dtype).value == "float32"

    cot = make_tensor(np.ones((2, 3), dtype=np.float32))
    grads = ensure_callable(vjp_fn)(cot)

    assert ensure_callable(len)(grads) == 2
    assert ensure_eager_tensor(grads)[0].shape == (2, 3)
    assert ensure_eager_tensor(grads)[1].shape == (2, 3)
