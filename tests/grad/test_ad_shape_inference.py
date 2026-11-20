"""Tests for test_ad_shape_inference."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.jvp_vjp import vjp
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op
from tests.testing_utils import extract_test_dtype, extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def make_tensor(val: BackendArg, shape: tuple[int, ...] | None = None) -> Tensor:
    """Test function make_tensor.

    Args:
        val: Test parameter `val`.
        shape: Test parameter `shape`.

    Returns:
        Tensor: Computed output value from `make_tensor`.
    """
    arr = np.array(val, dtype=np.float32)
    resolved_shape: tuple[int, ...] = shape if shape is not None else arr.shape
    return Tensor(arr, TensorConfig(resolved_shape, DType.Float32, Device("cpu")))


def test_vjp_shape_dtype_inference() -> None:
    """Test function test_vjp_shape_dtype_inference."""

    def f(x: BackendArg, y: BackendArg) -> BackendArg:
        """Test function f.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.

        Returns:
            BackendArg: Computed output value from `f`.
        """
        return dispatch_op("Multiply", x, y)

    x = make_tensor(np.ones((2, 3), dtype=np.float32))
    y = make_tensor(np.ones((2, 3), dtype=np.float32))

    out, vjp_fn = vjp(f, x, y)

    assert extract_test_shape(out) == (2, 3)
    out_dt = extract_test_dtype(out)
    assert isinstance(out_dt, DType) and out_dt.value == "float32"

    cot = make_tensor(np.ones((2, 3), dtype=np.float32))
    grads = vjp_fn(cot)

    assert len(grads) == 2
    assert extract_test_shape(grads[0]) == (2, 3)
    assert extract_test_shape(grads[1]) == (2, 3)
