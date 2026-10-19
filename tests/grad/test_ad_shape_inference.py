"""Tests for test_ad_shape_inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.jvp_vjp import vjp
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def make_tensor(val: EagerValue, shape: EagerValue = None) -> EagerValue:
    """Test function make_tensor.

    Args:
        val: Test parameter `val`.
        shape: Test parameter `shape`.
    """
    val = np.array(val, dtype=np.float32)
    if shape is None:
        shape = val.shape
    return cast("EagerCallableProtocol", Tensor)(val, cast("EagerCallableProtocol", TensorConfig)(shape, DType.Float32, cast("EagerCallableProtocol", Device)("cpu")))


def test_vjp_shape_dtype_inference() -> None:
    """Test function test_vjp_shape_dtype_inference."""

    def f(x: EagerValue, y: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.
        """
        return dispatch_op("Multiply", x, y)

    x = make_tensor(np.ones((2, 3), dtype=np.float32))
    y = make_tensor(np.ones((2, 3), dtype=np.float32))

    out, vjp_fn = cast("EagerCallableProtocol", vjp)(f, x, y)

    assert cast("EagerTensorProtocol", getattr(out, "shape")) == (2, 3)
    assert cast("EagerTensorProtocol", getattr(out, "dtype")).value == "float32"

    cot = make_tensor(np.ones((2, 3), dtype=np.float32))
    grads = cast("EagerCallableProtocol", vjp_fn)(cot)

    assert cast("EagerCallableProtocol", len)(grads) == 2
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(grads, "__getitem__")(0)), "shape")) == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(grads, "__getitem__")(1)), "shape")) == (2, 3)
