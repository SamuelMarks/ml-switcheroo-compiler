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
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def make_tensor(val: EagerValue, shape: EagerValue = None) -> EagerValue:
    """Test function make_tensor."""
    val = np.array(val, dtype=np.float32)
    if shape is None:
        shape = val.shape
    return cast("Callable[..., EagerTensorProtocol]", Tensor)(val, cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape, DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))


def test_vjp_shape_dtype_inference() -> None:
    """Test function test_vjp_shape_dtype_inference."""

    def f(x: EagerValue, y: EagerValue) -> EagerValue:
        """Test function f."""
        return dispatch_op("Multiply", x, y)

    x = make_tensor(np.ones((2, 3), dtype=np.float32))
    y = make_tensor(np.ones((2, 3), dtype=np.float32))

    out, vjp_fn = cast("Callable[..., EagerTensorProtocol]", vjp)(f, x, y)

    assert cast("EagerTensorProtocol", getattr(out, "shape")) == (2, 3)
    assert cast("EagerTensorProtocol", getattr(out, "dtype")).value == "float32"

    cot = make_tensor(np.ones((2, 3), dtype=np.float32))
    grads = cast("Callable[..., EagerTensorProtocol]", vjp_fn)(cot)

    assert cast("Callable[..., EagerTensorProtocol]", len)(grads) == 2
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(grads, "__getitem__")(0)), "shape")) == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(grads, "__getitem__")(1)), "shape")) == (2, 3)
