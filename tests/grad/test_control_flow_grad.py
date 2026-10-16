"""Tests for test_control_flow_grad."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import vjp
from ml_switcheroo_compiler.ops.control_flow import cond, while_loop

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_cond_grad() -> None:
    """Test function test_cond_grad."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f."""

        def true_fn(x: EagerValue) -> EagerValue:
            """Test function true_fn."""
            return (cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", x)) * cast("EagerTensorProtocol", x)

        def false_fn(x: EagerValue) -> EagerValue:
            """Test function false_fn."""
            return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", x)

        c = cast("EagerTensorProtocol", x) > cast("EagerTensorProtocol", 0.0)
        return cast("Callable[..., EagerTensorProtocol]", cond)(c, lambda: cast("Callable[..., EagerTensorProtocol]", true_fn)(x), lambda: cast("Callable[..., EagerTensorProtocol]", false_fn)(x))

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    out, vjp_fn = cast("Callable[..., EagerTensorProtocol]", vjp)(f, x)
    assert out is not None
    # Test backward pass graph construction
    gx = cast("Callable[..., EagerTensorProtocol]", vjp_fn)(
        cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", np.array)([1.0], dtype=np.float32), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((1,), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))
    )
    assert gx is not None


def test_while_loop_grad() -> None:
    """Test function test_while_loop_grad."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f."""

        def cond_fn(val: EagerValue) -> EagerValue:
            """Test function cond_fn."""
            return cast("EagerValue", getattr(val, "__getitem__")(0) < 10.0)

        def body_fn(val: EagerValue) -> EagerValue:
            """Test function body_fn."""
            return (getattr(val, "__getitem__")(0) + x,)

        res = cast("Callable[..., EagerTensorProtocol]", while_loop)(cond_fn, body_fn, (x,))
        return res[0]

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    out, vjp_fn = cast("Callable[..., EagerTensorProtocol]", vjp)(f, x)
    assert out is not None
    gx = cast("Callable[..., EagerTensorProtocol]", vjp_fn)(
        cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", np.array)([1.0], dtype=np.float32), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((1,), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))
    )
    assert gx is not None
