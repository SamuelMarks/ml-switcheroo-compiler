"""Tests for test_gradient_hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_gradient_hooks() -> None:
    """Test function test_gradient_hooks."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.api import hook_gradient, value_and_grad

    config.backend = "numpy"
    config.eager_mode = False

    def my_fun(x: EagerValue) -> EagerValue:
        # We hook the intermediate value
        """Test function my_fun."""
        y = cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", 2.0)

        # multiply gradient by 10
        def my_hook(grad: EagerValue) -> EagerValue:
            """Test function my_hook."""
            return cast("EagerTensorProtocol", grad) * cast("EagerTensorProtocol", 10.0)

        y_hooked = hook_gradient(y, my_hook)
        return cast("EagerTensorProtocol", y_hooked) * cast("EagerTensorProtocol", 3.0)

    vg = value_and_grad(my_fun)

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig(shape=(1,), dtype="float32", device="cpu"))

    from ml_switcheroo_compiler.grad.jit import jit

    jitted = cast("Callable[..., EagerTensorProtocol]", jit)(vg)

    v1, g1 = jitted(x)

    # Forward: 2.0 * 2.0 = 4.0, 4.0 * 3.0 = 12.0
    # Backward: grad of x = grad(x*2)*2.
    # out grad = 1.0. grad of y_hooked = 3.0.
    # hook multiplies by 10.0 -> 30.0
    # grad of x = 30.0 * 2.0 = 60.0

    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(v1, "numpy")))(), cast("Callable[..., EagerTensorProtocol]", np.array)([12.0], dtype=np.float32))
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(g1, "numpy")))(), cast("Callable[..., EagerTensorProtocol]", np.array)([60.0], dtype=np.float32))
