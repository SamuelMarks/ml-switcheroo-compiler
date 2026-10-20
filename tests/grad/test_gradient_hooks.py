"""Tests for test_gradient_hooks."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


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
        """Test function my_fun.

        Args:
            x: Test parameter `x`.
        """
        y = typing.cast("EagerTensorProtocol", x) * typing.cast("EagerTensorProtocol", 2.0)

        # multiply gradient by 10
        def my_hook(grad: EagerValue) -> EagerValue:
            """Test function my_hook.

            Args:
                grad: Test parameter `grad`.
            """
            return typing.cast("EagerTensorProtocol", grad) * typing.cast("EagerTensorProtocol", 10.0)

        y_hooked = hook_gradient(y, my_hook)
        return typing.cast("EagerTensorProtocol", y_hooked) * typing.cast("EagerTensorProtocol", 3.0)

    vg = value_and_grad(my_fun)

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig(shape=(1,), dtype="float32", device="cpu"))

    from ml_switcheroo_compiler.grad.jit import jit

    jitted = typing.cast("EagerCallableProtocol", jit)(vg)

    v1, g1 = jitted(x)

    # Forward: 2.0 * 2.0 = 4.0, 4.0 * 3.0 = 12.0
    # Backward: grad of x = grad(x*2)*2.
    # out grad = 1.0. grad of y_hooked = 3.0.
    # hook multiplies by 10.0 -> 30.0
    # grad of x = 30.0 * 2.0 = 60.0

    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", typing.cast("EagerTensorProtocol", getattr(v1, "numpy")))(), typing.cast("EagerCallableProtocol", np.array)([12.0], dtype=np.float32))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", typing.cast("EagerTensorProtocol", getattr(g1, "numpy")))(), typing.cast("EagerCallableProtocol", np.array)([60.0], dtype=np.float32))
