"""Tests for test_gradient_hooks."""

from __future__ import annotations

from tests.testing_utils import ensure_tensor, unpack_pair


def test_gradient_hooks() -> None:
    """Test function test_gradient_hooks."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.api import hook_gradient, value_and_grad

    config.backend = "numpy"
    config.eager_mode = False

    def my_fun(x: Tensor) -> Tensor:
        # We hook the intermediate value
        """Test function my_fun.

        Args:
            x (Tensor): Test parameter `x`.

        Returns:
            Tensor: Computed output value from `my_fun`.
        """
        y = x * 2.0

        # multiply gradient by 10
        def my_hook(grad: Tensor) -> Tensor:
            """Test function my_hook.

            Args:
                grad (Tensor): Test parameter `grad`.

            Returns:
                Tensor: Computed output value from `my_hook`.
            """
            return grad * 10.0

        y_hooked = hook_gradient(y, my_hook)
        return ensure_tensor(y_hooked) * 3.0

    vg = value_and_grad(my_fun)

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig(shape=(1,), dtype="float32", device="cpu"))

    from ml_switcheroo_compiler.grad.jit import jit

    jitted = jit(vg)

    v1, g1 = unpack_pair(jitted(x))

    # Forward: 2.0 * 2.0 = 4.0, 4.0 * 3.0 = 12.0
    # Backward: grad of x = grad(x*2)*2.
    # out grad = 1.0. grad of y_hooked = 3.0.
    # hook multiplies by 10.0 -> 30.0
    # grad of x = 30.0 * 2.0 = 60.0

    np.testing.assert_allclose(
        np.asarray(ensure_tensor(v1).numpy()),
        np.array([12.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(ensure_tensor(g1).numpy()),
        np.array([60.0], dtype=np.float32),
    )
