"""Tests for test_checkpoint_grad."""

from __future__ import annotations

from tests.testing_utils import ensure_tensor, unpack_pair


def test_checkpoint_grad() -> None:
    """Test function test_checkpoint_grad."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.api import value_and_grad
    from ml_switcheroo_compiler.grad.checkpointing import checkpoint

    config.backend = "numpy"
    config.eager_mode = False

    def my_fun(x: Tensor) -> Tensor:
        """Test function my_fun.

        Args:
            x (Tensor): Test parameter `x`.

        Returns:
            Tensor: Computed output value from `my_fun`.
        """
        return (x * x) * x

    cp_fun = checkpoint(my_fun)

    vg_std = value_and_grad(my_fun)
    vg_cp = value_and_grad(cp_fun)

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig(shape=(1,), dtype="float32", device="cpu"))

    from ml_switcheroo_compiler.grad.jit import jit

    jitted_std = jit(vg_std)
    jitted_cp = jit(vg_cp)

    v1, g1 = unpack_pair(jitted_std(x))
    v2, g2 = unpack_pair(jitted_cp(x))

    np.testing.assert_allclose(
        np.asarray(ensure_tensor(v1).numpy()),
        np.asarray(ensure_tensor(v2).numpy()),
    )
    np.testing.assert_allclose(
        np.asarray(ensure_tensor(g1).numpy()),
        np.asarray(ensure_tensor(g2).numpy()),
    )
