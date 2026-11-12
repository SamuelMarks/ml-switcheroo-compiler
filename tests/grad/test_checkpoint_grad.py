"""Tests for test_checkpoint_grad."""

from __future__ import annotations

import typing

from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_checkpoint_grad() -> None:
    """Test function test_checkpoint_grad."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.api import value_and_grad
    from ml_switcheroo_compiler.grad.checkpointing import checkpoint

    config.backend = "numpy"
    config.eager_mode = False

    def my_fun(x: EagerValue) -> EagerValue:
        """Test function my_fun.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `my_fun`.
        """
        return (ensure_eager_tensor(x) * ensure_eager_tensor(x)) * ensure_eager_tensor(x)

    cp_fun = checkpoint(my_fun)

    vg_std = value_and_grad(my_fun)
    vg_cp = value_and_grad(cp_fun)

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig(shape=(1,), dtype="float32", device="cpu"))

    from ml_switcheroo_compiler.grad.jit import jit

    jitted_std = ensure_callable(jit)(vg_std)
    jitted_cp = ensure_callable(jit)(vg_cp)

    v1, g1 = jitted_std(x)
    v2, g2 = jitted_cp(x)

    ensure_callable(np.testing.assert_allclose)(ensure_callable(ensure_eager_tensor(ensure_eager_tensor(v1).numpy))(), ensure_callable(ensure_eager_tensor(ensure_eager_tensor(v2).numpy))())
    ensure_callable(np.testing.assert_allclose)(ensure_callable(ensure_eager_tensor(ensure_eager_tensor(g1).numpy))(), ensure_callable(ensure_eager_tensor(ensure_eager_tensor(g2).numpy))())
