"""Tests for test_checkpoint_grad."""

from __future__ import annotations

import typing

from tests.testing_utils import as_callable, as_tensor

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
        """
        return (as_tensor(x) * as_tensor(x)) * as_tensor(x)

    cp_fun = checkpoint(my_fun)

    vg_std = value_and_grad(my_fun)
    vg_cp = value_and_grad(cp_fun)

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig(shape=(1,), dtype="float32", device="cpu"))

    from ml_switcheroo_compiler.grad.jit import jit

    jitted_std = as_callable(jit)(vg_std)
    jitted_cp = as_callable(jit)(vg_cp)

    v1, g1 = jitted_std(x)
    v2, g2 = jitted_cp(x)

    as_callable(np.testing.assert_allclose)(as_callable(as_tensor(getattr(v1, "numpy")))(), as_callable(as_tensor(getattr(v2, "numpy")))())
    as_callable(np.testing.assert_allclose)(as_callable(as_tensor(getattr(g1, "numpy")))(), as_callable(as_tensor(getattr(g2, "numpy")))())
