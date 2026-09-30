"""Tests for test_hadamard_transform."""

from __future__ import annotations

import numpy as np

import ml_switcheroo_compiler.ops.linalg.transform as transform_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


class DummyWithShape:
    """Mock operand providing shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 123


def test_hadamard_transform_full_coverage() -> None:
    """Test full coverage for Hadamard transform loops and scaling."""
    original_eager = config.eager_mode
    try:
        config.eager_mode = True
        t4 = Tensor(np.ones((1, 4), dtype=np.float32), TensorConfig((1, 4), "float32", "cpu"))
        res_scaled = transform_mod.hadamard_transform(t4, scale=0.5)
        assert res_scaled is not None
        assert res_scaled.shape == (1, 4)

        t2 = Tensor(np.ones((1, 2), dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
        res_noscale = transform_mod.hadamard_transform(t2, scale=1.0)
        assert res_noscale is not None
        assert res_noscale.shape == (1, 2)

        t1 = Tensor(np.ones((1, 1), dtype=np.float32), TensorConfig((1, 1), "float32", "cpu"))
        res_1 = transform_mod.hadamard_transform(t1, scale=1.0)
        assert res_1 is not None
        assert res_1.shape == (1, 1)
    finally:
        config.eager_mode = original_eager
