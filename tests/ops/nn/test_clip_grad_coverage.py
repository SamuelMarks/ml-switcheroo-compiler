"""Tests for test_clip_grad_coverage."""

from __future__ import annotations

import importlib

import numpy as np

import ml_switcheroo_compiler.ops.nn.clip_grad as clip_grad_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

gru_mod = importlib.import_module("ml_switcheroo_compiler.ops.nn.gru")


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 42


def test_nn_clip_grad_coverage() -> None:
    """Verify 100% line and branch coverage for clip_grad operations."""
    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        p1 = Tensor(np.array([1.0, -2.0, 3.0]), TensorConfig((3,), DType.Float32, Device("cpu")))
        p2 = Tensor(np.array([4.0, 0.0, -1.0]), TensorConfig((3,), DType.Float32, Device("cpu")))

        # Empty parameters list
        empty_clipped, empty_norm = clip_grad_mod.clip_grad_norm([], max_norm=1.0)
        assert empty_clipped == []
        assert empty_norm.shape == ()

        # Single tensor parameter with L2 norm (norm_type == 2.0)
        clipped_single, norm_single = clip_grad_mod.clip_grad_norm(p1, max_norm=2.0, norm_type=2.0)
        assert isinstance(clipped_single, Tensor)
        assert isinstance(norm_single, Tensor)

        # Iterable parameters with infinity norm (norm_type == inf)
        clipped_inf, norm_inf = clip_grad_mod.clip_grad_norm([p1, p2], max_norm=1.0, norm_type=float("inf"))
        assert isinstance(clipped_inf, list)
        assert len(clipped_inf) == 2
        assert isinstance(norm_inf, Tensor)

        # Iterable parameters with custom p-norm (e.g. norm_type == 1.0)
        clipped_l1, norm_l1 = clip_grad_mod.clip_grad_norm([p1, p2], max_norm=5.0, norm_type=1.0)
        assert isinstance(clipped_l1, list)
        assert len(clipped_l1) == 2
        assert isinstance(norm_l1, Tensor)
    finally:
        config.eager_mode = orig_eager
