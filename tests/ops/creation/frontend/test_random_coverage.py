"""Tests for test_random_coverage."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.creation.frontend_random as random_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType

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


def test_frontend_random_coverage() -> None:
    """Verify 100% line and branch coverage for frontend_random operations."""
    orig_eager = config.eager_mode
    try:
        # Eager mode tests
        config.eager_mode = True
        mock_backend = MagicMock()
        mock_backend.execute_op.return_value = np.ones((2, 3), dtype=np.float32)

        with patch("ml_switcheroo_compiler.ops.creation.frontend_random.get_active_backend", return_value=mock_backend):
            # rand with default dtype/device and custom dtype/device
            r1 = random_mod.rand(2, 3)
            assert r1.shape == (2, 3)
            r2 = random_mod.rand(2, 3, dtype=DType.Float64, device=Device("cpu"))
            assert r2.dtype == DType.Float64

            # randn
            rn1 = random_mod.randn(2, 3)
            assert rn1.shape == (2, 3)
            rn2 = random_mod.randn(2, 3, dtype=DType.Float64, device=Device("cpu"))
            assert rn2.dtype == DType.Float64

            # randint
            ri1 = random_mod.randint(0, 10, (2, 3))
            assert ri1.shape == (2, 3)
            ri2 = random_mod.randint(0, 10, [2, 3], dtype=DType.Int32, device=Device("cpu"))
            assert ri2.dtype == DType.Int32

            # manual_seed
            seed_val = random_mod.manual_seed(1234)
            assert seed_val == 1234
            mock_backend.execute_op.assert_called_with("Seed", 1234)

        # Non-eager mode tests
        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.creation.frontend_random._emit_creation_node") as mock_emit:
            mock_emit.side_effect = lambda *args, **kwargs: "node"
            res_rand = random_mod.rand(3, 4)
            assert res_rand == "node"

            res_randn = random_mod.randn(3, 4)
            assert res_randn == "node"

            res_randint = random_mod.randint(0, 5, (2, 2))
            assert res_randint == "node"

            res_seed = random_mod.manual_seed(99)
            assert res_seed == 99
    finally:
        config.eager_mode = orig_eager
