"""Tests for test_conv3d_coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.nn.conv3d as conv3d_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.configs import ConvConfig


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize dummy object with shape.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize dummy object without shape."""
        self.val: int = 42


def test_conv3d_full_coverage() -> None:
    """Test full coverage for conv3d functions and branches."""
    original_eager = config.eager_mode
    try:
        config.eager_mode = True
        t_in = Tensor(
            np.ones((1, 3, 3, 3, 2), dtype=np.float32),
            TensorConfig((1, 3, 3, 3, 2), DType("float32"), Device("cpu")),
        )
        t_w = Tensor(
            np.ones((2, 2, 2, 2, 4), dtype=np.float32),
            TensorConfig((2, 2, 2, 2, 4), DType("float32"), Device("cpu")),
        )

        cfg = ConvConfig(
            window_strides=(1, 1, 1),
            padding="VALID",
            dimension_numbers=((0, 4, 1, 2, 3), (4, 3, 0, 1, 2), (0, 4, 1, 2, 3)),
        )

        res_default_cfg = conv3d_mod.conv3d(t_in, t_w)
        assert res_default_cfg is not None

        res_explicit_cfg = conv3d_mod.conv3d(t_in, t_w, config_obj=cfg)
        assert res_explicit_cfg is not None

        with patch("ml_switcheroo_compiler.ops.nn.conv3d.get_op") as mock_get:
            mock_inst = MagicMock()
            mock_inst.return_value = t_in
            mock_get.return_value = MagicMock(return_value=mock_inst)
            assert conv3d_mod.conv3d_transpose(t_in, t_w, strides=2, padding="SAME") is t_in
    finally:
        config.eager_mode = original_eager
