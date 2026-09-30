"""Tests for test_conv_nd_coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.nn.conv_nd as conv_nd_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.conv_utils import GenericConvConfig


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


def test_conv_nd_full_coverage() -> None:
    """Test full coverage for generic multidimensional convolution operations."""
    t_1d = Tensor(np.ones((1, 4, 2), dtype=np.float32), TensorConfig((1, 4, 2), "float32", "cpu"))
    t_2d = Tensor(np.ones((1, 4, 4, 2), dtype=np.float32), TensorConfig((1, 4, 4, 2), "float32", "cpu"))
    t_3d = Tensor(np.ones((1, 4, 4, 4, 2), dtype=np.float32), TensorConfig((1, 4, 4, 4, 2), "float32", "cpu"))
    t_4d = Tensor(np.ones((1, 4, 4, 4, 4, 2), dtype=np.float32), TensorConfig((1, 4, 4, 4, 4, 2), "float32", "cpu"))
    k = Tensor(np.ones((2, 2, 2), dtype=np.float32), TensorConfig((2, 2, 2), "float32", "cpu"))

    original_eager = config.eager_mode
    try:
        config.eager_mode = True
        with patch("ml_switcheroo_compiler.ops.nn.conv_nd.get_active_backend") as mock_be:
            backend = MagicMock()
            backend.execute_op.return_value = np.ones((1, 4, 2), dtype=np.float32)
            mock_be.return_value = backend
            res_t = conv_nd_mod.conv_transpose(t_1d, k)
            assert isinstance(res_t, Tensor)

            backend.execute_op.return_value = DummyWithoutShape()
            res_noshape = conv_nd_mod.conv_transpose(t_1d, k)
            assert isinstance(res_noshape, Tensor)
            assert res_noshape.shape == ()

        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.nn.conv_nd._emit_linalg_node", return_value=t_1d) as mock_emit:
            assert conv_nd_mod.conv_transpose(t_1d, k) is t_1d
            mock_emit.assert_called_once()
    finally:
        config.eager_mode = original_eager

    with (
        patch("ml_switcheroo_compiler.ops.nn.conv_nd.conv1d", return_value=t_1d) as mock_c1,
        patch("ml_switcheroo_compiler.ops.nn.conv_nd.conv2d", return_value=t_2d) as mock_c2,
        patch("ml_switcheroo_compiler.ops.nn.conv_nd.conv3d", return_value=t_3d) as mock_c3,
    ):
        assert conv_nd_mod.conv(t_1d, k) is t_1d
        mock_c1.assert_called_once()

        custom_conf = GenericConvConfig(strides=2)
        assert conv_nd_mod.conv(t_2d, k, config=custom_conf) is t_2d
        mock_c2.assert_called_once()

        assert conv_nd_mod.conv(t_3d, k) is t_3d
        mock_c3.assert_called_once()

        with pytest.raises(ValueError, match="Unsupported spatial rank"):
            conv_nd_mod.conv(t_4d, k)

    with (
        patch("ml_switcheroo_compiler.ops.nn.conv_nd.depthwise_conv1d", return_value=t_1d) as mock_dw1,
        patch("ml_switcheroo_compiler.ops.nn.conv_nd.depthwise_conv2d", return_value=t_2d) as mock_dw2,
    ):
        assert conv_nd_mod.depthwise_conv(t_1d, k) is t_1d
        mock_dw1.assert_called_once()

        assert conv_nd_mod.depthwise_conv(t_2d, k, config=GenericConvConfig()) is t_2d
        mock_dw2.assert_called_once()

        with pytest.raises(ValueError, match="Unsupported spatial rank for depthwise_conv"):
            conv_nd_mod.depthwise_conv(t_3d, k)

    with (
        patch("ml_switcheroo_compiler.ops.nn.conv_nd.separable_conv1d", return_value=t_1d) as mock_sep1,
        patch("ml_switcheroo_compiler.ops.nn.conv_nd.separable_conv2d", return_value=t_2d) as mock_sep2,
    ):
        assert conv_nd_mod.separable_conv(t_1d, k, k) is t_1d
        mock_sep1.assert_called_once()

        assert conv_nd_mod.separable_conv(t_2d, k, k, config=GenericConvConfig()) is t_2d
        mock_sep2.assert_called_once()

        with pytest.raises(ValueError, match="Unsupported spatial rank for separable_conv"):
            conv_nd_mod.separable_conv(t_3d, k, k)
