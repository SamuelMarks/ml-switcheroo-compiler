"""Tests for test_conv."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.linalg.conv import (
    ConvLocalHyperparams,
    conv_general_dilated,
    conv_general_dilated_local,
    conv_general_dilated_patches,
    conv_with_general_padding,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


@patch("ml_switcheroo_compiler.ops.linalg.conv._emit_linalg_node")
@patch("ml_switcheroo_compiler.ops.linalg.conv.ConvGeneralDilated.infer_shape", return_value=(1, 1, 1, 1))
def test_conv_general_dilated(mock_infer: EagerTensorProtocol, mock_emit: EagerTensorProtocol) -> None:
    """Test function test_conv_general_dilated.

    Args:
        mock_infer: Test parameter `mock_infer`.
        mock_emit: Test parameter `mock_emit`.
    """
    lhs = MagicMock()
    lhs.dtype = "float32"
    rhs = MagicMock()
    rhs.dtype = "float32"
    config = ConvConfig(window_strides=[1, 1], padding="SAME", lhs_dilation=[1, 1], rhs_dilation=[1, 1], dimension_numbers=None, feature_group_count=1)

    typing.cast("EagerTensorProtocol", getattr(conv_general_dilated, "__wrapped__"))(lhs, rhs, config)
    mock_emit.assert_called()


@patch("ml_switcheroo_compiler.ops.linalg.conv._emit_linalg_node")
@patch("ml_switcheroo_compiler.ops.linalg.conv.ConvGeneralDilatedLocal.infer_shape", return_value=(1, 1, 1, 1))
def test_conv_general_dilated_local(mock_infer: EagerTensorProtocol, mock_emit: EagerTensorProtocol) -> None:
    """Test function test_conv_general_dilated_local.

    Args:
        mock_infer: Test parameter `mock_infer`.
        mock_emit: Test parameter `mock_emit`.
    """
    lhs = MagicMock()
    lhs.dtype = "float32"
    rhs = MagicMock()
    rhs.dtype = "float32"
    config = ConvLocalHyperparams(window_strides=[1, 1], padding="SAME", filter_shape=(1, 1))

    typing.cast("EagerTensorProtocol", getattr(conv_general_dilated_local, "__wrapped__"))(lhs, rhs, config)
    mock_emit.assert_called()


@patch("ml_switcheroo_compiler.ops.linalg.conv._emit_linalg_node")
@patch("ml_switcheroo_compiler.ops.linalg.conv.ConvGeneralDilatedPatches.infer_shape", return_value=(1, 1, 1, 1))
def test_conv_general_dilated_patches(mock_infer: EagerTensorProtocol, mock_emit: EagerTensorProtocol) -> None:
    """Test function test_conv_general_dilated_patches.

    Args:
        mock_infer: Test parameter `mock_infer`.
        mock_emit: Test parameter `mock_emit`.
    """
    lhs = MagicMock()
    lhs.dtype = "float32"

    typing.cast("EagerTensorProtocol", getattr(conv_general_dilated_patches, "__wrapped__"))(lhs, (1, 1), [1, 1], "SAME")
    mock_emit.assert_called()


@patch("ml_switcheroo_compiler.ops.linalg.conv._emit_linalg_node")
@patch("ml_switcheroo_compiler.ops.linalg.conv.ConvWithGeneralPadding.infer_shape", return_value=(1, 1, 1, 1))
def test_conv_with_general_padding(mock_infer: EagerTensorProtocol, mock_emit: EagerTensorProtocol) -> None:
    """Test function test_conv_with_general_padding.

    Args:
        mock_infer: Test parameter `mock_infer`.
        mock_emit: Test parameter `mock_emit`.
    """
    lhs = MagicMock()
    lhs.dtype = "float32"
    rhs = MagicMock()
    rhs.dtype = "float32"

    typing.cast("EagerTensorProtocol", getattr(conv_with_general_padding, "__wrapped__"))(lhs, rhs, [1, 1], "SAME")
    mock_emit.assert_called()
