from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.linalg.conv import (
    ConvLocalHyperparams,
    conv_general_dilated,
    conv_general_dilated_local,
    conv_general_dilated_patches,
    conv_with_general_padding,
)


@patch("ml_switcheroo_compiler.ops.linalg.conv._emit_linalg_node")
@patch("ml_switcheroo_compiler.ops.linalg.conv.ConvGeneralDilated.infer_shape", return_value=(1, 1, 1, 1))
def test_conv_general_dilated(mock_infer, mock_emit):
    lhs = MagicMock()
    lhs.dtype = "float32"
    rhs = MagicMock()
    rhs.dtype = "float32"
    config = ConvConfig([1, 1], "SAME", [1, 1], [1, 1], None, 1)

    conv_general_dilated.__wrapped__(lhs, rhs, config)
    mock_emit.assert_called()


@patch("ml_switcheroo_compiler.ops.linalg.conv._emit_linalg_node")
@patch("ml_switcheroo_compiler.ops.linalg.conv.ConvGeneralDilatedLocal.infer_shape", return_value=(1, 1, 1, 1))
def test_conv_general_dilated_local(mock_infer, mock_emit):
    lhs = MagicMock()
    lhs.dtype = "float32"
    rhs = MagicMock()
    rhs.dtype = "float32"
    config = ConvLocalHyperparams([1, 1], "SAME", (1, 1))

    conv_general_dilated_local.__wrapped__(lhs, rhs, config)
    mock_emit.assert_called()


@patch("ml_switcheroo_compiler.ops.linalg.conv._emit_linalg_node")
@patch("ml_switcheroo_compiler.ops.linalg.conv.ConvGeneralDilatedPatches.infer_shape", return_value=(1, 1, 1, 1))
def test_conv_general_dilated_patches(mock_infer, mock_emit):
    lhs = MagicMock()
    lhs.dtype = "float32"

    conv_general_dilated_patches.__wrapped__(lhs, (1, 1), [1, 1], "SAME")
    mock_emit.assert_called()


@patch("ml_switcheroo_compiler.ops.linalg.conv._emit_linalg_node")
@patch("ml_switcheroo_compiler.ops.linalg.conv.ConvWithGeneralPadding.infer_shape", return_value=(1, 1, 1, 1))
def test_conv_with_general_padding(mock_infer, mock_emit):
    lhs = MagicMock()
    lhs.dtype = "float32"
    rhs = MagicMock()
    rhs.dtype = "float32"

    conv_with_general_padding.__wrapped__(lhs, rhs, [1, 1], "SAME")
    mock_emit.assert_called()
