"""Behavioral tests for multidimensional `conv`, `conv_transpose`, `depthwise_conv`, and `separable_conv`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.conv import conv, conv_transpose, depthwise_conv, separable_conv
from ml_switcheroo_compiler.ops.nn.conv_utils import GenericConvConfig
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape


class _UnshapedResult:
    """Backend payload lacking `.shape`."""

    val: int = 123


def test_conv_transpose_eager_and_tracing_ir_emission() -> None:
    """Verify `conv_transpose` handles shaped/unshaped eager outputs and emits `ConvTranspose` IR nodes when tracing."""
    device = Device("cpu")
    t1 = Tensor(np.ones((1, 4, 2), dtype=np.float32), TensorConfig((1, 4, 2), "float32", device))
    k1 = Tensor(np.ones((2, 2, 2), dtype=np.float32), TensorConfig((2, 2, 2), "float32", device))

    backend = MagicMock()
    backend.execute_op.return_value = np.ones((1, 4, 2), dtype=np.float32)
    with (
        ConfigContext(eager_mode=True),
        patch("ml_switcheroo_compiler.ops.nn.conv.get_active_backend", return_value=backend),
    ):
        eager_shaped = conv_transpose(t1, k1, strides=1, padding="VALID")
        assert isinstance(eager_shaped, Tensor)
        assert eager_shaped.shape == (1, 4, 2)

        backend.execute_op.return_value = _UnshapedResult()
        eager_unshaped = conv_transpose(t1, k1)
        assert isinstance(eager_unshaped, Tensor)
        assert eager_unshaped.shape == ()

    t_sym = Tensor(ProxyTensor("ct_in", (1, 4, 2), "float32"), TensorConfig((1, 4, 2), "float32", device))
    k_sym = Tensor(ProxyTensor("ct_w", (2, 2, 2), "float32"), TensorConfig((2, 2, 2), "float32", device))
    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        traced_graph = tape.start_tracing("conv_transpose_trace")
        try:
            traced_out = conv_transpose(t_sym, k_sym, strides=1, padding="VALID")
        finally:
            tape.stop_tracing()

    assert isinstance(traced_out, Tensor)
    ct_nodes = [n for n in traced_graph.nodes.values() if n.op_type == "ConvTranspose"]
    assert len(ct_nodes) == 1
    assert ct_nodes[0].inputs == ["ct_in", "ct_w"]


def test_conv_depthwise_and_separable_spatial_rank_routing() -> None:
    """Verify `conv`, `depthwise_conv`, and `separable_conv` route 1D/2D/3D tensors and reject unsupported ranks."""
    device = Device("cpu")
    t_2d_rank = Tensor(np.ones((1, 2), dtype=np.float32), TensorConfig((1, 2), "float32", device))
    t_1d = Tensor(np.ones((1, 4, 2), dtype=np.float32), TensorConfig((1, 4, 2), "float32", device))
    t_2d = Tensor(np.ones((1, 4, 4, 2), dtype=np.float32), TensorConfig((1, 4, 4, 2), "float32", device))
    t_3d = Tensor(np.ones((1, 4, 4, 4, 2), dtype=np.float32), TensorConfig((1, 4, 4, 4, 2), "float32", device))
    t_4d = Tensor(np.ones((1, 4, 4, 4, 4, 2), dtype=np.float32), TensorConfig((1, 4, 4, 4, 4, 2), "float32", device))
    k = Tensor(np.ones((2, 2, 2), dtype=np.float32), TensorConfig((2, 2, 2), "float32", device))
    conf = GenericConvConfig(strides=1, padding="VALID", dilation_rate=1)

    with (
        patch("ml_switcheroo_compiler.ops.nn.conv.conv1d", return_value=t_1d) as mock_c1,
        patch("ml_switcheroo_compiler.ops.nn.conv.conv2d", return_value=t_2d) as mock_c2,
        patch("ml_switcheroo_compiler.ops.nn.conv.conv3d", return_value=t_3d) as mock_c3,
    ):
        assert conv(t_1d, k) is t_1d
        assert conv(t_1d, k, config=conf) is t_1d
        assert mock_c1.call_count == 2

        assert conv(t_2d, k, config=GenericConvConfig(strides=2)) is t_2d
        mock_c2.assert_called_once()

        assert conv(t_3d, k) is t_3d
        assert conv(t_3d, k, config=conf) is t_3d
        assert mock_c3.call_count == 2

        with pytest.raises(ValueError, match="Unsupported spatial rank"):
            conv(t_4d, k)
        with pytest.raises(ValueError, match="Unsupported spatial rank"):
            conv(t_2d_rank, t_2d_rank)

    with (
        patch("ml_switcheroo_compiler.ops.nn.conv.depthwise_conv1d", return_value=t_1d) as mock_dw1,
        patch("ml_switcheroo_compiler.ops.nn.conv.depthwise_conv2d", return_value=t_2d) as mock_dw2,
    ):
        assert depthwise_conv(t_1d, k) is t_1d
        assert depthwise_conv(t_1d, k, config=conf) is t_1d
        assert mock_dw1.call_count == 2

        assert depthwise_conv(t_2d, k, config=GenericConvConfig()) is t_2d
        mock_dw2.assert_called_once()

        with pytest.raises(ValueError, match="Unsupported spatial rank for depthwise_conv"):
            depthwise_conv(t_3d, k)

    with (
        patch("ml_switcheroo_compiler.ops.nn.conv.separable_conv1d", return_value=t_1d) as mock_sep1,
        patch("ml_switcheroo_compiler.ops.nn.conv.separable_conv2d", return_value=t_2d) as mock_sep2,
    ):
        assert separable_conv(t_1d, k, k) is t_1d
        assert separable_conv(t_1d, k, k, config=conf) is t_1d
        assert mock_sep1.call_count == 2

        assert separable_conv(t_2d, k, k, config=GenericConvConfig()) is t_2d
        mock_sep2.assert_called_once()

        with pytest.raises(ValueError, match="Unsupported spatial rank for separable_conv"):
            separable_conv(t_3d, k, k)
