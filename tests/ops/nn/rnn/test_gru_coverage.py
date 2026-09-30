"""Tests for test_gru_coverage."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import numpy as np

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


def test_nn_gru_coverage() -> None:
    """Verify 100% line and branch coverage for gru operations."""
    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        t_in = Tensor(np.ones((2, 3)), TensorConfig(shape=(2, 3), dtype=DType("float32"), device=Device("cpu")))
        t_state = Tensor(np.ones((2, 4)), TensorConfig(shape=(2, 4), dtype=DType("float32"), device=Device("cpu")))
        kernel = Tensor(np.ones((3, 12)), TensorConfig(shape=(3, 12), dtype=DType("float32"), device=Device("cpu")))
        recurrent_kernel = Tensor(np.ones((4, 12)), TensorConfig(shape=(4, 12), dtype=DType("float32"), device=Device("cpu")))
        bias = Tensor(np.ones((12,)), TensorConfig(shape=(12,), dtype=DType("float32"), device=Device("cpu")))

        # Test gru_cell with bias
        h_new, h_state = gru_mod.gru_cell(t_in, t_state, kernel, recurrent_kernel, bias)
        assert isinstance(h_new, Tensor)
        assert isinstance(h_state, Tensor)

        # Test gru_cell without bias
        h_new2, _ = gru_mod.gru_cell(t_in, t_state, kernel, recurrent_kernel, bias=None)
        assert isinstance(h_new2, Tensor)

        # Test Gru OpDef infer_shape
        gru_op = gru_mod.Gru()
        assert gru_op.infer_shape((2, 4)) == (2, 4)
        assert gru_op.infer_shape() == ()

        # Test gru eager call
        mock_backend = MagicMock()
        mock_backend.execute_op.return_value = "gru_backend_result"
        with patch.object(gru_mod, "get_active_backend", return_value=mock_backend):
            res_eager = gru_mod.gru(t_in, "extra_arg", foo="bar")
            assert res_eager == "gru_backend_result"

        # Test gru non-eager mode with tensor args
        config.eager_mode = False
        with patch.object(gru_mod, "_emit_shape_node", return_value="emitted_gru") as mock_emit:
            res_graph = gru_mod.gru(t_in)
            assert res_graph == "emitted_gru"
            mock_emit.assert_called_once_with("Gru", [t_in], {}, (2, 3), DType("float32"))

        # Test gru non-eager mode with empty/non-tensor args
        with patch.object(gru_mod, "_emit_shape_node", return_value="emitted_empty") as mock_emit_empty:
            res_graph_empty = gru_mod.gru("not_a_tensor")
            assert res_graph_empty == "emitted_empty"
            mock_emit_empty.assert_called_once_with("Gru", ["not_a_tensor"], {}, (), DType.Float32)

        # Test _sigmoid directly
        sig_val = gru_mod._sigmoid(t_in)
        assert sig_val is not None
    finally:
        config.eager_mode = orig_eager
