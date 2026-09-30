"""Tests for test_space_batch_coverage."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.shape.space_batch as space_batch_mod
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


def test_shape_space_batch_coverage() -> None:
    """Verify 100% line and branch coverage for space_batch operations."""
    orig_eager = config.eager_mode
    try:
        # OpDef infer_shape
        nd_op = space_batch_mod.SpaceToBatchND()
        assert nd_op.infer_shape(None, [2, 2], [[0, 0], [0, 0]]) == ()

        sb_op = space_batch_mod.SpaceToBatch()
        assert sb_op.infer_shape(None, 2, [[0, 0], [0, 0]]) == ()

        # space_to_batch eager mode
        config.eager_mode = True
        mock_backend = MagicMock()
        mock_backend.execute_op.return_value = np.zeros((4, 2, 2, 1))

        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend):
            # Input with data, dtype, device
            t_in = Tensor(np.zeros((1, 4, 4, 1)), TensorConfig((1, 4, 4, 1), DType.Float32, Device("cpu")))
            res1 = space_batch_mod.space_to_batch(t_in, 2, [[0, 0], [0, 0]])
            assert res1.shape == (4, 2, 2, 1)

            # Raw input without data/dtype/device
            raw_input = [[1, 2], [3, 4]]
            mock_backend.execute_op.return_value = 42
            res2 = space_batch_mod.space_to_batch(raw_input, 2, [[0, 0], [0, 0]])
            assert isinstance(res2, Tensor)

            # space_to_batch_nd eager mode
            mock_backend.execute_op.return_value = np.zeros((4, 2, 2, 1))
            res_nd1 = space_batch_mod.space_to_batch_nd(t_in, [2, 2], [[0, 0], [0, 0]])
            assert res_nd1.shape == (4, 2, 2, 1)

            mock_backend.execute_op.return_value = 42
            res_nd2 = space_batch_mod.space_to_batch_nd(raw_input, [2, 2], [[0, 0], [0, 0]])
            assert isinstance(res_nd2, Tensor)

        # Non-eager mode
        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted_sb") as mock_emit:
            res_sb_graph = space_batch_mod.space_to_batch(t_in, 2, [[0, 0], [0, 0]])
            assert res_sb_graph == "emitted_sb"
            mock_emit.assert_called_once()

        with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted_sb_nd") as mock_emit_nd:
            res_sb_nd_graph = space_batch_mod.space_to_batch_nd(t_in, [2, 2], [[0, 0], [0, 0]])
            assert res_sb_nd_graph == "emitted_sb_nd"
            mock_emit_nd.assert_called_once()
    finally:
        config.eager_mode = orig_eager
