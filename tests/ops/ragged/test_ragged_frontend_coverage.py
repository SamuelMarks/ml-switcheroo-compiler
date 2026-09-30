"""Tests for test_ragged_frontend_coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.ragged.frontend as ragged_fe_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


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


def test_ragged_frontend_coverage() -> None:
    """Verify 100% line and branch coverage for ragged frontend operations."""
    orig_eager = config.eager_mode
    try:
        # Eager mode
        config.eager_mode = True
        mock_backend = MagicMock()
        mock_backend.execute_op.return_value = np.zeros((2, 3))

        t_dummy = Tensor(np.zeros((2, 3)), TensorConfig((2, 3), DType.Float32, Device("cpu")))

        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend):
            # Test _ragged_op with args and kwargs
            res1 = ragged_fe_mod.ragged_constant(t_dummy, 42, foo="bar")
            assert isinstance(res1, Tensor)

            # Test _ragged_op with no args
            res_empty = ragged_fe_mod._ragged_op("RaggedConstant")
            assert isinstance(res_empty, Tensor)

            # Exercise all frontend function signatures in eager mode
            assert isinstance(ragged_fe_mod.ragged_cross(t_dummy, t_dummy), Tensor)
            assert isinstance(ragged_fe_mod.ragged_cross_hashed(t_dummy, t_dummy), Tensor)
            assert isinstance(ragged_fe_mod.ragged_range(0, 10, 2), Tensor)
            assert isinstance(ragged_fe_mod.ragged_row_splits_to_segment_ids(t_dummy), Tensor)
            assert isinstance(ragged_fe_mod.ragged_segment_ids_to_row_splits(t_dummy), Tensor)
            assert isinstance(ragged_fe_mod.ragged_stack([t_dummy, t_dummy]), Tensor)
            assert isinstance(ragged_fe_mod.ragged_stack_dynamic_partitions(t_dummy, t_dummy), Tensor)
            assert isinstance(ragged_fe_mod.ragged_gather(t_dummy, t_dummy), Tensor)
            assert isinstance(ragged_fe_mod.ragged_tensor_to_dense(t_dummy), Tensor)
            assert isinstance(ragged_fe_mod.ragged_add(t_dummy, t_dummy), Tensor)
            assert isinstance(ragged_fe_mod.ragged_matmul(t_dummy, t_dummy), Tensor)
            assert isinstance(ragged_fe_mod.ragged_dynamic_broadcast(t_dummy, (2, 3)), Tensor)
            assert isinstance(ragged_fe_mod.ragged_dot(t_dummy, t_dummy), Tensor)
            assert isinstance(ragged_fe_mod.boolean_mask(t_dummy, t_dummy), Tensor)
            assert isinstance(ragged_fe_mod.map_flat_values(lambda x: x, t_dummy), Tensor)

        # Non-eager mode
        config.eager_mode = False

        class MockRaggedOp:
            """Mock ragged op class for shape inference."""

            def infer_shape(self, *args, **kwargs):
                """Return mocked shape tuple.

                Args:
                    *args (object): Input arguments.
                    **kwargs (object): Keyword arguments.

                Returns:
                    tuple[int, ...]: Inferred shape.
                """
                return (2, 3)

        with patch("ml_switcheroo_compiler.ops.ragged.frontend.get_op", return_value=MockRaggedOp):
            with patch("ml_switcheroo_compiler.ops.ragged.frontend._emit_linalg_node", return_value="emitted_ragged"):
                res_graph = ragged_fe_mod.ragged_constant(t_dummy)
                assert res_graph == "emitted_ragged"

                res_graph_no_args = ragged_fe_mod._ragged_op("RaggedConstant")
                assert res_graph_no_args == "emitted_ragged"
    finally:
        config.eager_mode = orig_eager
