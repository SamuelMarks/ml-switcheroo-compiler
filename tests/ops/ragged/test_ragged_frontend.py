"""Unit tests for ragged frontend wrappers ensuring 100% statement and branch coverage."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.ragged.frontend import (
    _ragged_op,
    boolean_mask,
    map_flat_values,
    ragged_add,
    ragged_constant,
    ragged_cross,
    ragged_cross_hashed,
    ragged_dot,
    ragged_dynamic_broadcast,
    ragged_gather,
    ragged_matmul,
    ragged_range,
    ragged_row_splits_to_segment_ids,
    ragged_segment_ids_to_row_splits,
    ragged_stack,
    ragged_stack_dynamic_partitions,
    ragged_tensor_to_dense,
)


def _make_dummy_tensor(shape: tuple[int, ...] = (2, 2)) -> Tensor:
    """Create a dummy tensor for testing.

    Args:
        shape (tuple[int, ...]): Desired shape.

    Returns:
        Tensor: Dummy tensor instance.
    """
    data = np.ones(shape, dtype=np.float32)
    return Tensor(data, TensorConfig(shape, DType.Float32, Device("cpu")))


def test_ragged_frontend_eager_dispatch() -> None:
    """Verify eager dispatch for all ragged frontend operations."""
    config.eager_mode = True
    t = _make_dummy_tensor()

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2), dtype=np.float32)

        assert ragged_constant([[1, 2], [3]]) is not None
        assert ragged_cross(t, t) is not None
        assert ragged_cross_hashed(t, t) is not None
        assert ragged_range(t) is not None
        assert ragged_row_splits_to_segment_ids(t) is not None
        assert ragged_segment_ids_to_row_splits(t) is not None
        assert ragged_stack(t, t) is not None
        assert ragged_stack_dynamic_partitions(t, t) is not None
        assert ragged_gather(t, t) is not None
        assert ragged_tensor_to_dense(t) is not None
        assert ragged_add(t, t) is not None
        assert ragged_matmul(t, t) is not None
        assert ragged_dynamic_broadcast(t) is not None
        assert ragged_dot(t, t) is not None
        assert boolean_mask(t, t) is not None
        assert map_flat_values(None, t) is not None

        # Test with no args in eager mode (hits args[0] else branch)
        res_no_args = _ragged_op("RaggedConstant")
        assert res_no_args.dtype == "float32"
        assert "cpu" in str(res_no_args.device)


def test_ragged_frontend_tracing_dispatch() -> None:
    """Verify graph tracing mode dispatch for ragged frontend operations."""
    config.eager_mode = False
    t = _make_dummy_tensor()

    with patch("ml_switcheroo_compiler.ops.ragged.frontend._emit_linalg_node") as mock_emit:
        mock_emit.return_value = t

        assert ragged_constant([[1, 2], [3]]) is t
        assert ragged_cross(t, t) is t
        assert ragged_cross_hashed(t, t) is t
        assert ragged_range(t) is t
        assert ragged_row_splits_to_segment_ids(t) is t
        assert ragged_segment_ids_to_row_splits(t) is t
        assert ragged_stack(t, t) is t
        assert ragged_stack_dynamic_partitions(t, t) is t
        assert ragged_gather(t, t) is t
        assert ragged_tensor_to_dense(t) is t
        assert ragged_add(t, t) is t
        assert ragged_matmul(t, t) is t
        assert ragged_dynamic_broadcast(t) is t
        assert ragged_dot(t, t) is t
        assert boolean_mask(t, t) is t
        assert map_flat_values(None, t) is t

        # Test with no args in tracing mode (hits args[0] else branch)
        res_no_args = _ragged_op("RaggedConstant")
        assert res_no_args is t

    config.eager_mode = True
