"""Tests for sparse and ragged extras."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.ragged import BooleanMask, MapFlatValues, boolean_mask, map_flat_values
from ml_switcheroo_compiler.ops.sparse import (
    SparseConcat,
    SparseSplit,
    SparseToDense,
    sparse_concat,
    sparse_dense_matmul,
    sparse_split,
    sparse_to_dense,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_sparse_extras() -> None:
    """Test function test_sparse_extras."""
    backend = get_active_backend()
    dev = Device("cpu")
    t1 = Tensor(backend.array([1.0]), TensorConfig((1,), DType.Float32, dev))

    op1 = SparseConcat()
    assert cast("Callable[..., EagerTensorProtocol]", op1.infer_shape)(None) == ()
    op2 = SparseSplit()
    assert cast("Callable[..., EagerTensorProtocol]", op2.infer_shape)(None, None) == ((),)
    op3 = SparseToDense()
    assert cast("Callable[..., EagerTensorProtocol]", op3.infer_shape)(None, None, None, None) == ()

    config.eager_mode = False
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.sparse.frontend._emit_linalg_node") as mock_emit:
        sparse_concat(t1)
        sparse_split(t1, 1)
        sparse_to_dense(t1, t1, t1, t1)
        sparse_dense_matmul(t1, t1)
        assert mock_emit.call_count == 4


def test_ragged_extras() -> None:
    """Test function test_ragged_extras."""
    backend = get_active_backend()
    dev = Device("cpu")
    t1 = Tensor(backend.array([1.0]), TensorConfig((1,), DType.Float32, dev))

    op1 = BooleanMask()
    assert cast("Callable[..., EagerTensorProtocol]", len)(cast("Callable[..., EagerTensorProtocol]", op1.infer_shape)(None, None)) == 1
    op2 = MapFlatValues()
    assert op2.infer_shape(None) == ()

    config.eager_mode = False
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.ragged.frontend._emit_linalg_node") as mock_emit:
        boolean_mask(t1, t1)
        cast("Callable[..., EagerTensorProtocol]", map_flat_values)(lambda x: x, t1)
        assert mock_emit.call_count == 2
