# ruff: noqa: D103
"""Tests for sparse and ragged extras."""

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


def test_sparse_extras() -> None:
    backend = get_active_backend()
    dev = Device("cpu")
    t1 = Tensor(backend.array([1.0]), TensorConfig((1,), DType.Float32, dev))

    op1 = SparseConcat()
    assert op1.infer_shape(None) == ()
    op2 = SparseSplit()
    assert op2.infer_shape(None, None) == ()
    op3 = SparseToDense()
    assert op3.infer_shape(None, None, None, None) == ()

    config.eager_mode = False
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.sparse.frontend._emit_linalg_node") as mock_emit:
        sparse_concat(t1)
        sparse_split(t1, 1)
        sparse_to_dense(t1, t1, t1, t1)
        sparse_dense_matmul(t1, t1)
        assert mock_emit.call_count == 4


def test_ragged_extras() -> None:
    backend = get_active_backend()
    dev = Device("cpu")
    t1 = Tensor(backend.array([1.0]), TensorConfig((1,), DType.Float32, dev))

    op1 = BooleanMask()
    assert op1.infer_shape(None, None) == ()
    op2 = MapFlatValues()
    assert op2.infer_shape(None) == ()

    config.eager_mode = False
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.ragged.frontend._emit_linalg_node") as mock_emit:
        boolean_mask(t1, t1)
        map_flat_values(lambda x: x, t1)
        assert mock_emit.call_count == 2
