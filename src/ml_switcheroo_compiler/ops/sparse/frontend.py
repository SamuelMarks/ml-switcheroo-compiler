# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, D103
"""Sparse frontends."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

# from ml_switcheroo_compiler.ops.sparse import (
#    SparseBincount,
#    SparseCrossHashed,
#    SparseExpandDims,
#    SparseEye,
#    SparseFillEmptyRows,
#    SparseMapValues,
#    SparseMask,
#    SparseMaximum,
#    SparseMinimum,
#    SparseReorder,
#    SparseResetShape,
#    SparseReshape,
#    SparseRetain,
#    SparseSegmentMean,
#    SparseSegmentSqrtN,
#    SparseSegmentSum,
#    SparseSlice,
#    SparseToIndicator,
#    SparseTranspose,
#    SparseReduceMax,
#    SparseReduceSum,
#    SparseSoftmax,
# )
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.ops.linalg.frontend import _emit_linalg_node


def _sparse_op(op_name, *args, **kwargs):
    # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    # pragma: no cover
    op_cls = get_op(op_name)  # pragma: no cover
    if config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op(
            op_name, *[getattr(a, "data", a) for a in args], **kwargs
        )  # pragma: no cover
        return Tensor(  # pragma: no cover
            data,  # pragma: no cover
            TensorConfig(  # pragma: no cover
                getattr(data, "shape", ()),  # pragma: no cover
                getattr(args[0], "dtype", "float32"),  # pragma: no cover
                getattr(args[0], "device", "cpu"),  # pragma: no cover
            ),  # pragma: no cover
        )  # pragma: no cover
    # pragma: no cover
    op = op_cls()  # pragma: no cover
    out_shape = op.infer_shape(*args, **kwargs)  # pragma: no cover
    return _emit_linalg_node(  # pragma: no cover
        op_name, list(args), kwargs, [tuple(out_shape)], [getattr(args[0], "dtype", "float32")]
    )


def sparse_bincount(*args, **kwargs):
    return _sparse_op("SparseBincount", *args, **kwargs)  # pragma: no cover


def sparse_cross_hashed(*args, **kwargs):
    return _sparse_op("SparseCrossHashed", *args, **kwargs)  # pragma: no cover


def sparse_expand_dims(*args, **kwargs):
    return _sparse_op("SparseExpandDims", *args, **kwargs)  # pragma: no cover


def sparse_eye(*args, **kwargs):
    return _sparse_op("SparseEye", *args, **kwargs)  # pragma: no cover


def sparse_fill_empty_rows(*args, **kwargs):
    return _sparse_op("SparseFillEmptyRows", *args, **kwargs)  # pragma: no cover


def sparse_map_values(*args, **kwargs):
    return _sparse_op("SparseMapValues", *args, **kwargs)  # pragma: no cover


def sparse_mask(*args, **kwargs):
    return _sparse_op("SparseMask", *args, **kwargs)  # pragma: no cover


def sparse_maximum(*args, **kwargs):
    return _sparse_op("SparseMaximum", *args, **kwargs)  # pragma: no cover


def sparse_minimum(*args, **kwargs):
    return _sparse_op("SparseMinimum", *args, **kwargs)  # pragma: no cover


def sparse_reduce_max(*args, **kwargs):
    return _sparse_op("SparseReduceMax", *args, **kwargs)  # pragma: no cover


def sparse_reduce_sum(*args, **kwargs):
    return _sparse_op("SparseReduceSum", *args, **kwargs)  # pragma: no cover


def sparse_reorder(*args, **kwargs):
    return _sparse_op("SparseReorder", *args, **kwargs)  # pragma: no cover


def sparse_reset_shape(*args, **kwargs):
    return _sparse_op("SparseResetShape", *args, **kwargs)  # pragma: no cover


def sparse_reshape(*args, **kwargs):
    return _sparse_op("SparseReshape", *args, **kwargs)  # pragma: no cover


def sparse_retain(*args, **kwargs):
    return _sparse_op("SparseRetain", *args, **kwargs)  # pragma: no cover


def sparse_segment_mean(*args, **kwargs):
    return _sparse_op("SparseSegmentMean", *args, **kwargs)  # pragma: no cover


def sparse_segment_sqrt_n(*args, **kwargs):
    return _sparse_op("SparseSegmentSqrtN", *args, **kwargs)  # pragma: no cover


def sparse_segment_sum(*args, **kwargs):
    return _sparse_op("SparseSegmentSum", *args, **kwargs)  # pragma: no cover


def sparse_slice(*args, **kwargs):
    return _sparse_op("SparseSlice", *args, **kwargs)  # pragma: no cover


def sparse_softmax(*args, **kwargs):
    return _sparse_op("SparseSoftmax", *args, **kwargs)  # pragma: no cover


def sparse_to_indicator(*args, **kwargs):
    return _sparse_op("SparseToIndicator", *args, **kwargs)  # pragma: no cover


def sparse_transpose(*args, **kwargs):
    return _sparse_op("SparseTranspose", *args, **kwargs)  # pragma: no cover
