"""Sparse frontends."""

from __future__ import annotations

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
from ml_switcheroo_compiler.backends.registry import coerce_shape
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    extract_data,
    extract_device,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpArgValue, get_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


def _sparse_op(op_name: str, *args: OpArgValue, **kwargs: OpArgValue) -> Tensor | tuple[Tensor, ...]:
    """Compute the sparse op tensor operation.

    Args:
        op_name (str): Input `op_name` argument (op name).
        *args (OpArgValue): Positional args.
        **kwargs (OpArgValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    op_cls = get_op(op_name)
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(op_name, *[extract_data(a) for a in args], **kwargs)
        return Tensor(
            data,
            TensorConfig(
                extract_shape(data),
                extract_dtype(args[0], "float32"),
                extract_device(args[0], "cpu"),
            ),
        )

    op = op_cls()
    out_shape = op.dispatch_infer_shape(*args, **kwargs)
    out_dtype = extract_dtype(args[0], "float32")
    if isinstance(out_shape, (list, tuple)) and len(out_shape) > 0 and isinstance(out_shape[0], (list, tuple)):
        out_shapes = [coerce_shape(s) for s in out_shape]
        return _emit_linalg_node(op_name, list(args), kwargs, out_shapes, [out_dtype] * len(out_shapes))
    return _emit_linalg_node(op_name, list(args), kwargs, [coerce_shape(out_shape)], [out_dtype])


def sparse_bincount(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse bincount tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseBincount", *args, **kwargs)


def sparse_cross_hashed(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse cross hashed tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseCrossHashed", *args, **kwargs)


def sparse_expand_dims(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse expand dims tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseExpandDims", *args, **kwargs)


def sparse_eye(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse eye tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseEye", *args, **kwargs)


def sparse_fill_empty_rows(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse fill empty rows tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseFillEmptyRows", *args, **kwargs)


def sparse_map_values(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse map values tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseMapValues", *args, **kwargs)


def sparse_mask(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse mask tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseMask", *args, **kwargs)


def sparse_maximum(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse maximum tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseMaximum", *args, **kwargs)


def sparse_minimum(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse minimum tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseMinimum", *args, **kwargs)


def sparse_reduce_max(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse reduce max tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseReduceMax", *args, **kwargs)


def sparse_reduce_sum(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse reduce sum tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseReduceSum", *args, **kwargs)


def sparse_reorder(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse reorder tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseReorder", *args, **kwargs)


def sparse_reset_shape(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse reset shape tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseResetShape", *args, **kwargs)


def sparse_reshape(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse reshape tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseReshape", *args, **kwargs)


def sparse_retain(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse retain tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseRetain", *args, **kwargs)


def sparse_segment_mean(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse segment mean tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseSegmentMean", *args, **kwargs)


def sparse_segment_sqrt_n(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse segment sqrt n tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseSegmentSqrtN", *args, **kwargs)


def sparse_segment_sum(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse segment sum tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseSegmentSum", *args, **kwargs)


def sparse_slice(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse slice tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseSlice", *args, **kwargs)


def sparse_softmax(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse softmax tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseSoftmax", *args, **kwargs)


def sparse_to_indicator(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse to indicator tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseToIndicator", *args, **kwargs)


def sparse_transpose(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse transpose tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseTranspose", *args, **kwargs)


def sparse_add(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse add tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseAdd", *args, **kwargs)


def sparse_dense_matmul(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse dense matmul tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseDenseMatMul", *args, **kwargs)


def sparse_sampled_add(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the sparse sampled add tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseSampledAdd", *args, **kwargs)


def smm(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the smm tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("Smm", *args, **kwargs)


def sparse_concat(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Sparse concat.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseConcat", *args, **kwargs)


def sparse_split(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Sparse split.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseSplit", *args, **kwargs)


def sparse_to_dense(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Sparse to dense.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _sparse_op("SparseToDense", *args, **kwargs)
