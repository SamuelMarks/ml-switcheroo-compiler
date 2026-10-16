"""Sparse frontends."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import EagerValue

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
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpArgValue, get_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


def _sparse_op(op_name: str, *args: OpArgValue, **kwargs: OpArgValue) -> Tensor | tuple[Tensor, ...]:
    """Evaluate _sparse_op operation.

    Args:
        op_name (Any): The op_name parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    op_cls = get_op(op_name)
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(op_name, *[getattr(a, "data", a) for a in args], **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", ()),
                getattr(args[0], "dtype", "float32"),
                getattr(args[0], "device", "cpu"),
            ),
        )

    op = op_cls()
    out_shape = op.infer_shape(*args, **kwargs)
    return _emit_linalg_node(op_name, list(args), kwargs, [tuple(getattr(out_shape, "__iter__")())], [getattr(args[0], "dtype", "float32")])


def sparse_bincount(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_bincount operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseBincount", *args, **kwargs)


def sparse_cross_hashed(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_cross_hashed operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseCrossHashed", *args, **kwargs)


def sparse_expand_dims(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_expand_dims operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseExpandDims", *args, **kwargs)


def sparse_eye(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_eye operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseEye", *args, **kwargs)


def sparse_fill_empty_rows(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_fill_empty_rows operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseFillEmptyRows", *args, **kwargs)


def sparse_map_values(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_map_values operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseMapValues", *args, **kwargs)


def sparse_mask(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_mask operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseMask", *args, **kwargs)


def sparse_maximum(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_maximum operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseMaximum", *args, **kwargs)


def sparse_minimum(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_minimum operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseMinimum", *args, **kwargs)


def sparse_reduce_max(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_reduce_max operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseReduceMax", *args, **kwargs)


def sparse_reduce_sum(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_reduce_sum operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseReduceSum", *args, **kwargs)


def sparse_reorder(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_reorder operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseReorder", *args, **kwargs)


def sparse_reset_shape(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_reset_shape operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseResetShape", *args, **kwargs)


def sparse_reshape(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_reshape operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseReshape", *args, **kwargs)


def sparse_retain(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_retain operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseRetain", *args, **kwargs)


def sparse_segment_mean(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_segment_mean operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseSegmentMean", *args, **kwargs)


def sparse_segment_sqrt_n(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_segment_sqrt_n operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseSegmentSqrtN", *args, **kwargs)


def sparse_segment_sum(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_segment_sum operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseSegmentSum", *args, **kwargs)


def sparse_slice(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_slice operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseSlice", *args, **kwargs)


def sparse_softmax(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_softmax operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseSoftmax", *args, **kwargs)


def sparse_to_indicator(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_to_indicator operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseToIndicator", *args, **kwargs)


def sparse_transpose(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_transpose operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseTranspose", *args, **kwargs)


def sparse_add(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_add operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseAdd", *args, **kwargs)


def sparse_dense_matmul(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_dense_matmul operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseDenseMatMul", *args, **kwargs)


def sparse_sampled_add(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate sparse_sampled_add operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseSampledAdd", *args, **kwargs)


def smm(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate smm operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("Smm", *args, **kwargs)


def sparse_concat(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Sparse concat.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseConcat", *args, **kwargs)


def sparse_split(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Sparse split.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseSplit", *args, **kwargs)


def sparse_to_dense(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Sparse to dense.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _sparse_op("SparseToDense", *args, **kwargs)
