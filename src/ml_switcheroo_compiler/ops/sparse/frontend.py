"""Sparse frontends."""

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
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


def _sparse_op(op_name: object, *args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse op operation.

    Args:
        op_name (object): Required parameter for op_name.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
    return _emit_linalg_node(op_name, list(args), kwargs, [tuple(out_shape)], [getattr(args[0], "dtype", "float32")])


def sparse_bincount(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse bincount operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseBincount", *args, **kwargs)


def sparse_cross_hashed(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse cross hashed operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseCrossHashed", *args, **kwargs)


def sparse_expand_dims(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse expand dims operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseExpandDims", *args, **kwargs)


def sparse_eye(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse eye operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseEye", *args, **kwargs)


def sparse_fill_empty_rows(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse fill empty rows operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseFillEmptyRows", *args, **kwargs)


def sparse_map_values(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse map values operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseMapValues", *args, **kwargs)


def sparse_mask(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse mask operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseMask", *args, **kwargs)


def sparse_maximum(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse maximum operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseMaximum", *args, **kwargs)


def sparse_minimum(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse minimum operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseMinimum", *args, **kwargs)


def sparse_reduce_max(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse reduce max operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseReduceMax", *args, **kwargs)


def sparse_reduce_sum(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse reduce sum operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseReduceSum", *args, **kwargs)


def sparse_reorder(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse reorder operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseReorder", *args, **kwargs)


def sparse_reset_shape(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse reset shape operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseResetShape", *args, **kwargs)


def sparse_reshape(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse reshape operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseReshape", *args, **kwargs)


def sparse_retain(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse retain operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseRetain", *args, **kwargs)


def sparse_segment_mean(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse segment mean operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseSegmentMean", *args, **kwargs)


def sparse_segment_sqrt_n(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse segment sqrt n operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseSegmentSqrtN", *args, **kwargs)


def sparse_segment_sum(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse segment sum operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseSegmentSum", *args, **kwargs)


def sparse_slice(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse slice operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseSlice", *args, **kwargs)


def sparse_softmax(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse softmax operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseSoftmax", *args, **kwargs)


def sparse_to_indicator(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse to indicator operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseToIndicator", *args, **kwargs)


def sparse_transpose(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse transpose operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseTranspose", *args, **kwargs)


def sparse_add(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse add operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseAdd", *args, **kwargs)


def sparse_dense_matmul(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse dense matmul operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseDenseMatMul", *args, **kwargs)


def sparse_sampled_add(*args: object, **kwargs: object) -> object:
    """Evaluate and process the sparse sampled add operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("SparseSampledAdd", *args, **kwargs)


def smm(*args: object, **kwargs: object) -> object:
    """Evaluate and process the smm operation.

    Args:
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _sparse_op("Smm", *args, **kwargs)


def sparse_concat(*args: object, **kwargs: object) -> object:
    """Sparse concat."""
    return _sparse_op("SparseConcat", *args, **kwargs)


def sparse_split(*args: object, **kwargs: object) -> object:
    """Sparse split."""
    return _sparse_op("SparseSplit", *args, **kwargs)


def sparse_to_dense(*args: object, **kwargs: object) -> object:
    """Sparse to dense."""
    return _sparse_op("SparseToDense", *args, **kwargs)
