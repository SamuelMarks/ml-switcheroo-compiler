# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Sparse ops."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.ragged.core import RaggedDot as RaggedDot

from .frontend import smm as smm
from .frontend import sparse_add as sparse_add
from .frontend import sparse_bincount as sparse_bincount
from .frontend import sparse_concat as sparse_concat
from .frontend import sparse_cross_hashed as sparse_cross_hashed
from .frontend import sparse_dense_matmul as sparse_dense_matmul
from .frontend import sparse_expand_dims as sparse_expand_dims
from .frontend import sparse_eye as sparse_eye
from .frontend import sparse_fill_empty_rows as sparse_fill_empty_rows
from .frontend import sparse_map_values as sparse_map_values
from .frontend import sparse_mask as sparse_mask
from .frontend import sparse_maximum as sparse_maximum
from .frontend import sparse_minimum as sparse_minimum
from .frontend import sparse_reduce_max as sparse_reduce_max
from .frontend import sparse_reduce_sum as sparse_reduce_sum
from .frontend import sparse_reorder as sparse_reorder
from .frontend import sparse_reset_shape as sparse_reset_shape
from .frontend import sparse_reshape as sparse_reshape
from .frontend import sparse_retain as sparse_retain
from .frontend import sparse_sampled_add as sparse_sampled_add
from .frontend import sparse_segment_mean as sparse_segment_mean
from .frontend import sparse_segment_sqrt_n as sparse_segment_sqrt_n
from .frontend import sparse_segment_sum as sparse_segment_sum
from .frontend import sparse_slice as sparse_slice
from .frontend import sparse_softmax as sparse_softmax
from .frontend import sparse_split as sparse_split
from .frontend import sparse_to_dense as sparse_to_dense
from .frontend import sparse_to_indicator as sparse_to_indicator
from .frontend import sparse_transpose as sparse_transpose


def _get_dense_shape(obj: object) -> tuple[int, ...]:
    """Extract dense shape tuple from a tensor or shape structure.

    Args:
        obj (object): Target tensor or shape structure.

    Returns:
        tuple[int, ...]: Extracted dense shape tuple.
    """
    if obj is None:
        return ()
    for attr in ("dense_shape", "shape", "shape_metadata"):
        val = getattr(obj, attr, None)
        if val is not None:
            return tuple(int(d) for d in val)
    if isinstance(obj, (list, tuple)):
        return tuple(int(d) for d in obj)
    return ()


def _reduce_sparse_shape(shape: tuple[int, ...], axis: object = None, keepdims: bool = False) -> tuple[int, ...]:
    """Calculate shape after reduction along axis.

    Args:
        shape (tuple[int, ...]): Dense shape.
        axis (object): Reduction axis or axes.
        keepdims (bool): Whether to preserve reduced dimensions.

    Returns:
        tuple[int, ...]: Reduced shape.
    """
    if axis is None:
        return (1,) * len(shape) if keepdims else ()
    axes: set[int] = {int(axis)} if isinstance(axis, int) else {int(a) for a in axis}  # type: ignore[union-attr]
    normalized_axes = {a + len(shape) if a < 0 else a for a in axes}
    res: list[int] = []
    for i, dim in enumerate(shape):
        if i in normalized_axes:
            if keepdims:
                res.append(1)
        else:
            res.append(dim)
    return tuple(res)


def _broadcast_sparse_shapes(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> tuple[int, ...]:
    """Broadcast two dense equivalent shapes.

    Args:
        shape_a (tuple[int, ...]): First shape.
        shape_b (tuple[int, ...]): Second shape.

    Returns:
        tuple[int, ...]: Broadcasted shape.
    """
    if not shape_a:
        return shape_b
    if not shape_b:
        return shape_a
    from ml_switcheroo_compiler.core.shape import broadcast_shapes

    return broadcast_shapes(shape_a, shape_b)


@register_op("SparseDenseMatMul")
class SparseDenseMatMul(OpDef):
    """Operation class for sparse dense mat mul computation."""

    op_name = "SparseDenseMatMul"

    def infer_shape(self, sp_a, b, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse to dense operation.

        Args:
            sp_a (object): The sp_a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape_a = _get_dense_shape(sp_a)
        shape_b = _get_dense_shape(b)
        if len(shape_a) >= 2 and len(shape_b) >= 2:
            batch_a = shape_a[:-2]
            batch_b = shape_b[:-2]
            batch_out = _broadcast_sparse_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
            return batch_out + (shape_a[-2], shape_b[-1])
        return ()


@register_op("SparseAdd")
class SparseAdd(OpDef):
    """Operation class for sparse add computation."""

    op_name = "SparseAdd"

    def infer_shape(self, a, b, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse add operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape_a = _get_dense_shape(a)
        shape_b = _get_dense_shape(b)
        return _broadcast_sparse_shapes(shape_a, shape_b)


@register_op("SparseSampledAdd")
class SparseSampledAdd(OpDef):
    """Operation class for sparse sampled add computation."""

    op_name = "SparseSampledAdd"

    def infer_shape(self, a, b, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse sampled add operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape_a = _get_dense_shape(a)
        shape_b = _get_dense_shape(b)
        return _broadcast_sparse_shapes(shape_a, shape_b)


@register_op("Smm")
class Smm(OpDef):
    """Operation class for smm computation."""

    op_name = "Smm"

    def infer_shape(self, a, b, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the smm operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape_a = _get_dense_shape(a)
        shape_b = _get_dense_shape(b)
        if len(shape_a) >= 2 and len(shape_b) >= 2:
            batch_a = shape_a[:-2]
            batch_b = shape_b[:-2]
            batch_out = _broadcast_sparse_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
            return batch_out + (shape_a[-2], shape_b[-1])
        return ()


@register_op("SparseReduceSum")
class SparseReduceSum(OpDef):
    """Operation class for sparse reduce sum computation."""

    op_name = "SparseReduceSum"

    def infer_shape(self, sp_input, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse reduce sum operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape = _get_dense_shape(sp_input)
        if not shape:
            return ()
        axis = kwargs.get("axis")
        keepdims = bool(kwargs.get("keepdims", False))
        return _reduce_sparse_shape(shape, axis=axis, keepdims=keepdims)


@register_op("SparseSoftmax")
class SparseSoftmax(OpDef):
    """Operation class for sparse softmax computation."""

    op_name = "SparseSoftmax"

    def infer_shape(self, sp_input, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse softmax operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return _get_dense_shape(sp_input)


@register_op("SparseReduceMax")
class SparseReduceMax(OpDef):
    """Operation class for sparse reduce max computation."""

    op_name = "SparseReduceMax"

    def infer_shape(self, sp_input, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse reduce max operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape = _get_dense_shape(sp_input)
        if not shape:
            return ()
        axis = kwargs.get("axis")
        keepdims = bool(kwargs.get("keepdims", False))
        return _reduce_sparse_shape(shape, axis=axis, keepdims=keepdims)


@register_op("SparseBincount")
class SparseBincount(OpDef):
    """Operation class for sparse bincount computation."""

    op_name = "SparseBincount"

    def infer_shape(self, a, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse bincount operation.

        Args:
            a (object): The a parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape = _get_dense_shape(a)
        batch = shape[0] if shape else 1
        size = int(kwargs.get("size", kwargs.get("maxlength", 0)))
        return (batch, size) if size > 0 else (batch,)


@register_op("SparseCrossHashed")
class SparseCrossHashed(OpDef):
    """Operation class for sparse cross hashed computation."""

    op_name = "SparseCrossHashed"

    def infer_shape(self, inputs, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse cross hashed operation.

        Args:
            inputs (object): The inputs parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        inp_list = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        first_shape = _get_dense_shape(inp_list[0]) if inp_list else ()
        batch = first_shape[0] if first_shape else 1
        num_buckets = int(kwargs.get("num_buckets", 1))
        return (batch, num_buckets)


@register_op("SparseExpandDims")
class SparseExpandDims(OpDef):
    """Operation class for sparse expand dims computation."""

    op_name = "SparseExpandDims"

    def infer_shape(self, a, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse expand dims operation.

        Args:
            a (object): The a parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape = list(_get_dense_shape(a))
        axis = int(kwargs.get("axis", 0))
        if axis < 0:
            axis += len(shape) + 1
        shape.insert(axis, 1)
        return tuple(shape)


@register_op("SparseEye")
class SparseEye(OpDef):
    """Operation class for sparse eye computation."""

    op_name = "SparseEye"

    def infer_shape(self, num_rows, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse eye operation.

        Args:
            num_rows (object): The num_rows parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        r = int(getattr(num_rows, "value", num_rows)) if isinstance(num_rows, int) or hasattr(num_rows, "value") else 1
        c = int(kwargs.get("num_columns", r))
        return (r, c)


@register_op("SparseFillEmptyRows")
class SparseFillEmptyRows(OpDef):
    """Operation class for sparse fill empty rows computation."""

    op_name = "SparseFillEmptyRows"

    def infer_shape(self, sp_input, default_value=None, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse fill empty rows operation.

        Args:
            sp_input (object): The sp_input parameter.
            default_value (object): The default_value parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return _get_dense_shape(sp_input)


@register_op("SparseMapValues")
class SparseMapValues(OpDef):
    """Operation class for sparse map values computation."""

    op_name = "SparseMapValues"

    def infer_shape(self, sp_input, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse map values operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return _get_dense_shape(sp_input)


@register_op("SparseMask")
class SparseMask(OpDef):
    """Operation class for sparse mask computation."""

    op_name = "SparseMask"

    def infer_shape(self, a, mask=None, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse mask operation.

        Args:
            a (object): The a parameter.
            mask (object): The mask parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return _get_dense_shape(a)


@register_op("SparseMaximum")
class SparseMaximum(OpDef):
    """Operation class for sparse maximum computation."""

    op_name = "SparseMaximum"

    def infer_shape(self, a, b, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse maximum operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return _broadcast_sparse_shapes(_get_dense_shape(a), _get_dense_shape(b))


@register_op("SparseMinimum")
class SparseMinimum(OpDef):
    """Operation class for sparse minimum computation."""

    op_name = "SparseMinimum"

    def infer_shape(self, a, b, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse minimum operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return _broadcast_sparse_shapes(_get_dense_shape(a), _get_dense_shape(b))


@register_op("SparseReorder")
class SparseReorder(OpDef):
    """Operation class for sparse reorder computation."""

    op_name = "SparseReorder"

    def infer_shape(self, sp_input, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse reorder operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return _get_dense_shape(sp_input)


@register_op("SparseResetShape")
class SparseResetShape(OpDef):
    """Operation class for sparse reset shape computation."""

    op_name = "SparseResetShape"

    def infer_shape(self, sp_input, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse reset shape operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        new_shape = kwargs.get("new_shape")
        if new_shape is not None:
            return tuple(int(d) for d in new_shape)
        return _get_dense_shape(sp_input)


@register_op("SparseReshape")
class SparseReshape(OpDef):
    """Operation class for sparse reshape computation."""

    op_name = "SparseReshape"

    def infer_shape(self, sp_input, shape=None, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse reshape operation.

        Args:
            sp_input (object): The sp_input parameter.
            shape (object): The shape parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if shape is not None:
            return tuple(int(d) for d in shape) if isinstance(shape, (list, tuple)) else _get_dense_shape(shape)
        return _get_dense_shape(sp_input)


@register_op("SparseRetain")
class SparseRetain(OpDef):
    """Operation class for sparse retain computation."""

    op_name = "SparseRetain"

    def infer_shape(self, sp_input, to_retain=None, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse retain operation.

        Args:
            sp_input (object): The sp_input parameter.
            to_retain (object): The to_retain parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return _get_dense_shape(sp_input)


@register_op("SparseSegmentMean")
class SparseSegmentMean(OpDef):
    """Operation class for sparse segment mean computation."""

    op_name = "SparseSegmentMean"

    def infer_shape(self, data, indices=None, segment_ids=None, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse segment mean operation.

        Args:
            data (object): The data parameter.
            indices (object): The indices parameter.
            segment_ids (object): The segment_ids parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        d_shape = _get_dense_shape(data)
        num_segments = int(kwargs.get("num_segments", 1))
        return (num_segments,) + d_shape[1:]


@register_op("SparseSegmentSqrtN")
class SparseSegmentSqrtN(OpDef):
    """Operation class for sparse segment sqrt n computation."""

    op_name = "SparseSegmentSqrtN"

    def infer_shape(self, data, indices=None, segment_ids=None, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse segment sqrt n operation.

        Args:
            data (object): The data parameter.
            indices (object): The indices parameter.
            segment_ids (object): The segment_ids parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        d_shape = _get_dense_shape(data)
        num_segments = int(kwargs.get("num_segments", 1))
        return (num_segments,) + d_shape[1:]


@register_op("SparseSegmentSum")
class SparseSegmentSum(OpDef):
    """Operation class for sparse segment sum computation."""

    op_name = "SparseSegmentSum"

    def infer_shape(self, data, indices=None, segment_ids=None, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse segment sum operation.

        Args:
            data (object): The data parameter.
            indices (object): The indices parameter.
            segment_ids (object): The segment_ids parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        d_shape = _get_dense_shape(data)
        num_segments = int(kwargs.get("num_segments", 1))
        return (num_segments,) + d_shape[1:]


@register_op("SparseSlice")
class SparseSlice(OpDef):
    """Operation class for sparse slice computation."""

    op_name = "SparseSlice"

    def infer_shape(self, sp_input, start=None, size=None, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse slice operation.

        Args:
            sp_input (object): The sp_input parameter.
            start (object): The start parameter.
            size (object): The size parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        target_size = size if size is not None else kwargs.get("size")
        if target_size is not None:
            return tuple(int(d) for d in target_size) if isinstance(target_size, (list, tuple)) else _get_dense_shape(target_size)
        return _get_dense_shape(sp_input)


@register_op("SparseToIndicator")
class SparseToIndicator(OpDef):
    """Operation class for sparse to indicator computation."""

    op_name = "SparseToIndicator"

    def infer_shape(self, sp_input, vocab_size=None, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse to indicator operation.

        Args:
            sp_input (object): The sp_input parameter.
            vocab_size (object): The vocab_size parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape = _get_dense_shape(sp_input)
        batch = shape[0] if shape else 1
        v_size = vocab_size if vocab_size is not None else kwargs.get("vocab_size", 1)
        v = int(v_size) if isinstance(v_size, int) else 1
        return (batch, v)


@register_op("SparseTranspose")
class SparseTranspose(OpDef):
    """Operation class for sparse transpose computation."""

    op_name = "SparseTranspose"

    def infer_shape(self, sp_input, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse transpose operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape = _get_dense_shape(sp_input)
        perm = kwargs.get("perm")
        if perm is not None:
            return tuple(shape[int(p)] for p in perm)
        return tuple(reversed(shape))


__all__ = [
    "SparseDenseMatMul",
    "SparseMapValues",
    "SparseReduceMax",
    "SparseReshape",
    "SparseSegmentSum",
    "SparseTranspose",
    "sparse_eye",
    "sparse_maximum",
    "sparse_reorder",
    "sparse_segment_mean",
    "sparse_softmax",
    "SparseSampledAdd",
    "smm",
    "SparseConcat",
    "SparseSplit",
    "SparseToDense",
    "sparse_concat",
    "sparse_split",
    "sparse_to_dense",
]


@register_op("SparseConcat")
class SparseConcat(OpDef):
    """Operation class for sparse concat computation."""

    op_name = "SparseConcat"

    def infer_shape(self, sp_inputs, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse concat operation.

        Args:
            sp_inputs (object): The sp_inputs parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        inps = sp_inputs if isinstance(sp_inputs, (list, tuple)) else [sp_inputs]
        if not inps:
            return ()
        first_shape = list(_get_dense_shape(inps[0]))
        axis = int(kwargs.get("axis", 0))
        if axis < 0 and first_shape:
            axis += len(first_shape)
        total_dim = 0
        for inp in inps:
            s = _get_dense_shape(inp)
            if len(s) > axis:
                total_dim += s[axis]
        if first_shape and len(first_shape) > axis:
            first_shape[axis] = total_dim
        return tuple(first_shape)


@register_op("SparseSplit")
class SparseSplit(OpDef):
    """Operation class for sparse split computation."""

    op_name = "SparseSplit"

    def infer_shape(self, sp_input, num_split=1, **kwargs) -> tuple[tuple[int, ...], ...] | tuple[int, ...]:
        """Infers the output shape for the sparse split operation.

        Args:
            sp_input (object): The sp_input parameter.
            num_split (object): The num_split parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Result.
        """
        shape = list(_get_dense_shape(sp_input))
        n_split = num_split if num_split is not None else kwargs.get("num_split", 1)
        n = int(n_split) if isinstance(n_split, int) else 1
        axis = int(kwargs.get("axis", 0))
        if axis < 0 and shape:
            axis += len(shape)
        if shape and len(shape) > axis:
            shape[axis] = shape[axis] // n
        chunk = tuple(shape)
        return tuple(chunk for _ in range(n))


@register_op("SparseToDense")
class SparseToDense(OpDef):
    """Operation class for sparse to dense computation."""

    op_name = "SparseToDense"

    def infer_shape(self, sparse_indices, output_shape, sparse_values=None, default_value=None, **kwargs) -> tuple[int, ...]:
        """Infers the output shape for the sparse to dense operation.

        Args:
            sparse_indices (object): The sparse_indices parameter.
            output_shape (object): The output_shape parameter.
            sparse_values (object): The sparse_values parameter.
            default_value (object): The default_value parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return tuple(int(d) for d in output_shape) if isinstance(output_shape, (list, tuple)) else _get_dense_shape(output_shape)
