# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Sparse ops."""

from typing import Any

from ml_switcheroo_compiler.ops.base import OpDef, register_op

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


@register_op("SparseDenseMatMul")
class SparseDenseMatMul(OpDef):
    """Operation class for sparse dense mat mul computation."""

    op_name = "SparseDenseMatMul"

    def infer_shape(self, sp_a: Any, b: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse to dense operation.

        Args:
            sp_a (object): The sp_a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        shape_a = getattr(sp_a, "shape", getattr(sp_a, "dense_shape", ()))
        shape_b = getattr(b, "shape", ())
        if len(shape_a) >= 2 and len(shape_b) >= 2:
            return shape_a[:-1] + shape_b[1:]
        return ()


@register_op("SparseAdd")
class SparseAdd(OpDef):
    """Operation class for sparse add computation."""

    op_name = "SparseAdd"

    def infer_shape(self, a: Any, b: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse add operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(a, "dense_shape", getattr(a, "shape", ()))


@register_op("SparseSampledAdd")
class SparseSampledAdd(OpDef):
    """Operation class for sparse sampled add computation."""

    op_name = "SparseSampledAdd"

    def infer_shape(self, a: Any, b: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse sampled add operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(a, "dense_shape", getattr(a, "shape", ()))


@register_op("Smm")
class Smm(OpDef):
    """Operation class for smm computation."""

    op_name = "Smm"

    def infer_shape(self, a: Any, b: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the smm operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        shape_a = getattr(a, "shape", getattr(a, "dense_shape", ()))
        shape_b = getattr(b, "shape", ())
        if len(shape_a) >= 2 and len(shape_b) >= 2:
            return shape_a[:-1] + shape_b[1:]
        return ()


@register_op("SparseReduceSum")
class SparseReduceSum(OpDef):
    """Operation class for sparse reduce sum computation."""

    op_name = "SparseReduceSum"

    def infer_shape(self, sp_input: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse reduce sum operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseSoftmax")
class SparseSoftmax(OpDef):
    """Operation class for sparse softmax computation."""

    op_name = "SparseSoftmax"

    def infer_shape(self, sp_input: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse softmax operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(sp_input, "dense_shape", ())


@register_op("SparseReduceMax")
class SparseReduceMax(OpDef):
    """Operation class for sparse reduce max computation."""

    op_name = "SparseReduceMax"

    def infer_shape(self, sp_input: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse reduce max operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseBincount")
class SparseBincount(OpDef):
    """Operation class for sparse bincount computation."""

    op_name = "SparseBincount"

    def infer_shape(self, a: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse bincount operation.

        Args:
            a (object): The a parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseCrossHashed")
class SparseCrossHashed(OpDef):
    """Operation class for sparse cross hashed computation."""

    op_name = "SparseCrossHashed"

    def infer_shape(self, inputs: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse cross hashed operation.

        Args:
            inputs (object): The inputs parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseExpandDims")
class SparseExpandDims(OpDef):
    """Operation class for sparse expand dims computation."""

    op_name = "SparseExpandDims"

    def infer_shape(self, a: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse expand dims operation.

        Args:
            a (object): The a parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseEye")
class SparseEye(OpDef):
    """Operation class for sparse eye computation."""

    op_name = "SparseEye"

    def infer_shape(self, num_rows: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse eye operation.

        Args:
            num_rows (object): The num_rows parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseFillEmptyRows")
class SparseFillEmptyRows(OpDef):
    """Operation class for sparse fill empty rows computation."""

    op_name = "SparseFillEmptyRows"

    def infer_shape(self, sp_input: Any, default_value: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse fill empty rows operation.

        Args:
            sp_input (object): The sp_input parameter.
            default_value (object): The default_value parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(sp_input, "dense_shape", ())


@register_op("SparseMapValues")
class SparseMapValues(OpDef):
    """Operation class for sparse map values computation."""

    op_name = "SparseMapValues"

    def infer_shape(self, sp_input: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse map values operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(sp_input, "dense_shape", ())


@register_op("SparseMask")
class SparseMask(OpDef):
    """Operation class for sparse mask computation."""

    op_name = "SparseMask"

    def infer_shape(self, a: Any, mask: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse mask operation.

        Args:
            a (object): The a parameter.
            mask (object): The mask parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(a, "dense_shape", ())


@register_op("SparseMaximum")
class SparseMaximum(OpDef):
    """Operation class for sparse maximum computation."""

    op_name = "SparseMaximum"

    def infer_shape(self, a: Any, b: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse maximum operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(a, "dense_shape", ())


@register_op("SparseMinimum")
class SparseMinimum(OpDef):
    """Operation class for sparse minimum computation."""

    op_name = "SparseMinimum"

    def infer_shape(self, a: Any, b: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse minimum operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(a, "dense_shape", ())


@register_op("SparseReorder")
class SparseReorder(OpDef):
    """Operation class for sparse reorder computation."""

    op_name = "SparseReorder"

    def infer_shape(self, sp_input: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse reorder operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(sp_input, "dense_shape", ())


@register_op("SparseResetShape")
class SparseResetShape(OpDef):
    """Operation class for sparse reset shape computation."""

    op_name = "SparseResetShape"

    def infer_shape(self, sp_input: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse reset shape operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseReshape")
class SparseReshape(OpDef):
    """Operation class for sparse reshape computation."""

    op_name = "SparseReshape"

    def infer_shape(self, sp_input: Any, shape: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse reshape operation.

        Args:
            sp_input (object): The sp_input parameter.
            shape (object): The shape parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseRetain")
class SparseRetain(OpDef):
    """Operation class for sparse retain computation."""

    op_name = "SparseRetain"

    def infer_shape(self, sp_input: Any, to_retain: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse retain operation.

        Args:
            sp_input (object): The sp_input parameter.
            to_retain (object): The to_retain parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(sp_input, "dense_shape", ())


@register_op("SparseSegmentMean")
class SparseSegmentMean(OpDef):
    """Operation class for sparse segment mean computation."""

    op_name = "SparseSegmentMean"

    def infer_shape(self, data: Any, indices: Any, segment_ids: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse segment mean operation.

        Args:
            data (object): The data parameter.
            indices (object): The indices parameter.
            segment_ids (object): The segment_ids parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseSegmentSqrtN")
class SparseSegmentSqrtN(OpDef):
    """Operation class for sparse segment sqrt n computation."""

    op_name = "SparseSegmentSqrtN"

    def infer_shape(self, data: Any, indices: Any, segment_ids: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse segment sqrt n operation.

        Args:
            data (object): The data parameter.
            indices (object): The indices parameter.
            segment_ids (object): The segment_ids parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseSegmentSum")
class SparseSegmentSum(OpDef):
    """Operation class for sparse segment sum computation."""

    op_name = "SparseSegmentSum"

    def infer_shape(self, data: Any, indices: Any, segment_ids: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse segment sum operation.

        Args:
            data (object): The data parameter.
            indices (object): The indices parameter.
            segment_ids (object): The segment_ids parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseSlice")
class SparseSlice(OpDef):
    """Operation class for sparse slice computation."""

    op_name = "SparseSlice"

    def infer_shape(self, sp_input: Any, start: Any, size: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse slice operation.

        Args:
            sp_input (object): The sp_input parameter.
            start (object): The start parameter.
            size (object): The size parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseToIndicator")
class SparseToIndicator(OpDef):
    """Operation class for sparse to indicator computation."""

    op_name = "SparseToIndicator"

    def infer_shape(self, sp_input: Any, vocab_size: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse to indicator operation.

        Args:
            sp_input (object): The sp_input parameter.
            vocab_size (object): The vocab_size parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseTranspose")
class SparseTranspose(OpDef):
    """Operation class for sparse transpose computation."""

    op_name = "SparseTranspose"

    def infer_shape(self, sp_input: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse transpose operation.

        Args:
            sp_input (object): The sp_input parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


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


@register_op("RaggedDot")
class RaggedDot(OpDef):
    """Operation class for ragged dot computation."""

    op_name = "RaggedDot"

    def infer_shape(self, a: Any, b: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the ragged dot operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return a.shape


@register_op("SparseConcat")
class SparseConcat(OpDef):
    """Operation class for sparse concat computation."""

    op_name = "SparseConcat"

    def infer_shape(self, sp_inputs: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse concat operation.

        Args:
            sp_inputs (object): The sp_inputs parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseSplit")
class SparseSplit(OpDef):
    """Operation class for sparse split computation."""

    op_name = "SparseSplit"

    def infer_shape(self, sp_input: Any, num_split: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse split operation.

        Args:
            sp_input (object): The sp_input parameter.
            num_split (object): The num_split parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("SparseToDense")
class SparseToDense(OpDef):
    """Operation class for sparse to dense computation."""

    op_name = "SparseToDense"

    def infer_shape(self, sparse_indices: Any, output_shape: Any, sparse_values: Any, default_value: Any, **kwargs: Any) -> Any:
        """Infers the output shape for the sparse to dense operation.

        Args:
            sparse_indices (object): The sparse_indices parameter.
            output_shape (object): The output_shape parameter.
            sparse_values (object): The sparse_values parameter.
            default_value (object): The default_value parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()
