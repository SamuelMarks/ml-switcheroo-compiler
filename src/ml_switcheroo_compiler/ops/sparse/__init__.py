# ruff: noqa
"""Sparse ops."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .frontend import smm as smm
from .frontend import sparse_concat as sparse_concat
from .frontend import sparse_split as sparse_split
from .frontend import sparse_to_dense as sparse_to_dense
from .frontend import sparse_add as sparse_add
from .frontend import sparse_bincount as sparse_bincount
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
from .frontend import sparse_to_indicator as sparse_to_indicator
from .frontend import sparse_transpose as sparse_transpose


@register_op("SparseDenseMatMul")
class SparseDenseMatMul(OpDef):
    op_name = "SparseDenseMatMul"

    def infer_shape(self, sp_a: object, b: object, **kwargs: object) -> object:
        """Infers the shape for the SparseDenseMatMul operation.

        Args:
            sp_a (object): The sp a parameter.
            b (object): The b parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        shape_a = getattr(sp_a, "shape", getattr(sp_a, "dense_shape", ()))
        shape_b = getattr(b, "shape", ())
        if len(shape_a) >= 2 and len(shape_b) >= 2:
            return shape_a[:-1] + shape_b[1:]
        return ()


@register_op("SparseAdd")
class SparseAdd(OpDef):
    op_name = "SparseAdd"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infers the shape for the SparseAdd operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return getattr(a, "dense_shape", getattr(a, "shape", ()))


@register_op("SparseSampledAdd")
class SparseSampledAdd(OpDef):
    op_name = "SparseSampledAdd"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infers the shape for the SparseSampledAdd operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return getattr(a, "dense_shape", getattr(a, "shape", ()))


@register_op("Smm")
class Smm(OpDef):
    op_name = "Smm"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infers the shape for the Smm operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        shape_a = getattr(a, "shape", getattr(a, "dense_shape", ()))
        shape_b = getattr(b, "shape", ())
        if len(shape_a) >= 2 and len(shape_b) >= 2:
            return shape_a[:-1] + shape_b[1:]
        return ()


@register_op("SparseReduceSum")
class SparseReduceSum(OpDef):
    op_name = "SparseReduceSum"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infers the shape for the SparseReduceSum operation.

        Args:
            sp_input (object): The sp input parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("SparseSoftmax")
class SparseSoftmax(OpDef):
    op_name = "SparseSoftmax"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infers the shape for the SparseSoftmax operation.

        Args:
            sp_input (object): The sp input parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return getattr(sp_input, "dense_shape", ())


@register_op("SparseReduceMax")
class SparseReduceMax(OpDef):
    op_name = "SparseReduceMax"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infers the shape for the SparseReduceMax operation.

        Args:
            sp_input (object): The sp input parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("SparseBincount")
class SparseBincount(OpDef):
    op_name = "SparseBincount"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infers the shape for the SparseBincount operation.

        Args:
            a (object): The a parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("SparseCrossHashed")
class SparseCrossHashed(OpDef):
    op_name = "SparseCrossHashed"

    def infer_shape(self, inputs: object, **kwargs: object) -> object:
        """Infers the shape for the SparseCrossHashed operation.

        Args:
            inputs (object): The inputs parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("SparseExpandDims")
class SparseExpandDims(OpDef):
    op_name = "SparseExpandDims"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infers the shape for the SparseExpandDims operation.

        Args:
            a (object): The a parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("SparseEye")
class SparseEye(OpDef):
    op_name = "SparseEye"

    def infer_shape(self, num_rows: object, **kwargs: object) -> object:
        """Infers the shape for the SparseEye operation.

        Args:
            num_rows (object): The num rows parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("SparseFillEmptyRows")
class SparseFillEmptyRows(OpDef):
    op_name = "SparseFillEmptyRows"

    def infer_shape(self, sp_input: object, default_value: object, **kwargs: object) -> object:
        """Infers the shape for the SparseFillEmptyRows operation.

        Args:
            sp_input (object): The sp input parameter.
            default_value (object): The default value parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return getattr(sp_input, "dense_shape", ())


@register_op("SparseMapValues")
class SparseMapValues(OpDef):
    op_name = "SparseMapValues"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infers the shape for the SparseMapValues operation.

        Args:
            sp_input (object): The sp input parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return getattr(sp_input, "dense_shape", ())


@register_op("SparseMask")
class SparseMask(OpDef):
    op_name = "SparseMask"

    def infer_shape(self, a: object, mask: object, **kwargs: object) -> object:
        """Infers the shape for the SparseMask operation.

        Args:
            a (object): The a parameter.
            mask (object): The mask parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return getattr(a, "dense_shape", ())


@register_op("SparseMaximum")
class SparseMaximum(OpDef):
    op_name = "SparseMaximum"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infers the shape for the SparseMaximum operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return getattr(a, "dense_shape", ())


@register_op("SparseMinimum")
class SparseMinimum(OpDef):
    op_name = "SparseMinimum"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infers the shape for the SparseMinimum operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return getattr(a, "dense_shape", ())


@register_op("SparseReorder")
class SparseReorder(OpDef):
    op_name = "SparseReorder"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infers the shape for the SparseReorder operation.

        Args:
            sp_input (object): The sp input parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return getattr(sp_input, "dense_shape", ())


@register_op("SparseResetShape")
class SparseResetShape(OpDef):
    op_name = "SparseResetShape"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infers the shape for the SparseResetShape operation.

        Args:
            sp_input (object): The sp input parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("SparseReshape")
class SparseReshape(OpDef):
    op_name = "SparseReshape"

    def infer_shape(self, sp_input: object, shape: object, **kwargs: object) -> object:
        """Infers the shape for the SparseReshape operation.

        Args:
            sp_input (object): The sp input parameter.
            shape (object): The shape parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("SparseRetain")
class SparseRetain(OpDef):
    op_name = "SparseRetain"

    def infer_shape(self, sp_input: object, to_retain: object, **kwargs: object) -> object:
        """Infers the shape for the SparseRetain operation.

        Args:
            sp_input (object): The sp input parameter.
            to_retain (object): The to retain parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return getattr(sp_input, "dense_shape", ())


@register_op("SparseSegmentMean")
class SparseSegmentMean(OpDef):
    op_name = "SparseSegmentMean"

    def infer_shape(self, data: object, indices: object, segment_ids: object, **kwargs: object) -> object:
        """Infers the shape for the SparseSegmentMean operation.

        Args:
            data (object): The data parameter.
            indices (object): The indices parameter.
            segment_ids (object): The segment ids parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("SparseSegmentSqrtN")
class SparseSegmentSqrtN(OpDef):
    op_name = "SparseSegmentSqrtN"

    def infer_shape(self, data: object, indices: object, segment_ids: object, **kwargs: object) -> object:
        """Infers the shape for the SparseSegmentSqrtN operation.

        Args:
            data (object): The data parameter.
            indices (object): The indices parameter.
            segment_ids (object): The segment ids parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("SparseSegmentSum")
class SparseSegmentSum(OpDef):
    op_name = "SparseSegmentSum"

    def infer_shape(self, data: object, indices: object, segment_ids: object, **kwargs: object) -> object:
        """Infers the shape for the SparseSegmentSum operation.

        Args:
            data (object): The data parameter.
            indices (object): The indices parameter.
            segment_ids (object): The segment ids parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("SparseSlice")
class SparseSlice(OpDef):
    op_name = "SparseSlice"

    def infer_shape(self, sp_input: object, start: object, size: object, **kwargs: object) -> object:
        """Infers the shape for the SparseSlice operation.

        Args:
            sp_input (object): The sp input parameter.
            start (object): The start parameter.
            size (object): The size parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("SparseToIndicator")
class SparseToIndicator(OpDef):
    op_name = "SparseToIndicator"

    def infer_shape(self, sp_input: object, vocab_size: object, **kwargs: object) -> object:
        """Infers the shape for the SparseToIndicator operation.

        Args:
            sp_input (object): The sp input parameter.
            vocab_size (object): The vocab size parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("SparseTranspose")
class SparseTranspose(OpDef):
    op_name = "SparseTranspose"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infers the shape for the SparseTranspose operation.

        Args:
            sp_input (object): The sp input parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
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
    """RaggedDot ops."""

    op_name = "RaggedDot"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infers the shape for the RaggedDot operation.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape.
        """
        return a.shape


@register_op("SparseConcat")
class SparseConcat(OpDef):
    op_name = "SparseConcat"

    def infer_shape(self, sp_inputs: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("SparseSplit")
class SparseSplit(OpDef):
    op_name = "SparseSplit"

    def infer_shape(self, sp_input: object, num_split: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("SparseToDense")
class SparseToDense(OpDef):
    op_name = "SparseToDense"

    def infer_shape(
        self,
        sparse_indices: object,
        output_shape: object,
        sparse_values: object,
        default_value: object,
        **kwargs: object,
    ) -> object:
        return ()
