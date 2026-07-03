"""Sparse ops."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .frontend import (
    sparse_bincount as sparse_bincount,
)
from .frontend import (
    sparse_cross_hashed as sparse_cross_hashed,
)
from .frontend import (
    sparse_expand_dims as sparse_expand_dims,
)
from .frontend import (
    sparse_eye as sparse_eye,
)
from .frontend import (
    sparse_fill_empty_rows as sparse_fill_empty_rows,
)
from .frontend import (
    sparse_map_values as sparse_map_values,
)
from .frontend import (
    sparse_mask as sparse_mask,
)
from .frontend import (
    sparse_maximum as sparse_maximum,
)
from .frontend import (
    sparse_minimum as sparse_minimum,
)
from .frontend import (
    sparse_reduce_max as sparse_reduce_max,
)
from .frontend import (
    sparse_reduce_sum as sparse_reduce_sum,
)
from .frontend import (
    sparse_reorder as sparse_reorder,
)
from .frontend import (
    sparse_reset_shape as sparse_reset_shape,
)
from .frontend import (
    sparse_reshape as sparse_reshape,
)
from .frontend import (
    sparse_retain as sparse_retain,
)
from .frontend import (
    sparse_segment_mean as sparse_segment_mean,
)
from .frontend import (
    sparse_segment_sqrt_n as sparse_segment_sqrt_n,
)
from .frontend import (
    sparse_segment_sum as sparse_segment_sum,
)
from .frontend import (
    sparse_slice as sparse_slice,
)
from .frontend import (
    sparse_softmax as sparse_softmax,
)
from .frontend import (
    sparse_to_indicator as sparse_to_indicator,
)
from .frontend import (
    sparse_transpose as sparse_transpose,
)


@register_op("SparseDenseMatMul")
class SparseDenseMatMul(OpDef):
    """SparseDenseMatMul operation."""

    op_name = "SparseDenseMatMul"

    def infer_shape(self, sp_a: object, b: object, **kwargs: object) -> object:
        """Infer shape."""
        # Typically returns shape based on sp_a dense_shape and b
        return ()


@register_op("SparseAdd")
class SparseAdd(OpDef):
    """SparseAdd operation."""

    op_name = "SparseAdd"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(a, "dense_shape", ())  # pragma: no cover  # pragma: no cover


@register_op("SparseReduceSum")
class SparseReduceSum(OpDef):
    """SparseReduceSum operation."""

    op_name = "SparseReduceSum"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("SparseSoftmax")
class SparseSoftmax(OpDef):
    """SparseSoftmax operation."""

    op_name = "SparseSoftmax"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(sp_input, "dense_shape", ())  # pragma: no cover  # pragma: no cover


@register_op("SparseReduceMax")
class SparseReduceMax(OpDef):
    """SparseReduceMax operation."""

    op_name = "SparseReduceMax"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("SparseBincount")
class SparseBincount(OpDef):
    """SparseBincount operation."""

    op_name = "SparseBincount"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("SparseCrossHashed")
class SparseCrossHashed(OpDef):
    """SparseCrossHashed operation."""

    op_name = "SparseCrossHashed"

    def infer_shape(self, inputs: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("SparseExpandDims")
class SparseExpandDims(OpDef):
    """SparseExpandDims operation."""

    op_name = "SparseExpandDims"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("SparseEye")
class SparseEye(OpDef):
    """SparseEye operation."""

    op_name = "SparseEye"

    def infer_shape(self, num_rows: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("SparseFillEmptyRows")
class SparseFillEmptyRows(OpDef):
    """SparseFillEmptyRows operation."""

    op_name = "SparseFillEmptyRows"

    def infer_shape(self, sp_input: object, default_value: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(sp_input, "dense_shape", ())  # pragma: no cover  # pragma: no cover


@register_op("SparseMapValues")
class SparseMapValues(OpDef):
    """SparseMapValues operation."""

    op_name = "SparseMapValues"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(sp_input, "dense_shape", ())  # pragma: no cover  # pragma: no cover


@register_op("SparseMask")
class SparseMask(OpDef):
    """SparseMask operation."""

    op_name = "SparseMask"

    def infer_shape(self, a: object, mask: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(a, "dense_shape", ())  # pragma: no cover  # pragma: no cover


@register_op("SparseMaximum")
class SparseMaximum(OpDef):
    """SparseMaximum operation."""

    op_name = "SparseMaximum"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(a, "dense_shape", ())  # pragma: no cover  # pragma: no cover


@register_op("SparseMinimum")
class SparseMinimum(OpDef):
    """SparseMinimum operation."""

    op_name = "SparseMinimum"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(a, "dense_shape", ())  # pragma: no cover  # pragma: no cover


@register_op("SparseReorder")
class SparseReorder(OpDef):
    """SparseReorder operation."""

    op_name = "SparseReorder"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(sp_input, "dense_shape", ())  # pragma: no cover  # pragma: no cover


@register_op("SparseResetShape")
class SparseResetShape(OpDef):
    """SparseResetShape operation."""

    op_name = "SparseResetShape"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("SparseReshape")
class SparseReshape(OpDef):
    """SparseReshape operation."""

    op_name = "SparseReshape"

    def infer_shape(self, sp_input: object, shape: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("SparseRetain")
class SparseRetain(OpDef):
    """SparseRetain operation."""

    op_name = "SparseRetain"

    def infer_shape(self, sp_input: object, to_retain: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(sp_input, "dense_shape", ())  # pragma: no cover  # pragma: no cover


@register_op("SparseSegmentMean")
class SparseSegmentMean(OpDef):
    """SparseSegmentMean operation."""

    op_name = "SparseSegmentMean"

    def infer_shape(self, data: object, indices: object, segment_ids: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover


@register_op("SparseSegmentSqrtN")
class SparseSegmentSqrtN(OpDef):
    """SparseSegmentSqrtN operation."""

    op_name = "SparseSegmentSqrtN"

    def infer_shape(self, data: object, indices: object, segment_ids: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover


@register_op("SparseSegmentSum")
class SparseSegmentSum(OpDef):
    """SparseSegmentSum operation."""

    op_name = "SparseSegmentSum"

    def infer_shape(self, data: object, indices: object, segment_ids: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover


@register_op("SparseSlice")
class SparseSlice(OpDef):
    """SparseSlice operation."""

    op_name = "SparseSlice"

    def infer_shape(self, sp_input: object, start: object, size: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover


@register_op("SparseToIndicator")
class SparseToIndicator(OpDef):
    """SparseToIndicator operation."""

    op_name = "SparseToIndicator"

    def infer_shape(self, sp_input: object, vocab_size: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


@register_op("SparseTranspose")
class SparseTranspose(OpDef):
    """SparseTranspose operation."""

    op_name = "SparseTranspose"

    def infer_shape(self, sp_input: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()  # pragma: no cover  # pragma: no cover


__all__ = [
    "SparseAdd",
    "SparseBincount",
    "SparseCrossHashed",
    "SparseDenseMatMul",
    "SparseExpandDims",
    "SparseEye",
    "SparseFillEmptyRows",
    "SparseMapValues",
    "SparseMask",
    "SparseMaximum",
    "SparseMinimum",
    "SparseReduceMax",
    "SparseReduceSum",
    "SparseReorder",
    "SparseResetShape",
    "SparseReshape",
    "SparseRetain",
    "SparseSegmentMean",
    "SparseSegmentSqrtN",
    "SparseSegmentSum",
    "SparseSlice",
    "SparseSoftmax",
    "SparseToIndicator",
    "SparseTranspose",
    "sparse_bincount",
    "sparse_cross_hashed",
    "sparse_expand_dims",
    "sparse_eye",
    "sparse_fill_empty_rows",
    "sparse_map_values",
    "sparse_mask",
    "sparse_maximum",
    "sparse_minimum",
    "sparse_reduce_max",
    "sparse_reduce_sum",
    "sparse_reorder",
    "sparse_reset_shape",
    "sparse_reshape",
    "sparse_retain",
    "sparse_segment_mean",
    "sparse_segment_sqrt_n",
    "sparse_segment_sum",
    "sparse_slice",
    "sparse_softmax",
    "sparse_to_indicator",
    "sparse_transpose",
]
