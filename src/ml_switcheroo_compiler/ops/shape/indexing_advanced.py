"""Shape operations for Tensor objects."""

from __future__ import annotations

# pylint: disable=duplicate-code
from typing import TYPE_CHECKING

from ml_switcheroo_compiler.ops.base import OpDef, register_op

if TYPE_CHECKING:
    pass


def _normalize_k(k: object) -> int | object:
    """Function docstring.

    Args:
        k: Arg.
    """
    if hasattr(k, "__array__") and not isinstance(k, tuple):
        k = k.__array__()
    if hasattr(k, "item"):
        k = int(k.item())
    else:
        try:
            k = int(k)
        except (ValueError, TypeError):
            pass
    return k


@register_op("TopK")
class TopK(OpDef):
    """TopK operation."""

    op_name = "TopK"

    def infer_shape(self, x: object, k: object = None, **kwargs: object) -> object:
        """Infer shape.

        Args:
            x (object): The input x tensor.
            k (object): The k parameter for the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        if k is None:
            k = kwargs.get("k", 1)
        k = _normalize_k(k)

        if not hasattr(x, "shape") or not x.shape:
            return ()
        out_shape = list(x.shape)
        out_shape[-1] = k
        return tuple(out_shape)


@register_op("ArgSort")
class ArgSort(OpDef):
    """ArgSort operation."""

    op_name = "ArgSort"

    def infer_shape(
        self,
        x: object,
        dimension: object = -1,
        is_stable: object = True,
        **kwargs: object,
    ) -> object:
        """Infer shape for ArgSort."""
        if isinstance(x, tuple) and hasattr(x, "shape"):  # pragma: no branch
            return x.shape  # pragma: no cover
        return getattr(x, "shape", ())


@register_op("Sort")
class Sort(OpDef):
    """Sort operation."""

    op_name = "Sort"

    def infer_shape(
        self,
        x: object,
        dimension: object = -1,
        is_stable: object = True,
        **kwargs: object,
    ) -> object:
        """Infer shape.

        Args:
            x (object): The input x tensor.
            dimension (object): The dimension parameter for the operation.
            is_stable (object): The is_stable parameter for the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return getattr(x, "shape", ())


@register_op("Where")
class Where(OpDef):
    """Where operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Where."""
        return ()


@register_op("Gather")
class Gather(OpDef):
    """Gather operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Gather."""
        return ()


@register_op("Take")
class Take(OpDef):
    """Take operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Take."""
        return ()


@register_op("TakeAlongAxis")
class TakeAlongAxis(OpDef):
    """TakeAlongAxis operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for TakeAlongAxis."""
        return ()


@register_op("GatherNd")
class GatherNd(OpDef):
    """GatherNd operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for GatherNd."""
        return ()


@register_op("Scatter")
class Scatter(OpDef):
    """Scatter operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Scatter."""
        return ()


@register_op("ScatterNd")
class ScatterNd(OpDef):
    """ScatterNd operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for ScatterNd."""
        return ()


@register_op("ScatterAdd")
class ScatterAdd(OpDef):
    """ScatterAdd operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for ScatterAdd."""
        return ()


@register_op("Vdot")
class Vdot(OpDef):
    """Vdot operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Vdot."""
        return ()


@register_op("SearchSorted")
class SearchSorted(OpDef):
    """SearchSorted operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for SearchSorted."""
        return ()


@register_op("Select")
class Select(OpDef):
    """Select operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Select."""
        return ()


@register_op("Assign")
class Assign(OpDef):
    """Assign operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Assign."""
        return ()


@register_op("AssignAdd")
class AssignAdd(OpDef):
    """AssignAdd operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for AssignAdd."""
        return ()


@register_op("AssignSub")
class AssignSub(OpDef):
    """AssignSub operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for AssignSub."""
        return ()


@register_op("TensorScatterUpdate")
class TensorScatterUpdate(OpDef):
    """TensorScatterUpdate operator definition."""

    def infer_shape(self, tensor: object, indices: object, updates: object, **kwargs: object) -> object:
        """Infer shape for TensorScatterUpdate."""
        return getattr(tensor, "shape", ())  # pragma: no cover


@register_op("Argpartition")
class Argpartition(OpDef):
    """Perform an indirect partition along the given axis using the algorithm specified by the kind keyword."""

    op_name = "Argpartition"
    np_op_name = "argpartition"

    def infer_shape(self, a: object, kth: object, axis: int = -1, **kwargs: object) -> object:
        """Infer the output shape."""
        return a.shape if hasattr(a, "shape") else ()


@register_op("Partition")
class Partition(OpDef):
    """Return a partitioned copy of an array."""

    op_name = "Partition"
    np_op_name = "partition"

    def infer_shape(self, a: object, kth: object, axis: int = -1, **kwargs: object) -> object:
        """Infer the output shape."""
        return a.shape if hasattr(a, "shape") else ()


@register_op("Compress")
class Compress(OpDef):
    """Return selected slices of an array along given axis."""

    op_name = "Compress"
    np_op_name = "compress"

    def infer_shape(self, condition: object, a: object, axis: int = None, out: object = None, **kwargs: object) -> object:
        """Infer the output shape."""
        return (None,)


@register_op("Diagonal")
class Diagonal(OpDef):
    """Return specified diagonals."""

    op_name = "Diagonal"
    np_op_name = "diagonal"

    def infer_shape(self, a: object, offset: int = 0, axis1: int = 0, axis2: int = 1, **kwargs: object) -> object:
        """Infer the output shape."""
        return (None,)


@register_op("Diagflat")
class Diagflat(OpDef):
    """Create a two-dimensional array with the flattened input as a diagonal."""

    op_name = "Diagflat"
    np_op_name = "diagflat"

    def infer_shape(self, v: object, k: int = 0, **kwargs: object) -> object:
        """Infer the output shape."""
        return (None, None)


@register_op("DiagIndices")
class DiagIndices(OpDef):
    """Return the indices to access the main diagonal of an array."""

    op_name = "DiagIndices"
    np_op_name = "diag_indices"

    def infer_shape(self, n: int, ndim: int = 2, **kwargs: object) -> object:
        """Infer the output shape."""
        return (None,)


@register_op("DiagIndicesFrom")
class DiagIndicesFrom(OpDef):
    """Return the indices to access the main diagonal of an n-dimensional array."""

    op_name = "DiagIndicesFrom"
    np_op_name = "diag_indices_from"

    def infer_shape(self, arr: object, **kwargs: object) -> object:
        """Infer the output shape."""
        return (None,)


@register_op("BooleanMask")
class BooleanMask(OpDef):
    """BooleanMask operation."""

    op_name = "BooleanMask"

    def infer_shape(self, tensor: object, mask: object, axis: object = None, **kwargs: object) -> object:
        """Infer shape."""
        # Typically dynamic size
        return (None,)


@register_op("InvertPermutation")
class InvertPermutation(OpDef):
    """InvertPermutation operation."""

    op_name = "InvertPermutation"

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(x, "shape", ())
