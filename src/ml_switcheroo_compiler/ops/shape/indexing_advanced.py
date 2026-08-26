"""Module indexing_advanced.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Shape operations for advanced indexing and sorting Tensor objects."""

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.ops.base import OpDef, register_op


def _normalize_k(k):
    """Evaluate _normalize_k operation.

    Args:
        k (object): The k parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if hasattr(k, "__array__") and not isinstance(k, tuple):
        k = k.__array__()
    if hasattr(k, "item"):
        k = int(k.item())
    else:
        try:
            k = int(k)
        except (ValueError, TypeError):
            return k
    return k


@register_op("TopK")
class TopK(OpDef):
    """Operation for finding the top K elements along the last axis."""

    op_name = "TopK"

    def infer_shape(self, x, k=None, **kwargs):
        """Infer the output shape for the operation.

        Args:
            x (object): The x parameter.
            k (object): The k parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if k is None:
            k = kwargs.get("k", 1)
        k = _normalize_k(k)

        if not hasattr(x, "shape") or not x.shape:
            return (None,)
        out_shape = list(x.shape)
        out_shape[-1] = k
        return tuple(out_shape)


@register_op("Argsort")
class Argsort(OpDef):
    """Operation to return the indices that would sort an array."""

    op_name = "Argsort"

    def infer_shape(
        self,
        x,
        dimension=-1,
        is_stable=True,
        **kwargs,
    ):
        """Infer the output shape for sorting indices.

        Args:
            x (object): The x parameter.
            dimension (object): The dimension parameter.
            is_stable (object): The is_stable parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if isinstance(x, tuple) and hasattr(x, "shape"):
            return x.shape
        return getattr(x, "shape", ())


@register_op("Sort")
class Sort(OpDef):
    """Operation to sort a tensor along a specified dimension."""

    op_name = "Sort"

    def infer_shape(
        self,
        x,
        dimension=-1,
        is_stable=True,
        **kwargs,
    ):
        """Infer the output shape for the sorting operation.

        Args:
            x (object): The x parameter.
            dimension (object): The dimension parameter.
            is_stable (object): The is_stable parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(x, "shape", ())


@register_op("Where")
class Where(OpDef):
    """Operator to return elements chosen from x or y depending on a condition."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for conditionally chosen elements.

        Args:
            *args (object): Positional arguments, typically condition, x, and y.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple.
        """
        if len(args) > 1:
            return args[1]
        return (None,)


@register_op("Gather")
class Gather(OpDef):
    """Operator to gather slices from a tensor according to indices."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for gathered slices.

        Args:
            *args (object): Positional arguments, typically params and indices.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("Take")
class Take(OpDef):
    """Operator to take elements from an array along an axis."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for taken elements.

        Args:
            *args (object): Positional arguments, typically the input tensor and indices.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("TakeAlongAxis")
class TakeAlongAxis(OpDef):
    """Operator to take values from the input array by matching 1d index and data slices."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for elements taken along an axis.

        Args:
            *args (object): Positional arguments, typically the input array, indices, and axis.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("GatherNd")
class GatherNd(OpDef):
    """Operator to gather slices from a tensor into a tensor with specified shape."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for n-dimensional gathered slices.

        Args:
            *args (object): Positional arguments, typically params and n-dimensional indices.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("Scatter")
class Scatter(OpDef):
    """Operator to scatter updates into a new tensor according to indices."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for scattered updates.

        Args:
            *args (object): Positional arguments, typically input tensor, indices, and updates.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("ScatterNd")
class ScatterNd(OpDef):
    """Operator to apply sparse updates to individual values or slices within a tensor."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for n-dimensional scattered updates.

        Args:
            *args (object): Positional arguments, typically tensor, indices, and updates.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("ScatterAdd")
class ScatterAdd(OpDef):
    """Operator to add sparse updates to a tensor variable reference."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape after adding scattered updates.

        Args:
            *args (object): Positional arguments, typically tensor, indices, and updates.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("Vdot")
class Vdot(OpDef):
    """Operator to compute the dot product of two vectors."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the vector dot product.

        Args:
            *args (object): Positional arguments, typically the two input vectors.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("SearchSorted")
class SearchSorted(OpDef):
    """Operator to find indices where elements should be inserted to maintain order."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for searched sorted indices.

        Args:
            *args (object): Positional arguments, typically sorted sequence and values to insert.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("Select")
class Select(OpDef):
    """Operator to select elements from two arrays based on a boolean mask."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the selected elements.

        Args:
            *args (object): Positional arguments, typically condition, x, and y.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("Assign")
class Assign(OpDef):
    """Operator to assign a value to a tensor variable."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape after assignment.

        Args:
            *args (object): Positional arguments, typically the variable and the value.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("AssignAdd")
class AssignAdd(OpDef):
    """Operator to add a value to a tensor variable and assign the result."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape after addition assignment.

        Args:
            *args (object): Positional arguments, typically the variable and the value to add.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("AssignSub")
class AssignSub(OpDef):
    """Operator to subtract a value from a tensor variable and assign the result."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape after subtraction assignment.

        Args:
            *args (object): Positional arguments, typically the variable and the value to subtract.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("TensorScatterUpdate")
class TensorScatterUpdate(OpDef):
    """Operator to scatter updates into a new tensor using multi-dimensional indices."""

    def infer_shape(self, tensor, indices, updates, **kwargs):
        """Infer the output shape for scattered multidimensional updates.

        Args:
            tensor (object): The initial tensor to be updated.
            indices (object): The multidimensional indices for the updates.
            updates (object): The values to be updated in the tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: The inferred shape tuple matching the input tensor shape.
        """
        return getattr(tensor, "shape", ())


@register_op("Argpartition")
class Argpartition(OpDef):
    """Perform an indirect partition along the given axis using the algorithm specified by the kind keyword."""

    op_name = "Argpartition"
    np_op_name = "argpartition"

    def infer_shape(self, a, kth, axis: int = -1, **kwargs):
        """Infer the output shape for the partitioned indices.

        Args:
            a (object): The a parameter.
            kth (object): The kth parameter.
            axis (int): The axis parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return a.shape if hasattr(a, "shape") else ()


@register_op("Partition")
class Partition(OpDef):
    """Return a partitioned copy of an array."""

    op_name = "Partition"
    np_op_name = "partition"

    def infer_shape(self, a, kth, axis: int = -1, **kwargs):
        """Infer the output shape for the partitioned array.

        Args:
            a (object): The a parameter.
            kth (object): The kth parameter.
            axis (int): The axis parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return a.shape if hasattr(a, "shape") else ()


@register_op("Compress")
class Compress(OpDef):
    """Return selected slices of an array along given axis."""

    op_name = "Compress"
    np_op_name = "compress"

    def infer_shape(self, condition, a, axis=None, out=None, **kwargs):
        """Infer the output shape for the compressed array.

        Args:
            condition (object): The condition parameter.
            a (object): The a parameter.
            axis (int): The axis parameter.
            out (object): The out parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (None,)


@register_op("Diagonal")
class Diagonal(OpDef):
    """Return specified diagonals of a 2-D array."""

    op_name = "Diagonal"
    np_op_name = "diagonal"

    def infer_shape(self, a, offset: int = 0, axis1: int = 0, axis2: int = 1, **kwargs):
        """Infer the output shape for the diagonal extraction.

        Args:
            a (object): The a parameter.
            offset (int): The offset parameter.
            axis1 (int): The axis1 parameter.
            axis2 (int): The axis2 parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (None,)


@register_op("Diagflat")
class Diagflat(OpDef):
    """Create a two-dimensional array with the flattened input as a diagonal."""

    op_name = "Diagflat"
    np_op_name = "diagflat"

    def infer_shape(self, v, k: int = 0, **kwargs):
        """Infer the output shape for the flattened diagonal matrix.

        Args:
            v (object): The v parameter.
            k (int): The k parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (None, None)


@register_op("DiagIndices")
class DiagIndices(OpDef):
    """Return the indices to access the main diagonal of an array."""

    op_name = "DiagIndices"
    np_op_name = "diag_indices"

    def infer_shape(self, n: int, naxis: int = 2, **kwargs):
        """Infer the output shape for the main diagonal indices.

        Args:
            n (int): The n parameter.
            naxis (int): The naxis parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (None,)


@register_op("DiagIndicesFrom")
class DiagIndicesFrom(OpDef):
    """Return the indices to access the main diagonal of an n-dimensional array."""

    op_name = "DiagIndicesFrom"
    np_op_name = "diag_indices_from"

    def infer_shape(self, arr, **kwargs):
        """Infer the output shape for the n-dimensional diagonal indices.

        Args:
            arr (object): The array for which the indices will be generated.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: The inferred shape tuple representing a dynamic 1D size.
        """
        return (None,)


@register_op("BooleanMask")
class BooleanMask(OpDef):
    """Operation to mask a tensor with a boolean array."""

    op_name = "BooleanMask"

    def infer_shape(self, tensor, mask, axis=None, **kwargs):
        """Infer the output shape for the boolean masked array.

        Args:
            tensor (object): The tensor parameter.
            mask (object): The mask parameter.
            axis (object): The axis parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        # Typically dynamic size
        return (None,)


@register_op("InvertPermutation")
class InvertPermutation(OpDef):
    """Operation to compute the inverse permutation of a tensor."""

    op_name = "InvertPermutation"

    def infer_shape(self, x, **kwargs):
        """Infer the output shape for the inverted permutation.

        Args:
            x (object): The 1-D tensor indicating the permutation to invert.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: The inferred shape tuple matching the input tensor shape.
        """
        return getattr(x, "shape", ())
