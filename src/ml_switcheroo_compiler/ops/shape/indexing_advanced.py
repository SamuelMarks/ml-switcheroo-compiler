"""Shape operations for advanced indexing and sorting Tensor objects."""

from __future__ import annotations

# pylint: disable=duplicate-code
from typing import TYPE_CHECKING

from ml_switcheroo_compiler.ops.base import OpDef, register_op

if TYPE_CHECKING:
    pass


def _normalize_k(k: object) -> int | object:
    """Evaluate and process the normalize k operation.

    Args:
        k (object): The parameter k to be normalized, usually representing the number of top elements.

    Returns:
        int | object: The evaluated or processed output, typically casted to an integer.
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
    """Operation for finding the top K elements along the last axis."""

    op_name = "TopK"

    def infer_shape(self, x: object, k: object = None, **kwargs: object) -> object:
        """Infer the output shape for the operation.

        Args:
            x (object): The input tensor from which to compute the top elements.
            k (object, optional): The number of top elements to select. Defaults to None.
            **kwargs: Additional keyword arguments for shape inference.

        Returns:
            object: The inferred shape tuple replacing the last dimension with k.
        """
        if k is None:
            k = kwargs.get("k", 1)
        k = _normalize_k(k)

        if not hasattr(x, "shape") or not x.shape:
            return ()
        out_shape = list(x.shape)
        out_shape[-1] = k
        return tuple(out_shape)


@register_op("Argsort")
class Argsort(OpDef):
    """Operation to return the indices that would sort an array."""

    op_name = "Argsort"

    def infer_shape(
        self,
        x: object,
        dimension: object = -1,
        is_stable: object = True,
        **kwargs: object,
    ) -> object:
        """Infer the output shape for sorting indices.

        Args:
            x (object): The input tensor to be sorted.
            dimension (object, optional): The dimension along which to sort. Defaults to -1.
            is_stable (object, optional): Whether to use a stable sorting algorithm. Defaults to True.
            **kwargs: Additional keyword arguments for shape inference.

        Returns:
            object: The inferred shape tuple for the sorted indices, matching the input shape.
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
        x: object,
        dimension: object = -1,
        is_stable: object = True,
        **kwargs: object,
    ) -> object:
        """Infer the output shape for the sorting operation.

        Args:
            x (object): The input tensor to sort.
            dimension (object, optional): The dimension along which to sort. Defaults to -1.
            is_stable (object, optional): Whether to use a stable sorting algorithm. Defaults to True.
            **kwargs: Additional keyword arguments for shape inference.

        Returns:
            object: The inferred shape tuple for the sorted tensor, matching the input shape.
        """
        return getattr(x, "shape", ())


@register_op("Where")
class Where(OpDef):
    """Operator to return elements chosen from x or y depending on a condition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape for conditionally chosen elements.

        Args:
            *args (object): Positional arguments, typically condition, x, and y.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple.
        """
        if len(args) > 1:
            return args[1]
        return ()


@register_op("Gather")
class Gather(OpDef):
    """Operator to gather slices from a tensor according to indices."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape for gathered slices.

        Args:
            *args (object): Positional arguments, typically params and indices.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an empty tuple.
        """
        return ()


@register_op("Take")
class Take(OpDef):
    """Operator to take elements from an array along an axis."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape for taken elements.

        Args:
            *args (object): Positional arguments, typically the input tensor and indices.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an empty tuple.
        """
        return ()


@register_op("TakeAlongAxis")
class TakeAlongAxis(OpDef):
    """Operator to take values from the input array by matching 1d index and data slices."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape for elements taken along an axis.

        Args:
            *args (object): Positional arguments, typically the input array, indices, and axis.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an empty tuple.
        """
        return ()


@register_op("GatherNd")
class GatherNd(OpDef):
    """Operator to gather slices from a tensor into a tensor with specified shape."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape for n-dimensional gathered slices.

        Args:
            *args (object): Positional arguments, typically params and n-dimensional indices.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an empty tuple.
        """
        return ()


@register_op("Scatter")
class Scatter(OpDef):
    """Operator to scatter updates into a new tensor according to indices."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape for scattered updates.

        Args:
            *args (object): Positional arguments, typically input tensor, indices, and updates.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an empty tuple.
        """
        return ()


@register_op("ScatterNd")
class ScatterNd(OpDef):
    """Operator to apply sparse updates to individual values or slices within a tensor."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape for n-dimensional scattered updates.

        Args:
            *args (object): Positional arguments, typically tensor, indices, and updates.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an empty tuple.
        """
        return ()


@register_op("ScatterAdd")
class ScatterAdd(OpDef):
    """Operator to add sparse updates to a tensor variable reference."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape after adding scattered updates.

        Args:
            *args (object): Positional arguments, typically tensor, indices, and updates.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an empty tuple.
        """
        return ()


@register_op("Vdot")
class Vdot(OpDef):
    """Operator to compute the dot product of two vectors."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape for the vector dot product.

        Args:
            *args (object): Positional arguments, typically the two input vectors.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an empty tuple.
        """
        return ()


@register_op("SearchSorted")
class SearchSorted(OpDef):
    """Operator to find indices where elements should be inserted to maintain order."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape for searched sorted indices.

        Args:
            *args (object): Positional arguments, typically sorted sequence and values to insert.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an empty tuple.
        """
        return ()


@register_op("Select")
class Select(OpDef):
    """Operator to select elements from two arrays based on a boolean mask."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape for the selected elements.

        Args:
            *args (object): Positional arguments, typically condition, x, and y.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an empty tuple.
        """
        return ()


@register_op("Assign")
class Assign(OpDef):
    """Operator to assign a value to a tensor variable."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape after assignment.

        Args:
            *args (object): Positional arguments, typically the variable and the value.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an empty tuple.
        """
        return ()


@register_op("AssignAdd")
class AssignAdd(OpDef):
    """Operator to add a value to a tensor variable and assign the result."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape after addition assignment.

        Args:
            *args (object): Positional arguments, typically the variable and the value to add.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an empty tuple.
        """
        return ()


@register_op("AssignSub")
class AssignSub(OpDef):
    """Operator to subtract a value from a tensor variable and assign the result."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the output shape after subtraction assignment.

        Args:
            *args (object): Positional arguments, typically the variable and the value to subtract.
            **kwargs (object): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an empty tuple.
        """
        return ()


@register_op("TensorScatterUpdate")
class TensorScatterUpdate(OpDef):
    """Operator to scatter updates into a new tensor using multi-dimensional indices."""

    def infer_shape(self, tensor: object, indices: object, updates: object, **kwargs: object) -> object:
        """Infer the output shape for scattered multidimensional updates.

        Args:
            tensor (object): The initial tensor to be updated.
            indices (object): The multidimensional indices for the updates.
            updates (object): The values to be updated in the tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape tuple matching the input tensor shape.
        """
        return getattr(tensor, "shape", ())


@register_op("Argpartition")
class Argpartition(OpDef):
    """Perform an indirect partition along the given axis using the algorithm specified by the kind keyword."""

    op_name = "Argpartition"
    np_op_name = "argpartition"

    def infer_shape(self, a: object, kth: object, axis: int = -1, **kwargs: object) -> object:
        """Infer the output shape for the partitioned indices.

        Args:
            a (object): The array to partition.
            kth (object): Element index to partition by.
            axis (int, optional): The axis along which to sort. Defaults to -1.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape tuple matching the input array shape.
        """
        return a.shape if hasattr(a, "shape") else ()


@register_op("Partition")
class Partition(OpDef):
    """Return a partitioned copy of an array."""

    op_name = "Partition"
    np_op_name = "partition"

    def infer_shape(self, a: object, kth: object, axis: int = -1, **kwargs: object) -> object:
        """Infer the output shape for the partitioned array.

        Args:
            a (object): The array to partition.
            kth (object): Element index to partition by.
            axis (int, optional): The axis along which to sort. Defaults to -1.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape tuple matching the input array shape.
        """
        return a.shape if hasattr(a, "shape") else ()


@register_op("Compress")
class Compress(OpDef):
    """Return selected slices of an array along given axis."""

    op_name = "Compress"
    np_op_name = "compress"

    def infer_shape(self, condition: object, a: object, axis: int = None, out: object = None, **kwargs: object) -> object:
        """Infer the output shape for the compressed array.

        Args:
            condition (object): The boolean array that selects which entries to return.
            a (object): The input array from which elements are selected.
            axis (int, optional): The axis along which to take slices. Defaults to None.
            out (object, optional): The output array to place the result. Defaults to None.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape tuple with a single dynamic dimension.
        """
        return (None,)


@register_op("Diagonal")
class Diagonal(OpDef):
    """Return specified diagonals of a 2-D array."""

    op_name = "Diagonal"
    np_op_name = "diagonal"

    def infer_shape(self, a: object, offset: int = 0, axis1: int = 0, axis2: int = 1, **kwargs: object) -> object:
        """Infer the output shape for the diagonal extraction.

        Args:
            a (object): The array from which the diagonals are taken.
            offset (int, optional): Offset of the diagonal from the main diagonal. Defaults to 0.
            axis1 (int, optional): Axis to be used as the first axis of the 2-D sub-arrays. Defaults to 0.
            axis2 (int, optional): Axis to be used as the second axis of the 2-D sub-arrays. Defaults to 1.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape tuple representing a dynamic 1D size.
        """
        return (None,)


@register_op("Diagflat")
class Diagflat(OpDef):
    """Create a two-dimensional array with the flattened input as a diagonal."""

    op_name = "Diagflat"
    np_op_name = "diagflat"

    def infer_shape(self, v: object, k: int = 0, **kwargs: object) -> object:
        """Infer the output shape for the flattened diagonal matrix.

        Args:
            v (object): The input array data to be placed on the diagonal.
            k (int, optional): The diagonal to set. Defaults to 0.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape tuple representing a dynamic 2D size.
        """
        return (None, None)


@register_op("DiagIndices")
class DiagIndices(OpDef):
    """Return the indices to access the main diagonal of an array."""

    op_name = "DiagIndices"
    np_op_name = "diag_indices"

    def infer_shape(self, n: int, ndim: int = 2, **kwargs: object) -> object:
        """Infer the output shape for the main diagonal indices.

        Args:
            n (int): The size, along each dimension, of the arrays for which indices are returned.
            ndim (int, optional): The number of dimensions. Defaults to 2.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape tuple representing a dynamic 1D size.
        """
        return (None,)


@register_op("DiagIndicesFrom")
class DiagIndicesFrom(OpDef):
    """Return the indices to access the main diagonal of an n-dimensional array."""

    op_name = "DiagIndicesFrom"
    np_op_name = "diag_indices_from"

    def infer_shape(self, arr: object, **kwargs: object) -> object:
        """Infer the output shape for the n-dimensional diagonal indices.

        Args:
            arr (object): The array for which the indices will be generated.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape tuple representing a dynamic 1D size.
        """
        return (None,)


@register_op("BooleanMask")
class BooleanMask(OpDef):
    """Operation to mask a tensor with a boolean array."""

    op_name = "BooleanMask"

    def infer_shape(self, tensor: object, mask: object, axis: object = None, **kwargs: object) -> object:
        """Infer the output shape for the boolean masked array.

        Args:
            tensor (object): The N-D tensor to mask.
            mask (object): The K-D boolean mask.
            axis (object, optional): A 0-D int Tensor representing the axis in tensor to mask from. Defaults to None.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape tuple with a dynamic size dimension.
        """
        # Typically dynamic size
        return (None,)


@register_op("InvertPermutation")
class InvertPermutation(OpDef):
    """Operation to compute the inverse permutation of a tensor."""

    op_name = "InvertPermutation"

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer the output shape for the inverted permutation.

        Args:
            x (object): The 1-D tensor indicating the permutation to invert.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred shape tuple matching the input tensor shape.
        """
        return getattr(x, "shape", ())
