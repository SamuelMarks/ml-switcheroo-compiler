"""Shape operations for advanced indexing and sorting Tensor objects."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import (
    coerce_int,
)
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    extract_shape,
)

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@typing.runtime_checkable
class _ArrayCallableProtocol(typing.Protocol):
    """Protocol for objects exposing `__array__()`."""

    def __array__(self) -> OpValue:
        """Convert to array.

        Returns:
            OpValue: Array representation.
        """
        ...


@typing.runtime_checkable
class _ItemCallableProtocol(typing.Protocol):
    """Protocol for objects exposing `.item()`."""

    def item(self) -> OpValue:
        """Extract scalar item.

        Returns:
            OpValue: Scalar item.
        """
        ...


def _normalize_k(k: OpValue) -> OpValue:
    """Compute the normalize k tensor operation.

    Args:
        k (OpValue): Input `k` argument (k).

    Returns:
        OpValue: Normalized `k` value.
    """
    raw_k: OpValue | _ArrayCallableProtocol | _ItemCallableProtocol = k
    if not isinstance(raw_k, tuple) and isinstance(raw_k, _ArrayCallableProtocol):
        arr_fn = raw_k.__array__
        raw_k = arr_fn()
    if isinstance(raw_k, _ItemCallableProtocol):
        return int(coerce_int(raw_k.item()))
    try:
        return int(coerce_int(raw_k))
    except (ValueError, TypeError):
        return k


@register_op("TopK")
class TopK(OpDef):
    """Operation for finding the top K elements along the last axis."""

    op_name = "TopK"

    def infer_shape(self, x: OpValue, k: OpValue = None, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the operation.

        Args:
            x (OpValue): Input `x` argument (x).
            k (OpValue): Input `k` argument (k).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if k is None:
            k = kwargs.get("k", 1)
        k = _normalize_k(k)

        x_shape = extract_shape(x)
        if not x_shape:
            return (None,)
        out_shape = list(x_shape)
        out_shape[-1] = coerce_int(k)
        return tuple(out_shape)


@register_op("Argsort")
class Argsort(OpDef):
    """Operation to return the indices that would sort an array."""

    op_name = "Argsort"

    def infer_shape(
        self,
        x: OpValue,
        dimension: OpValue = -1,
        is_stable: OpValue = True,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer the output shape for sorting indices.

        Args:
            x (OpValue): Input `x` argument (x).
            dimension (OpValue): Input `dimension` argument (dimension).
            is_stable (OpValue): Input `is_stable` argument (is stable).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return extract_shape(x)


@register_op("Sort")
class Sort(OpDef):
    """Operation to sort a tensor along a specified dimension."""

    op_name = "Sort"

    def infer_shape(
        self,
        x: OpValue,
        dimension: OpValue = -1,
        is_stable: OpValue = True,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer the output shape for the sorting operation.

        Args:
            x (OpValue): Input `x` argument (x).
            dimension (OpValue): Input `dimension` argument (dimension).
            is_stable (OpValue): Input `is_stable` argument (is stable).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        _typed_ret_1: OpValue = extract_shape(x)
        return _typed_ret_1


@register_op("Where")
class Where(OpDef):
    """Operator to return elements chosen from x or y depending on a condition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for conditionally chosen elements.

        Args:
            *args (OpValue): Positional arguments, typically condition, x, and y.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple.
        """
        if len(args) > 1:
            return args[1]
        return (None,)


@register_op("Gather")
class Gather(OpDef):
    """Operator to gather slices from a tensor according to indices."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for gathered slices.

        Args:
            *args (OpValue): Positional arguments, typically params and indices.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("Take")
class Take(OpDef):
    """Operator to take elements from an array along an axis."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for taken elements.

        Args:
            *args (OpValue): Positional arguments, typically the input tensor and indices.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("TakeAlongAxis")
class TakeAlongAxis(OpDef):
    """Operator to take values from the input array by matching 1d index and data slices."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for elements taken along an axis.

        Args:
            *args (OpValue): Positional arguments, typically the input array, indices, and axis.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("GatherNd")
class GatherNd(OpDef):
    """Operator to gather slices from a tensor into a tensor with specified shape."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for n-dimensional gathered slices.

        Args:
            *args (OpValue): Positional arguments, typically params and n-dimensional indices.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("Scatter")
class Scatter(OpDef):
    """Operator to scatter updates into a new tensor according to indices."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for scattered updates.

        Args:
            *args (OpValue): Positional arguments, typically input tensor, indices, and updates.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("ScatterNd")
class ScatterNd(OpDef):
    """Operator to apply sparse updates to individual values or slices within a tensor."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for n-dimensional scattered updates.

        Args:
            *args (OpValue): Positional arguments, typically tensor, indices, and updates.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("ScatterAdd")
class ScatterAdd(OpDef):
    """Operator to add sparse updates to a tensor variable reference."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape after adding scattered updates.

        Args:
            *args (OpValue): Positional arguments, typically tensor, indices, and updates.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("Vdot")
class Vdot(OpDef):
    """Operator to compute the dot product of two vectors."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the vector dot product.

        Args:
            *args (OpValue): Positional arguments, typically the two input vectors.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("SearchSorted")
class SearchSorted(OpDef):
    """Operator to find indices where elements should be inserted to maintain order."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for searched sorted indices.

        Args:
            *args (OpValue): Positional arguments, typically sorted sequence and values to insert.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("Select")
class Select(OpDef):
    """Operator to select elements from two arrays based on a boolean mask."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the selected elements.

        Args:
            *args (OpValue): Positional arguments, typically condition, x, and y.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("Assign")
class Assign(OpDef):
    """Operator to assign a value to a tensor variable."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape after assignment.

        Args:
            *args (OpValue): Positional arguments, typically the variable and the value.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("AssignAdd")
class AssignAdd(OpDef):
    """Operator to add a value to a tensor variable and assign the result."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape after addition assignment.

        Args:
            *args (OpValue): Positional arguments, typically the variable and the value to add.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("AssignSub")
class AssignSub(OpDef):
    """Operator to subtract a value from a tensor variable and assign the result."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape after subtraction assignment.

        Args:
            *args (OpValue): Positional arguments, typically the variable and the value to subtract.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape tuple, currently returning an unknown dimension `(None,)`.
        """
        return (None,)


@register_op("TensorScatterUpdate")
class TensorScatterUpdate(OpDef):
    """Operator to scatter updates into a new tensor using multi-dimensional indices."""

    def infer_shape(self, tensor: OpValue, indices: OpValue, updates: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for scattered multidimensional updates.

        Args:
            tensor (OpValue): The initial tensor to be updated.
            indices (OpValue): The multidimensional indices for the updates.
            updates (OpValue): The values to be updated in the tensor.
            **kwargs (OpValue): Additional keyword arguments.

        Returns: Tensor: The inferred shape tuple matching the input tensor shape.
        """
        _typed_ret_2: OpValue = extract_shape(tensor)
        return _typed_ret_2


@register_op("Argpartition")
class Argpartition(OpDef):
    """Perform an indirect partition along the given axis using the algorithm specified by the kind keyword."""

    op_name = "Argpartition"
    np_op_name = "argpartition"

    def infer_shape(self, a: OpValue, kth: OpValue, axis: int = -1, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the partitioned indices.

        Args:
            a (OpValue): Input `a` argument (a).
            kth (OpValue): Input `kth` argument (kth).
            axis (int): Input `axis` argument (axis).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return extract_shape(a)


@register_op("Partition")
class Partition(OpDef):
    """Return a partitioned copy of an array."""

    op_name = "Partition"
    np_op_name = "partition"

    def infer_shape(self, a: OpValue, kth: OpValue, axis: int = -1, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the partitioned array.

        Args:
            a (OpValue): Input `a` argument (a).
            kth (OpValue): Input `kth` argument (kth).
            axis (int): Input `axis` argument (axis).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return extract_shape(a)


@register_op("Compress")
class Compress(OpDef):
    """Return selected slices of an array along given axis."""

    op_name = "Compress"
    np_op_name = "compress"

    def infer_shape(
        self,
        condition: OpValue,
        a: OpValue,
        axis: OpValue = None,
        out: OpValue = None,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer the output shape for the compressed array.

        Args:
            condition (OpValue): Input `condition` argument (condition).
            a (OpValue): Input `a` argument (a).
            axis (OpValue): Input `axis` argument (axis).
            out (OpValue): Input `out` argument (out).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return (None,)


@register_op("Diagonal")
class Diagonal(OpDef):
    """Return specified diagonals of a 2-D array."""

    op_name = "Diagonal"
    np_op_name = "diagonal"

    def infer_shape(self, a: OpValue, offset: int = 0, axis1: int = 0, axis2: int = 1, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the diagonal extraction.

        Args:
            a (OpValue): Input `a` argument (a).
            offset (int): Input `offset` argument (offset).
            axis1 (int): Input `axis1` argument (axis1).
            axis2 (int): Input `axis2` argument (axis2).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return (None,)


@register_op("Diagflat")
class Diagflat(OpDef):
    """Create a two-dimensional array with the flattened input as a diagonal."""

    op_name = "Diagflat"
    np_op_name = "diagflat"

    def infer_shape(self, v: OpValue, k: int = 0, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the flattened diagonal matrix.

        Args:
            v (OpValue): Input `v` argument (v).
            k (int): Input `k` argument (k).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return (None, None)


@register_op("DiagIndices")
class DiagIndices(OpDef):
    """Return the indices to access the main diagonal of an array."""

    op_name = "DiagIndices"
    np_op_name = "diag_indices"

    def infer_shape(self, n: int, naxis: int = 2, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the main diagonal indices.

        Args:
            n (int): Input `n` argument (n).
            naxis (int): Input `naxis` argument (naxis).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return (None,)


@register_op("DiagIndicesFrom")
class DiagIndicesFrom(OpDef):
    """Return the indices to access the main diagonal of an n-dimensional array."""

    op_name = "DiagIndicesFrom"
    np_op_name = "diag_indices_from"

    def infer_shape(self, arr: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the n-dimensional diagonal indices.

        Args:
            arr (OpValue): The array for which the indices will be generated.
            **kwargs (OpValue): Additional keyword arguments.

        Returns: Tensor: The inferred shape tuple representing a dynamic 1D size.
        """
        return (None,)


@register_op("BooleanMask")
class BooleanMask(OpDef):
    """Operation to mask a tensor with a boolean array."""

    op_name = "BooleanMask"

    def infer_shape(self, tensor: OpValue, mask: OpValue, axis: OpValue = None, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the boolean masked array.

        Args:
            tensor (OpValue): Input `tensor` argument (tensor).
            mask (OpValue): Input `mask` argument (mask).
            axis (OpValue): Input `axis` argument (axis).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        # Typically dynamic size
        return (None,)


@register_op("InvertPermutation")
class InvertPermutation(OpDef):
    """Operation to compute the inverse permutation of a tensor."""

    op_name = "InvertPermutation"

    def infer_shape(self, x: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the inverted permutation.

        Args:
            x (OpValue): The 1-D tensor indicating the permutation to invert.
            **kwargs (OpValue): Additional keyword arguments.

        Returns: Tensor: The inferred shape tuple matching the input tensor shape.
        """
        _typed_ret_3: OpValue = extract_shape(x)
        return _typed_ret_3
