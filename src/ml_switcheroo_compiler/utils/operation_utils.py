"""Operation lineage and shape inference utilities."""

from __future__ import annotations

import abc
from collections.abc import Iterable, Sequence
from typing import Protocol

from ml_switcheroo_compiler.core.constants import BINARY_OP_ARITY


class _OperationProtocol(Protocol):
    """Protocol representing an operation with input tensors."""

    @property
    def inputs(self) -> Sequence[_LineageTensorProtocol]:
        """Return operation inputs.

        Returns:
            Sequence[_LineageTensorProtocol]: Input tensor sequence.
        """


class _LineageNodeProtocol(Protocol):
    """Protocol representing a graph lineage node."""

    @property
    def operation(self) -> _OperationProtocol:
        """Return the producing operation.

        Returns:
            _OperationProtocol: Producing operation.
        """


class _LineageHistoryProtocol(Protocol):
    """Protocol representing tensor lineage history."""

    @property
    def node(self) -> _LineageNodeProtocol:
        """Return the lineage node.

        Returns:
            _LineageNodeProtocol: Lineage node.
        """


class _LineageTensorProtocol(Protocol):
    """Protocol representing a tensor with optional lineage metadata."""


def get_source_inputs(tensor: _LineageTensorProtocol) -> list[_LineageTensorProtocol]:
    """Return the list of root input tensors that a tensor depends on via graph lineage.

    Args:
        tensor (_LineageTensorProtocol): Target tensor to inspect.

    Returns:
        list[_LineageTensorProtocol]: Root source input tensors.
    """
    lineage: _LineageHistoryProtocol | None = getattr(tensor, "_lineage_history", None)
    if lineage is not None:
        node = lineage.node
        if not node.operation.inputs:
            return [tensor]
        res: list[_LineageTensorProtocol] = []
        for inp in node.operation.inputs:
            res.extend(get_source_inputs(inp))
        return res
    return [tensor]


class ShapeInferenceStrategy(abc.ABC):
    """Base class for operation shape inference strategies."""

    @abc.abstractmethod
    def __call__(
        self,
        shape: tuple[int | None, ...],
        args: tuple[int | str | Sequence[int] | None, ...],
        kwargs: dict[str, int | str | bool | Sequence[int] | None],
    ) -> tuple[int | None, ...] | list[tuple[int | None, ...]] | None:
        """Evaluate shape inference for an operation.

        Args:
            shape (tuple[int | None, ...]): Input tensor shape.
            args (tuple[int | str | Sequence[int] | None, ...]): Positional arguments.
            kwargs (dict[str, int | str | bool | Sequence[int] | None]): Keyword arguments.

        Returns:
            tuple[int | None, ...] | list[tuple[int | None, ...]] | None: Inferred shape or list of shapes.
        """
        _ = (shape, args, kwargs)
        return None


class ReshapeInference(ShapeInferenceStrategy):
    """Shape inference strategy for reshape."""

    def __call__(
        self,
        shape: tuple[int | None, ...],
        args: tuple[int | str | Sequence[int] | None, ...],
        kwargs: dict[str, int | str | bool | Sequence[int] | None],
    ) -> tuple[int | None, ...]:
        """Infer output shape for reshape operation.

        Args:
            shape (tuple[int | None, ...]): Input tensor shape.
            args (tuple[int | str | Sequence[int] | None, ...]): Positional arguments containing target shape at index 1.
            kwargs (dict[str, int | str | bool | Sequence[int] | None]): Keyword arguments.

        Returns:
            tuple[int | None, ...]: Reshaped dimensions tuple.
        """
        target = args[1]
        return tuple(target) if isinstance(target, Iterable) and not isinstance(target, str) else ()


class TransposeInference(ShapeInferenceStrategy):
    """Shape inference strategy for transpose."""

    def __call__(
        self,
        shape: tuple[int | None, ...],
        args: tuple[int | str | Sequence[int] | None, ...],
        kwargs: dict[str, int | str | bool | Sequence[int] | None],
    ) -> tuple[int | None, ...]:
        """Infer output shape for transpose operation.

        Args:
            shape (tuple[int | None, ...]): Input tensor shape.
            args (tuple[int | str | Sequence[int] | None, ...]): Positional arguments.
            kwargs (dict[str, int | str | bool | Sequence[int] | None]): Keyword arguments containing optional 'axes'.

        Returns:
            tuple[int | None, ...]: Transposed shape tuple.
        """
        axes = kwargs.get("axes", args[1] if len(args) > 1 else None)
        if isinstance(axes, Iterable) and not isinstance(axes, str):
            return tuple(shape[i] for i in axes)
        return tuple(reversed(shape))


class ExpandDimsInference(ShapeInferenceStrategy):
    """Shape inference strategy for expand_dims."""

    def __call__(
        self,
        shape: tuple[int | None, ...],
        args: tuple[int | str | Sequence[int] | None, ...],
        kwargs: dict[str, int | str | bool | Sequence[int] | None],
    ) -> tuple[int | None, ...]:
        """Infer output shape for expand_dims operation.

        Args:
            shape (tuple[int | None, ...]): Input tensor shape.
            args (tuple[int | str | Sequence[int] | None, ...]): Positional arguments.
            kwargs (dict[str, int | str | bool | Sequence[int] | None]): Keyword arguments containing optional 'axis'.

        Returns:
            tuple[int | None, ...]: Expanded shape tuple.
        """
        raw_axis = kwargs.get("axis", args[1] if len(args) > 1 else -1)
        axis: int = int(raw_axis) if isinstance(raw_axis, int) else -1
        if axis < 0:
            axis += len(shape) + 1
        return tuple(shape[:axis]) + (1,) + tuple(shape[axis:])


def _normalize_axes(axis: int | Sequence[int], ndim: int) -> list[int]:
    """Normalize axis or axes into non-negative indices.

    Args:
        axis (int | Sequence[int]): Single axis index or sequence of axis indices.
        ndim (int): Rank of the tensor.

    Returns:
        list[int]: Normalized list of non-negative axis indices.
    """
    axis_seq: Sequence[int] = [axis] if isinstance(axis, int) else axis
    return [a + ndim if a < 0 else a for a in axis_seq]


def _validate_squeeze_dims(shape: tuple[int | None, ...], axes: Sequence[int]) -> None:
    """Validate that squeezed dimensions are in bounds and have size 1.

    Args:
        shape (tuple[int | None, ...]): Input tensor shape.
        axes (Sequence[int]): Normalized axes to squeeze.

    Raises:
        ValueError: If any axis is out of bounds or its dimension size is not 1.
    """
    for a in axes:
        if a < 0 or a >= len(shape):
            raise ValueError(f"Squeeze axis {a} is out of bounds for shape {shape}")
        if shape[a] != 1:
            raise ValueError(f"Cannot squeeze dimension {a} of size {shape[a]}")


def _squeeze_all_ones(shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
    """Squeeze all dimensions of size 1.

    Args:
        shape (tuple[int | None, ...]): Input tensor shape.

    Returns:
        tuple[int | None, ...]: Shape with all unit dimensions removed.
    """
    return tuple(s for s in shape if s != 1)


def _squeeze_specific_axes(shape: tuple[int | None, ...], axes: int | Sequence[int]) -> tuple[int | None, ...]:
    """Squeeze only the specified axes.

    Args:
        shape (tuple[int | None, ...]): Input tensor shape.
        axes (int | Sequence[int]): Axis or sequence of axes to squeeze.

    Returns:
        tuple[int | None, ...]: Shape with specified unit axes removed.
    """
    normalized = _normalize_axes(axes, len(shape))
    _validate_squeeze_dims(shape, normalized)
    return tuple(s for i, s in enumerate(shape) if i not in normalized)


class SqueezeInference(ShapeInferenceStrategy):
    """Shape inference strategy for squeeze."""

    def __call__(
        self,
        shape: tuple[int | None, ...],
        args: tuple[int | str | Sequence[int] | None, ...],
        kwargs: dict[str, int | str | bool | Sequence[int] | None],
    ) -> tuple[int | None, ...]:
        """Infer output shape for squeeze operation.

        Args:
            shape (tuple[int | None, ...]): Input tensor shape.
            args (tuple[int | str | Sequence[int] | None, ...]): Positional arguments.
            kwargs (dict[str, int | str | bool | Sequence[int] | None]): Keyword arguments containing optional 'axis'.

        Returns:
            tuple[int | None, ...]: Squeezed shape tuple.
        """
        axis = kwargs.get("axis", args[1] if len(args) > 1 else None)
        if axis is None or isinstance(axis, (str, bool)):
            return _squeeze_all_ones(shape)
        return _squeeze_specific_axes(shape, axis)


class SplitInference(ShapeInferenceStrategy):
    """Shape inference strategy for split."""

    def __call__(
        self,
        shape: tuple[int | None, ...],
        args: tuple[int | str | Sequence[int] | None, ...],
        kwargs: dict[str, int | str | bool | Sequence[int] | None],
    ) -> tuple[int | None, ...] | list[tuple[int | None, ...]]:
        """Infer output shapes for split operation.

        Args:
            shape (tuple[int | None, ...]): Input tensor shape.
            args (tuple[int | str | Sequence[int] | None, ...]): Positional arguments.
            kwargs (dict[str, int | str | bool | Sequence[int] | None]): Keyword arguments containing optional 'axis'.

        Returns:
            tuple[int | None, ...] | list[tuple[int | None, ...]]: List of split shapes or original shape fallback.
        """
        num_or_size_splits = args[1]
        raw_axis = kwargs.get("axis", args[2] if len(args) > BINARY_OP_ARITY else 0)
        axis: int = int(raw_axis) if isinstance(raw_axis, int) else 0
        if isinstance(num_or_size_splits, int) and not isinstance(num_or_size_splits, bool):
            sub_shape: list[int | None] = list(shape)
            dim_val = sub_shape[axis]
            sub_shape[axis] = dim_val // num_or_size_splits if dim_val is not None else None
            return [tuple(sub_shape) for _ in range(num_or_size_splits)]
        return shape


class MeanInference(ShapeInferenceStrategy):
    """Shape inference strategy for mean reduction."""

    def _resolve_axis(self, axis: int | Sequence[int] | None, shape_len: int) -> set[int]:
        """Resolve the axis argument into a normalized set of axes.

        Args:
            axis (int | Sequence[int] | None): Axis or sequence of axes.
            shape_len (int): Number of dimensions in the input shape.

        Returns:
            set[int]: Normalized non-negative axis set.
        """
        if axis is None:
            return set()
        axis_list: Sequence[int] = [axis] if isinstance(axis, int) else axis
        return {ax if ax >= 0 else ax + shape_len for ax in axis_list}

    def _validate_datatype_promotion(
        self,
        kwargs: dict[str, int | str | bool | Sequence[int] | None],
    ) -> None:
        """Validate datatype promotion based on kwargs.

        Args:
            kwargs (dict[str, int | str | bool | Sequence[int] | None]): Keyword arguments.

        Returns:
            None
        """
        dtype = kwargs.get("dtype")
        if dtype is not None and not isinstance(dtype, str):
            return None

    def _compute_keepdims_shape(
        self,
        shape: tuple[int | None, ...],
        normalized_axis: set[int],
    ) -> tuple[int | None, ...]:
        """Compute reduced shape when keepdims is True.

        Args:
            shape (tuple[int | None, ...]): Input tensor shape.
            normalized_axis (set[int]): Set of reduced axes.

        Returns:
            tuple[int | None, ...]: Shape with reduced axes replaced by 1.
        """
        return tuple(1 if i in normalized_axis else s for i, s in enumerate(shape))

    def _compute_reduced_shape(
        self,
        shape: tuple[int | None, ...],
        normalized_axis: set[int],
    ) -> tuple[int | None, ...]:
        """Compute reduced shape when keepdims is False.

        Args:
            shape (tuple[int | None, ...]): Input tensor shape.
            normalized_axis (set[int]): Set of reduced axes.

        Returns:
            tuple[int | None, ...]: Shape with reduced axes removed.
        """
        return tuple(s for i, s in enumerate(shape) if i not in normalized_axis)

    def _extract_axis(
        self,
        args: tuple[int | str | Sequence[int] | None, ...],
        kwargs: dict[str, int | str | bool | Sequence[int] | None],
    ) -> int | Sequence[int] | None:
        """Extract axis argument from args or kwargs.

        Args:
            args (tuple[int | str | Sequence[int] | None, ...]): Positional arguments.
            kwargs (dict[str, int | str | bool | Sequence[int] | None]): Keyword arguments.

        Returns:
            int | Sequence[int] | None: Extracted axis specification.
        """
        raw = kwargs.get("axis", args[1] if len(args) > 1 else None)
        return raw if isinstance(raw, (int, Sequence)) and not isinstance(raw, (str, bool)) else None

    def __call__(
        self,
        shape: tuple[int | None, ...],
        args: tuple[int | str | Sequence[int] | None, ...],
        kwargs: dict[str, int | str | bool | Sequence[int] | None],
    ) -> tuple[int | None, ...]:
        """Infer output shape for mean reduction.

        Args:
            shape (tuple[int | None, ...]): Input tensor shape.
            args (tuple[int | str | Sequence[int] | None, ...]): Positional arguments.
            kwargs (dict[str, int | str | bool | Sequence[int] | None]): Keyword arguments.

        Returns:
            tuple[int | None, ...]: Reduced output shape.
        """
        self._validate_datatype_promotion(kwargs)
        axis = self._extract_axis(args, kwargs)
        if axis is None:
            return shape
        keepdims = bool(kwargs.get("keepdims", False))
        normalized_axis = self._resolve_axis(axis, len(shape))
        if keepdims:
            return self._compute_keepdims_shape(shape, normalized_axis)
        return self._compute_reduced_shape(shape, normalized_axis)


SHAPE_INFERENCE_REGISTRY: dict[str, ShapeInferenceStrategy] = {
    "reshape": ReshapeInference(),
    "transpose": TransposeInference(),
    "expand_dims": ExpandDimsInference(),
    "squeeze": SqueezeInference(),
    "split": SplitInference(),
    "mean": MeanInference(),
}


def compute_shape_propagation(
    name: str,
    shape: tuple[int | None, ...],
    args: tuple[int | str | Sequence[int] | None, ...],
    kwargs: dict[str, int | str | bool | Sequence[int] | None],
) -> tuple[int | None, ...] | list[tuple[int | None, ...]] | None:
    """Compute propagated output shape for a named operation.

    Args:
        name (str): Operation name.
        shape (tuple[int | None, ...]): Input tensor shape.
        args (tuple[int | str | Sequence[int] | None, ...]): Positional arguments.
        kwargs (dict[str, int | str | bool | Sequence[int] | None]): Keyword arguments.

    Returns:
        tuple[int | None, ...] | list[tuple[int | None, ...]] | None: Propagated output shape.
    """
    strategy = SHAPE_INFERENCE_REGISTRY.get(name)
    if strategy:
        return strategy(shape, args, kwargs)
    return shape
