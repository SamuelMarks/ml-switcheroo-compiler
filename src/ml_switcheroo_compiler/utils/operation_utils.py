"""Operation utilities."""

import abc
from typing import Any, Optional, Union

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2


def get_source_inputs(tensor: object) -> list[object]:
    """Return the list of input tensors that a tensor depends on.

    Args:
        tensor (object): The tensor parameter.
    """
    # This is a dummy implementation if no node history.
    if hasattr(tensor, "_keras_history"):
        node = tensor._keras_history.node
        if not node.operation.inputs:
            return [tensor]
        res = []
        for inp in node.operation.inputs:
            res.extend(get_source_inputs(inp))
        return res
    return [tensor]


class ShapeInferenceStrategy(abc.ABC):
    """Define base class for shape inference strategies."""

    @abc.abstractmethod
    def __call__(self, shape: tuple[int, ...], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Union[tuple[int, ...], list[tuple[int, ...]]]:
        """Evaluate __call__ operation.

        Args:
            shape (tuple): The shape parameter.
            args (tuple): The args parameter.
            kwargs (dict): The kwargs parameter.
            Union: Result.
        """
        _ = None


class ReshapeInference(ShapeInferenceStrategy):
    """Shape inference for reshape."""

    def __call__(self, shape: tuple[int, ...], args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int, ...]:
        """Evaluate __call__ operation.

        Args:
            shape (tuple): The shape parameter.
            args (tuple): The args parameter.
            kwargs (dict): The kwargs parameter.

        Returns:
            tuple: Result.
        """
        return tuple(args[1])


class TransposeInference(ShapeInferenceStrategy):
    """Shape inference for transpose."""

    def __call__(self, shape: tuple[int, ...], args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int, ...]:
        """Evaluate __call__ operation.

        Args:
            shape (tuple): The shape parameter.
            args (tuple): The args parameter.
            kwargs (dict): The kwargs parameter.

        Returns:
            tuple: Result.
        """
        axes = kwargs.get("axes", args[1] if len(args) > 1 else None)
        if axes is not None:
            return tuple(shape[i] for i in axes)
        return tuple(reversed(shape))


class ExpandDimsInference(ShapeInferenceStrategy):
    """Shape inference for expand_dims."""

    def __call__(self, shape: tuple[int, ...], args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int, ...]:
        """Evaluate __call__ operation.

        Args:
            shape (tuple): The shape parameter.
            args (tuple): The args parameter.
            kwargs (dict): The kwargs parameter.

        Returns:
            tuple: Result.
        """
        axis = kwargs.get("axis", args[1] if len(args) > 1 else -1)
        if axis < 0:
            axis += len(shape) + 1
        return tuple(shape[:axis]) + (1,) + tuple(shape[axis:])


def _normalize_axes(axis: object, ndim: int) -> list[int]:
    """Normalize axes, handling integers, negatives, and iterables.

    Args:
        axis (object): The axis parameter.
        ndim (int): The ndim parameter.
    """
    if isinstance(axis, int):
        axis = [axis]
    return [a + ndim if a < 0 else a for a in axis]


def _validate_squeeze_dims(shape: tuple[int, ...], axes: list[int]) -> None:
    """Validate that squeezed dimensions have size 1.

    Args:
        shape (tuple): The shape parameter.
        axes (list): The axes parameter.

    Raises:
        ValueError: An exception.
    """
    for a in axes:
        if a < 0 or a >= len(shape):
            raise ValueError(f"Squeeze axis {a} is out of bounds for shape {shape}")
        if shape[a] != 1:
            raise ValueError(f"Cannot squeeze dimension {a} of size {shape[a]}")


def _squeeze_all_ones(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Squeezes all dimensions of size 1.

    Args:
        shape (object): The shape parameter.
    """
    return tuple(filter(lambda s: s != 1, shape))


def _squeeze_specific_axes(shape: tuple[int, ...], axes: list[int]) -> tuple[int, ...]:
    """Squeezes only the specified axes.

    Args:
        shape (object): The shape parameter.
        axes (object): The axes parameter.
    """
    normalized = _normalize_axes(axes, len(shape))
    _validate_squeeze_dims(shape, normalized)
    # Using enumerate to keep track of index, filter out specified axes
    return tuple(s for i, s in enumerate(shape) if i not in normalized)


class SqueezeInference(ShapeInferenceStrategy):
    """Shape inference for squeeze."""

    def __call__(self, shape: tuple[int, ...], args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int, ...]:
        """Evaluate __call__ operation.

        Args:
            shape (tuple): The shape parameter.
            args (tuple): The args parameter.
            kwargs (dict): The kwargs parameter.

        Returns:
            tuple: Result.
        """
        axis = kwargs.get("axis", args[1] if len(args) > 1 else None)
        if axis is None:
            return _squeeze_all_ones(shape)
        return _squeeze_specific_axes(shape, axis)


class SplitInference(ShapeInferenceStrategy):
    """Shape inference for split."""

    def __call__(self, shape: tuple[int, ...], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Union[tuple[int, ...], list[tuple[int, ...]]]:
        """Evaluate __call__ operation.

        Args:
            shape (tuple): The shape parameter.
            args (tuple): The args parameter.
            kwargs (dict): The kwargs parameter.

        Returns:
            Union: Result.
        """
        num_or_size_splits = args[1]
        axis = kwargs.get("axis", args[2] if len(args) > MAGIC_VAL_2 else 0)
        if isinstance(num_or_size_splits, int):
            sub_shape = list(shape)
            sub_shape[axis] = sub_shape[axis] // num_or_size_splits if sub_shape[axis] is not None else None
            return [tuple(sub_shape) for _ in range(num_or_size_splits)]
        return shape  # (fallback if not int, based on original missing else)


class MeanInference(ShapeInferenceStrategy):
    """Shape inference for mean."""

    def _resolve_axis(self, axis: object, shape_len: int) -> set[int]:
        """Resolve the axis argument into a normalized set of axes.

        Args:
        axis (object): The axis parameter.
        shape_len (int): The shape_len parameter.
        """
        if axis is None:
            return set()
        axis_list = [axis] if isinstance(axis, int) else axis
        return {ax if ax >= 0 else ax + shape_len for ax in axis_list}

    def _validate_datatype_promotion(self, kwargs: dict[str, Any]) -> None:
        """Validate datatype promotion based on kwargs.

        Args:
        kwargs (object): The kwargs parameter.

        Returns:
        NoneType: Result.
        """
        dtype = kwargs.get("dtype")
        if dtype is not None and not isinstance(dtype, str):
            return None

    def _compute_keepdims_shape(self, shape: tuple[int, ...], normalized_axis: set[int]) -> tuple[int, ...]:
        """Evaluate _compute_keepdims_shape operation.

        Args:
        shape (object): The shape parameter.
        normalized_axis (object): The normalized_axis parameter.
        """
        return tuple(1 if i in normalized_axis else s for i, s in enumerate(shape))

    def _compute_reduced_shape(self, shape: tuple[int, ...], normalized_axis: set[int]) -> tuple[int, ...]:
        """Evaluate _compute_reduced_shape operation.

        Args:
        shape (object): The shape parameter.
        normalized_axis (object): The normalized_axis parameter.
        """
        return tuple(s for i, s in enumerate(shape) if i not in normalized_axis)

    def _extract_axis(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[object]:
        """Evaluate _extract_axis operation.

        Args:
        args (object): The args parameter.
        kwargs (object): The kwargs parameter.
        """
        if "axis" in kwargs:
            return kwargs["axis"]
        if len(args) > 1:
            return args[1]
        return None

    def __call__(self, shape: tuple[int, ...], args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int, ...]:
        """Evaluate __call__ operation.

        Args:
            shape (tuple): The shape parameter.
            args (tuple): The args parameter.
            kwargs (dict): The kwargs parameter.

        Returns:
            tuple: Result.
        """
        self._validate_datatype_promotion(kwargs)
        axis = self._extract_axis(args, kwargs)
        if axis is None:
            return shape  # (fallback if axis is None)
        keepdims = kwargs.get("keepdims", False)
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


def compute_shape_propagation(name: str, shape: tuple[int, ...], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Union[tuple[int, ...], list[tuple[int, ...]]]:
    """Evaluate compute_shape_propagation operation.

    Args:
        name (str): The name parameter.
        shape (object): The shape parameter.
        args (object): The args parameter.
        kwargs (object): The kwargs parameter.
    """
    strategy = SHAPE_INFERENCE_REGISTRY.get(name)
    if strategy:
        return strategy(shape, args, kwargs)
    return shape
