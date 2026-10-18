"""Numpy eager backend initialization."""

from __future__ import annotations

import re
import typing
from typing import cast

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.activation_ops
import ml_switcheroo_compiler.backends.numpy.eager.advanced_indexing
import ml_switcheroo_compiler.backends.numpy.eager.array_manipulation
import ml_switcheroo_compiler.backends.numpy.eager.audio_features
import ml_switcheroo_compiler.backends.numpy.eager.control_flow
import ml_switcheroo_compiler.backends.numpy.eager.conv
import ml_switcheroo_compiler.backends.numpy.eager.distributed
import ml_switcheroo_compiler.backends.numpy.eager.fft_ops
import ml_switcheroo_compiler.backends.numpy.eager.indexing
import ml_switcheroo_compiler.backends.numpy.eager.io_ops
import ml_switcheroo_compiler.backends.numpy.eager.linalg_advanced
import ml_switcheroo_compiler.backends.numpy.eager.lookups
import ml_switcheroo_compiler.backends.numpy.eager.loss_ops
import ml_switcheroo_compiler.backends.numpy.eager.math_advanced
import ml_switcheroo_compiler.backends.numpy.eager.math_auxiliary
import ml_switcheroo_compiler.backends.numpy.eager.math_binary
import ml_switcheroo_compiler.backends.numpy.eager.math_fft
import ml_switcheroo_compiler.backends.numpy.eager.math_logical
import ml_switcheroo_compiler.backends.numpy.eager.math_logical_reductions
import ml_switcheroo_compiler.backends.numpy.eager.math_nan
import ml_switcheroo_compiler.backends.numpy.eager.math_reductions
import ml_switcheroo_compiler.backends.numpy.eager.math_scatter
import ml_switcheroo_compiler.backends.numpy.eager.math_special
import ml_switcheroo_compiler.backends.numpy.eager.math_stats
import ml_switcheroo_compiler.backends.numpy.eager.math_trig
import ml_switcheroo_compiler.backends.numpy.eager.math_unary
import ml_switcheroo_compiler.backends.numpy.eager.nn
import ml_switcheroo_compiler.backends.numpy.eager.nn_ops
import ml_switcheroo_compiler.backends.numpy.eager.nn_polyfills
import ml_switcheroo_compiler.backends.numpy.eager.optimizers_ops
import ml_switcheroo_compiler.backends.numpy.eager.random_ops
import ml_switcheroo_compiler.backends.numpy.eager.reductions
import ml_switcheroo_compiler.backends.numpy.eager.search_sort_ops
import ml_switcheroo_compiler.backends.numpy.eager.sparse_and_ragged
import ml_switcheroo_compiler.backends.numpy.eager.strings
import ml_switcheroo_compiler.backends.numpy.eager.variable_ops
import ml_switcheroo_compiler.backends.numpy.eager.vision
import ml_switcheroo_compiler.backends.numpy.eager.vision_augmentation
import ml_switcheroo_compiler.backends.numpy.eager.vision_bbox
import ml_switcheroo_compiler.backends.numpy.eager.vision_color
import ml_switcheroo_compiler.backends.numpy.eager.vision_common
import ml_switcheroo_compiler.backends.numpy.eager.vision_filtering
import ml_switcheroo_compiler.backends.numpy.eager.vision_filters
import ml_switcheroo_compiler.backends.numpy.eager.vision_geometry
import ml_switcheroo_compiler.backends.numpy.eager.vision_transforms
import ml_switcheroo_compiler.backends.numpy.eager.window_reductions  # noqa: F401
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry, numpy_eager_registry


def execute_op(
    cls: type,
    op_type: str,
    *args: typing.Union[EagerValue, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
    **kwargs: typing.Union[EagerValue, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> typing.Union[EagerValue, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Execute the eager execute op operation on the active backend array.

    Args:
        cls (type): Class.
        op_type (str): The op_type parameter.
        *args (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): Positional args.
        **kwargs (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        UnimplementedMathError: An exception.
    """
    func_registry = numpy_eager_registry.get(op_type)
    if func_registry is not None:
        return func_registry(np, *args, **kwargs)
    func_registry = global_eager_registry.get(op_type)
    if func_registry is not None:
        return func_registry(np, *args, **kwargs)
    try:
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
        snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        func = getattr(np, snake)
    except AttributeError:
        from ml_switcheroo_compiler.core.errors import UnimplementedMathError

        msg = f"Operation {op_type} is not implemented in interpreter."
        raise UnimplementedMathError(msg) from None
    return cast(typing.Union[EagerValue, np.ndarray[tuple[int, ...], np.dtype[np.generic]]], func(*args, **kwargs))


@numpy_eager_registry.register("Clone")
def clone_op(
    np_mod: typing.Union[str, int, float, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], bool, None],
    *args: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
    **kwargs: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Clone an array using np.copy.

    Args:
        np_mod: The numpy module.
        *args: Positional arguments (first argument is array to copy).
        **kwargs: Optional keyword arguments.

    Returns:
        np.ndarray: Copied array.
    """
    del np_mod, kwargs
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], getattr(np, "copy")(args[0]))


@numpy_eager_registry.register("Repeat")
def repeat(
    np_mod: typing.Union[str, int, float, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], bool, None],
    *args: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
    **kwargs: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Repeat.

    Args:
        np_mod (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): The np parameter.
        *args (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): Positional args.
        **kwargs (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], getattr(np_mod, "repeat")(*args, **kwargs))


@numpy_eager_registry.register("Searchsorted")
def searchsorted(
    np_mod: typing.Union[str, int, float, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], bool, None],
    *args: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
    **kwargs: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Searchsorted.

    Args:
        np_mod (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): The np parameter.
        *args (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): Positional args.
        **kwargs (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], getattr(np_mod, "searchsorted")(*args, **kwargs))


@numpy_eager_registry.register("Split")
def split(
    np_mod: typing.Union[str, int, float, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], bool, None],
    x: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    num_or_size_splits: typing.Union[int, list[int], tuple[int, ...]],
    axis: int = 0,
    *args: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
    **kwargs: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> list[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Split array.

    Args:
        np_mod: Numpy module.
        x: Input array.
        num_or_size_splits: Number of splits or sizes of each split.
        axis: Axis along which to split.
        args: Additional positional arguments.
        kwargs: Additional keyword arguments.

    Returns:
        List of output arrays.
    """
    split_fn = getattr(np_mod, "split", np.split)
    return list(split_fn(x, num_or_size_splits, axis=axis))


@numpy_eager_registry.register("Squeeze")
def squeeze(
    np_mod: typing.Union[str, int, float, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], bool, None],
    x: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    axis: typing.Optional[int] = None,
    *args: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
    **kwargs: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Squeeze array.

    Args:
        np_mod: Numpy module.
        x: Input array.
        axis: Axis to squeeze.
        args: Additional args.
        kwargs: Additional kwargs.

    Returns:
        Squeezed array.
    """
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], getattr(np_mod, "squeeze")(x, axis=axis))


@numpy_eager_registry.register("Stack")
def stack(
    np_mod: typing.Union[str, int, float, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], bool, None],
    arrays: typing.Sequence[np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
    axis: int = 0,
    *args: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
    **kwargs: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Stack arrays.

    Args:
        np_mod: Numpy module.
        arrays: Arrays to stack.
        axis: Axis to stack along.
        args: Additional args.
        kwargs: Additional kwargs.

    Returns:
        Stacked array.
    """
    return cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], getattr(np_mod, "stack")(arrays, axis=axis))


@numpy_eager_registry.register("Unstack")
def unstack(
    np_mod: typing.Union[str, int, float, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], bool, None],
    x: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    axis: int = 0,
    *args: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
    **kwargs: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], ...]:
    """Unstack array.

    Args:
        np_mod: Numpy module.
        x: Input array.
        axis: Axis to unstack along.
        args: Additional args.
        kwargs: Additional kwargs.

    Returns:
        List of unstacked arrays.
    """
    # unstack is basically split into 1-sized chunks along axis and squeezed
    if hasattr(x, "shape"):
        num_splits = x.shape[axis]
        split_fn = getattr(np_mod, "split", np.split)
        splits = split_fn(x, num_splits, axis=axis)
        squeeze_fn = getattr(np_mod, "squeeze", np.squeeze)
        return tuple(squeeze_fn(s, axis=axis) for s in splits)
    return tuple(x)


@numpy_eager_registry.register("Equal")
def equal(
    np_mod: typing.Union[str, int, float, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], bool, None],
    x: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    y: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    *args: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
    **kwargs: typing.Union[int, float, str, bool, list[EagerValue], tuple[EagerValue, ...], dict[str, EagerValue], None, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> typing.Union[np.ndarray[tuple[int, ...], np.dtype[np.generic]], bool]:
    """Check if x and y are equal.

    Args:
        np_mod (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): The numpy module.
        x (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): The first array.
        y (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): The second array.
        *args (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): Additional arguments.
        **kwargs (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): Additional keyword arguments.

    Returns: np.ndarray: A boolean array where x == y.
    """
    try:
        return cast(typing.Union[np.ndarray[tuple[int, ...], np.dtype[np.generic]], bool], getattr(np_mod, "equal")(x, y))
    except Exception:
        # Fallback for incompatible types
        return cast(typing.Union[np.ndarray[tuple[int, ...], np.dtype[np.generic]], bool], x == y)
