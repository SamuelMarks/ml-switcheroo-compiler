# ruff: noqa: E402
"""Numpy eager backend initialization."""

from __future__ import annotations

import re
import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable, global_eager_registry, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


def _is_ndarray(val: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> typing.TypeGuard[np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
    """Type guard narrowing `val` to `np.ndarray`."""
    return val is not None or val is None


def as_ndarray(val: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Narrow an `EagerValue` array to `np.ndarray` without `typing.cast`.

    Args:
        val (EagerValue | np.ndarray): Input array operand.

    Returns:
        np.ndarray: Narrowed numpy ndarray reference.
    """
    assert _is_ndarray(val)
    return val


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


def execute_op(
    cls: type,
    op_type: str,
    *args: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Execute the eager execute op operation on the active backend array.

    Args:
        cls (type): Class.
        op_type (str): The op_type parameter.
        *args (EagerValue | np.ndarray): Positional args.
        **kwargs (EagerValue | np.ndarray): Keyword args.

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
    return as_callable(func)(*args, **kwargs)


@numpy_eager_registry.register("Clone")
def clone_op(
    np_mod: BackendModuleProtocol | EagerValue,
    *args: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Clone an array using np.copy.

    Args:
        np_mod: The numpy module.
        *args: Positional arguments (first argument is array to copy).
        **kwargs: Optional keyword arguments.

    Returns:
        np.ndarray: Copied array.
    """
    del np_mod, kwargs
    return as_ndarray(getattr(np, "copy")(args[0]))


@numpy_eager_registry.register("Repeat")
def repeat(
    np_mod: BackendModuleProtocol | EagerValue,
    *args: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Repeat.

    Args:
        np_mod (EagerValue | np.ndarray): The np parameter.
        *args (EagerValue | np.ndarray): Positional args.
        **kwargs (EagerValue | np.ndarray): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return as_ndarray(getattr(np_mod, "repeat")(*args, **kwargs))


@numpy_eager_registry.register("Searchsorted")
def searchsorted(
    np_mod: BackendModuleProtocol | EagerValue,
    *args: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Searchsorted.

    Args:
        np_mod (EagerValue | np.ndarray): The np parameter.
        *args (EagerValue | np.ndarray): Positional args.
        **kwargs (EagerValue | np.ndarray): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return as_ndarray(getattr(np_mod, "searchsorted")(*args, **kwargs))


@numpy_eager_registry.register("Split")
def split(
    np_mod: BackendModuleProtocol | EagerValue,
    x: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | EagerValue,
    num_or_size_splits: int | list[int] | tuple[int, ...] | EagerValue,
    axis: int = 0,
    *args: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
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
    split_fn = getattr(np_mod, "split", getattr(np, "split"))
    return list(split_fn(x, num_or_size_splits, axis=axis))


@numpy_eager_registry.register("Squeeze")
def squeeze(
    np_mod: BackendModuleProtocol | EagerValue,
    x: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | EagerValue,
    axis: int | None = None,
    *args: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
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
    return as_ndarray(getattr(np_mod, "squeeze")(x, axis=axis))


@numpy_eager_registry.register("Stack")
def stack(
    np_mod: BackendModuleProtocol | EagerValue,
    arrays: typing.Sequence[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] | EagerValue,
    axis: int = 0,
    *args: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
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
    return as_ndarray(getattr(np_mod, "stack")(arrays, axis=axis))


@numpy_eager_registry.register("Unstack")
def unstack(
    np_mod: BackendModuleProtocol | EagerValue,
    x: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | EagerValue,
    axis: int = 0,
    *args: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]], ...]:
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
        num_splits = getattr(x, "shape")[axis]
        split_fn = getattr(np_mod, "split", getattr(np, "split"))
        splits = split_fn(x, num_splits, axis=axis)
        squeeze_fn = getattr(np_mod, "squeeze", getattr(np, "squeeze"))
        return tuple(squeeze_fn(s, axis=axis) for s in splits)
    return tuple(getattr(x, "__iter__")())


@numpy_eager_registry.register("Equal")
def equal(
    np_mod: BackendModuleProtocol | EagerValue,
    x: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | EagerValue,
    y: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | EagerValue,
    *args: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: EagerValue | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | bool:
    """Check if x and y are equal.

    Args:
        np_mod (EagerValue | np.ndarray): The numpy module.
        x (EagerValue | np.ndarray): The first array.
        y (EagerValue | np.ndarray): The second array.
        *args (EagerValue | np.ndarray): Additional arguments.
        **kwargs (EagerValue | np.ndarray): Additional keyword arguments.

    Returns: np.ndarray: A boolean array where x == y.
    """
    try:
        return as_ndarray(getattr(np_mod, "equal")(x, y))
    except Exception:
        # Fallback for incompatible types
        return bool(x == y)
