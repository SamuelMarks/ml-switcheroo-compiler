# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy eager backend initialization."""

import re
import typing

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.activation_ops  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.advanced_indexing  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.array_manipulation  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.audio_features  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.control_flow  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.conv  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.distributed  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.fft_ops  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.indexing  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.io_ops  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.linalg_advanced  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.lookups  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.loss_ops  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_advanced  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_auxiliary  # noqa: F401  # noqa: F401  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_binary  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_fft  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_logical  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_logical_reductions  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_nan  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_reductions  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_scatter  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_special  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_stats  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_trig  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_unary  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.nn  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.nn_ops  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.nn_polyfills  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.optimizers_ops  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.random_ops  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.reductions  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.search_sort_ops  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.sparse_and_ragged  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.strings  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.variable_ops  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.vision  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.vision_augmentation  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.vision_bbox  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.vision_color  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.vision_common  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.vision_filtering  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.vision_filters  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.vision_geometry  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.vision_transforms  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.window_reductions  # noqa: F401
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry


def execute_op(cls: type, op_type: str, *args: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray], **kwargs: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray]) -> typing.Union[tuple[int, ...], np.ndarray, list, tuple, int, float, str, bool, None]:
    """Evaluate execute_op operation.

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
    return func(*args, **kwargs)


@numpy_eager_registry.register("Repeat")
def repeat(np_mod: typing.Union[str, int, float, list, tuple, dict, bool, None], *args: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray], **kwargs: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray]) -> np.ndarray:
    """Repeat.

    Args:
        np_mod (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): The np parameter.
        *args (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): Positional args.
        **kwargs (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return np_mod.repeat(*args, **kwargs)


@numpy_eager_registry.register("Searchsorted")
def searchsorted(np_mod: typing.Union[str, int, float, list, tuple, dict, bool, None], *args: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray], **kwargs: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray]) -> np.ndarray:
    """Searchsorted.

    Args:
        np_mod (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): The np parameter.
        *args (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): Positional args.
        **kwargs (typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray, typing.Union[str, int, float, list, tuple, dict, bool, None]]): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return np_mod.searchsorted(*args, **kwargs)


@numpy_eager_registry.register("Split")
def split(
    np_mod: typing.Union[str, int, float, list, tuple, dict, bool, None],
    x: np.ndarray,
    num_or_size_splits: typing.Union[int, list[int], tuple[int, ...]],
    axis: int = 0,
    *args: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray],
    **kwargs: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray],
) -> list[np.ndarray]:
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
    return np_mod.split(x, num_or_size_splits, axis=axis)


@numpy_eager_registry.register("Squeeze")
def squeeze(
    np_mod: typing.Union[str, int, float, list, tuple, dict, bool, None], x: np.ndarray, axis: typing.Optional[int] = None, *args: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray], **kwargs: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray]
) -> np.ndarray:
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
    return np_mod.squeeze(x, axis=axis)


@numpy_eager_registry.register("Stack")
def stack(
    np_mod: typing.Union[str, int, float, list, tuple, dict, bool, None], arrays: typing.Sequence[np.ndarray], axis: int = 0, *args: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray], **kwargs: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray]
) -> np.ndarray:
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
    return np_mod.stack(arrays, axis=axis)


@numpy_eager_registry.register("Unstack")
def unstack(
    np_mod: typing.Union[str, int, float, list, tuple, dict, bool, None], x: np.ndarray, axis: int = 0, *args: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray], **kwargs: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray]
) -> tuple[np.ndarray, ...]:
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
        splits = np_mod.split(x, num_splits, axis=axis)
        return tuple(np_mod.squeeze(s, axis=axis) for s in splits)
    return tuple(x)


@numpy_eager_registry.register("Equal")
def equal(
    np_mod: typing.Union[str, int, float, list, tuple, dict, bool, None], x: np.ndarray, y: np.ndarray, *args: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray], **kwargs: typing.Union[int, float, str, bool, list, tuple, dict, None, np.ndarray]
) -> typing.Union[np.ndarray, bool]:
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
        return np_mod.equal(x, y)
    except Exception:
        # Fallback for incompatible types
        return x == y
