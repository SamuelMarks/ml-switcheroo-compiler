# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy eager backend initialization."""

import re
import typing
from typing import Any

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
import ml_switcheroo_compiler.backends.numpy.eager.math_binary  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_extended  # noqa: F401  # noqa: F401  # noqa: F401
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


def execute_op(cls: type, op_type: str, *args: Any, **kwargs: Any) -> Any:
    """Evaluate execute_op operation.

    Args:
        cls (type): Class.
        op_type (str): The op_type parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

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
        try:
            raise AttributeError()
        except AttributeError:
            try:
                raise AttributeError()
            except AttributeError:
                from ml_switcheroo_compiler.core.errors import UnimplementedMathError

                msg = f"Operation {op_type} is not implemented in interpreter."
                raise UnimplementedMathError(msg) from None
    return func(*args, **kwargs)


@numpy_eager_registry.register("Repeat")
def repeat(np: Any, *args: Any, **kwargs: Any) -> Any:
    """Repeat.

    Args:
        np (object): The np parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return np.repeat(*args, **kwargs)


@numpy_eager_registry.register("Searchsorted")
def searchsorted(np: Any, *args: Any, **kwargs: Any) -> Any:
    """Searchsorted.

    Args:
        np (object): The np parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return np.searchsorted(*args, **kwargs)


@numpy_eager_registry.register("Split")
def split(
    np_mod: Any,
    x: Any,
    num_or_size_splits: typing.Union[int, list[int], tuple[int, ...]],
    axis: int = 0,
    *args: Any,
    **kwargs: Any,
) -> list[Any]:
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
def squeeze(np_mod: Any, x: Any, axis: typing.Optional[int] = None, *args: Any, **kwargs: Any) -> Any:
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
def stack(np_mod: Any, arrays: Any, axis: int = 0, *args: Any, **kwargs: Any) -> Any:
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
def unstack(np_mod: Any, x: Any, axis: int = 0, *args: Any, **kwargs: Any) -> Any:
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


@numpy_eager_registry.register("AllGather")
def all_gather(np_mod: Any, tensor: Any, *args: Any, **kwargs: Any) -> Any:
    """Simulate AllGather in eager mode.

    Args:
        np_mod (object): The numpy module.
        tensor (object): The tensor to gather.
        *args (object): Additional args.
        **kwargs (object): Additional kwargs.

    Returns: Any: The gathered tensor (in a simulated single-node env, just expanded).
    """
    axis = kwargs.get("axis", 0)
    return np_mod.expand_dims(tensor, axis=axis)


@numpy_eager_registry.register("AllReduce")
def all_reduce(np_mod: Any, tensor: Any, *args: Any, **kwargs: Any) -> Any:
    """Simulate AllReduce in eager mode.

    Args:
        np_mod (object): The numpy module.
        tensor (object): The tensor to reduce.
        *args (object): Additional args.
        **kwargs (object): Additional kwargs.

    Returns: Any: The reduced tensor.
    """
    return tensor


@numpy_eager_registry.register("ReduceScatter")
def reduce_scatter(np_mod: Any, tensor: Any, *args: Any, **kwargs: Any) -> Any:
    """Simulate ReduceScatter in eager mode.

    Args:
        np_mod (object): The numpy module.
        tensor (object): The tensor to scatter.
        *args (object): Additional args.
        **kwargs (object): Additional kwargs.

    Returns: Any: The scattered tensor.
    """
    return tensor


@numpy_eager_registry.register("AllToAll")
def all_to_all(np_mod: Any, tensor: Any, *args: Any, **kwargs: Any) -> Any:
    """Simulate AllToAll in eager mode.

    Args:
        np_mod (object): The numpy module.
        tensor (object): The tensor to all-to-all.
        *args (object): Additional args.
        **kwargs (object): Additional kwargs.

    Returns: Any: The result tensor.
    """
    return tensor


@numpy_eager_registry.register("Equal")
def equal(np_mod: Any, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
    """Check if x and y are equal.

    Args:
        np_mod (object): The numpy module.
        x (object): The first array.
        y (object): The second array.
        *args (object): Additional arguments.
        **kwargs (object): Additional keyword arguments.

    Returns: Any: A boolean array where x == y.
    """
    try:
        return np_mod.equal(x, y)
    except Exception:
        # Fallback for incompatible types
        return x == y
