# ruff: noqa: E501
"""Numpy eager backend initialization."""

import re
import typing

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.activation_ops  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.advanced_indexing  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.audio_extras  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.control_flow  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.conv  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.distributed  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.fft_ops  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.indexing  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.io_ops  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.linalg_extras  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.lookups  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.loss_ops  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_binary  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_extras  # noqa: F401  # noqa: F401  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_fft  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_logical  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_logical_reductions  # noqa: F401
import ml_switcheroo_compiler.backends.numpy.eager.math_misc  # noqa: F401
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
import ml_switcheroo_compiler.backends.numpy.eager.shape_ops_extra  # noqa: F401
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


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Execute execute_op.

    Args:
        cls (Any): The class.
        op_type (Any): Argument op_type.
        *args (Any): Argument *args.
        **kwargs (Any): Argument **kwargs.

    Returns:
    Any: The result.
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
            import scipy.special

            func = getattr(scipy.special, snake)
        except (ImportError, AttributeError):
            try:
                import scipy.signal

                func = getattr(scipy.signal, snake)
            except (ImportError, AttributeError):
                from ml_switcheroo_compiler.core.errors import UnimplementedMathError

                msg = f"Operation {op_type} is not implemented in interpreter."
                raise UnimplementedMathError(msg) from None

    return func(*args, **kwargs)


@numpy_eager_registry.register("Repeat")
def repeat(np: object, *args: object, **kwargs: object) -> object:
    """Repeat."""
    if "dim" in kwargs:
        kwargs["axis"] = kwargs.pop("dim")
    return np.repeat(*args, **kwargs)


@numpy_eager_registry.register("Searchsorted")
def searchsorted(np: object, *args: object, **kwargs: object) -> object:
    """Searchsorted."""
    return np.searchsorted(*args, **kwargs)


@numpy_eager_registry.register("Split")
def split(
    np_mod: object,
    x: object,
    num_or_size_splits: typing.Union[int, list[int], tuple[int, ...]],
    dim: int = 0,
    *args: object,
    **kwargs: object,
) -> list[object]:
    """Split array.

    Args:
        np_mod: Numpy module.
        x: Input array.
        num_or_size_splits: Number of splits or sizes of each split.
        dim: Axis along which to split.
        args: Additional positional arguments.
        kwargs: Additional keyword arguments.

    Returns:
        List of output arrays.
    """
    ax = dim if dim != 0 else kwargs.get("axis", 0)
    return np_mod.split(x, num_or_size_splits, axis=ax)


@numpy_eager_registry.register("Squeeze")
def squeeze(np_mod: object, x: object, dim: typing.Optional[int] = None, *args: object, **kwargs: object) -> object:
    """Squeeze array.

    Args:
        np_mod: Numpy module.
        x: Input array.
        dim: Axis to squeeze.
        args: Additional args.
        kwargs: Additional kwargs.

    Returns:
        Squeezed array.
    """
    ax = dim if dim is not None else kwargs.get("axis", None)
    return np_mod.squeeze(x, axis=ax)


@numpy_eager_registry.register("Stack")
def stack(np_mod: object, arrays: object, dim: int = 0, *args: object, **kwargs: object) -> object:
    """Stack arrays.

    Args:
        np_mod: Numpy module.
        arrays: Arrays to stack.
        dim: Axis to stack along.
        args: Additional args.
        kwargs: Additional kwargs.

    Returns:
        Stacked array.
    """
    ax = dim if dim != 0 else kwargs.get("axis", 0)
    return np_mod.stack(arrays, axis=ax)


@numpy_eager_registry.register("Unstack")
def unstack(np_mod: object, x: object, dim: int = 0, *args: object, **kwargs: object) -> list[object]:
    """Unstack array.

    Args:
        np_mod: Numpy module.
        x: Input array.
        dim: Axis to unstack along.
        args: Additional args.
        kwargs: Additional kwargs.

    Returns:
        List of unstacked arrays.
    """
    ax = dim if dim != 0 else kwargs.get("axis", 0)
    # unstack is basically split into 1-sized chunks along axis and squeezed

    if hasattr(x, "shape"):
        num_splits = x.shape[ax]
        splits = np_mod.split(x, num_splits, axis=ax)
        return tuple(np_mod.squeeze(s, axis=ax) for s in splits)
    return tuple(x)


@numpy_eager_registry.register("Equal")
def equal(np_mod: object, x: object, y: object, *args: object, **kwargs: object) -> object:
    """Check if x and y are equal.

    Args:
        np_mod (object): The numpy module.
        x (object): The first array.
        y (object): The second array.
        *args (object): Additional arguments.
        **kwargs (object): Additional keyword arguments.

    Returns:
        object: A boolean array where x == y.
    """
    try:
        return np_mod.equal(x, y)
    except Exception:
        # Fallback for incompatible types
        return x == y
