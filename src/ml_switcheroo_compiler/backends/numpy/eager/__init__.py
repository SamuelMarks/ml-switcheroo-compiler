# ruff: noqa: E402
"""Numpy eager backend initialization."""

from __future__ import annotations

import re
import typing

import numpy as np

_np_module = np

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    BackendOperand,
    numpy_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@typing.runtime_checkable
class _NumpyUnwrapProtocol(typing.Protocol):
    """Protocol for tensor wrappers exposing a `.numpy()` conversion method."""

    def numpy(self) -> BackendArg:
        """Convert wrapper into a NumPy array.

        Returns:
            BackendArg: Underlying NumPy array.
        """
        ...


def ensure_ndarray(
    val: BackendOperand | _NumpyUnwrapProtocol,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Convert or narrow an eager operand to `np.ndarray` without unsound TypeGuards.

    Args:
        val (BackendOperand | _NumpyUnwrapProtocol): Input array or scalar operand.

    Returns:
        np.ndarray: NumPy ndarray instance.
    """
    if isinstance(val, np.ndarray):
        return val
    unwrapped = val.numpy() if isinstance(val, _NumpyUnwrapProtocol) else val
    return unwrapped if isinstance(unwrapped, np.ndarray) else np.asarray(unwrapped)


import ml_switcheroo_compiler.backends.eager  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager._indexing_parsing_utils  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager.advanced_indexing  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager.distributed  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager.indexing  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager.io_ops  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager.scipy_linalg_kernels  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager.scipy_ndimage_kernels  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels  # noqa: E402, F401


def execute_op(
    cls: type | None,
    op_type: str,
    *args: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Dispatch and execute an eager operation on the NumPy backend.

    Args:
        cls (type | None): Backend class receiving the operation dispatch.
        op_type (str): Canonical IR operation name to execute.
        *args (BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Positional tensor or scalar operands.
        **kwargs (BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Keyword operation attributes.

    Returns:
        BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]]: Evaluated NumPy array or scalar result.

    Raises:
        UnimplementedMathError: If `op_type` is not registered and has no NumPy equivalent.
    """
    func_registry = numpy_eager_registry.get(op_type)
    if func_registry is not None:
        return func_registry(np, *args, **kwargs)
    try:
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
        snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        func = adapt_backend_module(np).require_symbol(snake)
    except AttributeError:
        from ml_switcheroo_compiler.core.errors import UnimplementedMathError

        msg = f"Operation {op_type} is not implemented in interpreter."
        raise UnimplementedMathError(msg) from None
    return func(*args, **kwargs)


@numpy_eager_registry.register("Clone")
def clone_op(
    np_mod: BackendModuleProtocol | BackendArg,
    *args: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
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
    return ensure_ndarray(adapt_backend_module(np).require_op("copy")(args[0]))


@numpy_eager_registry.register("Repeat")
def repeat(
    np_mod: BackendModuleProtocol | BackendArg,
    *args: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Repeat.

    Args:
        np_mod (BackendModuleProtocol | BackendArg): Input `np` argument (np).
        *args (BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Positional args.
        **kwargs (BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_ndarray(adapt_backend_module(np_mod).require_symbol("repeat")(*args, **kwargs))


@numpy_eager_registry.register("Searchsorted")
def searchsorted(
    np_mod: BackendModuleProtocol | BackendArg,
    *args: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Searchsorted.

    Args:
        np_mod (BackendModuleProtocol | BackendArg): Input `np` argument (np).
        *args (BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Positional args.
        **kwargs (BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_ndarray(adapt_backend_module(np_mod).require_symbol("searchsorted")(*args, **kwargs))


@numpy_eager_registry.register("Split")
def split(
    np_mod: BackendModuleProtocol | BackendArg,
    x: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | BackendArg,
    num_or_size_splits: int | list[int] | tuple[int, ...] | BackendArg,
    axis: int = 0,
    *args: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
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
    split_fn = adapt_backend_module(np_mod).get_symbol("split", np.split)
    return [ensure_ndarray(s) for s in split_fn(x, num_or_size_splits, axis=axis)]


@numpy_eager_registry.register("Squeeze")
def squeeze(
    np_mod: BackendModuleProtocol | BackendArg,
    x: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | BackendArg,
    axis: int | None = None,
    *args: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
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
    return ensure_ndarray(adapt_backend_module(np_mod).require_symbol("squeeze")(x, axis=axis))


@numpy_eager_registry.register("Stack")
def stack(
    np_mod: BackendModuleProtocol | BackendArg,
    arrays: typing.Sequence[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] | BackendArg,
    axis: int = 0,
    *args: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
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
    return ensure_ndarray(adapt_backend_module(np_mod).require_symbol("stack")(arrays, axis=axis))


@numpy_eager_registry.register("Unstack")
def unstack(
    np_mod: BackendModuleProtocol | BackendArg,
    x: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | BackendArg,
    axis: int = 0,
    *args: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
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
    if adapt_backend_module(x).has_symbol("shape"):
        num_splits = adapt_backend_module(x).require_symbol("shape")[axis]
        split_fn = adapt_backend_module(np_mod).get_symbol("split", np.split)
        splits = split_fn(x, num_splits, axis=axis)
        squeeze_fn = adapt_backend_module(np_mod).get_symbol("squeeze", np.squeeze)
        return tuple(ensure_ndarray(squeeze_fn(s, axis=axis)) for s in splits)
    from ml_switcheroo_compiler.backends.eager_registry import coerce_operand_list

    return tuple(ensure_ndarray(item) for item in coerce_operand_list(x))


@numpy_eager_registry.register("Equal")
def equal(
    np_mod: BackendModuleProtocol | BackendArg,
    x: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | BackendArg,
    y: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | BackendArg,
    *args: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    **kwargs: BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | bool:
    """Check if x and y are equal.

    Args:
        np_mod (BackendModuleProtocol | BackendArg): The numpy module.
        x (np.ndarray[tuple[int, ...], np.dtype[np.float64]] | BackendArg): The first array.
        y (np.ndarray[tuple[int, ...], np.dtype[np.float64]] | BackendArg): The second array.
        *args (BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Additional arguments.
        **kwargs (BackendArg | np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Additional keyword arguments.

    Returns: np.ndarray: A boolean array where x == y.
    """
    try:
        return ensure_ndarray(adapt_backend_module(np_mod).require_symbol("equal")(x, y))
    except (ImportError, ModuleNotFoundError, OSError, AttributeError, RuntimeError, TypeError, ValueError):
        # Fallback for incompatible types
        return bool(x == y)
