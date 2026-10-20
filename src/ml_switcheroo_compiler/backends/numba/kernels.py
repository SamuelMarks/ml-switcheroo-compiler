"""Native Numba array and mathematical kernels for CPU execution."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def _resolve_dense_backend() -> EagerValue:
    """Resolve the underlying array buffer backend via the BackendRegistry.

    Returns:
        EagerValue: Registered host array backend module.
    """
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    return typing.cast(EagerValue, BackendRegistry.get("numpy").get_module())


def asarray(data: EagerValue, dtype: EagerValue = None) -> EagerValue:
    """Convert input sequence or buffer into a Numba-compatible array buffer.

    Args:
        data (EagerValue): Input sequence, scalar, or array buffer.
        dtype (EagerValue): Optional target datatype specifier.

    Returns:
        EagerValue: Contiguous array buffer compatible with Numba JIT kernels.
    """
    backend = _resolve_dense_backend()
    if dtype is not None:
        return typing.cast(EagerValue, getattr(backend, "asarray")(data, dtype=dtype))
    return typing.cast(EagerValue, getattr(backend, "asarray")(data))


def array(data: EagerValue, dtype: EagerValue = None) -> EagerValue:
    """Construct a new Numba-compatible array from input data.

    Args:
        data (EagerValue): Input sequence or array buffer.
        dtype (EagerValue): Optional target datatype specifier.

    Returns:
        EagerValue: Newly allocated array buffer.
    """
    backend = _resolve_dense_backend()
    if dtype is not None:
        return typing.cast(EagerValue, getattr(backend, "array")(data, dtype=dtype))
    return typing.cast(EagerValue, getattr(backend, "array")(data))


def zeros(shape: tuple[int, ...] | int, dtype: EagerValue = "float32") -> EagerValue:
    """Allocate a zero-initialized array buffer of the given shape.

    Args:
        shape (tuple[int, ...] | int): Target shape dimensions.
        dtype (EagerValue): Target datatype. Defaults to 'float32'.

    Returns:
        EagerValue: Zero-initialized array buffer.
    """
    backend = _resolve_dense_backend()
    return typing.cast(EagerValue, getattr(backend, "zeros")(shape, dtype=dtype))


def zeros_like(arr: EagerValue) -> EagerValue:
    """Allocate a zero-initialized array matching the shape and dtype of arr.

    Args:
        arr (EagerValue): Reference array buffer.

    Returns:
        EagerValue: Zero-initialized array buffer.
    """
    backend = _resolve_dense_backend()
    return typing.cast(EagerValue, getattr(backend, "zeros_like")(arr))


def add(a: EagerValue, b: EagerValue) -> EagerValue:
    """Compute elementwise addition of two operands.

    Args:
        a (EagerValue): First operand.
        b (EagerValue): Second operand.

    Returns:
        EagerValue: Elementwise sum.
    """
    backend = _resolve_dense_backend()
    return typing.cast(EagerValue, getattr(backend, "add")(a, b))


def __getattr__(name: str) -> EagerValue:
    """Delegate attribute resolution to the underlying Numba array backend.

    Args:
        name (str): Attribute name to resolve.

    Returns:
        EagerValue: Resolved function or constant.

    Raises:
        AttributeError: If the attribute does not exist on the backend.
    """
    backend = _resolve_dense_backend()
    if hasattr(backend, name):
        return typing.cast(EagerValue, getattr(backend, name))
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
