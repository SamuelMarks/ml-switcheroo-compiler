"""DPNP backend tensor creation and array helpers."""

from __future__ import annotations

import importlib
from collections.abc import Sequence


def _get_dpnp_module() -> object:
    """Retrieve dpnp module dynamically with fallback.

    Returns:
        object: The dpnp or numpy module.
    """
    try:
        return importlib.import_module("dpnp")
    except Exception:
        return importlib.import_module("numpy")


def zeros(shape_or_cls: object, shape: tuple[int, ...] | None = None) -> object:
    """Create a DPNP array of zeros on SYCL target device.

    Args:
        shape_or_cls (object): Shape tuple or class context.
        shape (tuple[int, ...] | None): Optional shape when called as classmethod.

    Returns:
        object: Allocated zero-filled array.
    """
    actual_shape = shape_or_cls if shape is None and isinstance(shape_or_cls, tuple) else (shape or (0,))
    dpnp_mod = _get_dpnp_module()
    return dpnp_mod.zeros(actual_shape, dtype=dpnp_mod.float32)


def array(data_or_cls: object, data: Sequence[float] | None = None, dtype: str | None = None) -> object:
    """Create a DPNP array from data sequence.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (Sequence[float] | None): Optional data sequence when called as classmethod.
        dtype (str | None): Target datatype name.

    Returns:
        object: Constructed DPNP array.
    """
    actual_data = data_or_cls if data is None else data
    dpnp_mod = _get_dpnp_module()
    if dtype is not None and hasattr(dpnp_mod, str(dtype)):
        return dpnp_mod.array(actual_data, dtype=getattr(dpnp_mod, str(dtype)))
    return dpnp_mod.array(actual_data)


def asarray(data_or_cls: object, data: Sequence[float] | None = None) -> object:
    """Convert input sequence or array to DPNP format.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (Sequence[float] | None): Optional data sequence when called as classmethod.

    Returns:
        object: DPNP array representation.
    """
    actual_data = data_or_cls if data is None else data
    dpnp_mod = _get_dpnp_module()
    return dpnp_mod.asarray(actual_data)


def item(data_or_cls: object, data: object = None) -> float:
    """Extract scalar item value from single-element tensor.

    Args:
        data_or_cls (object): Target data or class context.
        data (object): Target data when called as classmethod.

    Returns:
        float: Extracted scalar value.
    """
    actual_data = data_or_cls if data is None else data
    if hasattr(actual_data, "item"):
        return float(actual_data.item())
    numpy_mod = importlib.import_module("numpy")
    return float(numpy_mod.asarray(actual_data).item())
