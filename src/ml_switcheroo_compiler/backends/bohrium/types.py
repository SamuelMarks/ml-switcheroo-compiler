"""Bohrium backend type conversions and lazy array creation utilities."""

from __future__ import annotations

import importlib
from collections.abc import Sequence


def _get_bh_module() -> object:
    """Retrieve bohrium module dynamically with fallback.

    Returns:
        object: The bohrium or numpy module.
    """
    try:
        return importlib.import_module("bohrium")
    except Exception:
        return importlib.import_module("numpy")


def zeros(shape_or_cls: object, shape: tuple[int, ...] | None = None) -> object:
    """Create a Bohrium zero-filled lazy array.

    Args:
        shape_or_cls (object): Shape tuple or class context.
        shape (tuple[int, ...] | None): Optional shape when called as classmethod.

    Returns:
        object: Allocated zero-filled lazy array.
    """
    actual_shape = shape_or_cls if shape is None and isinstance(shape_or_cls, tuple) else (shape or (0,))
    bh_mod = _get_bh_module()
    return bh_mod.zeros(actual_shape, dtype=bh_mod.float32)


def array(data_or_cls: object, data: Sequence[float] | None = None, dtype: str | None = None) -> object:
    """Create a Bohrium lazy array from data.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (Sequence[float] | None): Optional data sequence when called as classmethod.
        dtype (str | None): Optional datatype name.

    Returns:
        object: Constructed Bohrium array.
    """
    actual_data = data_or_cls if data is None else data
    bh_mod = _get_bh_module()
    if dtype is not None and hasattr(bh_mod, str(dtype)):
        return bh_mod.array(actual_data, dtype=getattr(bh_mod, str(dtype)))
    return bh_mod.array(actual_data)


def asarray(data_or_cls: object, data: Sequence[float] | None = None) -> object:
    """Convert input to a Bohrium lazy array.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (Sequence[float] | None): Optional data sequence when called as classmethod.

    Returns:
        object: Bohrium array representation.
    """
    actual_data = data_or_cls if data is None else data
    bh_mod = _get_bh_module()
    return bh_mod.asarray(actual_data)


def item(data_or_cls: object, data: object = None) -> float:
    """Extract scalar item value from a Bohrium array.

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
