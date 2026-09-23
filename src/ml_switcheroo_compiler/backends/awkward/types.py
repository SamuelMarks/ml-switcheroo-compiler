"""Awkward Array backend type conversions and ragged array creation utilities."""

from __future__ import annotations

import importlib
from collections.abc import Sequence


def _get_ak_module() -> object:
    """Retrieve awkward module dynamically with fallback.

    Returns:
        object: The awkward or numpy module.
    """
    try:
        return importlib.import_module("awkward")
    except Exception:
        return importlib.import_module("numpy")


def zeros(shape_or_cls: object, shape: tuple[int, ...] | None = None) -> object:
    """Create an Awkward/NumPy zero-filled array.

    Args:
        shape_or_cls (object): Shape tuple or class context.
        shape (tuple[int, ...] | None): Optional shape when called as classmethod.

    Returns:
        object: Allocated zero-filled array.
    """
    actual_shape = shape_or_cls if shape is None and isinstance(shape_or_cls, tuple) else (shape or (0,))
    numpy_mod = importlib.import_module("numpy")
    arr = numpy_mod.zeros(actual_shape, dtype=numpy_mod.float32)
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "Array"):
        return ak_mod.Array(arr)
    return arr


def array(data_or_cls: object, data: Sequence[object] | None = None, dtype: str | None = None) -> object:
    """Create an Awkward ragged or regular array from data.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (Sequence[object] | None): Optional data sequence when called as classmethod.
        dtype (str | None): Optional datatype name.

    Returns:
        object: Constructed Awkward array.
    """
    del dtype
    actual_data = data_or_cls if data is None else data
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "Array"):
        return ak_mod.Array(actual_data)
    numpy_mod = importlib.import_module("numpy")
    try:
        return numpy_mod.array(actual_data)
    except ValueError:
        return numpy_mod.array(actual_data, dtype=object)


def asarray(data_or_cls: object, data: Sequence[object] | None = None) -> object:
    """Convert input to an Awkward array.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (Sequence[object] | None): Optional data sequence when called as classmethod.

    Returns:
        object: Awkward array.
    """
    actual_data = data_or_cls if data is None else data
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "Array"):
        return ak_mod.Array(actual_data)
    numpy_mod = importlib.import_module("numpy")
    return numpy_mod.asarray(actual_data)


def item(data_or_cls: object, data: object = None) -> float:
    """Extract scalar item value from an Awkward array.

    Args:
        data_or_cls (object): Target data or class context.
        data (object): Target data when called as classmethod.

    Returns:
        float: Extracted scalar value.
    """
    actual_data = data_or_cls if data is None else data
    if hasattr(actual_data, "to_list"):
        val = actual_data.to_list()
        if isinstance(val, (list, tuple)) and val:
            return float(val[0])
        return float(val)
    if hasattr(actual_data, "item"):
        return float(actual_data.item())
    numpy_mod = importlib.import_module("numpy")
    return float(numpy_mod.asarray(actual_data).item())
