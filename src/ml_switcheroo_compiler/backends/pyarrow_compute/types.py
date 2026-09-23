"""Apache Arrow Compute backend type conversions and columnar array creation utilities."""

from __future__ import annotations

import importlib
from collections.abc import Sequence


def _get_pa_module() -> object:
    """Retrieve pyarrow module dynamically with fallback.

    Returns:
        object: The pyarrow or numpy module.
    """
    try:
        return importlib.import_module("pyarrow")
    except Exception:
        return importlib.import_module("numpy")


def zeros(shape_or_cls: object, shape: tuple[int, ...] | None = None) -> object:
    """Create a zero-filled array compatible with Arrow Compute.

    Args:
        shape_or_cls (object): Shape tuple or class context.
        shape (tuple[int, ...] | None): Optional shape when called as classmethod.

    Returns:
        object: Allocated zero-filled array.
    """
    actual_shape = shape_or_cls if shape is None and isinstance(shape_or_cls, tuple) else (shape or (0,))
    numpy_mod = importlib.import_module("numpy")
    arr = numpy_mod.zeros(actual_shape, dtype=numpy_mod.float32)
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "array"):
        try:
            return pa_mod.array(arr.flatten())
        except Exception:
            return arr
    return arr


def array(data_or_cls: object, data: Sequence[object] | None = None, dtype: str | None = None) -> object:
    """Create a columnar Arrow array from data.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (Sequence[object] | None): Optional data sequence when called as classmethod.
        dtype (str | None): Optional datatype name.

    Returns:
        object: Constructed Arrow array.
    """
    del dtype
    actual_data = data_or_cls if data is None else data
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "array"):
        try:
            return pa_mod.array(actual_data)
        except Exception:
            pass
    numpy_mod = importlib.import_module("numpy")
    return numpy_mod.array(actual_data)


def asarray(data_or_cls: object, data: Sequence[object] | None = None) -> object:
    """Convert input to an Arrow or NumPy array.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (Sequence[object] | None): Optional data sequence when called as classmethod.

    Returns:
        object: Columnar array representation.
    """
    actual_data = data_or_cls if data is None else data
    pa_mod = _get_pa_module()
    if hasattr(pa_mod, "array"):
        try:
            return pa_mod.array(actual_data)
        except Exception:
            pass
    numpy_mod = importlib.import_module("numpy")
    return numpy_mod.asarray(actual_data)


def item(data_or_cls: object, data: object = None) -> float:
    """Extract scalar item value from a columnar Arrow array.

    Args:
        data_or_cls (object): Target data or class context.
        data (object): Target data when called as classmethod.

    Returns:
        float: Extracted scalar value.
    """
    actual_data = data_or_cls if data is None else data
    if hasattr(actual_data, "as_py"):
        return float(actual_data.as_py())
    if hasattr(actual_data, "to_pylist"):
        lst = actual_data.to_pylist()
        if lst:
            return float(lst[0])
    if hasattr(actual_data, "item"):
        return float(actual_data.item())
    numpy_mod = importlib.import_module("numpy")
    return float(numpy_mod.asarray(actual_data).item())
