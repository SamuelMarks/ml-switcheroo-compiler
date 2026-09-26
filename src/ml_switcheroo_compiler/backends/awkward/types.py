"""Awkward Array backend type conversions and ragged array creation utilities."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence


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
        try:
            return ak_mod.Array(actual_data)
        except Exception:
            pass
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
        try:
            return ak_mod.Array(actual_data)
        except Exception:
            pass
    numpy_mod = importlib.import_module("numpy")
    return numpy_mod.asarray(actual_data)


def ragged_array(
    data_or_cls: object,
    data: Sequence[object] | None = None,
    dtype: str | None = None,
) -> object:
    """Construct an Awkward array preserving ragged nested dimensions.

    Args:
        data_or_cls (object): Nested sequence or class context.
        data (Sequence[object] | None): Nested sequence data when called as classmethod.
        dtype (str | None): Target primitive data type.

    Returns:
        object: Ragged Awkward array.
    """
    del dtype
    actual_data = data_or_cls if data is None else data
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "Array"):
        try:
            return ak_mod.Array(actual_data)
        except Exception:
            pass
    numpy_mod = importlib.import_module("numpy")
    try:
        return numpy_mod.array(actual_data)
    except ValueError:
        return numpy_mod.array(actual_data, dtype=object)


def record_array(
    data_or_cls: object,
    data: Mapping[str, object] | Sequence[object] | None = None,
    with_name: str | None = None,
) -> object:
    """Construct a nested record or zipped structure.

    Args:
        data_or_cls (object): Mapping/sequence of arrays or class context.
        data (Mapping[str, object] | Sequence[object] | None): Array mapping when called as classmethod.
        with_name (str | None): Optional record type name.

    Returns:
        object: Awkward record array.
    """
    actual_data = data_or_cls if data is None else data
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "zip") and isinstance(actual_data, Mapping):
        return ak_mod.zip(dict(actual_data), with_name=with_name)
    if hasattr(ak_mod, "Array"):
        return ak_mod.Array(actual_data)
    return actual_data


def flatten(
    data_or_cls: object,
    data: object = None,
    axis: int = 1,
) -> object:
    """Flatten a variable-length ragged dimension.

    Args:
        data_or_cls (object): Input array or class context.
        data (object): Input array when called as classmethod.
        axis (int): Axis along which to flatten ragged levels. Defaults to 1.

    Returns:
        object: Flattened array.
    """
    actual_data = data_or_cls if data is None else data
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "flatten"):
        return ak_mod.flatten(actual_data, axis=axis)
    numpy_mod = importlib.import_module("numpy")
    return numpy_mod.asarray(actual_data).flatten()


def unflatten(
    data_or_cls: object,
    data: object = None,
    counts: Sequence[int] | None = None,
    axis: int = 0,
) -> object:
    """Unflatten a 1D sequence into ragged partitions given counts.

    Args:
        data_or_cls (object): Input 1D array or class context.
        data (object): Input 1D array when called as classmethod.
        counts (Sequence[int] | None): Number of elements in each sub-partition.
        axis (int): Target axis. Defaults to 0.

    Returns:
        object: Ragged partitioned Awkward array.
    """
    if isinstance(data_or_cls, type):
        actual_data = data
        actual_counts = counts
    else:
        actual_data = data_or_cls
        actual_counts = counts if counts is not None else (data if isinstance(data, (list, tuple)) else None)
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "unflatten") and actual_counts is not None:
        return ak_mod.unflatten(actual_data, actual_counts, axis=axis)
    return actual_data


def with_field(
    base_or_cls: object,
    base: object = None,
    what: object = None,
    where: str = "",
) -> object:
    """Add or replace a field within a nested record structure.

    Args:
        base_or_cls (object): Base record structure or class context.
        base (object): Base record structure when called as classmethod.
        what (object): Field content array to insert.
        where (str): Field identifier name.

    Returns:
        object: Updated Awkward record array.
    """
    if isinstance(base_or_cls, type):
        actual_base = base
        actual_what = what
    else:
        actual_base = base_or_cls
        actual_what = what if what is not None else base
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "with_field"):
        return ak_mod.with_field(actual_base, actual_what, where)
    if isinstance(actual_base, dict):
        res = dict(actual_base)
        res[where] = actual_what
        return res
    return actual_base


def unzip(
    data_or_cls: object,
    data: object = None,
) -> tuple[object, ...]:
    """Unzip a record array into individual field arrays.

    Args:
        data_or_cls (object): Record array or class context.
        data (object): Record array when called as classmethod.

    Returns:
        tuple[object, ...]: Tuple of individual field arrays.
    """
    actual_data = data_or_cls if data is None else data
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "unzip"):
        unzipped = ak_mod.unzip(actual_data)
        return tuple(unzipped)
    if isinstance(actual_data, dict):
        return tuple(actual_data.values())
    return (actual_data,)


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
            first = val[0]
            if isinstance(first, (list, tuple)) and first:
                return float(first[0])
            return float(first)
        return float(val)
    if hasattr(actual_data, "item"):
        return float(actual_data.item())
    numpy_mod = importlib.import_module("numpy")
    return float(numpy_mod.asarray(actual_data).item())


def broadcast_ragged(
    a_or_cls: object,
    a: object = None,
    b: object = None,
) -> tuple[object, object]:
    """Broadcast two arrays across ragged dimensions without padding.

    Args:
        a_or_cls (object): First array operand or class context.
        a (object): First array operand when called as classmethod, or second array operand.
        b (object): Second array operand when called as classmethod.

    Returns:
        tuple[object, object]: Broadcast pair of Awkward or nested array structures.
    """
    if isinstance(a_or_cls, type):
        actual_a = a
        actual_b = b
    else:
        actual_a = a_or_cls
        actual_b = a if b is None else b

    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "broadcast_arrays"):
        try:
            res = ak_mod.broadcast_arrays(actual_a, actual_b)
            return (res[0], res[1])
        except Exception:
            pass

    if isinstance(actual_a, (int, float, bool)) and isinstance(actual_b, (list, tuple)):
        broadcasted_a = [[actual_a] * len(sub) if isinstance(sub, (list, tuple)) else actual_a for sub in actual_b]
        return (broadcasted_a, actual_b)
    if isinstance(actual_b, (int, float, bool)) and isinstance(actual_a, (list, tuple)):
        broadcasted_b = [[actual_b] * len(sub) if isinstance(sub, (list, tuple)) else actual_b for sub in actual_a]
        return (actual_a, broadcasted_b)

    return (actual_a, actual_b)


def ragged_shape(data_or_cls: object, data: object = None) -> tuple[int | None, ...]:
    """Infer the shape tuple of a regular or ragged array structure.

    Args:
        data_or_cls (object): Array or nested sequence or class context.
        data (object): Array or nested sequence when called as classmethod.

    Returns:
        tuple[int | None, ...]: Inferred shape with None for ragged variable dimensions.
    """
    actual_data = data_or_cls if data is None else data
    if hasattr(actual_data, "shape"):
        return tuple(int(d) for d in actual_data.shape)
    if isinstance(actual_data, (list, tuple)):
        outer_len = len(actual_data)
        if outer_len == 0:
            return (0,)
        first_sub = actual_data[0]
        if isinstance(first_sub, (list, tuple)):
            sub_len = len(first_sub)
            for sub in actual_data[1:]:
                if not isinstance(sub, (list, tuple)) or len(sub) != sub_len:
                    return (outer_len, None)
            inner_shape = ragged_shape(first_sub)
            return (outer_len, *inner_shape)
        return (outer_len,)
    return ()


def is_ragged(data_or_cls: object, data: object = None) -> bool:
    """Determine whether an array structure contains ragged variable-length dimensions.

    Args:
        data_or_cls (object): Array or nested sequence or class context.
        data (object): Array or nested sequence when called as classmethod.

    Returns:
        bool: True if structure has ragged inner dimensions, False otherwise.
    """
    shape = ragged_shape(data_or_cls, data)
    return any(d is None for d in shape)


def to_regular(
    data_or_cls: object,
    data: object = None,
    fill_value: object = None,
) -> object:
    """Convert ragged array to regular dense multidimensional structure.

    Args:
        data_or_cls (object): Ragged array structure or class context.
        data (object): Ragged array structure when called as classmethod.
        fill_value (object): Value to pad missing ragged elements.

    Returns:
        object: Regular dense array or padded nested list.
    """
    actual_data = data_or_cls if data is None else data
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "to_regular"):
        try:
            return ak_mod.to_regular(actual_data)
        except Exception:
            pass

    if isinstance(actual_data, (list, tuple)):
        lengths = [len(sub) if isinstance(sub, (list, tuple)) else 1 for sub in actual_data]
        max_len = max(lengths) if lengths else 0
        pad_val = 0 if fill_value is None else fill_value
        padded = []
        for sub in actual_data:
            if isinstance(sub, (list, tuple)):
                padded.append(list(sub) + [pad_val] * (max_len - len(sub)))
            else:
                padded.append([sub] + [pad_val] * (max_len - 1))
        numpy_mod = importlib.import_module("numpy")
        return numpy_mod.asarray(padded)

    numpy_mod = importlib.import_module("numpy")
    return numpy_mod.asarray(actual_data)


def from_regular(data_or_cls: object, data: object = None) -> object:
    """Convert regular dense structure to Awkward representation.

    Args:
        data_or_cls (object): Input array or class context.
        data (object): Input array when called as classmethod.

    Returns:
        object: Awkward array representation.
    """
    actual_data = data_or_cls if data is None else data
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "from_regular"):
        try:
            return ak_mod.from_regular(actual_data)
        except Exception:
            pass
    return array(actual_data)


def to_numpy(data_or_cls: object, data: object = None) -> object:
    """Convert Awkward structure to standard host NumPy ndarray.

    Args:
        data_or_cls (object): Awkward array or class context.
        data (object): Awkward array when called as classmethod.

    Returns:
        object: Host NumPy ndarray buffer.
    """
    actual_data = data_or_cls if data is None else data
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "to_numpy"):
        try:
            return ak_mod.to_numpy(actual_data)
        except Exception:
            pass
    numpy_mod = importlib.import_module("numpy")
    return numpy_mod.asarray(actual_data)


def from_numpy(data_or_cls: object, data: object = None) -> object:
    """Construct Awkward array from host NumPy ndarray.

    Args:
        data_or_cls (object): Host NumPy array or class context.
        data (object): Host NumPy array when called as classmethod.

    Returns:
        object: Awkward array.
    """
    actual_data = data_or_cls if data is None else data
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "from_numpy"):
        try:
            return ak_mod.from_numpy(actual_data)
        except Exception:
            pass
    return array(actual_data)


def to_list(data_or_cls: object, data: object = None) -> list[object]:
    """Convert Awkward array or structure to nested Python list.

    Args:
        data_or_cls (object): Awkward array or class context.
        data (object): Awkward array when called as classmethod.

    Returns:
        list[object]: Python nested list.
    """
    actual_data = data_or_cls if data is None else data
    if hasattr(actual_data, "to_list"):
        return actual_data.to_list()
    if hasattr(actual_data, "tolist"):
        return actual_data.tolist()
    if isinstance(actual_data, (list, tuple)):
        return list(actual_data)
    return [actual_data]


def from_iter(data_or_cls: object, data: object = None) -> object:
    """Construct Awkward array from an arbitrary Python iterable.

    Args:
        data_or_cls (object): Source iterable or class context.
        data (object): Source iterable when called as classmethod.

    Returns:
        object: Constructed Awkward array.
    """
    actual_data = data_or_cls if data is None else data
    ak_mod = _get_ak_module()
    if hasattr(ak_mod, "from_iter"):
        try:
            return ak_mod.from_iter(actual_data)
        except Exception:
            pass
    return array(actual_data)


def get_layout(data_or_cls: object, data: object = None) -> object:
    """Retrieve internal layout description of an Awkward structure.

    Args:
        data_or_cls (object): Awkward array or class context.
        data (object): Awkward array when called as classmethod.

    Returns:
        object: Layout object or shape layout description.
    """
    actual_data = data_or_cls if data is None else data
    if hasattr(actual_data, "layout"):
        return actual_data.layout
    return {"shape": ragged_shape(actual_data), "is_ragged": is_ragged(actual_data)}
