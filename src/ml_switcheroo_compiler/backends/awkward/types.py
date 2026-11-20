"""Awkward Array backend type conversions and ragged array creation utilities."""

from __future__ import annotations

import importlib
import logging
import typing
from collections.abc import Mapping, Sequence

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_float,
    coerce_int,
    coerce_operand_list,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


@typing.runtime_checkable
class _DunderArrayProtocol(typing.Protocol):
    """Protocol for objects exposing `__array__()`."""

    def __array__(self) -> BackendArg:
        """Convert to array buffer.

        Returns:
            BackendArg: Array buffer.
        """
        ...


def _get_ak_module() -> BackendArg:
    """Retrieve awkward module dynamically with fallback.

    Returns:
        BackendArg: The awkward module or None when unavailable.
    """
    try:
        return importlib.import_module("awkward")
    except (ImportError, ModuleNotFoundError, OSError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _build_nested_zeros(shape: tuple[int, ...] | int) -> BackendArg:
    """Construct nested zero-initialized Python lists for the given shape.

    Args:
        shape (tuple[int, ...] | int): Target shape tuple or integer size.

    Returns:
        BackendArg: Nested list of 0.0 values.
    """
    shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
    if len(shape_tuple) <= 1:
        return [0.0] * (shape_tuple[0] if shape_tuple else 0)
    return [_build_nested_zeros(shape_tuple[1:]) for _ in range(shape_tuple[0])]


def zeros(shape: tuple[int, ...] | int) -> BackendArg:
    """Create an Awkward zero-filled array.

    Args:
        shape (tuple[int, ...] | int): Target shape tuple or integer size.

    Returns:
        BackendArg: Allocated zero-filled Awkward array or nested list fallback.
    """
    nested = _build_nested_zeros(shape)
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("Array"):
        return adapt_backend_module(ak_mod).require_op("Array")(nested)
    return nested


def array(data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
    """Create an Awkward ragged or regular array from data.

    Args:
        data (BackendArg): Data sequence or buffer.
        dtype (BackendArg | None): Optional datatype name.

    Returns:
        BackendArg: Constructed Awkward array or sequence fallback.
    """
    del dtype
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("Array"):
        try:
            return adapt_backend_module(ak_mod).require_op("Array")(data)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return list(data) if isinstance(data, (list, tuple)) else [data]


def asarray(data: BackendArg) -> BackendArg:
    """Convert input to an Awkward array.

    Args:
        data (BackendArg): Data sequence or buffer.

    Returns:
        BackendArg: Awkward array or list fallback.
    """
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("Array"):
        try:
            return adapt_backend_module(ak_mod).require_op("Array")(data)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return list(data) if isinstance(data, (list, tuple)) else [data]


def ragged_array(
    data: BackendArg,
    dtype: str | None = None,
) -> BackendArg:
    """Construct an Awkward array preserving ragged nested dimensions.

    Args:
        data (BackendArg): Nested sequence.
        dtype (str | None): Target primitive data type.

    Returns:
        BackendArg: Ragged Awkward array or list fallback.
    """
    del dtype
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("Array"):
        try:
            return adapt_backend_module(ak_mod).require_op("Array")(data)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return list(data) if isinstance(data, (list, tuple)) else [data]


def record_array(
    data: BackendArg,
    with_name: str | None = None,
) -> BackendArg:
    """Construct a nested record or zipped structure.

    Args:
        data (BackendArg): Mapping or sequence of arrays.
        with_name (str | None): Optional record type name.

    Returns:
        BackendArg: Awkward record array.
    """
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("zip") and isinstance(data, Mapping):
        return adapt_backend_module(ak_mod).require_op("zip")(dict(data), with_name=with_name)
    if adapt_backend_module(ak_mod).has_symbol("Array"):
        return adapt_backend_module(ak_mod).require_op("Array")(data)
    return data


def flatten(
    data: BackendArg,
    axis: int = 1,
) -> BackendArg:
    """Flatten a variable-length ragged dimension.

    Args:
        data (BackendArg): Input array.
        axis (int): Axis along which to flatten ragged levels. Defaults to 1.

    Returns:
        BackendArg: Flattened array.
    """
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("flatten"):
        return adapt_backend_module(ak_mod).require_op("flatten")(data, axis=axis)
    return adapt_backend_module(data).require_op("flatten")()


def unflatten(
    data: BackendArg,
    counts: Sequence[int] | None = None,
    axis: int = 0,
) -> BackendArg:
    """Unflatten a 1D sequence into ragged partitions given counts.

    Args:
        data (BackendArg): Input 1D array.
        counts (Sequence[int] | None): Number of elements in each sub-partition.
        axis (int): Target axis. Defaults to 0.

    Returns:
        BackendArg: Ragged partitioned Awkward array.
    """
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("unflatten") and counts is not None:
        return adapt_backend_module(ak_mod).require_op("unflatten")(data, counts, axis=axis)
    return data


def with_field(
    base: BackendArg,
    what: BackendArg,
    where: str = "",
) -> BackendArg:
    """Add or replace a field within a nested record structure.

    Args:
        base (BackendArg): Base record structure.
        what (BackendArg): Field content array to insert.
        where (str): Field identifier name.

    Returns:
        BackendArg: Updated Awkward record array.
    """
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("with_field"):
        return adapt_backend_module(ak_mod).require_op("with_field")(base, what, where)
    if isinstance(base, dict):
        res = dict(base)
        res[where] = what
        return res
    return base


def unzip(
    data: BackendArg,
) -> tuple[BackendArg, ...]:
    """Unzip a record array into individual field arrays.

    Args:
        data (BackendArg): Record array.

    Returns:
        tuple[BackendArg, ...]: Tuple of individual field arrays.
    """
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("unzip"):
        unzipped = adapt_backend_module(ak_mod).require_op("unzip")(data)
        return tuple(coerce_operand_list(unzipped))
    if isinstance(data, dict):
        return tuple(data.values())
    return (data,)


def item(data: BackendArg) -> float:
    """Extract scalar item value from an Awkward array.

    Args:
        data (BackendArg): Target data.

    Returns:
        float: Extracted scalar value.
    """
    if adapt_backend_module(data).has_symbol("to_list"):
        val: BackendArg = adapt_backend_module(data).require_op("to_list")()
        if isinstance(val, (list, tuple)) and val:
            first = val[0]
            if isinstance(first, (list, tuple)) and first:
                return float(coerce_float(first[0]))
            return float(coerce_float(first))
        return float(coerce_float(val))
    if adapt_backend_module(data).has_symbol("item"):
        return float(coerce_float(adapt_backend_module(data).require_op("item")()))
    arr = to_numpy(data)
    return float(coerce_float(adapt_backend_module(arr).require_op("item")()))


def broadcast_ragged(
    a: BackendArg,
    b: BackendArg,
) -> tuple[BackendArg, BackendArg]:
    """Broadcast two arrays across ragged dimensions without padding.

    Args:
        a (BackendArg): First array operand.
        b (BackendArg): Second array operand.

    Returns:
        tuple[BackendArg, BackendArg]: Broadcast pair of Awkward or nested array structures.
    """
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("broadcast_arrays"):
        try:
            res = coerce_operand_list(adapt_backend_module(ak_mod).require_op("broadcast_arrays")(a, b))
            return (res[0], res[1])
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    if isinstance(a, (int, float, bool)) and isinstance(b, (list, tuple)):
        broadcasted_a = [[a] * len(sub) if isinstance(sub, (list, tuple)) else a for sub in b]
        return (broadcasted_a, b)
    if isinstance(b, (int, float, bool)) and isinstance(a, (list, tuple)):
        broadcasted_b = [[b] * len(sub) if isinstance(sub, (list, tuple)) else b for sub in a]
        return (a, broadcasted_b)

    return (a, b)


def ragged_shape(data: BackendArg) -> tuple[int | None, ...]:
    """Infer the shape tuple of a regular or ragged array structure.

    Args:
        data (BackendArg): Array or nested sequence.

    Returns:
        tuple[int | None, ...]: Inferred shape with None for ragged variable dimensions.
    """
    if adapt_backend_module(data).has_symbol("shape"):
        raw_sh: BackendArg = adapt_backend_module(data).require_symbol("shape")
        return tuple(int(coerce_int(d)) for d in (raw_sh if isinstance(raw_sh, (list, tuple)) else ()))
    if isinstance(data, (list, tuple)):
        outer_len = len(data)
        if outer_len == 0:
            return (0,)
        first_sub = data[0]
        if isinstance(first_sub, (list, tuple)):
            sub_len = len(first_sub)
            for sub in data[1:]:
                if not isinstance(sub, (list, tuple)) or len(sub) != sub_len:
                    return (outer_len, None)
            inner_shape = ragged_shape(first_sub)
            return (outer_len, *inner_shape)
        return (outer_len,)
    return ()


def is_ragged(data: BackendArg) -> bool:
    """Determine whether an array structure contains ragged variable-length dimensions.

    Args:
        data (BackendArg): Array or nested sequence.

    Returns:
        bool: True if structure has ragged inner dimensions, False otherwise.
    """
    shape = ragged_shape(data)
    return any(d is None for d in shape)


def to_regular(
    data: BackendArg,
    fill_value: BackendArg = None,
) -> BackendArg:
    """Convert ragged array to regular dense multidimensional structure.

    Args:
        data (BackendArg): Ragged array structure.
        fill_value (BackendArg): Value to pad missing ragged elements.

    Returns:
        BackendArg: Regular dense array or padded nested list.
    """
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("to_regular"):
        try:
            return adapt_backend_module(ak_mod).require_op("to_regular")(data)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    import awkward as ak

    if isinstance(data, (list, tuple)):
        lengths = [len(sub) if isinstance(sub, (list, tuple)) else 1 for sub in data]
        max_len = max(lengths) if lengths else 0
        pad_val = 0 if fill_value is None else fill_value
        padded = []
        for sub in data:
            if isinstance(sub, (list, tuple)):
                padded.append(list(sub) + [pad_val] * (max_len - len(sub)))
            else:
                padded.append([sub] + [pad_val] * (max_len - 1))
        return require_callable(ak.to_numpy)(require_callable(ak.Array)(padded))

    return require_callable(ak.to_numpy)(require_callable(ak.Array)(data))


def from_regular(data: BackendArg) -> BackendArg:
    """Convert regular dense structure to Awkward representation.

    Args:
        data (BackendArg): Input array.

    Returns:
        BackendArg: Awkward array representation.
    """
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("from_regular"):
        try:
            return adapt_backend_module(ak_mod).require_op("from_regular")(data)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return array(data)


def to_numpy(data: BackendArg) -> BackendArg:
    """Convert an Awkward array to a zero-copy or copied host buffer.

    Args:
        data (BackendArg): Awkward array or compatible input.

    Returns:
        BackendArg: Host buffer.
    """
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("to_numpy"):
        try:
            return adapt_backend_module(ak_mod).require_op("to_numpy")(data)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    cand_data: BackendArg | _DunderArrayProtocol = data
    if isinstance(cand_data, _DunderArrayProtocol):
        arr_fn = cand_data.__array__
        return arr_fn()
    import awkward as ak

    return require_callable(ak.to_numpy)(require_callable(ak.Array)(data))


def from_numpy(data: BackendArg) -> BackendArg:
    """Construct Awkward array from host buffer.

    Args:
        data (BackendArg): Host buffer.

    Returns:
        BackendArg: Awkward array.
    """
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("from_numpy"):
        try:
            return adapt_backend_module(ak_mod).require_op("from_numpy")(data)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return array(data)


def to_list(data: BackendArg) -> list[BackendArg]:
    """Convert Awkward array or structure to nested Python list.

    Args:
        data (BackendArg): Awkward array.

    Returns:
        list[BackendArg]: Python nested list.
    """
    if adapt_backend_module(data).has_symbol("to_list"):
        return coerce_operand_list(adapt_backend_module(data).require_op("to_list")())
    if adapt_backend_module(data).has_symbol("tolist"):
        return coerce_operand_list(adapt_backend_module(data).require_op("tolist")())
    if isinstance(data, (list, tuple)):
        return list(data)
    return [data]


def from_iter(data: BackendArg) -> BackendArg:
    """Construct Awkward array from an arbitrary Python iterable.

    Args:
        data (BackendArg): Source iterable.

    Returns:
        BackendArg: Constructed Awkward array.
    """
    ak_mod = _get_ak_module()
    if adapt_backend_module(ak_mod).has_symbol("from_iter"):
        try:
            return adapt_backend_module(ak_mod).require_op("from_iter")(data)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return array(data)


def get_layout(data: BackendArg) -> BackendArg:
    """Retrieve internal layout description of an Awkward structure.

    Args:
        data (BackendArg): Awkward array.

    Returns:
        BackendArg: Layout object or shape layout description.
    """
    if adapt_backend_module(data).has_symbol("layout"):
        return adapt_backend_module(data).require_symbol("layout")
    res_layout: dict[str, BackendArg] = {"shape": ragged_shape(data), "is_ragged": is_ragged(data)}
    return res_layout
