"""Awkward Array backend type conversions and ragged array creation utilities."""

from __future__ import annotations

import importlib
import logging
import typing
from collections.abc import Mapping, Sequence

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerValue,
    ensure_callable,
    ensure_eager_list,
    ensure_eager_tensor,
    ensure_float,
    ensure_int,
)
from ml_switcheroo_compiler.backends.protocols import get_backend_member, has_backend_member


@typing.runtime_checkable
class _DunderArrayProtocol(typing.Protocol):
    """Protocol for objects exposing `__array__()`."""

    def __array__(self) -> EagerValue:
        """Convert to array buffer.

        Returns:
            EagerValue: Array buffer.
        """
        ...


def _get_ak_module() -> EagerValue:
    """Retrieve awkward module dynamically with fallback.

    Returns:
        EagerValue: The awkward module or None when unavailable.
    """
    try:
        return importlib.import_module("awkward")
    except Exception:
        return None


def _build_nested_zeros(shape: tuple[int, ...] | int) -> EagerValue:
    """Construct nested zero-initialized Python lists for the given shape.

    Args:
        shape (tuple[int, ...] | int): Target shape tuple or integer size.

    Returns:
        EagerValue: Nested list of 0.0 values.
    """
    shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
    if len(shape_tuple) <= 1:
        return [0.0] * (shape_tuple[0] if shape_tuple else 0)
    return [_build_nested_zeros(shape_tuple[1:]) for _ in range(shape_tuple[0])]


def zeros(shape: tuple[int, ...] | int) -> EagerValue:
    """Create an Awkward zero-filled array.

    Args:
        shape (tuple[int, ...] | int): Target shape tuple or integer size.

    Returns:
        EagerValue: Allocated zero-filled Awkward array or nested list fallback.
    """
    nested = _build_nested_zeros(shape)
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "Array"):
        return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "Array"))(nested))
    return nested


def array(data: EagerValue, dtype: EagerValue | None = None) -> EagerValue:
    """Create an Awkward ragged or regular array from data.

    Args:
        data (EagerValue): Data sequence or buffer.
        dtype (str | None): Optional datatype name.

    Returns:
        EagerValue: Constructed Awkward array or sequence fallback.
    """
    del dtype
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "Array"):
        try:
            return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "Array"))(data))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return list(data) if isinstance(data, (list, tuple)) else [data]


def asarray(data: EagerValue) -> EagerValue:
    """Convert input to an Awkward array.

    Args:
        data (EagerValue): Data sequence or buffer.

    Returns:
        EagerValue: Awkward array or list fallback.
    """
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "Array"):
        try:
            return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "Array"))(data))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return list(data) if isinstance(data, (list, tuple)) else [data]


def ragged_array(
    data: EagerValue,
    dtype: str | None = None,
) -> EagerValue:
    """Construct an Awkward array preserving ragged nested dimensions.

    Args:
        data (EagerValue): Nested sequence.
        dtype (str | None): Target primitive data type.

    Returns:
        EagerValue: Ragged Awkward array or list fallback.
    """
    del dtype
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "Array"):
        try:
            return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "Array"))(data))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return list(data) if isinstance(data, (list, tuple)) else [data]


def record_array(
    data: EagerValue,
    with_name: str | None = None,
) -> EagerValue:
    """Construct a nested record or zipped structure.

    Args:
        data (EagerValue): Mapping or sequence of arrays.
        with_name (str | None): Optional record type name.

    Returns:
        EagerValue: Awkward record array.
    """
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "zip") and isinstance(data, Mapping):
        return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "zip"))(dict(data), with_name=with_name))
    if has_backend_member(ak_mod, "Array"):
        return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "Array"))(data))
    return data


def flatten(
    data: EagerValue,
    axis: int = 1,
) -> EagerValue:
    """Flatten a variable-length ragged dimension.

    Args:
        data (EagerValue): Input array.
        axis (int): Axis along which to flatten ragged levels. Defaults to 1.

    Returns:
        EagerValue: Flattened array.
    """
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "flatten"):
        return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "flatten"))(data, axis=axis))
    return ensure_eager_tensor(ensure_callable(get_backend_member(data, "flatten"))())


def unflatten(
    data: EagerValue,
    counts: Sequence[int] | None = None,
    axis: int = 0,
) -> EagerValue:
    """Unflatten a 1D sequence into ragged partitions given counts.

    Args:
        data (EagerValue): Input 1D array.
        counts (Sequence[int] | None): Number of elements in each sub-partition.
        axis (int): Target axis. Defaults to 0.

    Returns:
        EagerValue: Ragged partitioned Awkward array.
    """
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "unflatten") and counts is not None:
        return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "unflatten"))(data, counts, axis=axis))
    return data


def with_field(
    base: EagerValue,
    what: EagerValue,
    where: str = "",
) -> EagerValue:
    """Add or replace a field within a nested record structure.

    Args:
        base (EagerValue): Base record structure.
        what (EagerValue): Field content array to insert.
        where (str): Field identifier name.

    Returns:
        EagerValue: Updated Awkward record array.
    """
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "with_field"):
        return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "with_field"))(base, what, where))
    if isinstance(base, dict):
        res = dict(base)
        res[where] = what
        return res
    return base


def unzip(
    data: EagerValue,
) -> tuple[EagerValue, ...]:
    """Unzip a record array into individual field arrays.

    Args:
        data (EagerValue): Record array.

    Returns:
        tuple[EagerValue, ...]: Tuple of individual field arrays.
    """
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "unzip"):
        unzipped = ensure_callable(get_backend_member(ak_mod, "unzip"))(data)
        return tuple(ensure_eager_list(unzipped))
    if isinstance(data, dict):
        return tuple(data.values())
    return (data,)


def item(data: EagerValue) -> float:
    """Extract scalar item value from an Awkward array.

    Args:
        data (EagerValue): Target data.

    Returns:
        float: Extracted scalar value.
    """
    if has_backend_member(data, "to_list"):
        val: EagerValue = ensure_callable(get_backend_member(data, "to_list"))()
        if isinstance(val, (list, tuple)) and val:
            first = val[0]
            if isinstance(first, (list, tuple)) and first:
                return float(ensure_float(first[0]))
            return float(ensure_float(first))
        return float(ensure_float(val))
    if has_backend_member(data, "item"):
        return float(ensure_float(ensure_callable(get_backend_member(data, "item"))()))
    arr = to_numpy(data)
    return float(ensure_float(ensure_callable(get_backend_member(arr, "item"))()))


def broadcast_ragged(
    a: EagerValue,
    b: EagerValue,
) -> tuple[EagerValue, EagerValue]:
    """Broadcast two arrays across ragged dimensions without padding.

    Args:
        a (EagerValue): First array operand.
        b (EagerValue): Second array operand.

    Returns:
        tuple[EagerValue, EagerValue]: Broadcast pair of Awkward or nested array structures.
    """
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "broadcast_arrays"):
        try:
            res = ensure_eager_list(ensure_callable(get_backend_member(ak_mod, "broadcast_arrays"))(a, b))
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


def ragged_shape(data: EagerValue) -> tuple[int | None, ...]:
    """Infer the shape tuple of a regular or ragged array structure.

    Args:
        data (EagerValue): Array or nested sequence.

    Returns:
        tuple[int | None, ...]: Inferred shape with None for ragged variable dimensions.
    """
    if has_backend_member(data, "shape"):
        raw_sh: EagerValue = get_backend_member(data, "shape")
        return tuple(int(ensure_int(d)) for d in (raw_sh if isinstance(raw_sh, (list, tuple)) else ()))
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


def is_ragged(data: EagerValue) -> bool:
    """Determine whether an array structure contains ragged variable-length dimensions.

    Args:
        data (EagerValue): Array or nested sequence.

    Returns:
        bool: True if structure has ragged inner dimensions, False otherwise.
    """
    shape = ragged_shape(data)
    return any(d is None for d in shape)


def to_regular(
    data: EagerValue,
    fill_value: EagerValue = None,
) -> EagerValue:
    """Convert ragged array to regular dense multidimensional structure.

    Args:
        data (EagerValue): Ragged array structure.
        fill_value (EagerValue): Value to pad missing ragged elements.

    Returns:
        EagerValue: Regular dense array or padded nested list.
    """
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "to_regular"):
        try:
            return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "to_regular"))(data))
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
        return ensure_eager_tensor(ak.to_numpy(ak.Array(padded)))

    return ensure_eager_tensor(ak.to_numpy(ak.Array(data)))


def from_regular(data: EagerValue) -> EagerValue:
    """Convert regular dense structure to Awkward representation.

    Args:
        data (EagerValue): Input array.

    Returns:
        EagerValue: Awkward array representation.
    """
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "from_regular"):
        try:
            return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "from_regular"))(data))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return array(data)


def to_numpy(data: EagerValue) -> EagerValue:
    """Convert Awkward structure to standard host buffer via ak.to_numpy.

    Args:
        data (EagerValue): Awkward array.

    Returns:
        EagerValue: Host buffer converted by Awkward.
    """
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "to_numpy"):
        try:
            return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "to_numpy"))(data))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    import awkward as ak

    return ensure_eager_tensor(ak.to_numpy(ak.Array(data)))


def from_numpy(data: EagerValue) -> EagerValue:
    """Construct Awkward array from host buffer.

    Args:
        data (EagerValue): Host buffer.

    Returns:
        EagerValue: Awkward array.
    """
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "from_numpy"):
        try:
            return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "from_numpy"))(data))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return array(data)


def to_list(data: EagerValue) -> list[EagerValue]:
    """Convert Awkward array or structure to nested Python list.

    Args:
        data (EagerValue): Awkward array.

    Returns:
        list[EagerValue]: Python nested list.
    """
    if has_backend_member(data, "to_list"):
        return ensure_eager_list(ensure_callable(get_backend_member(data, "to_list"))())
    if has_backend_member(data, "tolist"):
        return ensure_eager_list(ensure_callable(get_backend_member(data, "tolist"))())
    if isinstance(data, (list, tuple)):
        return list(data)
    return [data]


def from_iter(data: EagerValue) -> EagerValue:
    """Construct Awkward array from an arbitrary Python iterable.

    Args:
        data (EagerValue): Source iterable.

    Returns:
        EagerValue: Constructed Awkward array.
    """
    ak_mod = _get_ak_module()
    if has_backend_member(ak_mod, "from_iter"):
        try:
            return ensure_eager_tensor(ensure_callable(get_backend_member(ak_mod, "from_iter"))(data))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return array(data)


def get_layout(data: EagerValue) -> EagerValue:
    """Retrieve internal layout description of an Awkward structure.

    Args:
        data (EagerValue): Awkward array.

    Returns:
        EagerValue: Layout object or shape layout description.
    """
    if has_backend_member(data, "layout"):
        return ensure_eager_tensor(get_backend_member(data, "layout"))
    res_layout: dict[str, EagerValue] = {"shape": ragged_shape(data), "is_ragged": is_ragged(data)}
    return res_layout
