"""DPNP backend tensor creation and array buffer helpers for Intel SYCL architectures."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.distributed.host_collectives import _np_mod


def _get_dpnp_module() -> EagerValue:
    """Retrieve dpnp module dynamically with fallback to numpy.

    Returns:
        object: The dpnp or numpy module.
    """
    try:
        return importlib.import_module("dpnp")
    except Exception:
        return _np_mod()


def _resolve_allocation_kwargs(
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> dict[str, EagerValue]:
    """Build keyword arguments for DPNP SYCL allocation when supported.

    Args:
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): Unified Shared Memory type ('device', 'shared', 'host').
        sycl_queue (object | None): Optional pre-initialized SYCL queue.

    Returns:
        dict[str, object]: Keyword arguments suitable for DPNP creation routines.
    """
    kwargs: dict[str, EagerValue] = {}
    if sycl_queue is not None:
        kwargs["sycl_queue"] = sycl_queue
    elif device != "auto":
        kwargs["device"] = device

    if usm_type in ("device", "shared", "host"):
        kwargs["usm_type"] = usm_type
    return kwargs


def zeros(
    shape_or_cls: EagerValue,
    shape: tuple[int, ...] | None = None,
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Create a DPNP array of zeros on SYCL target device with USM buffer.

    Args:
        shape_or_cls (object): Shape tuple or caller class context.
        shape (tuple[int, ...] | None): Optional shape when called as classmethod.
        dtype (str | None): Optional data type name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (object | None): Optional SYCL execution queue.

    Returns:
        object: Allocated zero-filled array in USM memory.
    """
    actual_shape = shape_or_cls if shape is None and isinstance(shape_or_cls, tuple) else (shape or (0,))
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    kwargs["dtype"] = getattr(dpnp_mod, str(dtype)) if dtype and hasattr(dpnp_mod, str(dtype)) else getattr(dpnp_mod, "float32", float)
    try:
        return cast(EagerValue, getattr(dpnp_mod, "zeros")(actual_shape, **kwargs))
    except TypeError:
        # Fallback when dpnp_mod is reference numpy without usm_type/device kwargs
        return cast(EagerValue, getattr(dpnp_mod, "zeros")(actual_shape, dtype=kwargs.get("dtype", float)))


def ones(
    shape_or_cls: EagerValue,
    shape: tuple[int, ...] | None = None,
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Create a DPNP array of ones on SYCL target device with USM buffer.

    Args:
        shape_or_cls (object): Shape tuple or caller class context.
        shape (tuple[int, ...] | None): Optional shape when called as classmethod.
        dtype (str | None): Optional data type name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (object | None): Optional SYCL execution queue.

    Returns:
        object: Allocated one-filled array in USM memory.
    """
    actual_shape = shape_or_cls if shape is None and isinstance(shape_or_cls, tuple) else (shape or (0,))
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    kwargs["dtype"] = getattr(dpnp_mod, str(dtype)) if dtype and hasattr(dpnp_mod, str(dtype)) else getattr(dpnp_mod, "float32", float)
    try:
        return cast(EagerValue, getattr(dpnp_mod, "ones")(actual_shape, **kwargs))
    except TypeError:
        return cast(EagerValue, getattr(dpnp_mod, "ones")(actual_shape, dtype=kwargs.get("dtype", float)))


def empty(
    shape_or_cls: EagerValue,
    shape: tuple[int, ...] | None = None,
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Allocate an uninitialized DPNP array on SYCL device.

    Args:
        shape_or_cls (object): Shape tuple or caller class context.
        shape (tuple[int, ...] | None): Optional shape when called as classmethod.
        dtype (str | None): Optional data type name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (object | None): Optional SYCL execution queue.

    Returns:
        object: Uninitialized array buffer in USM memory.
    """
    actual_shape = shape_or_cls if shape is None and isinstance(shape_or_cls, tuple) else (shape or (0,))
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    kwargs["dtype"] = getattr(dpnp_mod, str(dtype)) if dtype and hasattr(dpnp_mod, str(dtype)) else getattr(dpnp_mod, "float32", float)
    try:
        return cast(EagerValue, getattr(dpnp_mod, "empty")(actual_shape, **kwargs))
    except TypeError:
        return cast(EagerValue, getattr(dpnp_mod, "empty")(actual_shape, dtype=kwargs.get("dtype", float)))


def full(
    shape_or_cls: EagerValue,
    fill_value_or_shape: EagerValue = None,
    fill_value: EagerValue = None,
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Create a DPNP array filled with scalar value on SYCL target device.

    Args:
        shape_or_cls (object): Shape tuple or caller class context.
        fill_value_or_shape (object): Fill value or shape tuple when called via classmethod.
        fill_value (object): Fill scalar value when called via classmethod.
        dtype (str | None): Optional data type name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (object | None): Optional SYCL execution queue.

    Returns:
        object: Value-filled array in USM memory.
    """
    if isinstance(shape_or_cls, type):
        actual_shape = fill_value_or_shape if isinstance(fill_value_or_shape, tuple) else (0,)
        actual_val = fill_value
    else:
        actual_shape = shape_or_cls if isinstance(shape_or_cls, tuple) else (shape_or_cls,)
        actual_val = fill_value_or_shape

    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    if dtype is not None and hasattr(dpnp_mod, str(dtype)):
        kwargs["dtype"] = getattr(dpnp_mod, str(dtype))
    try:
        return cast(EagerValue, getattr(dpnp_mod, "full")(actual_shape, actual_val, **kwargs))
    except TypeError:
        return cast(EagerValue, getattr(dpnp_mod, "full")(actual_shape, actual_val, dtype=kwargs.get("dtype", None)))


def array(
    data_or_cls: EagerValue,
    data: Sequence[float] | None = None,
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Create a DPNP array from data sequence in SYCL USM memory.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (Sequence[float] | None): Optional data sequence when called as classmethod.
        dtype (str | None): Target datatype name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (object | None): Optional SYCL execution queue.

    Returns:
        object: Constructed DPNP array in USM buffer.
    """
    actual_data = data_or_cls if data is None else data
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    if dtype is not None and hasattr(dpnp_mod, str(dtype)):
        kwargs["dtype"] = getattr(dpnp_mod, str(dtype))
    try:
        return cast(EagerValue, getattr(dpnp_mod, "array")(actual_data, **kwargs))
    except TypeError:
        return cast(EagerValue, getattr(dpnp_mod, "array")(actual_data, dtype=kwargs.get("dtype", None)))


def asarray(
    data_or_cls: EagerValue,
    data: Sequence[float] | None = None,
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Convert input sequence or array to DPNP USM format.

    Args:
        data_or_cls (object): Data sequence or class context.
        data (Sequence[float] | None): Optional data sequence when called as classmethod.
        dtype (str | None): Target datatype name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (object | None): Optional SYCL execution queue.

    Returns:
        object: DPNP array representation.
    """
    actual_data = data_or_cls if data is None else data
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    if dtype is not None and hasattr(dpnp_mod, str(dtype)):
        kwargs["dtype"] = getattr(dpnp_mod, str(dtype))
    try:
        return cast(EagerValue, getattr(dpnp_mod, "asarray")(actual_data, **kwargs))
    except TypeError:
        return cast(EagerValue, getattr(dpnp_mod, "asarray")(actual_data, dtype=kwargs.get("dtype", None)))


def asnumpy(data_or_cls: EagerValue, data: EagerValue = None) -> EagerValue:
    """Transfer DPNP USM array back to host NumPy ndarray.

    Args:
        data_or_cls (object): Target array buffer or class context.
        data (object): Target array buffer when called as classmethod.

    Returns:
        object: Host NumPy ndarray buffer.
    """
    actual_data = data_or_cls if data is None else data
    dpnp_mod = _get_dpnp_module()
    if hasattr(dpnp_mod, "asnumpy"):
        try:
            return cast(EagerValue, getattr(dpnp_mod, "asnumpy")(actual_data))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    if hasattr(actual_data, "asnumpy"):
        return cast(EagerValue, getattr(actual_data, "asnumpy")())

    numpy_mod = _np_mod()
    return cast(EagerValue, numpy_mod.asarray(actual_data))


def to_numpy(data_or_cls: EagerValue, data: EagerValue = None) -> EagerValue:
    """Convert DPNP array to host NumPy array (alias for asnumpy).

    Args:
        data_or_cls (object): Target array buffer or class context.
        data (object): Target array buffer when called as classmethod.

    Returns:
        object: Host NumPy ndarray buffer.
    """
    return asnumpy(data_or_cls, data)


def from_numpy(
    data_or_cls: EagerValue,
    data: EagerValue = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Transfer host NumPy array to DPNP SYCL USM buffer.

    Args:
        data_or_cls (object): Source NumPy array or class context.
        data (object): Source NumPy array when called as classmethod.
        device (str): Target SYCL device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): Target USM type ('device', 'shared', 'host').
        sycl_queue (object | None): Optional target SYCL queue.

    Returns:
        object: DPNP array residing in target SYCL USM memory.
    """
    actual_data = data_or_cls if data is None else data
    return array(actual_data, device=device, usm_type=usm_type, sycl_queue=sycl_queue)


def get_device(data_or_cls: EagerValue, data: EagerValue = None) -> str:
    """Inspect the device of a DPNP array buffer.

    Args:
        data_or_cls (object): Array buffer or class context.
        data (object): Array buffer when called as classmethod.

    Returns:
        str: Device identifier string ('cpu', 'gpu', 'sycl', etc.).
    """
    actual_data = data_or_cls if data is None else data
    if hasattr(actual_data, "device"):
        return str(getattr(actual_data, "device"))
    if hasattr(actual_data, "sycl_device"):
        return str(getattr(actual_data, "sycl_device"))
    return "cpu"


def get_usm_type(data_or_cls: EagerValue, data: EagerValue = None) -> str | None:
    """Inspect the USM allocation type of a DPNP array buffer.

    Args:
        data_or_cls (object): Array buffer or class context.
        data (object): Array buffer when called as classmethod.

    Returns:
        str | None: USM type ('device', 'shared', 'host') or None.
    """
    actual_data = data_or_cls if data is None else data
    if hasattr(actual_data, "usm_type"):
        return str(getattr(actual_data, "usm_type"))
    return None


def is_sycl_array(data_or_cls: EagerValue, data: EagerValue = None) -> bool:
    """Determine whether an object is a DPNP/SYCL array buffer.

    Args:
        data_or_cls (object): Array candidate or class context.
        data (object): Array candidate when called as classmethod.

    Returns:
        bool: True if buffer exhibits DPNP/SYCL attributes, False otherwise.
    """
    actual_data = data_or_cls if data is None else data
    return bool(hasattr(actual_data, "usm_type") or hasattr(actual_data, "sycl_queue") or hasattr(actual_data, "sycl_device"))


def item(data_or_cls: EagerValue, data: EagerValue = None) -> float:
    """Extract scalar item value from single-element tensor.

    Args:
        data_or_cls (object): Target data or class context.
        data (object): Target data when called as classmethod.

    Returns:
        float: Extracted scalar value.
    """
    actual_data = data_or_cls if data is None else data
    if hasattr(actual_data, "item"):
        return float(getattr(actual_data, "item")())
    numpy_mod = _np_mod()
    return float(numpy_mod.asarray(actual_data).item())
