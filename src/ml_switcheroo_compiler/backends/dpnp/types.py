"""DPNP backend tensor creation and array buffer helpers for Intel SYCL architectures."""

from __future__ import annotations

import importlib
import logging
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def _get_dpnp_module() -> EagerValue:
    """Retrieve dpnp module dynamically.

    Returns:
        EagerValue: The dpnp module.
    """
    return importlib.import_module("dpnp")


def _resolve_allocation_kwargs(
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> dict[str, EagerValue]:
    """Build keyword arguments for DPNP SYCL allocation when supported.

    Args:
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): Unified Shared Memory type ('device', 'shared', 'host').
        sycl_queue (EagerValue | None): Optional pre-initialized SYCL queue.

    Returns:
        dict[str, EagerValue]: Keyword arguments suitable for DPNP creation routines.
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
    shape: tuple[int, ...],
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Create a DPNP array of zeros on SYCL target device with USM buffer.

    Args:
        shape (tuple[int, ...]): Shape tuple.
        dtype (str | None): Optional data type name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (EagerValue | None): Optional SYCL execution queue.

    Returns:
        EagerValue: Allocated zero-filled array in USM memory.
    """
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    kwargs["dtype"] = getattr(dpnp_mod, str(dtype)) if dtype and hasattr(dpnp_mod, str(dtype)) else getattr(dpnp_mod, "float32", float)
    try:
        return cast(EagerValue, getattr(dpnp_mod, "zeros")(shape, **kwargs))
    except TypeError:
        return cast(EagerValue, getattr(dpnp_mod, "zeros")(shape, dtype=kwargs.get("dtype", float)))


def ones(
    shape: tuple[int, ...],
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Create a DPNP array of ones on SYCL target device with USM buffer.

    Args:
        shape (tuple[int, ...]): Shape tuple.
        dtype (str | None): Optional data type name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (EagerValue | None): Optional SYCL execution queue.

    Returns:
        EagerValue: Allocated one-filled array in USM memory.
    """
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    kwargs["dtype"] = getattr(dpnp_mod, str(dtype)) if dtype and hasattr(dpnp_mod, str(dtype)) else getattr(dpnp_mod, "float32", float)
    try:
        return cast(EagerValue, getattr(dpnp_mod, "ones")(shape, **kwargs))
    except TypeError:
        return cast(EagerValue, getattr(dpnp_mod, "ones")(shape, dtype=kwargs.get("dtype", float)))


def empty(
    shape: tuple[int, ...],
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Allocate an uninitialized DPNP array on SYCL device.

    Args:
        shape (tuple[int, ...]): Shape tuple.
        dtype (str | None): Optional data type name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (EagerValue | None): Optional SYCL execution queue.

    Returns:
        EagerValue: Uninitialized array buffer in USM memory.
    """
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    kwargs["dtype"] = getattr(dpnp_mod, str(dtype)) if dtype and hasattr(dpnp_mod, str(dtype)) else getattr(dpnp_mod, "float32", float)
    try:
        return cast(EagerValue, getattr(dpnp_mod, "empty")(shape, **kwargs))
    except TypeError:
        return cast(EagerValue, getattr(dpnp_mod, "empty")(shape, dtype=kwargs.get("dtype", float)))


def full(
    shape: tuple[int, ...],
    fill_value: EagerValue = 0.0,
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Create a DPNP array filled with scalar value on SYCL target device.

    Args:
        shape (tuple[int, ...]): Shape tuple.
        fill_value (EagerValue): Fill scalar value.
        dtype (str | None): Optional data type name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (EagerValue | None): Optional SYCL execution queue.

    Returns:
        EagerValue: Value-filled array in USM memory.
    """
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    if dtype is not None and hasattr(dpnp_mod, str(dtype)):
        kwargs["dtype"] = getattr(dpnp_mod, str(dtype))
    try:
        return cast(EagerValue, getattr(dpnp_mod, "full")(shape, fill_value, **kwargs))
    except TypeError:
        return cast(EagerValue, getattr(dpnp_mod, "full")(shape, fill_value, dtype=kwargs.get("dtype", None)))


def array(
    data: EagerValue,
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Create a DPNP array from data sequence in SYCL USM memory.

    Args:
        data (EagerValue): Data sequence or buffer.
        dtype (str | None): Target datatype name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (EagerValue | None): Optional SYCL execution queue.

    Returns:
        EagerValue: Constructed DPNP array in USM buffer.
    """
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    if dtype is not None and hasattr(dpnp_mod, str(dtype)):
        kwargs["dtype"] = getattr(dpnp_mod, str(dtype))
    try:
        return cast(EagerValue, getattr(dpnp_mod, "array")(data, **kwargs))
    except TypeError:
        return cast(EagerValue, getattr(dpnp_mod, "array")(data, dtype=kwargs.get("dtype", None)))


def asarray(
    data: EagerValue,
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Convert input sequence or array to DPNP USM format.

    Args:
        data (EagerValue): Data sequence or buffer.
        dtype (str | None): Target datatype name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (EagerValue | None): Optional SYCL execution queue.

    Returns:
        EagerValue: DPNP array representation.
    """
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    if dtype is not None and hasattr(dpnp_mod, str(dtype)):
        kwargs["dtype"] = getattr(dpnp_mod, str(dtype))
    try:
        return cast(EagerValue, getattr(dpnp_mod, "asarray")(data, **kwargs))
    except TypeError:
        return cast(EagerValue, getattr(dpnp_mod, "asarray")(data, dtype=kwargs.get("dtype", None)))


def asnumpy(data: EagerValue) -> EagerValue:
    """Transfer DPNP USM array back to host ndarray.

    Args:
        data (EagerValue): Target array buffer.

    Returns:
        EagerValue: Host ndarray buffer.
    """
    dpnp_mod = _get_dpnp_module()
    if hasattr(dpnp_mod, "asnumpy"):
        try:
            return cast(EagerValue, getattr(dpnp_mod, "asnumpy")(data))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    if hasattr(data, "asnumpy"):
        return cast(EagerValue, getattr(data, "asnumpy")())

    return cast(EagerValue, getattr(dpnp_mod, "asarray")(data))


def to_numpy(data: EagerValue) -> EagerValue:
    """Convert DPNP array to host array (alias for asnumpy).

    Args:
        data (EagerValue): Target array buffer.

    Returns:
        EagerValue: Host ndarray buffer.
    """
    return asnumpy(data)


def from_numpy(
    data: EagerValue,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: EagerValue | None = None,
) -> EagerValue:
    """Transfer host array to DPNP SYCL USM buffer.

    Args:
        data (EagerValue): Source host array.
        device (str): Target SYCL device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): Target USM type ('device', 'shared', 'host').
        sycl_queue (EagerValue | None): Optional target SYCL queue.

    Returns:
        EagerValue: DPNP array residing in target SYCL USM memory.
    """
    return array(data, device=device, usm_type=usm_type, sycl_queue=sycl_queue)


def get_device(data: EagerValue) -> str:
    """Inspect the device of a DPNP array buffer.

    Args:
        data (EagerValue): Array buffer.

    Returns:
        str: Device identifier string ('cpu', 'gpu', 'sycl', etc.).
    """
    if hasattr(data, "device"):
        return str(getattr(data, "device"))
    if hasattr(data, "sycl_device"):
        return str(getattr(data, "sycl_device"))
    return "cpu"


def get_usm_type(data: EagerValue) -> str | None:
    """Inspect the USM allocation type of a DPNP array buffer.

    Args:
        data (EagerValue): Array buffer.

    Returns:
        str | None: USM type ('device', 'shared', 'host') or None.
    """
    if hasattr(data, "usm_type"):
        return str(getattr(data, "usm_type"))
    return None


def is_sycl_array(data: EagerValue) -> bool:
    """Determine whether an object is a DPNP/SYCL array buffer.

    Args:
        data (EagerValue): Array candidate.

    Returns:
        bool: True if buffer exhibits DPNP/SYCL attributes, False otherwise.
    """
    return bool(hasattr(data, "usm_type") or hasattr(data, "sycl_queue") or hasattr(data, "sycl_device"))


def item(data: EagerValue) -> float:
    """Extract scalar item value from single-element tensor.

    Args:
        data (EagerValue): Target data.

    Returns:
        float: Extracted scalar value.
    """
    if hasattr(data, "item"):
        return float(getattr(data, "item")())
    dpnp_mod = _get_dpnp_module()
    return float(getattr(dpnp_mod, "asarray")(data).item())
