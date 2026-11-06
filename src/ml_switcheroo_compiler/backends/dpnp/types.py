"""DPNP backend tensor creation and array buffer helpers for Intel SYCL architectures."""

from __future__ import annotations

import importlib
import logging
import typing

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerValue,
    ensure_callable,
    ensure_eager_tensor,
    ensure_float,
)
from ml_switcheroo_compiler.backends.protocols import get_backend_member, has_backend_member


@typing.runtime_checkable
class DpnpArrayProtocol(typing.Protocol):
    """Protocol for DPNP/SYCL USM array buffers."""

    @property
    def usm_type(self) -> str:
        """Return USM allocation type ('device', 'shared', 'host').

        Returns:
            str: USM allocation type.
        """
        ...

    @property
    def sycl_device(self) -> EagerValue:
        """Return SYCL device object or identifier.

        Returns:
            EagerValue: SYCL device.
        """
        ...

    @property
    def sycl_queue(self) -> EagerValue:
        """Return SYCL execution queue.

        Returns:
            EagerValue: SYCL queue.
        """
        ...

    def asnumpy(self) -> EagerValue:
        """Convert USM array to host ndarray.

        Returns:
            EagerValue: Host ndarray.
        """
        ...

    def item(self) -> float | int | bool:
        """Extract scalar value.

        Returns:
            float | int | bool: Scalar value.
        """
        ...


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
    kwargs["dtype"] = get_backend_member(dpnp_mod, str(dtype)) if dtype and has_backend_member(dpnp_mod, str(dtype)) else get_backend_member(dpnp_mod, "float32", float)
    zeros_fn = ensure_callable(get_backend_member(dpnp_mod, "zeros"))
    try:
        return ensure_eager_tensor(zeros_fn(shape, **kwargs))
    except TypeError:
        return ensure_eager_tensor(zeros_fn(shape, dtype=kwargs.get("dtype", float)))


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
    kwargs["dtype"] = get_backend_member(dpnp_mod, str(dtype)) if dtype and has_backend_member(dpnp_mod, str(dtype)) else get_backend_member(dpnp_mod, "float32", float)
    ones_fn = ensure_callable(get_backend_member(dpnp_mod, "ones"))
    try:
        return ensure_eager_tensor(ones_fn(shape, **kwargs))
    except TypeError:
        return ensure_eager_tensor(ones_fn(shape, dtype=kwargs.get("dtype", float)))


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
    kwargs["dtype"] = get_backend_member(dpnp_mod, str(dtype)) if dtype and has_backend_member(dpnp_mod, str(dtype)) else get_backend_member(dpnp_mod, "float32", float)
    empty_fn = ensure_callable(get_backend_member(dpnp_mod, "empty"))
    try:
        return ensure_eager_tensor(empty_fn(shape, **kwargs))
    except TypeError:
        return ensure_eager_tensor(empty_fn(shape, dtype=kwargs.get("dtype", float)))


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
    if dtype is not None and has_backend_member(dpnp_mod, str(dtype)):
        kwargs["dtype"] = get_backend_member(dpnp_mod, str(dtype))
    full_fn = ensure_callable(get_backend_member(dpnp_mod, "full"))
    try:
        return ensure_eager_tensor(full_fn(shape, fill_value, **kwargs))
    except TypeError:
        return ensure_eager_tensor(full_fn(shape, fill_value, dtype=kwargs.get("dtype", None)))


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
    if dtype is not None and has_backend_member(dpnp_mod, str(dtype)):
        kwargs["dtype"] = get_backend_member(dpnp_mod, str(dtype))
    array_fn = ensure_callable(get_backend_member(dpnp_mod, "array"))
    try:
        return ensure_eager_tensor(array_fn(data, **kwargs))
    except TypeError:
        return ensure_eager_tensor(array_fn(data, dtype=kwargs.get("dtype", None)))


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
    if dtype is not None and has_backend_member(dpnp_mod, str(dtype)):
        kwargs["dtype"] = get_backend_member(dpnp_mod, str(dtype))
    asarray_fn = ensure_callable(get_backend_member(dpnp_mod, "asarray"))
    try:
        return ensure_eager_tensor(asarray_fn(data, **kwargs))
    except TypeError:
        return ensure_eager_tensor(asarray_fn(data, dtype=kwargs.get("dtype", None)))


def asnumpy(data: EagerValue) -> EagerValue:
    """Transfer DPNP USM array back to host ndarray.

    Args:
        data (EagerValue): Target array buffer.

    Returns:
        EagerValue: Host ndarray buffer.
    """
    dpnp_mod = _get_dpnp_module()
    if has_backend_member(dpnp_mod, "asnumpy"):
        try:
            return ensure_eager_tensor(ensure_callable(get_backend_member(dpnp_mod, "asnumpy"))(data))
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    raw_data: EagerValue = data
    if isinstance(raw_data, DpnpArrayProtocol) or has_backend_member(data, "asnumpy"):
        return ensure_eager_tensor(ensure_callable(get_backend_member(data, "asnumpy"))())

    return ensure_eager_tensor(ensure_callable(get_backend_member(dpnp_mod, "asarray"))(data))


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
    if has_backend_member(data, "device"):
        return str(get_backend_member(data, "device"))
    if has_backend_member(data, "sycl_device"):
        return str(get_backend_member(data, "sycl_device"))
    return "cpu"


def get_usm_type(data: EagerValue) -> str | None:
    """Inspect the USM allocation type of a DPNP array buffer.

    Args:
        data (EagerValue): Array buffer.

    Returns:
        str | None: USM type ('device', 'shared', 'host') or None.
    """
    if has_backend_member(data, "usm_type"):
        return str(get_backend_member(data, "usm_type"))
    return None


def is_sycl_array(data: EagerValue) -> bool:
    """Determine whether an object is a DPNP/SYCL array buffer.

    Args:
        data (EagerValue): Array candidate.

    Returns:
        bool: True if buffer exhibits DPNP/SYCL attributes, False otherwise.
    """
    raw_data: EagerValue = data
    return bool(isinstance(raw_data, DpnpArrayProtocol) or has_backend_member(data, "usm_type") or has_backend_member(data, "sycl_queue") or has_backend_member(data, "sycl_device"))


def item(data: EagerValue) -> float:
    """Extract scalar item value from single-element tensor.

    Args:
        data (EagerValue): Target data.

    Returns:
        float: Extracted scalar value.
    """
    if has_backend_member(data, "item"):
        return float(ensure_float(ensure_callable(get_backend_member(data, "item"))()))
    dpnp_mod = _get_dpnp_module()
    arr = ensure_eager_tensor(ensure_callable(get_backend_member(dpnp_mod, "asarray"))(data))
    return float(ensure_float(arr.item()))
