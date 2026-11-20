"""DPNP backend tensor creation and array buffer helpers for Intel SYCL architectures."""

from __future__ import annotations

import importlib
import logging
import typing

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_float,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


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
    def sycl_device(self) -> BackendArg:
        """Return SYCL device object or identifier.

        Returns:
            BackendArg: SYCL device.
        """
        ...

    @property
    def sycl_queue(self) -> BackendArg:
        """Return SYCL execution queue.

        Returns:
            BackendArg: SYCL queue.
        """
        ...

    def asnumpy(self) -> BackendArg:
        """Convert USM array to host ndarray.

        Returns:
            BackendArg: Host ndarray.
        """
        ...

    def item(self) -> float | int | bool:
        """Extract scalar value.

        Returns:
            float | int | bool: Scalar value.
        """
        ...


def _get_dpnp_module() -> BackendArg:
    """Retrieve dpnp module dynamically.

    Returns:
        BackendArg: The dpnp module.
    """
    return importlib.import_module("dpnp")


def _resolve_allocation_kwargs(
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: BackendArg | None = None,
) -> dict[str, BackendArg]:
    """Build keyword arguments for DPNP SYCL allocation when supported.

    Args:
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): Unified Shared Memory type ('device', 'shared', 'host').
        sycl_queue (BackendArg | None): Optional pre-initialized SYCL queue.

    Returns:
        dict[str, BackendArg]: Keyword arguments suitable for DPNP creation routines.
    """
    kwargs: dict[str, BackendArg] = {}
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
    sycl_queue: BackendArg | None = None,
) -> BackendArg:
    """Create a DPNP array of zeros on SYCL target device with USM buffer.

    Args:
        shape (tuple[int, ...]): Shape tuple.
        dtype (str | None): Optional data type name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (BackendArg | None): Optional SYCL execution queue.

    Returns:
        BackendArg: Allocated zero-filled array in USM memory.
    """
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    kwargs["dtype"] = (
        adapt_backend_module(dpnp_mod).require_symbol(str(dtype))
        if dtype and adapt_backend_module(dpnp_mod).has_symbol(str(dtype))
        else adapt_backend_module(dpnp_mod).get_symbol("float32", float)
    )
    zeros_fn = adapt_backend_module(dpnp_mod).require_op("zeros")
    try:
        return zeros_fn(shape, **kwargs)
    except TypeError:
        return zeros_fn(shape, dtype=kwargs.get("dtype", float))


def ones(
    shape: tuple[int, ...],
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: BackendArg | None = None,
) -> BackendArg:
    """Create a DPNP array of ones on SYCL target device with USM buffer.

    Args:
        shape (tuple[int, ...]): Shape tuple.
        dtype (str | None): Optional data type name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (BackendArg | None): Optional SYCL execution queue.

    Returns:
        BackendArg: Allocated one-filled array in USM memory.
    """
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    kwargs["dtype"] = (
        adapt_backend_module(dpnp_mod).require_symbol(str(dtype))
        if dtype and adapt_backend_module(dpnp_mod).has_symbol(str(dtype))
        else adapt_backend_module(dpnp_mod).get_symbol("float32", float)
    )
    ones_fn = adapt_backend_module(dpnp_mod).require_op("ones")
    try:
        return ones_fn(shape, **kwargs)
    except TypeError:
        return ones_fn(shape, dtype=kwargs.get("dtype", float))


def empty(
    shape: tuple[int, ...],
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: BackendArg | None = None,
) -> BackendArg:
    """Allocate an uninitialized DPNP array on SYCL device.

    Args:
        shape (tuple[int, ...]): Shape tuple.
        dtype (str | None): Optional data type name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (BackendArg | None): Optional SYCL execution queue.

    Returns:
        BackendArg: Uninitialized array buffer in USM memory.
    """
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    kwargs["dtype"] = (
        adapt_backend_module(dpnp_mod).require_symbol(str(dtype))
        if dtype and adapt_backend_module(dpnp_mod).has_symbol(str(dtype))
        else adapt_backend_module(dpnp_mod).get_symbol("float32", float)
    )
    empty_fn = adapt_backend_module(dpnp_mod).require_op("empty")
    try:
        return empty_fn(shape, **kwargs)
    except TypeError:
        return empty_fn(shape, dtype=kwargs.get("dtype", float))


def full(
    shape: tuple[int, ...],
    fill_value: BackendArg = 0.0,
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: BackendArg | None = None,
) -> BackendArg:
    """Create a DPNP array filled with scalar value on SYCL target device.

    Args:
        shape (tuple[int, ...]): Shape tuple.
        fill_value (BackendArg): Fill scalar value.
        dtype (str | None): Optional data type name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (BackendArg | None): Optional SYCL execution queue.

    Returns:
        BackendArg: Value-filled array in USM memory.
    """
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    if dtype is not None and adapt_backend_module(dpnp_mod).has_symbol(str(dtype)):
        kwargs["dtype"] = adapt_backend_module(dpnp_mod).require_symbol(str(dtype))
    full_fn = adapt_backend_module(dpnp_mod).require_op("full")
    try:
        return full_fn(shape, fill_value, **kwargs)
    except TypeError:
        return full_fn(shape, fill_value, dtype=kwargs.get("dtype", None))


def array(
    data: BackendArg,
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: BackendArg | None = None,
) -> BackendArg:
    """Create a DPNP array from data sequence in SYCL USM memory.

    Args:
        data (BackendArg): Data sequence or buffer.
        dtype (str | None): Target datatype name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (BackendArg | None): Optional SYCL execution queue.

    Returns:
        BackendArg: Constructed DPNP array in USM buffer.
    """
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    if dtype is not None and adapt_backend_module(dpnp_mod).has_symbol(str(dtype)):
        kwargs["dtype"] = adapt_backend_module(dpnp_mod).require_symbol(str(dtype))
    array_fn = adapt_backend_module(dpnp_mod).require_op("array")
    try:
        return array_fn(data, **kwargs)
    except TypeError:
        return array_fn(data, dtype=kwargs.get("dtype", None))


def asarray(
    data: BackendArg,
    dtype: str | None = None,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: BackendArg | None = None,
) -> BackendArg:
    """Convert input sequence or array to DPNP USM format.

    Args:
        data (BackendArg): Data sequence or buffer.
        dtype (str | None): Target datatype name.
        device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): USM memory type ('device', 'shared', 'host').
        sycl_queue (BackendArg | None): Optional SYCL execution queue.

    Returns:
        BackendArg: DPNP array representation.
    """
    dpnp_mod = _get_dpnp_module()
    kwargs = _resolve_allocation_kwargs(device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    if dtype is not None and adapt_backend_module(dpnp_mod).has_symbol(str(dtype)):
        kwargs["dtype"] = adapt_backend_module(dpnp_mod).require_symbol(str(dtype))
    asarray_fn = adapt_backend_module(dpnp_mod).require_op("asarray")
    try:
        return asarray_fn(data, **kwargs)
    except TypeError:
        return asarray_fn(data, dtype=kwargs.get("dtype", None))


def asnumpy(data: BackendArg) -> BackendArg:
    """Transfer DPNP USM array back to host ndarray.

    Args:
        data (BackendArg): Target array buffer.

    Returns:
        BackendArg: Host ndarray buffer.
    """
    dpnp_mod = _get_dpnp_module()
    if adapt_backend_module(dpnp_mod).has_symbol("asnumpy"):
        try:
            return adapt_backend_module(dpnp_mod).require_op("asnumpy")(data)
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    raw_data: BackendArg | DpnpArrayProtocol = data
    if isinstance(raw_data, DpnpArrayProtocol) or adapt_backend_module(data).has_symbol("asnumpy"):
        return adapt_backend_module(data).require_op("asnumpy")()

    return adapt_backend_module(dpnp_mod).require_op("asarray")(data)


def to_numpy(data: BackendArg) -> BackendArg:
    """Convert DPNP array to host array (alias for asnumpy).

    Args:
        data (BackendArg): Target array buffer.

    Returns:
        BackendArg: Host ndarray buffer.
    """
    return asnumpy(data)


def from_numpy(
    data: BackendArg,
    device: str = "auto",
    usm_type: str = "device",
    sycl_queue: BackendArg | None = None,
) -> BackendArg:
    """Transfer host array to DPNP SYCL USM buffer.

    Args:
        data (BackendArg): Source host array.
        device (str): Target SYCL device ('auto', 'cpu', 'gpu', 'fpga').
        usm_type (str): Target USM type ('device', 'shared', 'host').
        sycl_queue (BackendArg | None): Optional target SYCL queue.

    Returns:
        BackendArg: DPNP array residing in target SYCL USM memory.
    """
    return array(data, device=device, usm_type=usm_type, sycl_queue=sycl_queue)


def get_device(data: BackendArg) -> str:
    """Inspect the device of a DPNP array buffer.

    Args:
        data (BackendArg): Array buffer.

    Returns:
        str: Device identifier string ('cpu', 'gpu', 'sycl', etc.).
    """
    if adapt_backend_module(data).has_symbol("device"):
        return str(adapt_backend_module(data).require_symbol("device"))
    if adapt_backend_module(data).has_symbol("sycl_device"):
        return str(adapt_backend_module(data).require_symbol("sycl_device"))
    return "cpu"


def get_usm_type(data: BackendArg) -> str | None:
    """Inspect the USM allocation type of a DPNP array buffer.

    Args:
        data (BackendArg): Array buffer.

    Returns:
        str | None: USM type ('device', 'shared', 'host') or None.
    """
    if adapt_backend_module(data).has_symbol("usm_type"):
        return str(adapt_backend_module(data).require_symbol("usm_type"))
    return None


def is_sycl_array(data: BackendArg) -> bool:
    """Determine whether an object is a DPNP/SYCL array buffer.

    Args:
        data (BackendArg): Array candidate.

    Returns:
        bool: True if buffer exhibits DPNP/SYCL attributes, False otherwise.
    """
    raw_data: BackendArg | DpnpArrayProtocol = data
    return bool(
        isinstance(raw_data, DpnpArrayProtocol)
        or adapt_backend_module(data).has_symbol("usm_type")
        or adapt_backend_module(data).has_symbol("sycl_queue")
        or adapt_backend_module(data).has_symbol("sycl_device")
    )


def item(data: BackendArg) -> float:
    """Extract scalar item value from single-element tensor.

    Args:
        data (BackendArg): Target data.

    Returns:
        float: Extracted scalar value.
    """
    if adapt_backend_module(data).has_symbol("item"):
        return float(coerce_float(adapt_backend_module(data).require_op("item")()))
    dpnp_mod = _get_dpnp_module()
    arr = adapt_backend_module(dpnp_mod).require_op("asarray")(data)
    return float(coerce_float(arr.item()))
