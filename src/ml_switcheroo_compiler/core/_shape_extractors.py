"""Runtime metadata extractors and structural member checks for Shape Layer 2."""

from __future__ import annotations

import typing
from types import ModuleType

from ml_switcheroo_ir import DType as IRDType
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core._shape_protocols import (
    BackendBufferProtocol,
    DataHolderProtocol,
    DeviceHolderProtocol,
    DTypeHolderProtocol,
    ShapeMetadataProtocol,
    ShapeProtocol,
    ShapeTuple,
    TracedDataProtocol,
    VariadicCallableProtocol,
    _IdMemberProtocol,
    _ItemMemberProtocol,
    _SizeMemberProtocol,
    _TolistMemberProtocol,
)
from ml_switcheroo_compiler.core._shape_types import OpValue
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType


def _has_shape_attr(val: OpValue) -> typing.TypeGuard[ShapeProtocol]:
    """Check whether `val` exposes a `shape` attribute.

    Args:
        val (OpValue): Candidate object.

    Returns:
        bool: True if `val` satisfies `ShapeProtocol`.
    """
    return isinstance(val, ShapeProtocol)


def _has_shape_metadata_attr(val: OpValue) -> typing.TypeGuard[ShapeMetadataProtocol]:
    """Check whether `val` exposes a `shape_metadata` attribute.

    Args:
        val (OpValue): Candidate object.

    Returns:
        bool: True if `val` satisfies `ShapeMetadataProtocol`.
    """
    return isinstance(val, ShapeMetadataProtocol)


def _has_dtype_holder_attr(val: OpValue) -> typing.TypeGuard[DTypeHolderProtocol]:
    """Check whether `val` exposes a `dtype` attribute.

    Args:
        val (OpValue): Candidate object.

    Returns:
        bool: True if `val` satisfies `DTypeHolderProtocol`.
    """
    return isinstance(val, DTypeHolderProtocol)


def _has_device_holder_attr(val: OpValue) -> typing.TypeGuard[DeviceHolderProtocol]:
    """Check whether `val` exposes a `device` attribute.

    Args:
        val (OpValue): Candidate object.

    Returns:
        bool: True if `val` satisfies `DeviceHolderProtocol`.
    """
    return isinstance(val, DeviceHolderProtocol)


def _has_data_holder_attr(val: OpValue) -> typing.TypeGuard[DataHolderProtocol]:
    """Check whether `val` exposes a `data` attribute.

    Args:
        val (OpValue): Candidate object.

    Returns:
        bool: True if `val` satisfies `DataHolderProtocol`.
    """
    return isinstance(val, DataHolderProtocol)


def _is_logical_node(val: OpValue) -> typing.TypeGuard[ShapeMetadataProtocol]:
    """Check whether `val` is a `LogicalNode` without negative union narrowing.

    Args:
        val (OpValue): Candidate value.

    Returns:
        bool: True if `val` is a `LogicalNode`.
    """
    return isinstance(val, LogicalNode)


def extract_symbolic_shape(
    val: OpValue,
    default: ShapeTuple = (),
) -> ShapeTuple:
    """Safely extract `.shape` or `.shape_metadata` from `val`, preserving symbolic/dynamic dimensions.

    Args:
        val (OpValue): Tensor, IRNode, or shape-bearing object.
        default (ShapeTuple): Default shape tuple if `val` has no shape.

    Returns:
        ShapeTuple: Extracted symbolic shape tuple.
    """
    if _is_logical_node(val):
        return tuple(val.shape_metadata) if val.shape_metadata is not None else default
    if _has_shape_attr(val) and val.shape is not None:
        return tuple(val.shape)
    if _has_shape_metadata_attr(val) and val.shape_metadata is not None:
        return tuple(val.shape_metadata)
    return default


def extract_shape(
    val: OpValue,
    default: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """Extract `.shape` or `.shape_metadata` coerced to concrete integers (with non-int dims as ``-1``).

    Args:
        val (OpValue): Tensor, IRNode, or shape-bearing object.
        default (tuple[int, ...]): Default concrete shape tuple if ``val`` has no shape.

    Returns:
        tuple[int, ...]: Concrete integer shape tuple.
    """
    if _is_logical_node(val):
        if val.shape_metadata is None:
            return default
        return tuple(int(d) if isinstance(d, int) else -1 for d in val.shape_metadata)
    if _has_shape_attr(val) and val.shape is not None:
        return tuple(int(d) if isinstance(d, int) else -1 for d in val.shape)
    if _has_shape_metadata_attr(val) and val.shape_metadata is not None:
        return tuple(int(d) if isinstance(d, int) else -1 for d in val.shape_metadata)
    return default


extract_int_shape = extract_shape


def extract_dtype(
    val: OpValue,
    default: DType | str = "float32",
) -> DType | str:
    """Safely extract `.dtype` from `val`, falling back to `default`.

    Args:
        val (OpValue): Tensor or array object.
        default (DType | str): Fallback dtype if `val` has no `.dtype`.

    Returns:
        DType | str: Extracted dtype or fallback.
    """
    if _has_dtype_holder_attr(val) and val.dtype is not None:
        raw_dt: OpValue = val.dtype
        if isinstance(raw_dt, IRDType):
            return raw_dt.value
        return raw_dt if isinstance(raw_dt, (DType, str)) else str(raw_dt)
    return default


def extract_device(
    val: OpValue,
    default: Device | str | None = None,
) -> Device | str | None:
    """Safely extract `.device` from `val`, falling back to `default`.

    Args:
        val (OpValue): Tensor or array object.
        default (Device | str | None): Fallback device if `val` has no `.device`.

    Returns:
        Device | str | None: Extracted device or fallback.
    """
    if _has_device_holder_attr(val) and val.device is not None:
        dev: Device | str = val.device
        return dev
    return default


def extract_data(val: OpValue) -> OpValue:
    """Safely unwrap `.data` from `val` if present, otherwise return `val`.

    Args:
        val (OpValue): Tensor or raw value.

    Returns:
        OpValue: Unwrapped `.data` or `val`.
    """
    if _has_data_holder_attr(val):
        raw_d: OpValue = val.data
        return raw_d
    return val


def is_traced_data(val: OpValue) -> typing.TypeGuard[TracedDataProtocol]:
    """Check whether `val` satisfies `TracedDataProtocol` or exposes `.id`.

    Args:
        val (OpValue): Candidate runtime object.

    Returns:
        bool: True if `val` is a traced proxy data object.
    """
    return isinstance(val, _IdMemberProtocol)


def is_backend_buffer(val: OpValue) -> typing.TypeGuard[BackendBufferProtocol]:
    """Check whether `val` satisfies `BackendBufferProtocol`.

    Args:
        val (OpValue): Candidate runtime buffer.

    Returns:
        bool: True if `val` is a backend array buffer.
    """
    return isinstance(val, (_TolistMemberProtocol, _ItemMemberProtocol, _SizeMemberProtocol))


def is_variadic_callable(val: OpValue) -> typing.TypeGuard[VariadicCallableProtocol]:
    """Verify at runtime that `val` satisfies `VariadicCallableProtocol` and is not a plain module.

    Args:
        val (OpValue): Candidate runtime object.

    Returns:
        bool: True if `val` implements `VariadicCallableProtocol` and is not a plain `ModuleType`.
    """
    return isinstance(val, VariadicCallableProtocol) and type(val) is not ModuleType
