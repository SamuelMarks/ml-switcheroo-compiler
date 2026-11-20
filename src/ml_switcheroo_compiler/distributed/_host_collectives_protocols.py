"""Protocols and low-level buffer helpers for host collective communication."""

from __future__ import annotations

import typing
from collections.abc import Sequence
from types import ModuleType

from ml_switcheroo_compiler.core.shape import (
    OpValue,
    ShapeDim,
)

TensorLike = OpValue


def _hc_to_int(val: OpValue, default: int = 0) -> int:
    """Coerce a parameter value to int.

    Args:
        val (OpValue): Raw parameter value.
        default (int): Default fallback integer.

    Returns:
        int: Coerced integer value.
    """
    return int(val) if isinstance(val, (int, float)) else default


def _hc_extract_shape(seq: Sequence[ShapeDim]) -> tuple[int, ...]:
    """Convert a sequence of dimensions into a concrete integer shape tuple.

    Args:
        seq (Sequence[ShapeDim]): Shape sequence.

    Returns:
        tuple[int, ...]: Concrete integer shape tuple.
    """
    return tuple(int(d) for d in seq if isinstance(d, int))


def _hc_opt_str(val: OpValue) -> str | None:
    """Coerce an optional dtype or string value to `str | None`.

    Args:
        val (OpValue): Raw dtype or string value.

    Returns:
        str | None: String representation or None.
    """
    return str(val) if val is not None else None


@typing.runtime_checkable
class _CopyableProtocol(typing.Protocol):
    """Protocol for tensor buffers supporting `.copy()`."""

    def copy(self) -> TensorLike:
        """Return shallow copy of tensor buffer.

        Returns:
            TensorLike: Copied tensor buffer.
        """
        ...


@typing.runtime_checkable
class _HasDTypeAttrProtocol(typing.Protocol):
    """Protocol for tensor buffers exposing `.dtype`."""

    @property
    def dtype(self) -> OpValue:
        """Return buffer dtype.

        Returns:
            OpValue: Buffer dtype.
        """
        ...


@typing.runtime_checkable
class _NamedBackendObjProtocol(typing.Protocol):
    """Protocol for backend objects exposing `.name`."""

    @property
    def name(self) -> str:
        """Return backend name.

        Returns:
            str: Backend name.
        """
        ...


@typing.runtime_checkable
class _SizedDeviceBufferProtocol(typing.Protocol):
    """Protocol for device-transferable buffers exposing ``nbytes`` and ``size``."""

    @property
    def nbytes(self) -> int:
        """Return buffer size in bytes.

        Returns:
            int: Byte count.
        """
        ...

    @property
    def size(self) -> int:
        """Return element count.

        Returns:
            int: Element count.
        """
        ...


def _buffer_nbytes_and_size(buf: TensorLike | _SizedDeviceBufferProtocol) -> tuple[int, int]:
    """Extract ``(nbytes, size)`` from a device-transferable tensor buffer.

    Args:
        buf (TensorLike | _SizedDeviceBufferProtocol): Candidate tensor buffer.

    Returns:
        tuple[int, int]: ``(nbytes, size)`` integer pair.
    """
    assert isinstance(buf, _SizedDeviceBufferProtocol) and not isinstance(buf, ModuleType)
    return int(buf.nbytes), int(buf.size)


@typing.runtime_checkable
class _CollectiveDriverProtocol(typing.Protocol):
    """Protocol for hardware NCCL/RCCL collective drivers."""

    def is_available(self) -> bool:
        """Return True if driver is available.

        Returns:
            bool: Availability flag.
        """
        ...

    def allocate_buffer(self, size_bytes: int, /) -> int:
        """Allocate device memory buffer.

        Args:
            size_bytes (int): Buffer size in bytes.

        Returns:
            int: Device pointer.
        """
        ...

    def free_buffer(self, ptr: int, /) -> None:
        """Free device memory buffer.

        Args:
            ptr (int): Device pointer.
        """
        ...

    def copy_host_to_device(self, host_array: TensorLike, dev_ptr: int, /) -> None:
        """Copy host array to device pointer.

        Args:
            host_array (TensorLike): Source host array.
            dev_ptr (int): Destination device pointer.
        """
        ...

    def copy_device_to_host(self, dev_ptr: int, out_array: TensorLike, /) -> None:
        """Copy device pointer to host array.

        Args:
            dev_ptr (int): Source device pointer.
            out_array (TensorLike): Destination host array.
        """
        ...

    def stream_synchronize(self, stream: int | None = None) -> int | None:
        """Synchronize device stream.

        Args:
            stream (int | None): Stream pointer.

        Returns:
            int | None: Status code if returned by driver.
        """
        ...

    def all_reduce(
        self,
        sendbuff: int,
        recvbuff: int,
        /,
        count: int,
        datatype: int = ...,
        op: int = ...,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int | None:
        """Execute all_reduce on device.

        Args:
            sendbuff (int): Send buffer pointer.
            recvbuff (int): Receive buffer pointer.
            count (int): Element count.
            datatype (int): Data type code.
            op (int): Reduction operator code.
            comm (int | None): Communicator pointer.
            stream (int | None): Stream pointer.

        Returns:
            int | None: Status code if returned by driver.
        """
        ...

    def all_gather(
        self,
        sendbuff: int,
        recvbuff: int,
        /,
        sendcount: int,
        datatype: int = ...,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int | None:
        """Execute all_gather on device.

        Args:
            sendbuff (int): Send buffer pointer.
            recvbuff (int): Receive buffer pointer.
            sendcount (int): Send element count.
            datatype (int): Data type code.
            comm (int | None): Communicator pointer.
            stream (int | None): Stream pointer.

        Returns:
            int | None: Status code if returned by driver.
        """
        ...

    def reduce_scatter(
        self,
        sendbuff: int,
        recvbuff: int,
        /,
        recvcount: int,
        datatype: int = ...,
        op: int = ...,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int | None:
        """Execute reduce_scatter on device.

        Args:
            sendbuff (int): Send buffer pointer.
            recvbuff (int): Receive buffer pointer.
            recvcount (int): Receive element count.
            datatype (int): Data type code.
            op (int): Reduction operator code.
            comm (int | None): Communicator pointer.
            stream (int | None): Stream pointer.

        Returns:
            int | None: Status code if returned by driver.
        """
        ...

    def broadcast(
        self,
        sendbuff: int,
        recvbuff: int,
        /,
        count: int,
        datatype: int = ...,
        root: int = 0,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int | None:
        """Execute broadcast on device.

        Args:
            sendbuff (int): Send buffer pointer.
            recvbuff (int): Receive buffer pointer.
            count (int): Element count.
            datatype (int): Data type code.
            root (int): Root rank.
            comm (int | None): Communicator pointer.
            stream (int | None): Stream pointer.

        Returns:
            int | None: Status code if returned by driver.
        """
        ...

    def all_to_all(
        self,
        sendbuff: int,
        recvbuff: int,
        /,
        count: int,
        datatype: int = ...,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int | None:
        """Execute all_to_all on device.

        Args:
            sendbuff (int): Send buffer pointer.
            recvbuff (int): Receive buffer pointer.
            count (int): Element count.
            datatype (int): Data type code.
            comm (int | None): Communicator pointer.
            stream (int | None): Stream pointer.

        Returns:
            int | None: Status code if returned by driver.
        """
        ...


def _copy_if_possible(data: TensorLike) -> TensorLike:
    """Return `data.copy()` if supported, else `data`.

    Args:
        data (TensorLike): Input tensor buffer.

    Returns:
        TensorLike: Copied or original tensor buffer.
    """
    raw: OpValue = data
    return raw.copy() if isinstance(raw, _CopyableProtocol) else data
