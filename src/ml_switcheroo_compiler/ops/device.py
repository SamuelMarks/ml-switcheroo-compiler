"""Device and system operations."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import ensure_int
from ml_switcheroo_compiler.core.shape import TensorOrScalar, extract_data, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef
from ml_switcheroo_compiler.ops.registry import register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


@typing.runtime_checkable
class _BackendEvalProtocol(typing.Protocol):
    """Protocol for backends exposing `.eval(*args)`."""

    def eval(self, *args: TensorOrScalar) -> None:
        """Evaluate lazy tensors.

        Args:
            *args (TensorOrScalar): Tensors to evaluate.
        """
        ...


@typing.runtime_checkable
class _BackendSynchronizeProtocol(typing.Protocol):
    """Protocol for backends exposing `.synchronize()`."""

    def synchronize(self) -> None:
        """Synchronize the active device."""
        ...


@typing.runtime_checkable
class _BackendPeakMemoryProtocol(typing.Protocol):
    """Protocol for backends exposing `.get_peak_memory()`."""

    def get_peak_memory(self) -> int:
        """Return peak device memory in bytes.

        Returns:
            int: Peak memory in bytes.
        """
        ...


def eval(*args: TensorOrScalar) -> None:
    """Force the evaluation of the given tensors.

    Args:
        *args (TensorOrScalar): Positional args.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend: TensorOrScalar | _BackendEvalProtocol = get_active_backend()
    if isinstance(backend, _BackendEvalProtocol):
        backend.eval(*args)
    else:
        # Fallback for backends that evaluate eagerly
        for arg in args:
            extract_data(arg)


def synchronize() -> None:
    """Synchronize the default device."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend: TensorOrScalar | _BackendSynchronizeProtocol = get_active_backend()
    if isinstance(backend, _BackendSynchronizeProtocol):
        backend.synchronize()


def get_peak_memory() -> int:
    """Get the peak memory usage of the device in bytes.

    Returns:
        int: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend: TensorOrScalar | _BackendPeakMemoryProtocol = get_active_backend()
    if isinstance(backend, _BackendPeakMemoryProtocol):
        return ensure_int(backend.get_peak_memory())
    return 0


__all__ = [
    "eval",
    "get_peak_memory",
    "synchronize",
]


@register_op("DeviceContext")
class DeviceContextOp(OpDef):
    """Operation definition for setting or altering a generic device context."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return extract_shape(args[0])


@register_op("DeviceTransfer")
class DeviceTransferOp(OpDef):
    """Operation definition for migrating data between logical devices (e.g., host-to-device)."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return extract_shape(args[0])


def device_transfer(input: Tensor, target_device: str, stream: TensorOrScalar = None) -> TensorOrScalar:
    """Simulate data migration between devices for a tensor.

    Args:
        input: The tensor to transfer.
        target_device: The target device string representation.
        stream: Optional stream identifier for asynchronous transfers.

    Returns:
        Tensor: The tensor on the new device context.
    """
    attrs = {"target_device": target_device, "stream": stream}
    return _emit_shape_node("DeviceTransfer", [input], attrs, input.shape, input.dtype)
