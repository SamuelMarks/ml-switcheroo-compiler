"""Device and system operations."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef
from ml_switcheroo_compiler.ops.registry import register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def eval(*args: object) -> None:
    """Forces the evaluation of the given tensors."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "eval"):
        backend.eval(*args)
    else:
        # Fallback for backends that evaluate eagerly
        for arg in args:
            getattr(arg, "data", arg)


def synchronize() -> None:
    """Synchronizes the default device."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "synchronize"):
        backend.synchronize()


def get_peak_memory() -> int:
    """Gets the peak memory usage of the device in bytes."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "get_peak_memory"):
        return backend.get_peak_memory()
    return 0


__all__ = [
    "eval",
    "get_peak_memory",
    "synchronize",
]


@register_op("DeviceContext")
class DeviceContextOp(OpDef):
    """Operation definition for setting or altering a generic device context."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(args[0], "shape", ())


@register_op("DeviceTransfer")
class DeviceTransferOp(OpDef):
    """Operation definition for migrating data between logical devices (e.g., host-to-device)."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(args[0], "shape", ())


def device_transfer(input: Tensor, target_device: str, stream: str = None) -> Tensor:
    """Simulates data migration between devices for a tensor.

    Args:
        input: The tensor to transfer.
        target_device: The target device string representation.
        stream: Optional stream identifier for asynchronous transfers.

    Returns:
        Tensor: The tensor on the new device context.
    """
    attrs = {"target_device": target_device, "stream": stream}
    return _emit_shape_node("DeviceTransfer", [input], attrs, input.shape, input.dtype)
