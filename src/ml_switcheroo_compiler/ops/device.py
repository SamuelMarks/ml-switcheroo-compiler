from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Device and system operations."""
from typing import Any

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef
from ml_switcheroo_compiler.ops.registry import register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def eval(*args: Any) -> None:
    """Force the evaluation of the given tensors.

    Args:
        *args (object): Positional args.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "eval"):
        backend.eval(*args)
    else:
        # Fallback for backends that evaluate eagerly
        for arg in args:
            getattr(arg, "data", arg)


def synchronize() -> None:
    """Synchronize the default device."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    if hasattr(backend, "synchronize"):
        backend.synchronize()


def get_peak_memory() -> int:
    """Get the peak memory usage of the device in bytes.

    Returns:
        int: Result.
    """
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

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(args[0], "shape", ())


@register_op("DeviceTransfer")
class DeviceTransferOp(OpDef):
    """Operation definition for migrating data between logical devices (e.g., host-to-device)."""

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(args[0], "shape", ())


def device_transfer(input: Tensor, target_device: str, stream: Any = None) -> Any:
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
