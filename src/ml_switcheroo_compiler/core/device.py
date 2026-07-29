"""Device and DeviceType classes for the ml-switcheroo compiler."""

from dataclasses import dataclass
from enum import Enum


class DeviceType(Enum):
    """Enumeration of supported device types."""

    CPU = "cpu"
    GPU = "gpu"
    WEBGPU = "webgpu"


@dataclass(frozen=True)
class Device:
    """Represents a specific hardware device and index."""

    device_type: DeviceType
    index: int = 0

    def __repr__(self) -> str:
        """Return the string representation of the Device.

        Returns:
            str: The evaluated output resulting from this operation.
        """
        return f"Device({self.device_type.value}:{self.index})"


class Stream:
    """A stream for running operations on a given device."""

    def __init__(self, device: "Device | None" = None) -> None:
        """Initialize Stream.

        Args:
            device (Device | None): The device
        """
        self.device = device


class StreamContext:
    """A context manager for setting the current device and stream."""

    def __init__(self, stream: "Stream") -> None:
        """Initialize StreamContext.

        Args:
            stream (Stream): The stream
        """
        self.stream = stream

    def __enter__(self) -> "StreamContext":
        """Enter context.

        Returns:
            StreamContext: The evaluated output resulting from this operation.
        """
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context.

        Args:
            exc_type (object): exception type
            exc_val (object): exception value
            exc_tb (object): exception traceback
        """
        self.stream = None


def clear_cache() -> None:
    """Clear the memory cache."""
    import warnings

    import ml_switcheroo_compiler.backends.registry as registry

    backend = registry.get_active_backend()
    if hasattr(backend, "clear_cache"):
        backend.clear_cache()
    else:
        warnings.warn(f"Active backend '{getattr(backend, '__name__', type(backend).__name__)}' does not support clear_cache()", stacklevel=2)


class FunctionExporter:
    """A context managing class for exporting multiple traces of the same function."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize FunctionExporter.

        Args:
            args (object): args
            kwargs (object): kwargs
        """
        self.args = args
        self.kwargs = kwargs

    def __enter__(self) -> "FunctionExporter":
        """Enter context.

        Returns:
            FunctionExporter: The evaluated output resulting from this operation.
        """
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context.

        Args:
            exc_type (object): exception type
            exc_val (object): exception value
            exc_tb (object): exception traceback
        """
        self.args = ()
        self.kwargs = {}


def export_function(*args: object, **kwargs: object) -> None:
    """Export an MLX function.

    Args:
        args (object): args
        kwargs (object): kwargs
    """
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend = registry.get_active_backend()
    if hasattr(backend, "export_function"):
        backend.export_function(*args, **kwargs)
    else:
        raise BackendNotSupportedError(f"Active backend '{getattr(backend, '__name__', type(backend).__name__)}' does not support export_function()")


def exporter(*args: object, **kwargs: object) -> FunctionExporter:
    """Make a callable object to export multiple traces of a function to a file.

    Args:
        args (object): args
        kwargs (object): kwargs

    Returns:
        FunctionExporter: The exporter
    """
    return FunctionExporter(*args, **kwargs)


def get_logical_devices(device_type: str = None) -> list[Device]:
    """Get logical devices."""
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend = registry.get_active_backend()
    if hasattr(backend, "get_logical_devices"):
        return backend.get_logical_devices(device_type)
    raise BackendNotSupportedError(f"Active backend '{getattr(backend, '__name__', type(backend).__name__)}' does not support get_logical_devices()")


def get_physical_devices(device_type: str = None) -> list[Device]:
    """Get physical devices."""
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend = registry.get_active_backend()
    if hasattr(backend, "get_physical_devices"):
        return backend.get_physical_devices(device_type)
    raise BackendNotSupportedError(f"Active backend '{getattr(backend, '__name__', type(backend).__name__)}' does not support get_physical_devices()")


def get_memory_info(device: str = None) -> dict[str, int]:
    """Get memory statistics tracking (allocation bytes, peak usage)."""
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend = registry.get_active_backend()
    if hasattr(backend, "get_memory_info"):
        return backend.get_memory_info(device)
    raise BackendNotSupportedError(f"Active backend '{getattr(backend, '__name__', type(backend).__name__)}' does not support get_memory_info()")
