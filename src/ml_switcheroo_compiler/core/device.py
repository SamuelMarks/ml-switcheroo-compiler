# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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
        """Return the string representation of the Device."""
        if isinstance(self.device_type, str):
            return f"Device({self.device_type}:{self.index})"
        """Return the string representation of the Device.

        Returns:
            str: The device representation (e.g. Device(cpu:0)).
        """
        return f"Device({self.device_type.value}:{self.index})"


class Stream:
    """Provide a stream for running operations on a given device."""

    def __init__(self, device: "Device | None" = None) -> None:
        """Initialize Stream.

        Args:
            device (Device | None): The device
        """
        self.device = device


class StreamContext:
    """Provide a context manager for setting the current device and stream."""

    def __init__(self, stream: "Stream") -> None:
        """Initialize StreamContext.

        Args:
            stream (Stream): The stream
        """
        self.stream = stream

    def __enter__(self) -> "StreamContext":
        """Enter the context manager.

        Returns:
            StreamContext: The context manager instance.
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

    backend: object = registry.get_active_backend()
    if hasattr(backend, "clear_cache"):
        backend.clear_cache()
    else:
        warnings.warn(f"Active backend '{getattr(backend, '__name__', type(backend).__name__)}' does not support clear_cache()", stacklevel=2)


class FunctionExporter:
    """Provide a context managing class for exporting multiple traces of the same function."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize FunctionExporter.

        Args:
            args (object): args
            kwargs (object): kwargs
        """
        self.args = args
        self.kwargs = kwargs

    def __enter__(self) -> "FunctionExporter":
        """Enter the context manager.

        Returns:
            FunctionExporter: The context manager instance.
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
        args (object): Positional arguments.
        kwargs (object): Keyword arguments.

    Raises:
        BackendNotSupportedError: If the backend does not support exporting.
    """
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend: object = registry.get_active_backend()
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


def get_logical_devices(device_type: object = None) -> list[Device]:
    """Get logical devices for the current backend.

    Args:
        device_type (str): The type of device to filter by.

    Returns:
        list[Device]: A list of logical devices.

    Raises:
        BackendNotSupportedError: If the backend does not support this.
    """
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend: object = registry.get_active_backend()
    if hasattr(backend, "get_logical_devices"):
        return backend.get_logical_devices(device_type)
    raise BackendNotSupportedError(f"Active backend '{getattr(backend, '__name__', type(backend).__name__)}' does not support get_logical_devices()")


def get_physical_devices(device_type: object = None) -> list[Device]:
    """Get physical devices for the current backend.

    Args:
        device_type (str): The type of device to filter by.

    Returns:
        list[Device]: A list of physical devices.

    Raises:
        BackendNotSupportedError: If the backend does not support this.
    """
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend: object = registry.get_active_backend()
    if hasattr(backend, "get_physical_devices"):
        return backend.get_physical_devices(device_type)
    raise BackendNotSupportedError(f"Active backend '{getattr(backend, '__name__', type(backend).__name__)}' does not support get_physical_devices()")


def get_memory_info(device: object = None) -> dict[str, int]:
    """Get memory statistics tracking (allocation bytes, peak usage).

    Args:
        device (str): The device to get info for.

    Returns:
        dict[str, int]: Memory statistics dictionary.

    Raises:
        BackendNotSupportedError: If the backend does not support memory info.
    """
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend: object = registry.get_active_backend()
    if hasattr(backend, "get_memory_info"):
        return backend.get_memory_info(device)
    raise BackendNotSupportedError(f"Active backend '{getattr(backend, '__name__', type(backend).__name__)}' does not support get_memory_info()")
