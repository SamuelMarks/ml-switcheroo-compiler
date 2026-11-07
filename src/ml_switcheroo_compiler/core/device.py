"""Device and DeviceType classes for the ml-switcheroo compiler."""

from __future__ import annotations

import string
import typing
from enum import Enum
from types import TracebackType

import pydantic

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.shape import AttrValue, TensorOrScalar

_DEVICE_ATTR_FORMATTER = string.Formatter()


def _has_device_member(obj: TensorOrScalar, attr: str) -> bool:
    """Check whether `obj` exposes `attr` via `string.Formatter().get_field`.

    Args:
        obj (TensorOrScalar): Target object.
        attr (str): Attribute name.

    Returns:
        bool: True if `obj` has `attr`.
    """
    try:
        _DEVICE_ATTR_FORMATTER.get_field(f"0.{attr}", (obj,), {})
        return True
    except Exception:
        return False


class _ClearCacheBackendProtocol(typing.Protocol):
    """Protocol for backends supporting `clear_cache()`."""

    def clear_cache(self) -> None:
        """Clear backend memory cache."""
        ...


class _ExportFunctionBackendProtocol(typing.Protocol):
    """Protocol for backends supporting `export_function()`."""

    def export_function(self, *args: TensorOrScalar, **kwargs: AttrValue) -> None:
        """Export computational trace or function.

        Args:
            *args (TensorOrScalar): Positional arguments.
            **kwargs (AttrValue): Keyword arguments.
        """
        ...


class _LogicalDevicesBackendProtocol(typing.Protocol):
    """Protocol for backends supporting `get_logical_devices()`."""

    def get_logical_devices(self, device_type: str | None = None) -> list[Device]:
        """Return logical devices.

        Args:
            device_type (str | None): Device type filter.

        Returns:
            list[Device]: Logical devices.
        """
        ...


class _PhysicalDevicesBackendProtocol(typing.Protocol):
    """Protocol for backends supporting `get_physical_devices()`."""

    def get_physical_devices(self, device_type: str | None = None) -> list[Device]:
        """Return physical devices.

        Args:
            device_type (str | None): Device type filter.

        Returns:
            list[Device]: Physical devices.
        """
        ...


class _MemoryInfoBackendProtocol(typing.Protocol):
    """Protocol for backends supporting `get_memory_info()`."""

    def get_memory_info(self, device: str | None = None) -> dict[str, int]:
        """Return memory usage statistics.

        Args:
            device (str | None): Device identifier.

        Returns:
            dict[str, int]: Memory telemetry dictionary.
        """
        ...


class _NamedBackendObjectProtocol(typing.Protocol):
    """Protocol for backend classes or mocks exposing a `__name__` attribute."""

    __name__: str


def _is_clear_cache_backend(val: TensorOrScalar) -> typing.TypeGuard[_ClearCacheBackendProtocol]:
    """Check whether `val` exposes a `clear_cache` method.

    Args:
        val (TensorOrScalar): Active backend.

    Returns:
        bool: True if `val` has `clear_cache`.
    """
    return _has_device_member(val, "clear_cache")


def _is_export_function_backend(val: TensorOrScalar) -> typing.TypeGuard[_ExportFunctionBackendProtocol]:
    """Check whether `val` exposes an `export_function` method.

    Args:
        val (TensorOrScalar): Active backend.

    Returns:
        bool: True if `val` has `export_function`.
    """
    return _has_device_member(val, "export_function")


def _is_logical_devices_backend(val: TensorOrScalar) -> typing.TypeGuard[_LogicalDevicesBackendProtocol]:
    """Check whether `val` exposes a `get_logical_devices` method.

    Args:
        val (TensorOrScalar): Active backend.

    Returns:
        bool: True if `val` has `get_logical_devices`.
    """
    return _has_device_member(val, "get_logical_devices")


def _is_physical_devices_backend(val: TensorOrScalar) -> typing.TypeGuard[_PhysicalDevicesBackendProtocol]:
    """Check whether `val` exposes a `get_physical_devices` method.

    Args:
        val (TensorOrScalar): Active backend.

    Returns:
        bool: True if `val` has `get_physical_devices`.
    """
    return _has_device_member(val, "get_physical_devices")


def _is_memory_info_backend(val: TensorOrScalar) -> typing.TypeGuard[_MemoryInfoBackendProtocol]:
    """Check whether `val` exposes a `get_memory_info` method.

    Args:
        val (TensorOrScalar): Active backend.

    Returns:
        bool: True if `val` has `get_memory_info`.
    """
    return _has_device_member(val, "get_memory_info")


def _is_named_backend_object(val: TensorOrScalar) -> typing.TypeGuard[_NamedBackendObjectProtocol]:
    """Check whether `val` exposes a `__name__` attribute.

    Args:
        val (TensorOrScalar): Active backend.

    Returns:
        bool: True if `val` has `__name__`.
    """
    return _has_device_member(val, "__name__")


def _backend_name(backend: TensorOrScalar) -> str:
    """Resolve display name of a backend class or instance.

    Args:
        backend (TensorOrScalar): Active backend class or instance.

    Returns:
        str: Backend name.
    """
    if _is_named_backend_object(backend) and isinstance(backend.__name__, str):
        return str(backend.__name__)
    cls_name = type(backend).__name__
    return cls_name if isinstance(cls_name, str) else str(cls_name)


class DeviceType(Enum):
    """Enumeration of supported device types."""

    CPU = "cpu"
    GPU = "gpu"
    WEBGPU = "webgpu"


class Device(pydantic.BaseModel):
    """Represents a specific hardware device and index."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    device_type: DeviceType | str
    index: int = 0

    def __init__(self, device_type: DeviceType | str, index: int = 0) -> None:
        """Initialize Device.

        Args:
            device_type (DeviceType | str): Hardware device type or identifier string.
            index (int): Zero-based device ordinal.
        """
        super().__init__(device_type=device_type, index=index)

    def __repr__(self) -> str:
        """Return the string representation of the Device.

        Returns:
            str: The device representation (e.g. Device(cpu:0)).
        """
        if isinstance(self.device_type, str):
            return f"Device({self.device_type}:{self.index})"
        return f"Device({self.device_type.value}:{self.index})"


class Stream:
    """Provide a stream for running operations on a given device."""

    def __init__(self, device: Device | None = None) -> None:
        """Initialize Stream.

        Args:
            device (Device | None): The device
        """
        self.device = device


class StreamContext:
    """Provide a context manager for setting the current device and stream."""

    def __init__(self, stream: Stream) -> None:
        """Initialize StreamContext.

        Args:
            stream (Stream): The stream
        """
        self.stream: Stream | None = stream

    def __enter__(self) -> StreamContext:
        """Enter the context manager.

        Returns:
            StreamContext: The context manager instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context.

        Args:
            exc_type (type[BaseException] | None): Exception type.
            exc_val (BaseException | None): Exception value.
            exc_tb (TracebackType | None): Exception traceback.
        """
        self.stream = None


def clear_cache() -> None:
    """Clear the memory cache."""
    import warnings

    import ml_switcheroo_compiler.backends.registry as registry

    backend: TensorOrScalar = registry.get_active_backend()
    if _is_clear_cache_backend(backend):
        backend.clear_cache()
    else:
        warnings.warn(f"Active backend '{_backend_name(backend)}' does not support clear_cache()", stacklevel=2)


class FunctionExporter:
    """Provide a context managing class for exporting multiple traces of the same function."""

    def __init__(self, *args: TensorOrScalar, **kwargs: AttrValue) -> None:
        """Initialize FunctionExporter.

        Args:
            *args (TensorOrScalar): Positional arguments for function export.
            **kwargs (AttrValue): Keyword arguments for function export.
        """
        self.args = args
        self.kwargs = kwargs

    def __enter__(self) -> FunctionExporter:
        """Enter the context manager.

        Returns:
            FunctionExporter: The context manager instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context.

        Args:
            exc_type (type[BaseException] | None): Exception type.
            exc_val (BaseException | None): Exception value.
            exc_tb (TracebackType | None): Exception traceback.
        """
        self.args = ()
        self.kwargs = {}


def export_function(*args: TensorOrScalar, **kwargs: AttrValue) -> None:
    """Export a computational trace or function for unified persistence.

    Args:
        *args (TensorOrScalar): Positional arguments.
        **kwargs (AttrValue): Keyword arguments.

    Raises:
        BackendNotSupportedError: If the backend does not support exporting.
    """
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend: TensorOrScalar = registry.get_active_backend()
    if _is_export_function_backend(backend):
        backend.export_function(*args, **kwargs)
    else:
        raise BackendNotSupportedError(f"Active backend '{_backend_name(backend)}' does not support export_function()")


def exporter(*args: TensorOrScalar, **kwargs: AttrValue) -> FunctionExporter:
    """Make a callable object to export multiple traces of a function to a file.

    Args:
        *args (TensorOrScalar): Positional arguments.
        **kwargs (AttrValue): Keyword arguments.

    Returns:
        FunctionExporter: The exporter context manager.
    """
    return FunctionExporter(*args, **kwargs)


def get_logical_devices(device_type: str | None = None) -> list[Device]:
    """Get logical devices for the current backend.

    Args:
        device_type (str | None): The type of device to filter by.

    Returns:
        list[Device]: A list of logical devices.

    Raises:
        BackendNotSupportedError: If the backend does not support this.
    """
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend: TensorOrScalar = registry.get_active_backend()
    if _is_logical_devices_backend(backend):
        return list(backend.get_logical_devices(device_type))
    raise BackendNotSupportedError(f"Active backend '{_backend_name(backend)}' does not support get_logical_devices()")


def get_physical_devices(device_type: str | None = None) -> list[Device]:
    """Get physical devices for the current backend.

    Args:
        device_type (str | None): The type of device to filter by.

    Returns:
        list[Device]: A list of physical devices.

    Raises:
        BackendNotSupportedError: If the backend does not support this.
    """
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend: TensorOrScalar = registry.get_active_backend()
    if _is_physical_devices_backend(backend):
        return list(backend.get_physical_devices(device_type))
    raise BackendNotSupportedError(f"Active backend '{_backend_name(backend)}' does not support get_physical_devices()")


def get_memory_info(device: str | None = None) -> dict[str, int]:
    """Get memory statistics tracking (allocation bytes, peak usage).

    Args:
        device (str | None): The device to get info for.

    Returns:
        dict[str, int]: Memory statistics dictionary.

    Raises:
        BackendNotSupportedError: If the backend does not support memory info.
    """
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend: TensorOrScalar = registry.get_active_backend()
    if _is_memory_info_backend(backend):
        return dict(backend.get_memory_info(device))
    raise BackendNotSupportedError(f"Active backend '{_backend_name(backend)}' does not support get_memory_info()")
