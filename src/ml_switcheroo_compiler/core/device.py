"""Device and DeviceType classes for the ml-switcheroo compiler."""

from enum import Enum


class DeviceType(Enum):
    """Enumeration of supported device types."""

    CPU = "cpu"
    GPU = "gpu"
    WEBGPU = "webgpu"


class Device:
    """Represents a specific hardware device and index."""

    def __init__(self, device_type: DeviceType, index: int = 0) -> None:
        """Initialize the Device.

        device_type (DeviceType): The device to store the tensor on._type
            index (int): Argument index

        Args:
            device_type (DeviceType): The device to store the tensor on._type
            index (int): Argument index
        """
        self.device_type = device_type
        self.index = index

    def __eq__(self, other: object) -> bool:
        """Check for equality with another Device.

        Args:
            other (object): The other.

        Returns:
            bool: The computed result.
        """
        if not isinstance(other, Device):
            return False
        return self.device_type == other.device_type and self.index == other.index

    def __hash__(self) -> int:
        """Hash the Device.

        Returns:
            int: The computed result.
        """
        return hash((self.device_type, self.index))

    def __repr__(self) -> str:
        """Return the string representation of the Device.

        Returns:
            str: The computed result.
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
            StreamContext: The computed result.
        """
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context.

        Args:
            exc_type (object): exception type
            exc_val (object): exception value
            exc_tb (object): exception traceback
        """


def clear_cache() -> None:
    """Clear the memory cache."""


class FunctionExporter:
    """A context managing class for exporting multiple traces of the same function."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize FunctionExporter.

        Args:
            args (object): args
            kwargs (object): kwargs
        """

    def __enter__(self) -> "FunctionExporter":
        """Enter context.

        Returns:
            FunctionExporter: The computed result.
        """
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context.

        Args:
            exc_type (object): exception type
            exc_val (object): exception value
            exc_tb (object): exception traceback
        """


def export_function(*args: object, **kwargs: object) -> None:
    """Export an MLX function.

    Args:
        args (object): args
        kwargs (object): kwargs
    """


def exporter(*args: object, **kwargs: object) -> FunctionExporter:
    """Make a callable object to export multiple traces of a function to a file.

    Args:
        args (object): args
        kwargs (object): kwargs

    Returns:
        FunctionExporter: The exporter
    """
    return FunctionExporter(*args, **kwargs)
