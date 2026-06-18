"""Device and DeviceType classes for the ml-switcheroo compiler."""

from enum import Enum


from dataclasses import dataclass


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
