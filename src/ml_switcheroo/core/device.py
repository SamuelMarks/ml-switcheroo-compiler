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
        """Initialize the Device."""
        self.device_type = device_type
        self.index = index

    def __eq__(self, other: object) -> bool:
        """Check for equality with another Device."""
        if not isinstance(other, Device):
            return False
        return self.device_type == other.device_type and self.index == other.index

    def __hash__(self) -> int:
        """Hash the Device."""
        return hash((self.device_type, self.index))

    def __repr__(self) -> str:
        """Return the string representation of the Device."""
        return f"Device({self.device_type.value}:{self.index})"
