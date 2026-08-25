# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Apply logical hardware topology abstraction."""

from collections.abc import Sequence
from typing import Optional


class DeviceMesh:
    """Apply logical hardware topology abstraction for distributed execution."""

    def __init__(
        self,
        shape: Sequence[int],
        axis_names: Sequence[str],
        devices: Optional[Sequence[object]] = None,
    ) -> None:
        """Initialize DeviceMesh.

        Args:
            shape (Sequence): The shape parameter.
            axis_names (Sequence): The axis_names parameter.
            devices (Optional): The devices parameter.

        Raises:
            ValueError: An exception.
        """
        if len(shape) != len(axis_names):
            msg: object = "Length of shape and axis_names must match."
            raise ValueError(msg)

        self.shape = tuple(shape)
        self.axis_names = tuple(axis_names)

        expected_devices: object = 1
        for dim in shape:
            expected_devices *= dim

        if devices is not None:
            if len(devices) != expected_devices:
                msg: object = f"Expected {expected_devices} devices, got {len(devices)}."
                raise ValueError(msg)
            self.devices = tuple(devices)
        else:
            self.devices = tuple(range(expected_devices))

    def __repr__(self) -> str:
        """Return representation.

        Returns:
        str: Result.
        """
        return f"DeviceMesh(shape={self.shape}, axis_names={self.axis_names})"

    def __eq__(self, other: object) -> bool:
        """Equality check.

        Args:
        other (object): The other parameter.

        Returns:
        bool: Result.
        """
        if not isinstance(other, DeviceMesh):
            return False
        return self.shape == other.shape and self.axis_names == other.axis_names and self.devices == other.devices
