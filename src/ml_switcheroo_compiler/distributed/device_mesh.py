"""Apply logical hardware topology abstraction."""

import os
from collections.abc import Sequence
from typing import Optional

import yaml

from ml_switcheroo_compiler.distributed.config_models import DeviceMeshYamlConfig


class DeviceMesh:
    """Apply logical hardware topology abstraction for distributed execution."""

    shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    devices: tuple[int, ...]

    def __init__(
        self,
        shape: Sequence[int],
        axis_names: Sequence[str],
        devices: Optional[Sequence[int]] = None,
    ) -> None:
        """Initialize DeviceMesh.

        Args:
            shape (Sequence[int]): The shape of the device mesh.
            axis_names (Sequence[str]): Names of each dimension/axis.
            devices (Optional[Sequence[int]]): Flat list of device ranks.

        Raises:
            ValueError: If lengths of shape and axis_names mismatch or devices count is incorrect.
        """
        if len(shape) != len(axis_names):
            msg = "Length of shape and axis_names must match."
            raise ValueError(msg)

        self.shape = tuple(shape)
        self.axis_names = tuple(axis_names)

        expected_devices = 1
        for dim in shape:
            expected_devices *= dim

        if devices is not None:
            if len(devices) != expected_devices:
                msg = f"Expected {expected_devices} devices, got {len(devices)}."
                raise ValueError(msg)
            self.devices = tuple(devices)
        else:
            self.devices = tuple(range(expected_devices))

    @classmethod
    def from_yaml(cls, name: str = "default", yaml_path: Optional[str] = None) -> "DeviceMesh":
        """Load a DeviceMesh configuration from a declarative YAML specification.

        Args:
            name (str): The name of the mesh in cluster_meshes.
            yaml_path (Optional[str]): Explicit path to device_mesh.yaml.

        Returns:
            DeviceMesh: Instantiated DeviceMesh instance.

        Raises:
            KeyError: If mesh name is not found in YAML.
            FileNotFoundError: If yaml_path does not exist.
        """
        path = yaml_path or os.path.join(os.path.dirname(__file__), "device_mesh.yaml")
        with open(path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        validated = DeviceMeshYamlConfig.model_validate(raw_data)
        if name not in validated.cluster_meshes:
            raise KeyError(f"Cluster mesh '{name}' not found in {path}")

        mesh_cfg = validated.cluster_meshes[name]
        return cls(
            shape=mesh_cfg.shape,
            axis_names=mesh_cfg.axis_names,
            devices=mesh_cfg.devices,
        )

    def get_coords(self, rank: int) -> tuple[int, ...]:
        """Convert a flat device rank into n-dimensional mesh coordinates.

        Args:
            rank (int): The flat device rank.

        Returns:
            tuple[int, ...]: Coordinate indices corresponding to shape dimensions.

        Raises:
            ValueError: If rank is not part of this mesh.
        """
        if rank not in self.devices:
            raise ValueError(f"Rank {rank} is not part of devices {self.devices}")

        idx = self.devices.index(rank)
        coords = []
        rem = idx
        for dim in reversed(self.shape):
            coords.append(rem % dim)
            rem //= dim
        return tuple(reversed(coords))

    def get_rank(self, coords: Sequence[int]) -> int:
        """Convert n-dimensional mesh coordinates into a flat device rank.

        Args:
            coords (Sequence[int]): N-dimensional mesh coordinate indices.

        Returns:
            int: The flat device rank.

        Raises:
            ValueError: If coordinate length or dimensions are out of bounds.
        """
        if len(coords) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} coordinates, got {len(coords)}")

        idx = 0
        multiplier = 1
        for dim, c in zip(reversed(self.shape), reversed(coords)):
            if not 0 <= c < dim:
                raise ValueError(f"Coordinate {c} out of bounds for dimension size {dim}")
            idx += c * multiplier
            multiplier *= dim

        return self.devices[idx]

    def get_neighbors(self, rank: int, axis_name: str) -> tuple[int, int]:
        """Get cyclic left and right neighbor ranks along a designated axis.

        Args:
            rank (int): The flat device rank.
            axis_name (str): Named axis to traverse.

        Returns:
            tuple[int, int]: Tuple of (prev_rank, next_rank).

        Raises:
            ValueError: If axis_name is invalid or rank is not in devices.
        """
        if axis_name not in self.axis_names:
            raise ValueError(f"Unknown axis '{axis_name}'. Available: {self.axis_names}")

        axis_idx = self.axis_names.index(axis_name)
        dim_size = self.shape[axis_idx]
        coords = list(self.get_coords(rank))

        c = coords[axis_idx]
        prev_coords = list(coords)
        prev_coords[axis_idx] = (c - 1) % dim_size

        next_coords = list(coords)
        next_coords[axis_idx] = (c + 1) % dim_size

        return self.get_rank(prev_coords), self.get_rank(next_coords)

    def to_dict(self) -> dict[str, Sequence[object]]:
        """Serialize DeviceMesh properties to dictionary.

        Returns:
            dict[str, Sequence[object]]: Dictionary of shape, axis_names, and devices.
        """
        return {
            "shape": self.shape,
            "axis_names": self.axis_names,
            "devices": self.devices,
        }

    def __repr__(self) -> str:
        """Return representation.

        Returns:
            str: String representation of DeviceMesh.
        """
        return f"DeviceMesh(shape={self.shape}, axis_names={self.axis_names})"

    def __eq__(self, other: object) -> bool:
        """Equality check.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if shapes, axis names, and devices are identical.
        """
        if not isinstance(other, DeviceMesh):
            return False
        return self.shape == other.shape and self.axis_names == other.axis_names and self.devices == other.devices
