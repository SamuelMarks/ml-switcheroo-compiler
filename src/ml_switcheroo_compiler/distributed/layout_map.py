"""LayoutMap API for annotating tensors with sharding constraints."""

from collections.abc import Sequence
from typing import Optional


class ShardingSpec:
    """Specification of how a tensor is sharded across a DeviceMesh."""

    def __init__(self, mesh: object, mesh_mapping: Sequence[Optional[str]]) -> None:
        """Initialize ShardingSpec.

        Args:
            mesh: The DeviceMesh.
            mesh_mapping: Sequence of mesh axis names or None for replicated dimensions.
        """
        self.mesh = mesh
        self.mesh_mapping = tuple(mesh_mapping)

    def __repr__(self) -> str:
        """Return representation."""
        return f"ShardingSpec(mesh={self.mesh}, mapping={self.mesh_mapping})"

    def __eq__(self, other: object) -> bool:
        """Equality check."""
        if not isinstance(other, ShardingSpec):
            return False
        return self.mesh == other.mesh and self.mesh_mapping == other.mesh_mapping


class LayoutMap:
    """Mapping of logical tensor paths to ShardingSpecs."""

    def __init__(self, device_mesh: Optional[object] = None) -> None:
        """Initialize LayoutMap.

        Args:
            device_mesh: Optional device mesh.
        """
        self.device_mesh = device_mesh
        self._map: dict[str, ShardingSpec] = {}

    def insert(self, path: str, spec: ShardingSpec) -> None:
        """Insert a sharding specification for a tensor path.

        Args:
            path: The tensor path (e.g., regex or explicit name).
            spec: The ShardingSpec.
        """
        self._map[path] = spec

    def get(self, path: str) -> Optional[ShardingSpec]:
        """Get the sharding specification for a tensor path.

        Args:
            path: The tensor path.

        Returns:
            The ShardingSpec or None if not found.
        """
        # Simple exact match for now, could be extended to regex
        for k, v in self._map.items():
            import re

            if re.fullmatch(k, path):
                return v
        return None

    def __repr__(self) -> str:
        """Return representation."""
        return f"LayoutMap(size={len(self._map)})"

    def __setitem__(self, key: str, value: ShardingSpec) -> None:
        """Set a sharding spec for a given key.

        Args:
            key: Path key.
            value: ShardingSpec.
        """
        self._map[key] = value  # pragma: no cover

    def __getitem__(self, key: str) -> Optional[ShardingSpec]:
        """Get the sharding spec for a given key.

        Args:
            key: Path key.

        Returns:
            The ShardingSpec or None.
        """
        return self._map.get(key)  # pragma: no cover
