# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""LayoutMap API for annotating tensors with sharding constraints."""

import re
from collections.abc import Sequence
from typing import Optional


class ShardingSpec:
    """Specification of how a tensor is sharded across a DeviceMesh."""

    def __init__(self, mesh, mesh_mapping: Sequence[Optional[str]]) -> None:
        """Initialize ShardingSpec.

        Args:
            mesh: The DeviceMesh.
            mesh_mapping: Sequence of mesh axis names or None for replicated dimensions.
        """
        self.mesh = mesh
        self.mesh_mapping = tuple(mesh_mapping)

    def __repr__(self) -> str:
        """Return representation.

        Returns:
        str: Result.
        """
        return f"ShardingSpec(mesh={self.mesh}, mapping={self.mesh_mapping})"

    def __eq__(self, other) -> bool:
        """Equality check.

        Args:
        other (object): The other parameter.

        Returns:
        bool: Result.
        """
        if not isinstance(other, ShardingSpec):
            return False
        return self.mesh == other.mesh and self.mesh_mapping == other.mesh_mapping


class LayoutMap:
    """Mapping of logical tensor paths to ShardingSpecs."""

    def __init__(self, device_mesh=None) -> None:
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
            if re.fullmatch(k, path):
                return v
        return None

    def __repr__(self) -> str:
        """Return representation.

        Returns:
        str: Result.
        """
        return f"LayoutMap(size={len(self._map)})"

    def __setitem__(self, key: str, value: ShardingSpec) -> None:
        """Set a sharding spec for a given key.

        Args:
            key: Path key.
            value: ShardingSpec.
        """
        self._map[key] = value

    def __getitem__(self, key: str) -> Optional[ShardingSpec]:
        """Get the sharding spec for a given key.

        Args:
            key: Path key.

        Returns:
            The ShardingSpec or None.
        """
        return self._map.get(key)

    def to_dict(self) -> dict[str, object]:
        """Convert LayoutMap to a serializable dictionary.

        Returns:
            dict[str, object]: Serialized layout map representation.
        """
        specs_dict: dict[str, object] = {}
        for path, spec in self._map.items():
            specs_dict[path] = {
                "mesh_name": getattr(spec.mesh, "name", "default") if spec.mesh else None,
                "mesh_mapping": list(spec.mesh_mapping),
            }
        return {
            "version": "1.0",
            "device_mesh": getattr(self.device_mesh, "name", None) if self.device_mesh else None,
            "specs": specs_dict,
        }

    def to_yaml(self) -> str:
        """Serialize LayoutMap to declarative YAML text.

        Returns:
            str: Valid YAML formatted string.
        """
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_content: str, device_mesh: Optional[object] = None) -> "LayoutMap":
        """Construct a LayoutMap from declarative YAML content.

        Args:
            yaml_content (str): YAML string or filesystem path.
            device_mesh (Optional[object]): Associated DeviceMesh instance.

        Returns:
            LayoutMap: Initialized and populated LayoutMap.
        """
        import os

        import yaml

        from ml_switcheroo_compiler.distributed.config_models import LayoutMapYamlConfig

        raw_text: str = yaml_content
        if os.path.isfile(yaml_content):
            with open(yaml_content, encoding="utf-8") as f:
                raw_text = f.read()

        parsed = yaml.safe_load(raw_text) or {}
        validated = LayoutMapYamlConfig.model_validate(parsed)

        layout = cls(device_mesh=device_mesh)
        for path, spec_cfg in validated.specs.items():
            layout.insert(path, ShardingSpec(mesh=device_mesh, mesh_mapping=spec_cfg.mesh_mapping))
        return layout
