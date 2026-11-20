"""SavedModel bundle serialization, sharding policies, and SHA-256 fingerprinting."""

from __future__ import annotations

import hashlib
import json
import os

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.shape import OpValue as AttributeValue
from ml_switcheroo_compiler.serialization.protocols import (
    _ConfigProviderProtocol,
    _GraphProviderProtocol,
    _ToJsonProviderProtocol,
    graph_to_json,
)


class TrackableResource:
    """Trackable resource for asset extraction."""

    def __init__(self) -> None:
        """Initialize TrackableResource."""
        self.resource_id = None
        self.tracked: bool = False


class PythonState:
    """Python state synchronization capabilities."""

    def __init__(self) -> None:
        """Initialize PythonState."""
        self.state: dict[str, AttributeValue] = {}


class MaxShardSizePolicy:
    """Sharded saving protocol by max size."""

    def __init__(self, max_shard_size: int) -> None:
        """Initialize MaxShardSizePolicy.

        Args:
            max_shard_size (int): Maximum size per shard in bytes.
        """
        self.max_shard_size = max_shard_size


class ShardByTaskPolicy:
    """Sharded saving protocol by task."""

    def __init__(self) -> None:
        """Initialize ShardByTaskPolicy."""
        self.policy = "task"


class SavedModel:
    """SavedModel proto serialization/deserialization and asset bundle manager."""

    def __init__(self, model: AttributeValue = None, signatures: dict[str, AttributeValue] | None = None) -> None:
        """Initialize SavedModel instance.

        Args:
            model (AttributeValue): Model or graph structure to encapsulate.
            signatures (Optional[dict[str, AttributeValue]]): Mapping of endpoint names to signatures.
        """
        self.model = model
        self.signatures = signatures or {}
        self.assets: dict[str, AttributeValue] = {}

    def save(self, path: str) -> None:
        """Serialize model manifest, topology, fingerprint, and assets to directory.

        Args:
            path (str): Destination directory path.
        """
        os.makedirs(path, exist_ok=True)
        graph_def: dict[str, AttributeValue] = {}
        if self.model is not None:
            if isinstance(self.model, _GraphProviderProtocol):
                graph_def = json.loads(graph_to_json(self.model.graph))
            elif isinstance(self.model, _ToJsonProviderProtocol):
                graph_def = json.loads(self.model.to_json())
            elif isinstance(self.model, _ConfigProviderProtocol):
                graph_def = dict(self.model.get_config())

        manifest = {
            "format": "SavedModel",
            "version": "2.0",
            "graph_def": graph_def,
            "signatures": self.signatures,
            "assets": self.assets,
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
        with open(os.path.join(path, "saved_model.pb"), "wb") as f:
            f.write(manifest_bytes)

        fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
        with open(os.path.join(path, "fingerprint.pb"), "w", encoding="utf-8") as f:
            f.write(fingerprint)

    @classmethod
    def load(cls, path: str) -> SavedModel:
        """Load and reconstruct SavedModel bundle from disk.

        Args:
            path (str): Path to the SavedModel directory.

        Returns:
            SavedModel: Fully reconstructed SavedModel container.

        Raises:
            FileNotFoundError: If path or saved_model.pb is missing.
            ValueError: If saved_model.pb is corrupted.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"SavedModel directory '{path}' not found")

        pb_path = os.path.join(path, "saved_model.pb") if os.path.isdir(path) else path
        if not os.path.exists(pb_path):
            raise FileNotFoundError(f"SavedModel proto '{pb_path}' not found")

        with open(pb_path, "rb") as f:
            raw_bytes = f.read()

        if len(raw_bytes) == 0:
            manifest: dict[str, AttributeValue] = {
                "format": "SavedModel",
                "version": "2.0",
                "graph_def": {},
                "signatures": {},
                "assets": {},
            }
        else:
            try:
                manifest = json.loads(raw_bytes.decode("utf-8"))
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                raise ValueError(f"Corrupted SavedModel proto at '{pb_path}': {e}") from e

        raw_sigs = manifest.get("signatures", {})
        sigs_dict: dict[str, OpValue] = dict(raw_sigs) if isinstance(raw_sigs, dict) else {}
        raw_assets = manifest.get("assets", {})
        assets_dict: dict[str, OpValue] = dict(raw_assets) if isinstance(raw_assets, dict) else {}
        instance = cls(model=manifest.get("graph_def"), signatures=sigs_dict)
        instance.assets = assets_dict
        return instance


def read_fingerprint(path: str) -> str:
    """Read or compute deterministic cryptographic SHA-256 fingerprint for a model or checkpoint.

    Args:
        path (str): The model directory path or file path.

    Returns:
        str: Cryptographic fingerprint hex digest, or 'fingerprint' fallback if nonexistent.
    """
    if not os.path.exists(path):
        return "fingerprint"

    fp_path = os.path.join(path, "fingerprint.pb") if os.path.isdir(path) else None
    if fp_path and os.path.exists(fp_path):
        with open(fp_path, encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            return content

    hasher = hashlib.sha256()
    if os.path.isdir(path):
        for root, _, files in sorted(os.walk(path)):
            for file in sorted(files):
                if file == "fingerprint.pb":
                    continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, path)
                hasher.update(rel_path.encode("utf-8"))
                with open(file_path, "rb") as f:
                    data = f.read()
                hasher.update(data)
    else:
        with open(path, "rb") as f:
            data = f.read()
        hasher.update(data)

    return hasher.hexdigest()
