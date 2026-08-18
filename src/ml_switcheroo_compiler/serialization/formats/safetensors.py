# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module safetensors.py."""

from typing import Any

"""Safetensors format serialization."""

import json
import struct

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_8
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


class SafetensorsWeightFormat(WeightLoader, WeightSaver):
    """Safetensors weight format handler."""

    def load(self, filepath: str) -> dict[str, Any]:
        """Load safetensors weights.

        Args:
        filepath (str): The filepath parameter.

        Returns:
        dict: Result.
        """
        with open(filepath, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < MAGIC_VAL_8:
                return {}
            header_size = struct.unpack("<Q", header_size_bytes)[0]

            header_bytes = f.read(header_size)
            header = json.loads(header_bytes.decode("utf-8"))

            weights = {}
            for k, v in header.items():
                if k == "__metadata__":
                    continue
                offsets = v["data_offsets"]
                f.seek(8 + header_size + offsets[0])
                buffer = f.read(offsets[1] - offsets[0])

                from ml_switcheroo_compiler.backends.registry import get_active_backend

                backend = get_active_backend()
                if hasattr(backend, "from_buffer"):
                    weights[k] = backend.from_buffer(buffer, dtype=v["dtype"], shape=v["shape"])
                else:
                    # Pass the raw buffer if no converter is available
                    weights[k] = {"buffer": buffer, "dtype": v["dtype"], "shape": v["shape"]}

            return weights

    def save(self, weights_np: dict[str, Any], filepath: str) -> None:
        """Save safetensors weights.

        Args:
            weights_np (dict): The weights_np parameter.
            filepath (str): The filepath parameter.
        """
        header = {}
        offset = 0
        buffers = []

        for k, v in weights_np.items():
            if not hasattr(v, "dtype") or not hasattr(v, "shape") or not hasattr(v, "tobytes"):
                continue

            import os

            import yaml

            from ml_switcheroo_compiler.serialization.formats.config_models import SerializationSchemaConfig

            path = os.path.join(os.path.dirname(__file__), "serialization_schema.yaml")
            with open(path) as f:
                data = yaml.safe_load(f)
                schema = SerializationSchemaConfig(**data)
            dtype_map = schema.safetensors.dtype_map

            dtype_str = str(v.dtype)
            st_dtype = dtype_map.get(dtype_str, "F32")

            buffer = v.tobytes()
            length = len(buffer)

            header[k] = {
                "dtype": st_dtype,
                "shape": list[Any](v.shape),
                "data_offsets": [offset, offset + length],
            }

            buffers.append(buffer)
            offset += length

        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_length = len(header_bytes)

        padding_length = (8 - (header_length % 8)) % 8
        header_bytes += b" " * padding_length
        header_length += padding_length

        with open(filepath, "wb") as f:
            f.write(struct.pack("<Q", header_length))
            f.write(header_bytes)
            for buffer in buffers:
                f.write(buffer)
