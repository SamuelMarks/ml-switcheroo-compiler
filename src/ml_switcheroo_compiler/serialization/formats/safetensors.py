# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module safetensors.py."""

"""Safetensors format serialization."""

import json
import struct

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_8
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


class SafetensorsWeightFormat(WeightLoader, WeightSaver):
    """Safetensors weight format handler."""

    def load(self, filepath: str) -> dict[str, object]:
        """Load safetensors weights.

        Args:
        filepath (str): The filepath parameter.

        Returns:
        dict: Result.
        """
        with open(filepath, "rb") as f:
            header_size_bytes: object = f.read(8)
            if len(header_size_bytes) < MAGIC_VAL_8:
                return {}
            header_size: object = struct.unpack("<Q", header_size_bytes)[0]

            header_bytes: object = f.read(header_size)
            header: object = json.loads(header_bytes.decode("utf-8"))

            weights: object = {}
            for k, v in header.items():
                if k == "__metadata__":
                    continue
                offsets: object = v["data_offsets"]
                f.seek(8 + header_size + offsets[0])
                buffer: object = f.read(offsets[1] - offsets[0])

                from ml_switcheroo_compiler.backends.registry import get_active_backend

                backend: object = get_active_backend()
                if hasattr(backend, "from_buffer"):
                    weights[k] = backend.from_buffer(buffer, dtype=v["dtype"], shape=v["shape"])
                else:
                    # Pass the raw buffer if no converter is available
                    weights[k] = {"buffer": buffer, "dtype": v["dtype"], "shape": v["shape"]}

            return weights

    def save(self, weights_np: dict[str, object], filepath: str) -> None:
        """Save safetensors weights.

        Args:
            weights_np (dict): The weights_np parameter.
            filepath (str): The filepath parameter.
        """
        header: object = {}
        offset: object = 0
        buffers: object = []

        for k, v in weights_np.items():
            if not hasattr(v, "dtype") or not hasattr(v, "shape") or not hasattr(v, "tobytes"):
                continue

            import os

            import yaml

            from ml_switcheroo_compiler.serialization.formats.config_models import SerializationSchemaConfig

            path: object = os.path.join(os.path.dirname(__file__), "serialization_schema.yaml")
            with open(path) as f:
                data: object = yaml.safe_load(f)
                schema: object = SerializationSchemaConfig(**data)
            dtype_map: object = schema.safetensors.dtype_map

            dtype_str: object = str(v.dtype)
            st_dtype: object = dtype_map.get(dtype_str, "F32")

            buffer: object = v.tobytes()
            length: object = len(buffer)

            header[k] = {
                "dtype": st_dtype,
                "shape": list[object](v.shape),
                "data_offsets": [offset, offset + length],
            }

            buffers.append(buffer)
            offset += length

        header_bytes: object = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_length: object = len(header_bytes)

        padding_length: object = (8 - (header_length % 8)) % 8
        header_bytes += b" " * padding_length
        header_length += padding_length

        with open(filepath, "wb") as f:
            f.write(struct.pack("<Q", header_length))
            f.write(header_bytes)
            for buffer in buffers:
                f.write(buffer)
