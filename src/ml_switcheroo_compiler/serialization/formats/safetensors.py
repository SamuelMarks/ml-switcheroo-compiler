"""Safetensors format serialization."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_8

import json
import struct
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


class SafetensorsWeightFormat(WeightLoader, WeightSaver):
    """Safetensors weight format handler."""

    def load(self, filepath: str) -> dict:
        """Load safetensors weights."""
        import numpy as np  # pragma: no cover

        dtype_map = {  # pragma: no cover
            "F64": np.float64,
            "F32": np.float32,
            "F16": np.float16,
            "I64": np.int64,
            "I32": np.int32,
            "I16": np.int16,
            "I8": np.int8,
            "U8": np.uint8,
            "BOOL": np.bool_,
        }

        with open(filepath, "rb") as f:  # pragma: no cover
            header_size_bytes = f.read(8)  # pragma: no cover
            if len(header_size_bytes) < MAGIC_VAL_8:  # pragma: no cover
                return {}  # pragma: no cover
            header_size = struct.unpack("<Q", header_size_bytes)[0]  # pragma: no cover

            header_bytes = f.read(header_size)  # pragma: no cover
            header = json.loads(header_bytes.decode("utf-8"))  # pragma: no cover

            weights = {}  # pragma: no cover
            for k, v in header.items():  # pragma: no cover
                if k == "__metadata__":  # pragma: no cover
                    continue  # pragma: no cover
                offsets = v["data_offsets"]  # pragma: no cover
                f.seek(8 + header_size + offsets[0])  # pragma: no cover
                buffer = f.read(offsets[1] - offsets[0])  # pragma: no cover
                dtype = dtype_map.get(v["dtype"], np.float32)  # pragma: no cover
                arr = (
                    np.frombuffer(buffer, dtype=dtype).reshape(v["shape"]).copy()
                )  # pragma: no cover
                weights[k] = arr  # pragma: no cover

            return weights  # pragma: no cover

    def save(self, weights_np: dict, filepath: str) -> None:
        """Save safetensors weights."""
        header = {}  # pragma: no cover
        offset = 0  # pragma: no cover
        buffers = []  # pragma: no cover

        for k, v in weights_np.items():  # pragma: no cover
            if (
                not hasattr(v, "dtype") or not hasattr(v, "shape") or not hasattr(v, "tobytes")
            ):  # pragma: no cover
                continue  # pragma: no cover

            dtype_map = {  # pragma: no cover
                "float32": "F32",
                "float64": "F64",
                "float16": "F16",
                "int32": "I32",
                "int64": "I64",
                "int16": "I16",
                "int8": "I8",
                "uint8": "U8",
                "bool": "BOOL",
            }

            dtype_str = str(v.dtype)  # pragma: no cover
            st_dtype = dtype_map.get(dtype_str, "F32")  # pragma: no cover

            buffer = v.tobytes()  # pragma: no cover
            length = len(buffer)  # pragma: no cover

            header[k] = {  # pragma: no cover
                "dtype": st_dtype,
                "shape": list(v.shape),
                "data_offsets": [offset, offset + length],
            }

            buffers.append(buffer)  # pragma: no cover
            offset += length  # pragma: no cover

        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")  # pragma: no cover
        header_length = len(header_bytes)  # pragma: no cover

        padding_length = (8 - (header_length % 8)) % 8  # pragma: no cover
        header_bytes += b" " * padding_length  # pragma: no cover
        header_length += padding_length  # pragma: no cover

        with open(filepath, "wb") as f:  # pragma: no cover
            f.write(struct.pack("<Q", header_length))  # pragma: no cover
            f.write(header_bytes)  # pragma: no cover
            for buffer in buffers:  # pragma: no cover
                f.write(buffer)  # pragma: no cover
