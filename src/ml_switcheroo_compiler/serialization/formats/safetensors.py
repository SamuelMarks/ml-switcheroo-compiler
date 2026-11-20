"""Safetensors format serialization."""

from __future__ import annotations

import json
import struct
import typing
from collections.abc import Mapping, Sequence

from ml_switcheroo_compiler.core.constants import SAFETENSORS_HEADER_SIZE_BYTES
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver

SafetensorsEntry = typing.Union[TensorLike, dict[str, typing.Union[bytes, str, list[int]]]]  # noqa: UP007


@typing.runtime_checkable
class _FromBufferBackendProtocol(typing.Protocol):
    """Protocol for backends implementing `from_buffer(buffer, dtype, shape)`."""

    def from_buffer(self, buffer: bytes, dtype: str, shape: Sequence[int]) -> TensorLike:
        """Construct tensor from raw bytes buffer.

        Args:
            buffer (bytes): Raw bytes.
            dtype (str): Safetensors dtype identifier.
            shape (Sequence[int]): Tensor shape dimensions.

        Returns:
            TensorLike: Reconstructed tensor.
        """
        ...


@typing.runtime_checkable
class _SafetensorsArrayProtocol(typing.Protocol):
    """Protocol for array-like objects serializable into Safetensors format."""

    dtype: OpValue
    shape: Sequence[int]

    def tobytes(self) -> bytes:
        """Serialize array contents to raw bytes.

        Returns:
            bytes: Raw array bytes.
        """
        ...


def _is_from_buffer_backend(val: OpValue) -> typing.TypeGuard[_FromBufferBackendProtocol]:
    """Check whether `val` exposes `from_buffer`.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` has `from_buffer`.
    """
    return isinstance(val, _FromBufferBackendProtocol)


def _is_safetensors_array(val: OpValue) -> typing.TypeGuard[_SafetensorsArrayProtocol]:
    """Check whether `val` exposes `dtype`, `shape`, and `tobytes`.

    Args:
        val (OpValue): Candidate array object.

    Returns:
        bool: True if `val` has `dtype`, `shape`, and `tobytes`.
    """
    return isinstance(val, _SafetensorsArrayProtocol)


class SafetensorsWeightFormat(WeightLoader, WeightSaver):
    """Safetensors weight format handler."""

    def load(self, filepath: str) -> dict[str, SafetensorsEntry]:
        """Load safetensors weights.

        Args:
            filepath (str): Input `filepath` argument (filepath).

        Returns:
            dict[str, SafetensorsEntry]: Loaded weights dictionary.
        """
        with open(filepath, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < SAFETENSORS_HEADER_SIZE_BYTES:
                return {}
            header_size = struct.unpack("<Q", header_size_bytes)[0]

            header_bytes = f.read(header_size)
            header = json.loads(header_bytes.decode("utf-8"))

            weights: dict[str, SafetensorsEntry] = {}
            for k, v in header.items():
                if k == "__metadata__":
                    continue
                offsets = v["data_offsets"]
                f.seek(8 + header_size + offsets[0])
                buffer = f.read(offsets[1] - offsets[0])

                from ml_switcheroo_compiler.backends.registry import get_active_backend

                backend: OpValue = get_active_backend()
                if _is_from_buffer_backend(backend):
                    weights[k] = backend.from_buffer(buffer, dtype=v["dtype"], shape=v["shape"])
                else:
                    # Pass the raw buffer if no converter is available
                    fallback_entry: dict[str, bytes | str | list[int]] = {
                        "buffer": buffer,
                        "dtype": v["dtype"],
                        "shape": v["shape"],
                    }
                    weights[k] = fallback_entry

            return weights

    def save(self, weights_np: OpValue, filepath: str) -> None:
        """Save safetensors weights.

        Args:
            weights_np (OpValue): Input `weights_np` argument (weights np).
            filepath (str): Input `filepath` argument (filepath).
        """
        header = {}
        offset = 0
        buffers: list[bytes] = []

        assert isinstance(weights_np, Mapping)
        for k, v in weights_np.items():
            raw_v: OpValue = v
            if not _is_safetensors_array(raw_v):
                continue

            import os

            import yaml

            from ml_switcheroo_compiler.serialization.formats.config_models import SerializationSchemaConfig

            path = os.path.join(os.path.dirname(__file__), "serialization_schema.yaml")
            with open(path) as f:
                data = yaml.safe_load(f)
                schema = SerializationSchemaConfig(**data)
            dtype_map = schema.safetensors.dtype_map

            dtype_str = str(raw_v.dtype)
            st_dtype = dtype_map.get(dtype_str, "F32")

            buffer = raw_v.tobytes()
            length = len(buffer)

            header[str(k)] = {
                "dtype": st_dtype,
                "shape": list(raw_v.shape),
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
            f.writelines(buffers)
