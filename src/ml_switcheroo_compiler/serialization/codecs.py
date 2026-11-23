"""Composable bidirectional wire Codecs for UTF-8 bytes and deterministic float tensor maps."""

from __future__ import annotations

import dataclasses
import json
import math

from ml_switcheroo_ir import Codec, CodecError, Eq, Err, Ok, Result
from ml_switcheroo_ir.typeclasses import compose_codec, imap_codec


def _decode_utf8_bytes(wire: bytes) -> Result[CodecError, str]:
    """Decode UTF-8 ``bytes`` into ``str``, returning ``Err(CodecError)`` on invalid byte sequences.

    Args:
        wire (bytes): Raw UTF-8 wire bytes.

    Returns:
        Result[CodecError, str]: ``Ok(str)`` or ``Err(CodecError)``.
    """
    try:
        return Ok(wire.decode("utf-8"))
    except UnicodeDecodeError as exc:
        return Err(
            CodecError(
                path=(),
                expected="valid UTF-8 bytes",
                actual_description="invalid UTF-8 bytes",
                message=str(exc),
            )
        )


UTF8_BYTES_STR_CODEC: Codec[bytes, str] = Codec(
    encode=lambda text: text.encode("utf-8"),
    decode=_decode_utf8_bytes,
)


def _encode_tensor_map_json(tensor_map: dict[str, tuple[float, ...]]) -> str:
    """Serialize ``dict[str, tuple[float, ...]]`` into canonical sorted-key JSON text.

    Args:
        tensor_map (dict[str, tuple[float, ...]]): Mapping of parameter names to flat float tuples.

    Returns:
        str: Canonical JSON payload string.
    """
    serializable = {k: list(vals) for k, vals in sorted(tensor_map.items())}
    return json.dumps(serializable, sort_keys=True)


def _decode_tensor_map_json(wire: str) -> Result[CodecError, dict[str, tuple[float, ...]]]:
    """Decode and validate a JSON string into ``dict[str, tuple[float, ...]]``.

    Args:
        wire (str): Raw JSON string payload.

    Returns:
        Result[CodecError, dict[str, tuple[float, ...]]]: ``Ok`` tensor map or ``Err(CodecError)``.
    """
    try:
        parsed = json.loads(wire)
    except json.JSONDecodeError as exc:
        return Err(
            CodecError(
                path=(),
                expected="JSON object",
                actual_description="malformed JSON",
                message=str(exc),
            )
        )
    if not isinstance(parsed, dict):
        return Err(
            CodecError(
                path=(),
                expected="JSON object mapping str -> list[float]",
                actual_description=type(parsed).__name__,
                message="Top-level tensor map payload must be a JSON object.",
            )
        )
    out: dict[str, tuple[float, ...]] = {}
    for raw_key, raw_vals in parsed.items():
        key_str = str(raw_key)
        if not isinstance(raw_vals, list):
            return Err(
                CodecError(
                    path=(key_str,),
                    expected="list[float]",
                    actual_description=type(raw_vals).__name__,
                    message=f"Entry {key_str!r} must be a JSON array of finite numbers.",
                )
            )
        nums: list[float] = []
        for idx, item in enumerate(raw_vals):
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                return Err(
                    CodecError(
                        path=(key_str, str(idx)),
                        expected="finite float",
                        actual_description=repr(item),
                        message=f"Element {idx} of {key_str!r} is not a finite number.",
                    )
                )
            nums.append(float(item))
        out[key_str] = tuple(nums)
    return Ok(out)


TENSOR_MAP_JSON_CODEC: Codec[str, dict[str, tuple[float, ...]]] = Codec(
    encode=_encode_tensor_map_json,
    decode=_decode_tensor_map_json,
)

TENSOR_MAP_BYTES_CODEC: Codec[bytes, dict[str, tuple[float, ...]]] = compose_codec(
    UTF8_BYTES_STR_CODEC,
    TENSOR_MAP_JSON_CODEC,
)


@dataclasses.dataclass(frozen=True)
class FrozenTensorArchive:
    """Immutable sorted tensor archive record isomorphic to ``dict[str, tuple[float, ...]]``."""

    entries: tuple[tuple[str, tuple[float, ...]], ...]

    @staticmethod
    def from_dict(mapping: dict[str, tuple[float, ...]]) -> FrozenTensorArchive:
        """Build a canonical sorted ``FrozenTensorArchive`` from a dictionary.

        Args:
            mapping (dict[str, tuple[float, ...]]): Tensor name to flat float values mapping.

        Returns:
            FrozenTensorArchive: Canonical immutable archive.
        """
        return FrozenTensorArchive(entries=tuple(sorted(mapping.items())))

    def to_dict(self) -> dict[str, tuple[float, ...]]:
        """Project ``FrozenTensorArchive`` back to a mutable ``dict[str, tuple[float, ...]]``.

        Returns:
            dict[str, tuple[float, ...]]: Reconstructed dictionary.
        """
        return dict(self.entries)


FROZEN_TENSOR_ARCHIVE_EQ: Eq[FrozenTensorArchive] = Eq(eq=lambda a, b: a == b)

FROZEN_TENSOR_ARCHIVE_CODEC: Codec[bytes, FrozenTensorArchive] = imap_codec(
    TENSOR_MAP_BYTES_CODEC,
    FrozenTensorArchive.from_dict,
    FrozenTensorArchive.to_dict,
)

__all__ = [
    "FROZEN_TENSOR_ARCHIVE_CODEC",
    "FROZEN_TENSOR_ARCHIVE_EQ",
    "FrozenTensorArchive",
    "TENSOR_MAP_BYTES_CODEC",
    "TENSOR_MAP_JSON_CODEC",
    "UTF8_BYTES_STR_CODEC",
]
