"""Summary generation utilities for TensorBoard integration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ml_switcheroo_compiler.backends.registry import BackendRegistry

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor


def write_raw_pb(pb_data: bytes, logdir: str) -> None:
    """Write raw protobuf data to a file.

    Args:
        pb_data (bytes): The raw protobuf data to write.
        logdir (str): Destination directory for the event file.
    """
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "events.out.tfevents.pb"), "ab") as f:
        f.write(pb_data)


def _pack_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Pack a single PNG chunk with length, type, data, and CRC32.

    Args:
        chunk_type (bytes): 4-byte ASCII chunk identifier.
        data (bytes): Raw chunk payload bytes.

    Returns:
        bytes: Formatted PNG chunk bytes.
    """
    import struct
    import zlib

    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def encode_image(tensor: Tensor) -> bytes:
    """Encode an image tensor into PNG byte format suitable for summary.image.

    Args:
        tensor (Tensor): The image tensor to encode (2D, 3D, or 4D).

    Returns:
        bytes: PNG encoded image bytes starting with valid PNG header.
    """
    import struct
    import zlib

    np_mod = BackendRegistry.get("numpy").get_module()
    raw_data = getattr(tensor, "data", tensor)
    arr = np_mod.asarray(raw_data)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np_mod.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    if np_mod.issubdtype(arr.dtype, np_mod.floating):
        arr = np_mod.clip(arr * 255.0, 0.0, 255.0).astype(np_mod.uint8)
    else:
        arr = np_mod.clip(arr, 0, 255).astype(np_mod.uint8)

    h, w = int(arr.shape[0]), int(arr.shape[1])
    c = int(arr.shape[2]) if arr.ndim == 3 else 1
    color_type = 0 if c == 1 else (6 if c == 4 else 2)

    raw_rows = bytearray()
    row_bytes = w * c
    flat_bytes = arr.tobytes()
    for y in range(h):
        raw_rows.append(0)  # Filter type 0 (None)
        raw_rows.extend(flat_bytes[y * row_bytes : (y + 1) * row_bytes])

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    idat = zlib.compress(bytes(raw_rows))
    return b"\x89PNG\r\n\x1a\n" + _pack_png_chunk(b"IHDR", ihdr) + _pack_png_chunk(b"IDAT", idat) + _pack_png_chunk(b"IEND", b"")
