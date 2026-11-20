"""Universal binary record and tabular I/O operations."""

from __future__ import annotations

import gzip
import os
import struct
import typing
import zlib
from types import TracebackType

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    ShapeProtocol,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op


def decode_csv(
    records: Tensor,
    record_defaults: OpValue,
    field_delim: str = ",",
    use_quote_delim: bool = True,
    na_value: str = "",
    select_cols: OpValue = None,
    name: OpValue = None,
) -> OpValue:
    """Decode delimited CSV records into a sequence of column tensors.

    Args:
        records (Tensor): String tensor containing CSV record lines.
        record_defaults (OpValue): Default values and dtypes per CSV column.
        field_delim (str): Single-character delimiter separating fields.
        use_quote_delim (bool): Whether quoted fields are treated specially.
        na_value (str): String token recognized as NA/missing value.
        select_cols (OpValue): Optional sorted list of column indices to extract.
        name (OpValue): Optional operation name.

    Returns:
        OpValue: Decoded column tensor(s).
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op(
            "DecodeCsv",
            records,
            record_defaults=record_defaults,
            field_delim=field_delim,
            use_quote_delim=use_quote_delim,
            na_value=na_value,
            select_cols=select_cols,
            name=name,
        )
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "DecodeCsv",
        [records],
        {
            "record_defaults": record_defaults,
            "field_delim": field_delim,
            "use_quote_delim": use_quote_delim,
            "na_value": na_value,
            "select_cols": select_cols,
            "name": name,
        },
        extract_shape(records),
        extract_dtype(records),
    )


def serialize_tensor(tensor: Tensor, name: OpValue = None) -> OpValue:
    """Serialize a tensor into a binary byte-string representation.

    Args:
        tensor (Tensor): Input tensor to serialize.
        name (OpValue): Optional operation name.

    Returns:
        OpValue: Scalar string/bytes tensor containing the serialized payload.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SerializeTensor", tensor, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "SerializeTensor",
        [tensor],
        {"name": name},
        extract_shape(tensor),
        extract_dtype(tensor),
    )


def parse_tensor(serialized: Tensor, out_type: DType, name: OpValue = None) -> OpValue:
    """Deserialize a binary byte-string tensor into a typed tensor.

    Args:
        serialized (Tensor): Scalar binary string tensor containing serialized tensor data.
        out_type (DType): Target data type of the deserialized tensor.
        name (OpValue): Optional operation name.

    Returns:
        OpValue: Deserialized tensor of dtype ``out_type``.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ParseTensor", serialized, out_type=out_type, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "ParseTensor",
        [serialized],
        {"out_type": out_type, "name": name},
        extract_shape(serialized),
        extract_dtype(serialized),
    )


class BinaryRecordOptions:
    """Compression configuration options for length-prefixed CRC-framed binary record streams."""

    def __init__(self, compression_type: str = "") -> None:
        """Initialize BinaryRecordOptions.

        Args:
            compression_type (str): Compression format (``''``, ``'GZIP'``, or ``'ZLIB'``).
        """
        self.compression_type = compression_type.upper() if compression_type else ""


class _ZlibCompressorProtocol(typing.Protocol):
    """Protocol for streaming ZLIB compressor objects."""

    def compress(self, data: bytes) -> bytes:
        """Compress an input byte chunk.

        Args:
            data (bytes): Uncompressed bytes.

        Returns:
            bytes: Compressed byte chunk.
        """
        ...

    def flush(self) -> bytes:
        """Flush remaining compressed bytes.

        Returns:
            bytes: Flushed compressed trailer bytes.
        """
        ...


class BinaryRecordWriter:
    """Writer for length-prefixed binary record files supporting masked CRC32 and compression."""

    def __init__(self, path: str, options: BinaryRecordOptions | None = None) -> None:
        """Initialize BinaryRecordWriter.

        Args:
            path (str): Destination file path or directory for records.
            options (BinaryRecordOptions | None): Optional compression configuration options.
        """
        self.path = path
        self.options = options or BinaryRecordOptions()
        self._closed = False

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        actual_path = os.path.join(path, "records.bin") if os.path.isdir(path) else path

        self._file = open(actual_path, "wb")
        self._gzip_compressor: gzip.GzipFile | None = None
        self._zlib_compressor: _ZlibCompressorProtocol | None = None
        if self.options.compression_type == "GZIP":
            self._gzip_compressor = gzip.GzipFile(fileobj=self._file, mode="wb")
        elif self.options.compression_type == "ZLIB":
            self._zlib_compressor = zlib.compressobj()

    @staticmethod
    def _mask_crc(crc: int) -> int:
        """Compute 32-bit masked CRC32 checksum.

        Args:
            crc (int): Unmasked CRC32 checksum.

        Returns:
            int: 32-bit masked CRC checksum.
        """
        return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF

    def write(self, record: bytes | str | None) -> None:
        """Write a record using length-prefixed binary framing with CRC32 checksums.

        Args:
            record (bytes | str | None): Record data to encode and write.

        Raises:
            ValueError: If writing to a closed writer.
        """
        if self._closed:
            raise ValueError("I/O operation on closed BinaryRecordWriter")

        if record is None:
            data = b""
        elif isinstance(record, str):
            data = record.encode("utf-8")
        else:
            data = bytes(record)
        length = len(data)

        len_bytes = struct.pack("<Q", length)
        len_crc = struct.pack("<I", self._mask_crc(zlib.crc32(len_bytes)))
        data_crc = struct.pack("<I", self._mask_crc(zlib.crc32(data)))

        framed_record = len_bytes + len_crc + data + data_crc

        if self.options.compression_type == "GZIP" and self._gzip_compressor is not None:
            self._gzip_compressor.write(framed_record)
        elif self.options.compression_type == "ZLIB" and self._zlib_compressor is not None:
            compressed: bytes = self._zlib_compressor.compress(framed_record)
            if compressed:
                self._file.write(compressed)
        else:
            self._file.write(framed_record)

    def close(self) -> None:
        """Flush and close underlying compression and file streams."""
        if self._closed:
            return

        if self.options.compression_type == "GZIP" and self._gzip_compressor is not None:
            self._gzip_compressor.close()
        elif self.options.compression_type == "ZLIB" and self._zlib_compressor is not None:
            self._file.write(self._zlib_compressor.flush())

        if not self._file.closed:
            self._file.flush()
            self._file.close()

        self._closed = True

    def __enter__(self) -> BinaryRecordWriter:
        """Enter context manager.

        Returns:
            BinaryRecordWriter: The open writer instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager and flush all buffers.

        Args:
            exc_type (type[BaseException] | None): Exception type if raised.
            exc_val (BaseException | None): Exception value if raised.
            exc_tb (TracebackType | None): Exception traceback if raised.
        """
        self.close()


def read_binary_records(path: str, compression_type: str = "") -> list[bytes]:
    """Read all records from a framed binary record file and verify masked CRC checksums.

    Args:
        path (str): Path to the binary record file.
        compression_type (str): Compression type (``''``, ``'GZIP'``, or ``'ZLIB'``).

    Returns:
        list[bytes]: List of verified record byte payloads.

    Raises:
        ValueError: If any length CRC or data CRC fails verification or EOF is truncated.
    """
    comp = compression_type.upper()
    if comp == "GZIP":
        with gzip.open(path, "rb") as gf:
            content = gf.read()
    elif comp == "ZLIB":
        with open(path, "rb") as f:
            content = zlib.decompress(f.read())
    else:
        with open(path, "rb") as f:
            content = f.read()

    records: list[bytes] = []
    offset = 0
    total = len(content)

    def _mask(crc: int) -> int:
        """Mask a 32-bit CRC checksum.

        Args:
            crc (int): Unmasked 32-bit CRC checksum.

        Returns:
            int: Masked 32-bit CRC checksum integer.
        """
        return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF

    while offset < total:
        if offset + 12 > total:
            break
        (length,) = struct.unpack("<Q", content[offset : offset + 8])
        (len_crc,) = struct.unpack("<I", content[offset + 8 : offset + 12])
        computed_len_crc = _mask(zlib.crc32(content[offset : offset + 8]))
        if len_crc != computed_len_crc:
            raise ValueError(f"Corrupted length CRC at offset {offset}")

        offset += 12
        if offset + length + 4 > total:
            raise ValueError(f"Unexpected EOF reading record data of length {length}")

        data = content[offset : offset + length]
        offset += length
        (data_crc,) = struct.unpack("<I", content[offset : offset + 4])
        computed_data_crc = _mask(zlib.crc32(data))
        if data_crc != computed_data_crc:
            raise ValueError("Corrupted data CRC in record")

        offset += 4
        records.append(data)

    return records


def _infer_broadcast_io_shape(*args: OpValue) -> OpValue:
    """Infer broadcasted output shape across positional tensor operands.

    Args:
        *args (OpValue): Positional tensor or scalar operands.

    Returns:
        OpValue: Broadcasted output shape tuple.
    """
    from ml_switcheroo_compiler.core.shape import broadcast_shapes

    shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
    if not shapes:
        return ()
    res = shapes[0]
    for s in shapes[1:]:
        res = broadcast_shapes(res, s)
    return res


@register_op("DecodeCsv")
class DecodeCsv(OpDef):
    """Universal delimited CSV record decoding operation definition."""

    op_name = "DecodeCsv"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer broadcasted output shape for DecodeCsv.

        Args:
            *args (OpValue): Positional tensor arguments.
            **kwargs (OpValue): Keyword operation attributes.

        Returns:
            OpValue: Inferred output shape tuple.
        """
        del kwargs
        return _infer_broadcast_io_shape(*args)


@register_op("SerializeTensor")
class SerializeTensor(OpDef):
    """Universal binary tensor serialization operation definition."""

    op_name = "SerializeTensor"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer broadcasted output shape for SerializeTensor.

        Args:
            *args (OpValue): Positional tensor arguments.
            **kwargs (OpValue): Keyword operation attributes.

        Returns:
            OpValue: Inferred output shape tuple.
        """
        del kwargs
        return _infer_broadcast_io_shape(*args)


@register_op("ParseTensor")
class ParseTensor(OpDef):
    """Universal binary tensor deserialization operation definition."""

    op_name = "ParseTensor"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer broadcasted output shape for ParseTensor.

        Args:
            *args (OpValue): Positional tensor arguments.
            **kwargs (OpValue): Keyword operation attributes.

        Returns:
            OpValue: Inferred output shape tuple.
        """
        del kwargs
        return _infer_broadcast_io_shape(*args)
