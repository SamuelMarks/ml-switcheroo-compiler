# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""I/O and memory operations."""

from __future__ import annotations

import glob
import os
import shutil
from typing import Any

from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from ml_switcheroo_compiler.serialization.utils import load_npz


def decode_csv(records: Tensor, record_defaults, field_delim=",", use_quote_delim=True, na_value="", select_cols=None, name=None) -> Any:
    """Decode csv.

    Args:
        records (Tensor): The records parameter.
        record_defaults (list): The record_defaults parameter.
        field_delim (str): The field_delim parameter.
        use_quote_delim (bool): The use_quote_delim parameter.
        na_value (str): The na_value parameter.
        select_cols (list): The select_cols parameter.
        name (str): The name parameter.

    Returns:
        Any: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DecodeCsv", records, record_defaults=record_defaults, field_delim=field_delim, use_quote_delim=use_quote_delim, na_value=na_value, select_cols=select_cols, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DecodeCsv", [records], {"record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}, getattr(records, "shape", ()), getattr(records, "dtype", "float32"))


def parse_example(serialized: Tensor, features, example_names=None, name=None) -> Any:
    """Parse example.

    Args:
        serialized (Tensor): The serialized parameter.
        features (dict): The features parameter.
        example_names (Tensor): The example_names parameter.
        name (str): The name parameter.

    Returns:
        Any: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ParseExample", serialized, features=features, example_names=example_names, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ParseExample", [serialized], {"features": features, "example_names": example_names, "name": name}, getattr(serialized, "shape", ()), getattr(serialized, "dtype", "float32"))


def serialize_tensor(tensor: Tensor, name=None):
    """Serialize tensor.

    Args:
        tensor (Tensor): The tensor parameter.
        name (str): The name parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SerializeTensor", tensor, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SerializeTensor", [tensor], {"name": name}, getattr(tensor, "shape", ()), getattr(tensor, "dtype", "float32"))


def parse_tensor(serialized: Tensor, out_type: DType, name=None):
    """Parse tensor.

    Args:
        serialized (Tensor): The serialized parameter.
        out_type (DType): The out_type parameter.
        name (str): The name parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ParseTensor", serialized, out_type=out_type, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ParseTensor", [serialized], {"out_type": out_type, "name": name}, getattr(serialized, "shape", ()), getattr(serialized, "dtype", "float32"))


def parse_single_sequence_example(serialized: Tensor, context_features=None, sequence_features=None, example_names=None, name=None) -> Any:
    """Parse single sequence example.

    Args:
        serialized (Tensor): The serialized parameter.
        context_features (dict): The context_features parameter.
        sequence_features (dict): The sequence_features parameter.
        example_names (Tensor): The example_names parameter.
        name (str): The name parameter.

    Returns:
        Any: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ParseSequenceExample", serialized, context_features=context_features, sequence_features=sequence_features, example_names=example_names, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ParseSequenceExample", [serialized], {"context_features": context_features, "sequence_features": sequence_features, "example_names": example_names, "name": name}, getattr(serialized, "shape", ()), getattr(serialized, "dtype", "float32"))


parse_sequence_example = parse_single_sequence_example


class TFRecordOptions:
    """Options for TFRecordWriter."""

    def __init__(self, compression_type: str = "") -> None:
        """Initialize.

        Args:
            compression_type (str): Compression format ('', 'GZIP', or 'ZLIB').
        """
        self.compression_type = compression_type.upper() if compression_type else ""


class TFRecordWriter:
    """Writer for TFRecord format supporting binary framing, masked CRC32, and compression."""

    def __init__(self, path: str, options: TFRecordOptions | None = None) -> None:
        """Initialize TFRecordWriter.

        Args:
            path (str): Destination file path for records.
            options (TFRecordOptions): Optional compression configuration options.
        """
        import os

        self.path = path
        self.options = options or TFRecordOptions()
        self._closed = False

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # If path points to an existing directory (e.g. repo directory 'path/'), write to a file inside it
        actual_path = os.path.join(path, "records.tfrecord") if os.path.isdir(path) else path

        self._file = open(actual_path, "wb")
        self._compressor: Any = None
        if self.options.compression_type == "GZIP":
            import gzip

            self._compressor = gzip.GzipFile(fileobj=self._file, mode="wb")
        elif self.options.compression_type == "ZLIB":
            import zlib

            self._compressor = zlib.compressobj()

    @staticmethod
    def _mask_crc(crc: int) -> int:
        """Compute masked CRC32 according to TensorFlow TFRecord standard.

        Args:
            crc (int): Unmasked CRC32 checksum.

        Returns:
            int: 32-bit masked CRC checksum.
        """
        return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF

    def write(self, record: bytes | str) -> None:
        """Write record using standard length-prefixed binary framing.

        Args:
            record (bytes | str): Record data to encode and write.

        Raises:
            ValueError: If writing to a closed writer.
        """
        if self._closed:
            raise ValueError("I/O operation on closed TFRecordWriter")

        import struct
        import zlib

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

        if self.options.compression_type == "GZIP":
            self._compressor.write(framed_record)
        elif self.options.compression_type == "ZLIB":
            compressed = self._compressor.compress(framed_record)
            if compressed:
                self._file.write(compressed)
        else:
            self._file.write(framed_record)

    def close(self) -> None:
        """Flush and close underlying file streams."""
        if self._closed:
            return

        if self.options.compression_type == "GZIP" and self._compressor is not None:
            self._compressor.close()
        elif self.options.compression_type == "ZLIB" and self._compressor is not None:
            flushed = self._compressor.flush()
            if flushed:
                self._file.write(flushed)

        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

        self._closed = True

    def __enter__(self) -> TFRecordWriter:
        """Enter context manager.

        Returns:
            TFRecordWriter: The open writer instance.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and flush all buffers.

        Args:
            exc_type (Any): Exception type if raised.
            exc_val (Any): Exception value if raised.
            exc_tb (Any): Exception traceback if raised.
        """
        self.close()


def read_tfrecords(path: str, compression_type: str = "") -> list[bytes]:
    """Read all records from a TFRecord file and verify masked CRC checksums.

    Args:
        path (str): Path to the TFRecord file.
        compression_type (str): Compression type ('', 'GZIP', or 'ZLIB').

    Returns:
        list[bytes]: List of verified record byte payloads.

    Raises:
        ValueError: If any length CRC or data CRC fails verification.
    """
    import struct
    import zlib

    comp = compression_type.upper()
    if comp == "GZIP":
        import gzip

        with gzip.open(path, "rb") as f:
            content = f.read()
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
        """Mask a 32-bit CRC checksum according to the TFRecord protocol.

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


@register_op("DecodeCsv")
class DecodeCsv(OpDef):
    """DecodeCsv operation."""

    op_name = "DecodeCsv"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("ParseExample")
class ParseExample(OpDef):
    """ParseExample operation."""

    op_name = "ParseExample"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("SerializeTensor")
class SerializeTensor(OpDef):
    """SerializeTensor operation."""

    op_name = "SerializeTensor"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("ParseTensor")
class ParseTensor(OpDef):
    """ParseTensor operation."""

    op_name = "ParseTensor"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("ParseSequenceExample")
class ParseSequenceExample(OpDef):
    """ParseSequenceExample operation."""

    op_name = "ParseSequenceExample"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res
