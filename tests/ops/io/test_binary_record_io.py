"""Exhaustive unit tests for BinaryRecordWriter binary framing, masked CRC, compression, and universal record I/O.

ops.
"""

from __future__ import annotations

import os
import tempfile
import typing

import pytest

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.ops.io.record_io import BinaryRecordOptions, BinaryRecordWriter, read_binary_records
from tests.testing_utils import require_callable

if typing.TYPE_CHECKING:
    pass


def test_binary_record_writer_uncompressed() -> None:
    """Test uncompressed BinaryRecordWriter writing, directory target, and readback with CRC validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec_path = os.path.join(tmpdir, "test.bin")

        records_to_write: list[bytes | str] = [
            b"First binary record payload",
            b"Second record with binary data: \x00\x01\x02\xff",
            "String payload that gets encoded as UTF-8",
            b"",
            b"Final payload",
        ]

        with BinaryRecordWriter(rec_path) as writer:
            for rec in records_to_write:
                writer.write(rec)

        loaded = read_binary_records(rec_path)
        assert len(loaded) == len(records_to_write)
        for written, read_val in zip(records_to_write, loaded):
            expected = written.encode("utf-8") if isinstance(written, str) else written
            assert read_val == expected

        dir_target = os.path.join(tmpdir, "subdir_records")
        os.makedirs(dir_target, exist_ok=True)
        with BinaryRecordWriter(dir_target) as dir_writer:
            dir_writer.write(b"dir-record")
        assert read_binary_records(os.path.join(dir_target, "records.bin")) == [b"dir-record"]


def test_binary_record_writer_compression_modes() -> None:
    """Test BinaryRecordWriter writing and reading under GZIP and ZLIB compression formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gzip_path = os.path.join(tmpdir, "test_gzip.bin")
        gzip_opts = BinaryRecordOptions("GZIP")
        assert gzip_opts.compression_type == "GZIP"

        with BinaryRecordWriter(gzip_path, gzip_opts) as writer:
            writer.write(b"GZIP Compressed Record 1")
            writer.write(b"GZIP Compressed Record 2")

        loaded_gzip = read_binary_records(gzip_path, compression_type="GZIP")
        assert loaded_gzip == [b"GZIP Compressed Record 1", b"GZIP Compressed Record 2"]

        zlib_path = os.path.join(tmpdir, "test_zlib.bin")
        zlib_opts = BinaryRecordOptions("zlib")
        assert zlib_opts.compression_type == "ZLIB"

        with BinaryRecordWriter(zlib_path, zlib_opts) as writer:
            writer.write(b"ZLIB Compressed Record A")
            writer.write(b"ZLIB Compressed Record B")

        loaded_zlib = read_binary_records(zlib_path, compression_type="ZLIB")
        assert loaded_zlib == [b"ZLIB Compressed Record A", b"ZLIB Compressed Record B"]


def test_binary_record_writer_closed_error() -> None:
    """Test that writing to a closed BinaryRecordWriter raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec_path = os.path.join(tmpdir, "closed.bin")
        writer = BinaryRecordWriter(rec_path)
        writer.close()
        writer.close()

        with pytest.raises(ValueError, match="closed BinaryRecordWriter"):
            writer.write(b"Data")


def test_binary_record_crc_corruption_detection() -> None:
    """Test corrupted CRC length, CRC data, and truncated records detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec_path = os.path.join(tmpdir, "corrupt.bin")

        with BinaryRecordWriter(rec_path) as writer:
            writer.write(b"Valid data payload")

        with open(rec_path, "rb") as f:
            valid_bytes = bytearray(f.read())

        corrupt_len_bytes = bytearray(valid_bytes)
        corrupt_len_bytes[9] ^= 0xFF
        corrupt_len_path = os.path.join(tmpdir, "corrupt_len.bin")
        with open(corrupt_len_path, "wb") as f:
            f.write(corrupt_len_bytes)

        with pytest.raises(ValueError, match="Corrupted length CRC"):
            read_binary_records(corrupt_len_path)

        corrupt_data_bytes = bytearray(valid_bytes)
        corrupt_data_bytes[-1] ^= 0xFF
        corrupt_data_path = os.path.join(tmpdir, "corrupt_data.bin")
        with open(corrupt_data_path, "wb") as f:
            f.write(corrupt_data_bytes)

        with pytest.raises(ValueError, match="Corrupted data CRC"):
            read_binary_records(corrupt_data_path)

        trunc_bytes = valid_bytes[:15]
        trunc_path = os.path.join(tmpdir, "truncated.bin")
        with open(trunc_path, "wb") as f:
            f.write(trunc_bytes)

        with pytest.raises(ValueError, match="Unexpected EOF"):
            read_binary_records(trunc_path)

        trailing_path = os.path.join(tmpdir, "trailing.bin")
        with open(trailing_path, "wb") as f:
            f.write(b"1234")
        assert read_binary_records(trailing_path) == []


def test_binary_record_writer_edge_cases() -> None:
    """Test BinaryRecordWriter with None record, already-closed underlying file, and empty compressor flush."""
    with tempfile.TemporaryDirectory() as tmpdir:
        none_path = os.path.join(tmpdir, "none.bin")
        with BinaryRecordWriter(none_path) as writer:
            writer.write(None)
        records = read_binary_records(none_path)
        assert records == [b""]

        closed_file_path = os.path.join(tmpdir, "closed_file.bin")
        writer_closed = BinaryRecordWriter(closed_file_path)
        writer_closed._file.close()
        writer_closed.close()
        assert writer_closed._closed

        class DummyCompressor:
            """Stub zlib compressor returning empty bytes on compress and flush."""

            def compress(self, data: bytes) -> bytes:
                """Return empty compressed chunk.

                Args:
                    data (bytes): Uncompressed bytes.

                Returns:
                    bytes: Empty bytes.
                """
                del data
                return b""

            def flush(self) -> bytes:
                """Return empty flushed bytes.

                Returns:
                    bytes: Empty bytes.
                """
                return b""

        zlib_empty_path = os.path.join(tmpdir, "zlib_empty.bin")
        writer_zlib = BinaryRecordWriter(zlib_empty_path, BinaryRecordOptions("ZLIB"))
        writer_zlib._zlib_compressor = DummyCompressor()
        writer_zlib.write(b"buffered")
        writer_zlib.close()
        assert writer_zlib._closed


def test_generic_decode_csv_and_serialize_tensor_eager() -> None:
    """Verify generic DecodeCsv and SerializeTensor handlers in global_eager_registry."""
    import numpy as np

    decode_fn = global_eager_registry.get("DecodeCsv")
    cols = require_callable(decode_fn)(np, require_callable("\n".join)(["1.5,2.5", ",4.0"]), record_defaults=[0.0, 9.0])
    np.testing.assert_allclose(cols[0], np.array([1.5, 0.0]))
    np.testing.assert_allclose(cols[1], np.array([2.5, 4.0]))

    empty_cols = require_callable(decode_fn)(np, "", record_defaults=[3.0], field_delim="invalid_multi")
    np.testing.assert_allclose(empty_cols[0], np.array(3.0))
    assert require_callable(decode_fn)(np, "1,2", record_defaults=123) == ()

    ser_fn = global_eager_registry.get("SerializeTensor")
    assert require_callable(ser_fn)(np) == b"[]"
    assert require_callable(ser_fn)(np, np.array([1, 2])) == b"[1, 2]"
    assert require_callable(ser_fn)(np, 42) == b"42"
