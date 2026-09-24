"""Exhaustive unit tests for TFRecordWriter binary framing, masked CRC, and compression."""

from __future__ import annotations

import os
import tempfile

import pytest

from ml_switcheroo_compiler.ops.io.tf_io import (
    TFRecordOptions,
    TFRecordWriter,
    read_tfrecords,
)


def test_tfrecord_writer_uncompressed() -> None:
    """Test uncompressed TFRecord writing and readback with CRC validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec_path = os.path.join(tmpdir, "test.tfrecord")

        records_to_write = [
            b"First TFRecord payload",
            b"Second record with binary data: \x00\x01\x02\xff",
            "String payload that gets encoded as UTF-8",
            b"",  # Empty payload
            b"Final payload",
        ]

        with TFRecordWriter(rec_path) as writer:
            for rec in records_to_write:
                writer.write(rec)

        # Read back and verify exact byte equality
        loaded = read_tfrecords(rec_path)
        assert len(loaded) == len(records_to_write)
        for written, read_val in zip(records_to_write, loaded):
            expected = written.encode("utf-8") if isinstance(written, str) else written
            assert read_val == expected


def test_tfrecord_writer_compression_modes() -> None:
    """Test TFRecord writing and reading under GZIP and ZLIB compression formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. GZIP
        gzip_path = os.path.join(tmpdir, "test_gzip.tfrecord")
        gzip_opts = TFRecordOptions("GZIP")
        assert gzip_opts.compression_type == "GZIP"

        with TFRecordWriter(gzip_path, gzip_opts) as writer:
            writer.write(b"GZIP Compressed Record 1")
            writer.write(b"GZIP Compressed Record 2")

        loaded_gzip = read_tfrecords(gzip_path, compression_type="GZIP")
        assert len(loaded_gzip) == 2
        assert loaded_gzip[0] == b"GZIP Compressed Record 1"
        assert loaded_gzip[1] == b"GZIP Compressed Record 2"

        # 2. ZLIB
        zlib_path = os.path.join(tmpdir, "test_zlib.tfrecord")
        zlib_opts = TFRecordOptions("zlib")
        assert zlib_opts.compression_type == "ZLIB"

        with TFRecordWriter(zlib_path, zlib_opts) as writer:
            writer.write(b"ZLIB Compressed Record A")
            writer.write(b"ZLIB Compressed Record B")

        loaded_zlib = read_tfrecords(zlib_path, compression_type="ZLIB")
        assert len(loaded_zlib) == 2
        assert loaded_zlib[0] == b"ZLIB Compressed Record A"
        assert loaded_zlib[1] == b"ZLIB Compressed Record B"


def test_tfrecord_writer_closed_error() -> None:
    """Test that writing to a closed TFRecordWriter raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec_path = os.path.join(tmpdir, "closed.tfrecord")
        writer = TFRecordWriter(rec_path)
        writer.close()
        writer.close()  # Double close is safe

        with pytest.raises(ValueError, match="closed TFRecordWriter"):
            writer.write(b"Data")


def test_tfrecord_crc_corruption_detection() -> None:
    """Test corrupted CRC length, CRC data, and truncated records detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rec_path = os.path.join(tmpdir, "corrupt.tfrecord")

        with TFRecordWriter(rec_path) as writer:
            writer.write(b"Valid data payload")

        with open(rec_path, "rb") as f:
            valid_bytes = bytearray(f.read())

        # 1. Corrupt length CRC (bytes 8..11)
        corrupt_len_bytes = bytearray(valid_bytes)
        corrupt_len_bytes[9] ^= 0xFF
        corrupt_len_path = os.path.join(tmpdir, "corrupt_len.tfrecord")
        with open(corrupt_len_path, "wb") as f:
            f.write(corrupt_len_bytes)

        with pytest.raises(ValueError, match="Corrupted length CRC"):
            read_tfrecords(corrupt_len_path)

        # 2. Corrupt data CRC (last 4 bytes)
        corrupt_data_bytes = bytearray(valid_bytes)
        corrupt_data_bytes[-1] ^= 0xFF
        corrupt_data_path = os.path.join(tmpdir, "corrupt_data.tfrecord")
        with open(corrupt_data_path, "wb") as f:
            f.write(corrupt_data_bytes)

        with pytest.raises(ValueError, match="Corrupted data CRC"):
            read_tfrecords(corrupt_data_path)

        # 3. Truncated record data (short length)
        trunc_bytes = valid_bytes[:15]
        trunc_path = os.path.join(tmpdir, "truncated.tfrecord")
        with open(trunc_path, "wb") as f:
            f.write(trunc_bytes)

        with pytest.raises(ValueError, match="Unexpected EOF"):
            read_tfrecords(trunc_path)
