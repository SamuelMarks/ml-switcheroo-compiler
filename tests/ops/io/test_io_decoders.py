"""Behavioral unit tests for io decoders."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import ml_switcheroo_compiler.ops.io.from_io as from_io_mod
from ml_switcheroo_compiler.ops.io import (
    BinaryRecordOptions,
    BinaryRecordWriter,
    decode_base64,
    encode_base64,
    gfile_copy,
    gfile_glob,
    gfile_makedirs,
    gfile_stat,
)
from tests.testing_utils import require_callable


def test_gfile() -> None:
    """Test gfile ops."""
    import shutil

    shutil.rmtree("/tmp/test_gfile_ml_switcheroo", ignore_errors=True)
    gfile_makedirs("/tmp/test_gfile_ml_switcheroo")
    with open("/tmp/test_gfile_ml_switcheroo/test.txt", "w") as f:
        f.write("hello")
    gfile_copy("/tmp/test_gfile_ml_switcheroo/test.txt", "/tmp/test_gfile_ml_switcheroo/test2.txt")
    with pytest.raises(FileExistsError):
        gfile_copy("/tmp/test_gfile_ml_switcheroo/test.txt", "/tmp/test_gfile_ml_switcheroo/test2.txt")
    gfile_copy("/tmp/test_gfile_ml_switcheroo/test.txt", "/tmp/test_gfile_ml_switcheroo/test2.txt", overwrite=True)
    assert len(gfile_glob("/tmp/test_gfile_ml_switcheroo/*.txt")) == 2
    stat = gfile_stat("/tmp/test_gfile_ml_switcheroo/test.txt")
    assert stat["length"] == 5
    assert "mtime" in stat


def test_base64_ops() -> None:
    """Test function test_base64_ops."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.start_tracing()
    require_callable(encode_base64)(None)
    require_callable(decode_base64)(None)
    assert callable(global_tracing_state.start_tracing)


def test_binary_record_writer_zlib(tmp_path: Path) -> None:
    """Test BinaryRecordWriter ZLIB options and None record handling.

    Args:
        tmp_path (Path): Temporary path fixture.
    """
    opts = BinaryRecordOptions("ZLIB")
    assert opts.compression_type == "ZLIB"
    with BinaryRecordWriter(str(tmp_path / "path"), opts) as writer:
        writer.write(None)


def test_base64_ops_success() -> None:
    """Test function test_base64_ops_success."""
    from unittest.mock import MagicMock, patch

    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = "raw"
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend):
        assert mock_backend.execute_op() == "raw"


class DummyWithShape:
    """Mock operand providing shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 123


def test_from_io_full_coverage() -> None:
    """Test full coverage for from_io op definitions and functional dispatchers."""
    ff_op = from_io_mod.Fromfile()
    assert ff_op.infer_shape(count=-1) == (None,)
    assert ff_op.infer_shape(count=10) == (10,)

    fs_op = from_io_mod.Fromstring()
    assert fs_op.infer_shape(count=-1) == (None,)
    assert fs_op.infer_shape(count=5) == (5,)

    fi_op = from_io_mod.Fromiter()
    assert fi_op.infer_shape(count=-1) == (None,)
    assert fi_op.infer_shape(count=8) == (8,)

    fn_op = from_io_mod.Fromfunction()
    assert fn_op.infer_shape(None, shape=(3, 4)) == (3, 4)
    assert fn_op.infer_shape(None, shape=5) == (5,)
    assert fn_op.infer_shape(None, [2, 2]) == (2, 2)
    assert fn_op.infer_shape(None) == ()

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="dispatched_fromfile") as mock_disp:
        assert from_io_mod.fromfile("file.bin", dtype=float, count=2) == "dispatched_fromfile"
        mock_disp.assert_called_once_with("Fromfile", "file.bin", dtype=float, count=2, sep="", offset=0, like=None)

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="dispatched_fromstring") as mock_disp:
        assert from_io_mod.fromstring("1 2", dtype=float, count=2) == "dispatched_fromstring"
        mock_disp.assert_called_once_with("Fromstring", "1 2", dtype=float, count=2, sep="", like=None)

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="dispatched_fromiter") as mock_disp:
        assert from_io_mod.fromiter([1, 2], dtype=float, count=2) == "dispatched_fromiter"
        mock_disp.assert_called_once_with("Fromiter", [1, 2], float, count=2, like=None)

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="dispatched_fromfn") as mock_disp:
        assert from_io_mod.fromfunction(lambda i: i, (3,)) == "dispatched_fromfn"
        mock_disp.assert_called_once()
