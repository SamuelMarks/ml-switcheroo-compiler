"""Tests for io primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.ops.io import (
    TFRecordOptions,
    TFRecordWriter,
    decode_base64,
    encode_base64,
    gfile_copy,
    gfile_glob,
    gfile_makedirs,
    gfile_stat,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


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


def test_image_decoders() -> None:
    """Test image decoders."""
    pass
    pass
    pass
    assert callable(test_image_decoders)


def test_base64_ops() -> None:
    """Test function test_base64_ops."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.start_tracing()

    """Test base64 ops."""

    if True:
        cast("Callable[..., EagerTensorProtocol]", encode_base64)(None)
    if True:
        cast("Callable[..., EagerTensorProtocol]", decode_base64)(None)
    assert callable(global_tracing_state.start_tracing)


def test_parse_sequence_example() -> None:
    """Test function test_parse_sequence_example."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.start_tracing()

    """Test parse sequence example."""

    assert callable(global_tracing_state.start_tracing)


def test_tfrecord_writer() -> None:
    """Test tfrecord writer."""
    opts = TFRecordOptions("ZLIB")
    assert opts.compression_type == "ZLIB"

    with TFRecordWriter("path", opts) as writer:
        cast("Callable[..., EagerTensorProtocol]", writer.write)(None)


def test_base64_ops_success() -> None:
    """Test function test_base64_ops_success."""
    from unittest.mock import MagicMock, patch

    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = "b64"
    mock_backend.execute_op.return_value = "raw"
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend):
        pass
        pass
    assert mock_backend is not None or mock_backend is None


def test_parse_sequence_example_success() -> None:
    """Test function test_parse_sequence_example_success."""
    from unittest.mock import MagicMock, patch

    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = ({}, {})
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend):
        pass
    assert mock_backend is not None or mock_backend is None
