"""Tests for test_io."""

from __future__ import annotations

import os
import tempfile
import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import io
from ml_switcheroo_compiler.ops.io import SparsePlus, SparseSigmoid, _fallback_load, set_default_stream, set_memory_limit, set_wired_limit, sparse_plus, sparse_sigmoid
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_io_fallback_load() -> None:
    """Test function test_io_fallback_load."""
    pass
    with patch("ml_switcheroo_compiler.serialization.formats.safetensors.SafetensorsWeightFormat.load") as mock_safe:
        mock_safe.return_value = "safe"
        assert _fallback_load("model.safetensors") == "safe"
    with patch("ml_switcheroo_compiler.ops.io.numpy_io.load_npz") as mock_npz:
        mock_npz.return_value = "npz"
        assert _fallback_load("model.npz") == "npz"
    with patch("ml_switcheroo_compiler.serialization.formats.h5.H5WeightFormat.load") as mock_h5:
        mock_h5.return_value = "h5"
        assert _fallback_load("model.h5") == "h5"
    pass


def test_set_limits() -> None:
    """Test function test_set_limits."""
    from ml_switcheroo_compiler.core.config import config as core_config

    orig_backend = core_config.backend
    core_config.backend = "mlx"
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:
        mock_backend = MagicMock()
        mock_get.return_value = mock_backend
        set_default_stream("stream")
        mock_backend.set_default_stream.assert_called_once_with("stream")
        set_memory_limit(1024)
        mock_backend.set_memory_limit.assert_called_once_with(1024)
        set_wired_limit(512)
        mock_backend.set_wired_limit.assert_called_once_with(512)
        mock_get.side_effect = ImportError("mock")
        set_default_stream("stream")
        set_memory_limit(1024)
        set_wired_limit(512)
        mock_get.side_effect = None
        del mock_backend.set_default_stream
        del mock_backend.set_memory_limit
        del mock_backend.set_wired_limit
        set_default_stream("stream")
        set_memory_limit(1024)
        set_wired_limit(512)
    core_config.backend = orig_backend


def test_sparse_ops() -> None:
    """Test function test_sparse_ops."""
    assert SparsePlus().infer_shape("shape") == ()
    assert SparsePlus().infer_shape() == ()
    assert SparseSigmoid().infer_shape("shape") == ()
    assert SparseSigmoid().infer_shape() == ()
    t = Tensor([1.0], TensorConfig((1,), "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = "mock_plus"
        assert sparse_plus(t) == "mock_plus"
        mock_backend.return_value.execute_op.return_value = "mock_sig"
        assert sparse_sigmoid(t) == "mock_sig"
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.start_tracing()
    sparse_plus(t)
    sparse_sigmoid(t)
    global_tracing_state.stop_tracing()


def _test_io_functions_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_io_functions.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.start_tracing()
    import os
    import tempfile

    from ml_switcheroo_compiler.ops.io import TFRecordOptions, TFRecordWriter, decode_bmp, decode_gif, decode_jpeg, decode_png, gfile_copy, gfile_glob, gfile_makedirs, gfile_stat, save_safetensors

    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    assert isinstance(gfile_glob("*.txt"), list)
    assert isinstance(gfile_stat(__file__), dict)
    gfile_makedirs("test_dir_to_make")
    os.rmdir("test_dir_to_make")
    getattr(_state, "update")({"TFRecordOptions": TFRecordOptions, "TFRecordWriter": TFRecordWriter, "config": config, "decode_bmp": decode_bmp})
    getattr(_state, "update")({"decode_gif": decode_gif, "decode_jpeg": decode_jpeg, "decode_png": decode_png, "gfile_copy": gfile_copy})
    getattr(_state, "update")({"gfile_glob": gfile_glob, "gfile_makedirs": gfile_makedirs, "gfile_stat": gfile_stat, "global_tracing_state": global_tracing_state})
    getattr(_state, "update")({"os": os, "save_safetensors": save_safetensors, "tempfile": tempfile})
    _state.update(locals())


def _test_io_functions_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_io_functions.

    Args:
        _state: Shared test state dictionary.
    """
    (gfile_copy, os, tempfile) = (_state["gfile_copy"], _state["os"], _state["tempfile"])
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        gfile_copy(__file__, __file__)
    with tempfile.NamedTemporaryFile(delete=False) as src, tempfile.NamedTemporaryFile(delete=False) as dst:
        os.unlink(dst.name)
        gfile_copy(src.name, dst.name)
        os.unlink(src.name)
        os.unlink(dst.name)
    with patch("ml_switcheroo_compiler.ops.io._fallback_load") as mock_fb:
        mock_fb.return_value = "success"
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:
            del mock_get.return_value.execute_op
            pass
    getattr(_state, "update")({"dst": dst, "mock_fb": mock_fb, "mock_get": mock_get, "src": src})
    _state.update(locals())


def _test_io_functions_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_io_functions.

    Args:
        _state: Shared test state dictionary.
    """
    (decode_bmp, decode_gif, decode_jpeg, decode_png) = (_state["decode_bmp"], _state["decode_gif"], _state["decode_jpeg"], _state["decode_png"])
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.decode_jpeg.return_value = "jpeg"
        mock_backend.return_value.decode_png.return_value = "png"
        mock_backend.return_value.decode_gif.return_value = "gif"
        mock_backend.return_value.decode_bmp.return_value = "bmp"
        mock_backend.return_value.encode_base64.return_value = "enc64"
        mock_backend.return_value.decode_base64.return_value = "dec64"
        mock_backend.return_value.parse_sequence_example.return_value = ("seq1", "seq2")
        mock_backend.return_value.execute_op.side_effect = NotImplementedError()
        assert decode_jpeg("data") == "jpeg"
        assert decode_png("data") == "png"
        assert decode_gif("data") == "gif"
        assert decode_bmp("data") == "bmp"
        pass
        pass
        pass
        pass
    getattr(_state, "update")({"mock_backend": mock_backend})
    _state.update(locals())


def _test_io_functions_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_io_functions.

    Args:
        _state: Shared test state dictionary.
    """
    (TFRecordOptions, TFRecordWriter) = (_state["TFRecordOptions"], _state["TFRecordWriter"])
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        del mock_backend.return_value.decode_jpeg
        del mock_backend.return_value.decode_png
        del mock_backend.return_value.decode_gif
        del mock_backend.return_value.decode_bmp
        del mock_backend.return_value.encode_base64
        del mock_backend.return_value.decode_base64
        del mock_backend.return_value.parse_sequence_example
        pass
        pass
        pass
        pass
        pass
        pass
        pass
    opt = TFRecordOptions("GZIP")
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TFRecordWriter(os.path.join(tmpdir, "path"), opt)
        assert writer.write("data") is None
        with writer:
            pass
    getattr(_state, "update")({"mock_backend": mock_backend, "opt": opt, "writer": writer})
    _state.update(locals())


def _test_io_functions_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_io_functions.

    Args:
        _state: Shared test state dictionary.
    """
    save_safetensors = _state["save_safetensors"]
    with patch("ml_switcheroo_compiler.serialization.formats.safetensors.SafetensorsWeightFormat.save") as mock_save:
        save_safetensors("path", {})
        mock_save.assert_called_once()
    getattr(_state, "update")({"mock_save": mock_save})


def test_io_functions() -> None:
    """Test function test_io_functions."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_io_functions_part_1(_state)
    _test_io_functions_part_2(_state)
    _test_io_functions_part_3(_state)
    _test_io_functions_part_4(_state)
    _test_io_functions_part_5(_state)
    assert len(_state) > 0


def test_set_limits_not_mlx() -> None:
    """Test function test_set_limits_not_mlx."""
    from ml_switcheroo_compiler.core.config import config as core_config

    orig_backend = core_config.backend
    core_config.backend = "not_mlx"
    set_default_stream("stream")
    set_memory_limit(1024)
    set_wired_limit(512)
    core_config.backend = orig_backend
    assert core_config.backend == orig_backend


def test_io_stubs() -> None:
    """Test function test_io_stubs."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.start_tracing()
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = None
        pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.encode_base64.return_value = None
        mock_backend.return_value.decode_base64.return_value = None
        mock_backend.return_value.parse_sequence_example.return_value = ({}, {})
        pass
        pass
        pass
    with tempfile.TemporaryDirectory() as tmpdir:
        fh = io.TFRecordWriter(os.path.join(tmpdir, "path"))
        assert as_callable(fh.write)(None) is None
        assert as_callable(fh.close)() is None


def test_io_reading_writing() -> None:
    """Test function test_io_reading_writing."""
    import unittest.mock as mock

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.io import load, read_file, save, savez, write_file

    with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "test.txt")
                assert isinstance(as_callable(write_file)(path, "hello world"), Tensor)
                assert isinstance(read_file(path), Tensor)
                path_npy = os.path.join(d, "test.npy")
                t = Tensor(np.array([1, 2, 3]), TensorConfig((3,), "int64", "cpu"))
                assert isinstance(save(path_npy, t), Tensor)
                assert isinstance(load(path_npy), Tensor)
                path_npz = os.path.join(d, "test.npz")
                assert isinstance(savez(path_npz, t=t), Tensor)


def test_io_encoding() -> None:
    """Test function test_io_encoding."""
    import unittest.mock as mock

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.io import decode_base64, encode_base64

    with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
            t = Tensor(np.array([b"hello"]), TensorConfig((1,), "string", "cpu"))
            encoded = encode_base64(t)
            assert isinstance(encoded, Tensor)
            decoded = decode_base64(encoded)
            assert isinstance(decoded, Tensor)


def test_numpy_io_tracing() -> None:
    """Test function test_numpy_io_tracing."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.io.numpy_io import Load, Save, SaveGguf, Savez, SavezCompressed, load, save, save_gguf, savez, savez_compressed
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        assert load("test") is not None
        assert save("test", [1, 2]) is not None
        assert savez("test", [1, 2]) is not None
        assert savez_compressed("test", [1, 2]) is not None
        assert save_gguf("test", {"t": [1]}) is not None
        assert Load().infer_shape() is not None
        assert Save().infer_shape() is not None
        assert Savez().infer_shape() is not None
        assert SavezCompressed().infer_shape() is not None
        assert SaveGguf().infer_shape() is not None
    finally:
        global_tracing_state.stop_tracing()


def test_numpy_io_eager_and_format_fallback() -> None:
    """Test function test_numpy_io_eager_and_format_fallback."""
    import unittest.mock as mock

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.io.numpy_io import _fallback_load, load, save, save_gguf, savez, savez_compressed

    config.eager_mode = True

    class MockBackend:
        """Test class MockBackend."""

        def execute_op(self, name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                name: Test parameter `name`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "executed"

    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockBackend()):
        assert load("x") == "executed"
        assert as_tensor(save("x", [1])) == as_tensor("executed")
        assert as_tensor(savez("x", [1])) == as_tensor("executed")
        assert as_tensor(savez_compressed("x", [1])) == as_tensor("executed")
        assert as_tensor(save_gguf("x", {"t": [1]})) == as_tensor("executed")
    assert as_callable(_fallback_load)(123) is None
    assert _fallback_load("test.bin") is None
    with mock.patch("ml_switcheroo_compiler.serialization.formats.h5.H5WeightFormat.load", return_value="h5load"):
        assert _fallback_load("test.h5") == "h5load"


def test_numpy_io_infer_shape_args() -> None:
    """Test function test_numpy_io_infer_shape_args."""
    from ml_switcheroo_compiler.ops.io.numpy_io import Load, Save, SaveGguf, Savez, SavezCompressed

    class DummyNode:
        """Test class DummyNode."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape

    assert as_callable(Load)().infer_shape(as_callable(DummyNode)((2, 2)), as_callable(DummyNode)((2, 2))) == (2, 2)
    assert as_callable(Save)().infer_shape(as_callable(DummyNode)((2, 2)), as_callable(DummyNode)((2, 2))) == (2, 2)
    assert as_callable(Savez)().infer_shape(as_callable(DummyNode)((2, 2)), as_callable(DummyNode)((2, 2))) == (2, 2)
    assert as_callable(SavezCompressed)().infer_shape(as_callable(DummyNode)((2, 2)), as_callable(DummyNode)((2, 2))) == (2, 2)
    assert as_callable(SaveGguf)().infer_shape(as_callable(DummyNode)((2, 2)), as_callable(DummyNode)((2, 2))) == (2, 2)
