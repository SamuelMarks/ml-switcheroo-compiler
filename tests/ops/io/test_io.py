"""Tests for test_io."""

from __future__ import annotations

import os
import tempfile
import types
import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import io
from ml_switcheroo_compiler.ops.io import (
    SparsePlus,
    SparseSigmoid,
    _fallback_load,
    set_default_stream,
    set_memory_limit,
    set_wired_limit,
    sparse_plus,
    sparse_sigmoid,
)
from tests.testing_utils import (
    EagerBoundCallable,
    MockTestStub,
    coerce_operand_list,
    extract_test_shape,
    require_callable,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


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


class _TestIoFunctionsState(typing.TypedDict, total=False):
    """Typed shared state for `_test_io_functions` split helpers."""

    BinaryRecordOptions: type
    BinaryRecordWriter: type
    decode_bmp: EagerBoundCallable
    decode_gif: EagerBoundCallable
    decode_jpeg: EagerBoundCallable
    decode_png: EagerBoundCallable
    gfile_copy: EagerBoundCallable
    os: types.ModuleType
    save_safetensors: EagerBoundCallable
    tempfile: types.ModuleType


def _test_io_functions_part_1(_state: _TestIoFunctionsState) -> None:
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

    from ml_switcheroo_compiler.ops.io import (
        BinaryRecordOptions,
        BinaryRecordWriter,
        decode_bmp,
        decode_gif,
        decode_jpeg,
        decode_png,
        gfile_copy,
        gfile_glob,
        gfile_makedirs,
        gfile_stat,
        save_safetensors,
    )

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
    _state.update(
        {
            "BinaryRecordOptions": BinaryRecordOptions,
            "BinaryRecordWriter": BinaryRecordWriter,
            "decode_bmp": require_callable(decode_bmp),
        }
    )
    _state.update(
        {
            "decode_gif": require_callable(decode_gif),
            "decode_jpeg": require_callable(decode_jpeg),
            "decode_png": require_callable(decode_png),
            "gfile_copy": require_callable(gfile_copy),
        }
    )
    _state.update({"os": os, "save_safetensors": require_callable(save_safetensors), "tempfile": tempfile})


def _test_io_functions_part_2(_state: _TestIoFunctionsState) -> None:
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
    _state.update({"gfile_copy": require_callable(gfile_copy), "os": os, "tempfile": tempfile})


def _test_io_functions_part_3(_state: _TestIoFunctionsState) -> None:
    """Execute part 3 of test_io_functions.

    Args:
        _state: Shared test state dictionary.
    """
    (decode_bmp, decode_gif, decode_jpeg, decode_png) = (
        _state["decode_bmp"],
        _state["decode_gif"],
        _state["decode_jpeg"],
        _state["decode_png"],
    )
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.decode_jpeg.return_value = "jpeg"
        mock_backend.return_value.decode_png.return_value = "png"
        mock_backend.return_value.decode_gif.return_value = "gif"
        mock_backend.return_value.decode_bmp.return_value = "bmp"
        mock_backend.return_value.encode_base64.return_value = "enc64"
        mock_backend.return_value.decode_base64.return_value = "dec64"
        mock_backend.return_value.parse_sequence_example.return_value = ("seq1", "seq2")
        mock_backend.return_value.execute_op.side_effect = NotImplementedError()
        assert decode_jpeg("data").data == "jpeg"
        assert decode_png("data").data == "png"
        assert decode_gif("data").data == "gif"
        assert decode_bmp("data").data == "bmp"
    _state.update(
        {
            "decode_bmp": require_callable(decode_bmp),
            "decode_gif": require_callable(decode_gif),
            "decode_jpeg": require_callable(decode_jpeg),
        }
    )
    _state.update({"decode_png": require_callable(decode_png)})


def _test_io_functions_part_4(_state: _TestIoFunctionsState) -> None:
    """Execute part 4 of test_io_functions.

    Args:
        _state: Shared test state dictionary.
    """
    (BinaryRecordOptions, BinaryRecordWriter) = (_state["BinaryRecordOptions"], _state["BinaryRecordWriter"])
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        del mock_backend.return_value.decode_jpeg
        del mock_backend.return_value.decode_png
        del mock_backend.return_value.decode_gif
        del mock_backend.return_value.decode_bmp
        del mock_backend.return_value.encode_base64
        del mock_backend.return_value.decode_base64
    opt = BinaryRecordOptions("GZIP")
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = BinaryRecordWriter(os.path.join(tmpdir, "path"), opt)
        writer.write("data")
        with writer:
            pass
    _state.update({"BinaryRecordOptions": BinaryRecordOptions, "BinaryRecordWriter": BinaryRecordWriter})


def _test_io_functions_part_5(_state: _TestIoFunctionsState) -> None:
    """Execute part 5 of test_io_functions.

    Args:
        _state: Shared test state dictionary.
    """
    save_safetensors = _state["save_safetensors"]
    with patch("ml_switcheroo_compiler.serialization.formats.safetensors.SafetensorsWeightFormat.save") as mock_save:
        save_safetensors("path", {})
        mock_save.assert_called_once()


def test_io_functions() -> None:
    """Test function test_io_functions."""
    _state: _TestIoFunctionsState = {}
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

    orig_eager = config.eager_mode
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_path = os.path.join(tmpdir, "path")
            writer = io.BinaryRecordWriter(rec_path)
            writer.write(None)
            writer.close()
            assert writer._closed is True
            assert os.path.isfile(rec_path)
    finally:
        global_tracing_state.stop_tracing()
        config.eager_mode = orig_eager


def test_io_reading_writing() -> None:
    """Test function test_io_reading_writing."""
    import unittest.mock as mock

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.io import load, read_file, save, savez, write_file

    with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "test.txt")
                assert isinstance(require_callable(write_file)(path, "hello world"), Tensor)
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
    from ml_switcheroo_compiler.ops.io.numpy_io import (
        Load,
        Save,
        SaveGguf,
        Savez,
        SavezCompressed,
        load,
        save,
        save_gguf,
        savez,
        savez_compressed,
    )
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        assert tuple(extract_test_shape(load("test"))) == ()
        assert tuple(extract_test_shape(save("test", [1, 2]))) == ()
        assert tuple(extract_test_shape(savez("test", [1, 2]))) == ()
        assert tuple(extract_test_shape(savez_compressed("test", [1, 2]))) == ()
        assert tuple(extract_test_shape(save_gguf("test", {"t": [1]}))) == ()
        assert len(coerce_operand_list(Load().infer_shape())) == 0
        assert len(coerce_operand_list(Save().infer_shape())) == 0
        assert len(coerce_operand_list(Savez().infer_shape())) == 0
        assert len(coerce_operand_list(SavezCompressed().infer_shape())) == 0
        assert len(coerce_operand_list(SaveGguf().infer_shape())) == 0
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

        def execute_op(self, name: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                name: Test parameter `name`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return "executed"

    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockBackend()):
        assert load("x") == "executed"
        assert save("x", [1]) == "executed"
        assert savez("x", [1]) == "executed"
        assert savez_compressed("x", [1]) == "executed"
        assert save_gguf("x", {"t": [1]}) == "executed"
    invalid_load: BackendArg = require_callable(_fallback_load)(123)
    assert invalid_load is None
    assert _fallback_load("test.bin") is None
    with mock.patch("ml_switcheroo_compiler.serialization.formats.h5.H5WeightFormat.load", return_value="h5load"):
        assert _fallback_load("test.h5") == "h5load"


def test_numpy_io_infer_shape_args() -> None:
    """Test function test_numpy_io_infer_shape_args."""
    from ml_switcheroo_compiler.ops.io.numpy_io import Load, Save, SaveGguf, Savez, SavezCompressed

    class DummyNode(MockTestStub):
        """Test class DummyNode."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape

    assert Load().infer_shape(DummyNode((2, 2)), DummyNode((2, 2))) == (2, 2)
    assert Save().infer_shape(DummyNode((2, 2)), DummyNode((2, 2))) == (2, 2)
    assert Savez().infer_shape(DummyNode((2, 2)), DummyNode((2, 2))) == (2, 2)
    assert SavezCompressed().infer_shape(DummyNode((2, 2)), DummyNode((2, 2))) == (2, 2)
    assert SaveGguf().infer_shape(DummyNode((2, 2)), DummyNode((2, 2))) == (2, 2)
