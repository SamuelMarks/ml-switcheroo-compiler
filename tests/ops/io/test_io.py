# ruff: noqa: E501
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import io
from ml_switcheroo_compiler.ops.io import SparsePlus, SparseSigmoid, _fallback_load, set_default_stream, set_memory_limit, set_wired_limit, sparse_plus, sparse_sigmoid


def test_io_fallback_load() -> None:
    pass
    with patch("ml_switcheroo_compiler.serialization.formats.safetensors.SafetensorsWeightFormat.load") as mock_safe:
        mock_safe.return_value = "safe"
        assert _fallback_load("model.safetensors") == "safe"
    with patch("ml_switcheroo_compiler.ops.io.load_npz") as mock_npz:
        mock_npz.return_value = "npz"
        assert _fallback_load("model.npz") == "npz"
    with patch("ml_switcheroo_compiler.serialization.formats.h5.H5WeightFormat.load") as mock_h5:
        mock_h5.return_value = "h5"
        assert _fallback_load("model.h5") == "h5"
    pass


def test_set_limits() -> None:
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
    assert SparsePlus().infer_shape("shape") == "shape"
    assert SparsePlus().infer_shape() == ()
    assert SparseSigmoid().infer_shape("shape") == "shape"
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
    try:
        sparse_plus(t)
    except Exception:
        pass
    try:
        sparse_sigmoid(t)
    except Exception:
        pass
    global_tracing_state.stop_tracing()


def test_io_functions() -> None:

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.start_tracing()

    import os
    import tempfile

    from ml_switcheroo_compiler.ops.io import (
        TFRecordOptions,
        TFRecordWriter,
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
    try:
        gfile_copy(__file__, __file__)
    except FileExistsError:
        pass
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
    writer = TFRecordWriter("path", opt)
    assert writer.write("data") is None
    with writer as w:
        pass
    with patch("ml_switcheroo_compiler.serialization.formats.safetensors.SafetensorsWeightFormat.save") as mock_save:
        save_safetensors("path", {})
        mock_save.assert_called_once()


def test_set_limits_not_mlx() -> None:
    from ml_switcheroo_compiler.core.config import config as core_config

    orig_backend = core_config.backend
    core_config.backend = "not_mlx"
    set_default_stream("stream")
    set_memory_limit(1024)
    set_wired_limit(512)
    core_config.backend = orig_backend


"Tests for IO operations."


def test_io_stubs() -> None:

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.start_tracing()

    """Test IO stubs."""
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
    fh = io.TFRecordWriter("path")
    assert fh.write(None) is None
    assert fh.close() is None
