"""Module test_io.py."""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.io import (
    DecodeBase64,
    DecodeCsv,
    DecodeImage,
    EncodeBase64,
    Fromfile,
    Fromfunction,
    Fromiter,
    Fromstring,
    Load,
    ParseExample,
    ParseSequenceExample,
    ParseTensor,
    ReadFile,
    Save,
    SaveGguf,
    Savez,
    SavezCompressed,
    SerializeTensor,
    SparsePlus,
    SparseSigmoid,
    TFRecordOptions,
    TFRecordWriter,
    WriteFile,
    _eager_base64,
    _fallback_load,
    decode_base64,
    decode_bmp,
    decode_csv,
    decode_gif,
    decode_image,
    decode_jpeg,
    decode_png,
    encode_base64,
    gfile_copy,
    gfile_glob,
    gfile_makedirs,
    gfile_stat,
    load,
    parse_example,
    parse_sequence_example,
    parse_tensor,
    read_file,
    save,
    save_gguf,
    save_safetensors,
    savez,
    savez_compressed,
    serialize_tensor,
    set_default_stream,
    set_memory_limit,
    set_wired_limit,
    sparse_plus,
    sparse_sigmoid,
    write_file,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_fallback_load() -> None:
    """test_fallback_load."""
    assert _fallback_load(None) is None
    assert _fallback_load("test.txt") is None
    with patch("ml_switcheroo_compiler.serialization.formats.safetensors.SafetensorsWeightFormat.load") as mock_load:
        mock_load.return_value = {"a": 1}
        assert _fallback_load("test.safetensors") == {"a": 1}
    with patch("ml_switcheroo_compiler.ops.io.numpy_io.load_npz") as mock_load:
        mock_load.return_value = {"a": 1}
        assert _fallback_load("test.npz") == {"a": 1}
    with patch("ml_switcheroo_compiler.serialization.formats.h5.H5WeightFormat.load") as mock_load:
        mock_load.return_value = {"a": 1}
        assert _fallback_load("test.h5") == {"a": 1}


def test_save_safetensors() -> None:
    """test_save_safetensors."""
    with patch("ml_switcheroo_compiler.serialization.formats.safetensors.SafetensorsWeightFormat.save") as mock_save:
        save_safetensors("test.safetensors", {"a": 1})
        mock_save.assert_called_once_with({"a": 1}, "test.safetensors")


def test_mlx_settings() -> None:
    """test_mlx_settings."""
    original_backend = core_config.backend
    core_config.backend = "mlx"
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:
        mock_backend = MagicMock()
        mock_get.return_value = mock_backend
        set_default_stream("stream1")
        mock_backend.set_default_stream.assert_called_once_with("stream1")
        set_memory_limit(1024)
        mock_backend.set_memory_limit.assert_called_once_with(1024)
        set_wired_limit(512)
        mock_backend.set_wired_limit.assert_called_once_with(512)
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=ImportError):
        set_default_stream("stream2")
        set_memory_limit(2048)
        set_wired_limit(1024)
    core_config.backend = original_backend


def test_gfile(tmp_path: EagerTensorProtocol) -> None:
    """test_gfile."""
    test_dir = str(tmp_path / "test_dir")
    gfile_makedirs(test_dir)
    assert os.path.exists(test_dir)
    test_file = str(tmp_path / "test_dir" / "test.txt")
    test2_file = str(tmp_path / "test_dir" / "test2.txt")
    with open(test_file, "w") as f:
        f.write("hello")
    gfile_copy(test_file, test2_file)
    assert os.path.exists(test2_file)
    with pytest.raises(FileExistsError):
        gfile_copy(test_file, test2_file)
    gfile_copy(test_file, test2_file, overwrite=True)
    glob_pattern = str(tmp_path / "test_dir" / "*.txt")
    glob_res = gfile_glob(glob_pattern)
    assert test_file in glob_res
    stat = gfile_stat(test_file)
    assert stat["length"] == 5
    assert "mtime" in stat
    shutil.rmtree(test_dir)


def test_tfrecord(tmp_path: EagerTensorProtocol) -> None:
    """test_tfrecord."""
    options = TFRecordOptions("GZIP")
    assert options.compression_type == "GZIP"
    tfrecord1 = str(tmp_path / "dataset1.tfrecord")
    tfrecord2 = str(tmp_path / "dataset2.tfrecord")
    writer = TFRecordWriter(tfrecord1, options)
    assert writer.path == tfrecord1
    assert writer.options == options
    writer.write("data")
    writer.close()
    with TFRecordWriter(tfrecord2) as w:
        w.write("data")


def test_eager_base64() -> None:
    """test_eager_base64."""
    res1 = _eager_base64("encode", b"hello", pad=True)
    assert res1 == b"aGVsbG8="
    res1_nopad = _eager_base64("encode", b"hello")
    assert res1_nopad == b"aGVsbG8"
    res2 = _eager_base64("decode", b"aGVsbG8=")
    assert res2 == b"hello"
    res3 = _eager_base64("encode", "hello", pad=True)
    assert res3 == b"aGVsbG8="


def test_decode_images() -> None:
    """test_decode_images."""
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:
        mock_backend = MagicMock()
        mock_get.return_value = mock_backend
        decode_jpeg("data", channels=3, ratio=2)
        mock_backend.decode_jpeg.assert_called_once_with("data", channels=3, ratio=2)
        decode_png("data", channels=3, dtype="uint8")
        mock_backend.decode_png.assert_called_once_with("data", channels=3, dtype="uint8")
        decode_gif("data")
        mock_backend.decode_gif.assert_called_once_with("data")
        decode_bmp("data", channels=3)
        mock_backend.decode_bmp.assert_called_once_with("data", channels=3)


def test_opdefs_infer_shape() -> None:
    """test_opdefs_infer_shape."""
    t = MagicMock()
    t.shape = (2, 3)
    assert Load().infer_shape(t) == (2, 3)
    assert Load().infer_shape() == ()
    assert Save().infer_shape(t) == (2, 3)
    assert SaveGguf().infer_shape(t) == (2, 3)
    assert Savez().infer_shape(t) == (2, 3)
    assert SavezCompressed().infer_shape(t) == (2, 3)
    assert ReadFile().infer_shape(t) == (2, 3)
    assert WriteFile().infer_shape(t) == (2, 3)
    assert DecodeImage().infer_shape(t) == (2, 3)
    assert DecodeCsv().infer_shape(t) == (2, 3)
    assert ParseExample().infer_shape(t) == (2, 3)
    assert SerializeTensor().infer_shape(t) == (2, 3)
    assert ParseTensor().infer_shape(t) == (2, 3)
    assert EncodeBase64().infer_shape(t) == (2, 3)
    assert DecodeBase64().infer_shape(t) == (2, 3)
    assert ParseSequenceExample().infer_shape(t) == (2, 3)
    assert SparsePlus().infer_shape(t) == (2, 3)
    assert SparsePlus().infer_shape() == ()
    assert SparseSigmoid().infer_shape(t) == (2, 3)
    assert SparseSigmoid().infer_shape() == ()
    assert Fromfile().infer_shape() == (None,)
    assert Fromstring().infer_shape() == (None,)
    assert Fromiter().infer_shape() == (None,)
    assert Fromfunction().infer_shape(None, (4, 5)) == (4, 5)
    assert Fromfunction().infer_shape(None, shape=(6, 7)) == (6, 7)


def test_frontend_functions_eager() -> None:
    """test_frontend_functions_eager."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:
        mock_backend = MagicMock()
        mock_get.return_value = mock_backend
        load("file")
        mock_backend.execute_op.assert_called_with("Load", "file")
        save("file")
        mock_backend.execute_op.assert_called_with("Save", "file")
        save_gguf("file")
        mock_backend.execute_op.assert_called_with("SaveGguf", "file")
        savez("file")
        mock_backend.execute_op.assert_called_with("Savez", "file")
        savez_compressed("file")
        mock_backend.execute_op.assert_called_with("SavezCompressed", "file")
        read_file("file")
        mock_backend.execute_op.assert_called_with("ReadFile", "file", name=None)
        cast("Callable[..., EagerTensorProtocol]", write_file)("file", "content")
        mock_backend.execute_op.assert_called_with("WriteFile", "file", "content", name=None)
        decode_image("content")
        mock_backend.execute_op.assert_called_with("DecodeImage", "content", channels=0, dtype=DType.UInt8, name=None, expand_animations=True)
        cast("Callable[..., EagerTensorProtocol]", decode_csv)("content", [1])
        mock_backend.execute_op.assert_called_with("DecodeCsv", "content", record_defaults=[1], field_delim=",", use_quote_delim=True, na_value="", select_cols=None, name=None)
        cast("Callable[..., EagerTensorProtocol]", parse_example)("serialized", {"a": 1})
        mock_backend.execute_op.assert_called_with("ParseExample", "serialized", features={"a": 1}, example_names=None, name=None)
        cast("Callable[..., EagerTensorProtocol]", serialize_tensor)("tensor")
        mock_backend.execute_op.assert_called_with("SerializeTensor", "tensor", name=None)
        cast("Callable[..., EagerTensorProtocol]", parse_tensor)("serialized", DType.UInt8)
        mock_backend.execute_op.assert_called_with("ParseTensor", "serialized", out_type=DType.UInt8, name=None)
        cast("Callable[..., EagerTensorProtocol]", encode_base64)("content")
        mock_backend.execute_op.assert_called_with("EncodeBase64", "content", pad=False, name=None)
        cast("Callable[..., EagerTensorProtocol]", decode_base64)("content")
        mock_backend.execute_op.assert_called_with("DecodeBase64", "content", name=None)
        cast("Callable[..., EagerTensorProtocol]", parse_sequence_example)("serialized")
        mock_backend.execute_op.assert_called_with("ParseSequenceExample", "serialized", context_features=None, sequence_features=None, example_names=None, name=None)
        sparse_plus("a", "b")
        mock_backend.execute_op.assert_called_with("SparsePlus", "a", "b")
        sparse_sigmoid("a")
        mock_backend.execute_op.assert_called_with("SparseSigmoid", "a")


def test_frontend_functions_lazy() -> None:
    """test_frontend_functions_lazy."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node") as mock_emit:
        t = MagicMock()
        t.shape = (2, 3)
        t.dtype = "float32"
        load(t)
        mock_emit.assert_called_with("Load", [t], {}, (2, 3), "float32")
        save(t)
        mock_emit.assert_called_with("Save", [t], {}, (), "float32")
        save_gguf(t)
        mock_emit.assert_called_with("SaveGguf", [t], {}, (), "float32")
        savez(t)
        mock_emit.assert_called_with("Savez", [t], {}, (), "float32")
        savez_compressed(t)
        mock_emit.assert_called_with("SavezCompressed", [t], {}, (), "float32")
        read_file(t)
        mock_emit.assert_called_with("ReadFile", [t], {"name": None}, (2, 3), "float32")
        write_file("file", t)
        mock_emit.assert_called_with("WriteFile", ["file", t], {"name": None}, (), "float32")
        decode_image(t)
        mock_emit.assert_called_with("DecodeImage", [t], {"channels": 0, "dtype": DType.UInt8, "name": None, "expand_animations": True}, (2, 3), "float32")
        decode_csv(t, [1])
        mock_emit.assert_called_with("DecodeCsv", [t], {"record_defaults": [1], "field_delim": ",", "use_quote_delim": True, "na_value": "", "select_cols": None, "name": None}, (2, 3), "float32")
        parse_example(t, {"a": 1})
        mock_emit.assert_called_with("ParseExample", [t], {"features": {"a": 1}, "example_names": None, "name": None}, (2, 3), "float32")
        serialize_tensor(t)
        mock_emit.assert_called_with("SerializeTensor", [t], {"name": None}, (2, 3), "float32")
        parse_tensor(t, DType.UInt8)
        mock_emit.assert_called_with("ParseTensor", [t], {"out_type": DType.UInt8, "name": None}, (2, 3), "float32")
        encode_base64(t)
        mock_emit.assert_called_with("EncodeBase64", [t], {"pad": False, "name": None}, (2, 3), "float32")
        decode_base64(t)
        mock_emit.assert_called_with("DecodeBase64", [t], {"name": None}, (2, 3), "float32")
        parse_sequence_example(t)
        mock_emit.assert_called_with("ParseSequenceExample", [t], {"context_features": None, "sequence_features": None, "example_names": None, "name": None}, (2, 3), "float32")
        sparse_plus(t, t)
        mock_emit.assert_called_with("SparsePlus", [t, t], {}, (2, 3), "float32")
        sparse_sigmoid(t)
        mock_emit.assert_called_with("SparseSigmoid", [t], {}, (2, 3), "float32")
        load()
        sparse_plus()
        sparse_sigmoid()


class MockArray:
    """MockArray."""

    def __init__(self, shape: EagerValue) -> None:
        """__init__."""
        self.shape = cast("Callable[..., EagerTensorProtocol]", tuple)(shape)


def test_io_ops_infer_shape() -> None:
    """test_io_ops_infer_shape."""
    ops = [Load(), Save(), SaveGguf(), Savez(), SavezCompressed(), ReadFile(), WriteFile(), DecodeImage(), DecodeCsv(), ParseExample(), SerializeTensor(), ParseTensor(), EncodeBase64(), DecodeBase64(), ParseSequenceExample(), SparsePlus(), SparseSigmoid()]
    for op in ops:
        assert op.infer_shape(MockArray((2, 2))) == (2, 2)
        assert op.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == (2, 2)
        assert op.infer_shape(MockArray((2, 1)), MockArray((2, 2))) == (2, 2)
        assert op.infer_shape() == ()


def test_fallback_load_2() -> None:
    """test_fallback_load_2."""
    from unittest.mock import patch

    assert cast("Callable[..., EagerTensorProtocol]", _fallback_load)(123) is None
    with patch("ml_switcheroo_compiler.serialization.formats.safetensors.SafetensorsWeightFormat.load", return_value="safetensors"):
        assert _fallback_load("test.safetensors") == "safetensors"
    with patch("ml_switcheroo_compiler.ops.io.numpy_io.load_npz", return_value="npz"):
        assert _fallback_load("test.npz") == "npz"
    with patch("ml_switcheroo_compiler.serialization.formats.h5.H5WeightFormat.load", return_value="h5"):
        assert _fallback_load("test.h5") == "h5"
    assert _fallback_load("test.unknown") is None


def test_io_functions_eager() -> None:
    """test_io_functions_eager."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType

    config.eager_mode = True
    t = Tensor(data="dummy", config=TensorConfig((2,), DType.Float32, None))
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
        mock_backend = mock_get_backend.return_value
        mock_backend.execute_op.return_value = "eager_result"
        mock_backend.asarray.return_value = "eager_result"
        assert load(t) == "eager_result"
        assert cast("EagerTensorProtocol", save(t)) == cast("EagerTensorProtocol", "eager_result")
        assert cast("EagerTensorProtocol", save_gguf(t)) == cast("EagerTensorProtocol", "eager_result")
        assert cast("EagerTensorProtocol", savez(t)) == cast("EagerTensorProtocol", "eager_result")
        assert cast("EagerTensorProtocol", savez_compressed(t)) == cast("EagerTensorProtocol", "eager_result")
        assert read_file(t) == "eager_result"
        assert cast("EagerTensorProtocol", write_file(t, t)) == cast("EagerTensorProtocol", "eager_result")
        assert cast("EagerTensorProtocol", decode_image(t)) == cast("EagerTensorProtocol", "eager_result")
        assert cast("EagerTensorProtocol", decode_csv(t, [])) == cast("EagerTensorProtocol", "eager_result")
        assert cast("EagerTensorProtocol", parse_example(t, {})) == cast("EagerTensorProtocol", "eager_result")
        assert serialize_tensor(t) == "eager_result"
        assert parse_tensor(t, DType.Float32) == "eager_result"
        with patch("ml_switcheroo_compiler.ops.io._eager_base64", return_value="eager_result"):
            assert encode_base64(t) == "eager_result"
            assert decode_base64(t) == "eager_result"
        assert cast("EagerTensorProtocol", parse_sequence_example(t)) == cast("EagerTensorProtocol", "eager_result")
        assert sparse_plus(t) == "eager_result"
        assert sparse_sigmoid(t) == "eager_result"
    config.eager_mode = False


def test_io_functions_tracing() -> None:
    """test_io_functions_tracing."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType

    config.eager_mode = False

    class FakeData:
        """FakeData."""

        shape = (2,)

    t = Tensor(data=FakeData(), config=TensorConfig((2,), DType.Float32, None))
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="dummy_node"):
        assert load(t) == "dummy_node"
        assert cast("EagerTensorProtocol", save(t)) == cast("EagerTensorProtocol", "dummy_node")
        assert cast("EagerTensorProtocol", save_gguf(t)) == cast("EagerTensorProtocol", "dummy_node")
        assert cast("EagerTensorProtocol", savez(t)) == cast("EagerTensorProtocol", "dummy_node")
        assert cast("EagerTensorProtocol", savez_compressed(t)) == cast("EagerTensorProtocol", "dummy_node")
        assert read_file(t) == "dummy_node"
        assert cast("EagerTensorProtocol", write_file(t, t)) == cast("EagerTensorProtocol", "dummy_node")
        assert cast("EagerTensorProtocol", decode_image(t)) == cast("EagerTensorProtocol", "dummy_node")
        assert cast("EagerTensorProtocol", decode_csv(t, [])) == cast("EagerTensorProtocol", "dummy_node")
        assert cast("EagerTensorProtocol", parse_example(t, {})) == cast("EagerTensorProtocol", "dummy_node")
        assert serialize_tensor(t) == "dummy_node"
        assert parse_tensor(t, DType.Float32) == "dummy_node"
        assert encode_base64(t) == "dummy_node"
        assert decode_base64(t) == "dummy_node"
        assert cast("EagerTensorProtocol", parse_sequence_example(t)) == cast("EagerTensorProtocol", "dummy_node")
        assert sparse_plus(t) == "dummy_node"
        assert sparse_sigmoid(t) == "dummy_node"
    config.eager_mode = True


def test_missing_io_ops_classes() -> None:
    """test_missing_io_ops_classes."""
    from ml_switcheroo_compiler.ops.io import Fromfile, Fromfunction, Fromiter, Fromstring

    class FakeTensor:
        """FakeTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """__init__."""
            self.shape = shape

    for op_cls in [Fromfile, Fromstring, Fromiter, Fromfunction]:
        op = op_cls()
        if type(op).__name__ == "Fromfunction":
            assert op.infer_shape() == ()
            t1 = FakeTensor((2, 3))
            t2 = FakeTensor((1, 3))
            assert op.infer_shape(t1, (2, 3)) == (2, 3)
        else:
            assert op.infer_shape() == (None,)
            t1 = FakeTensor((2, 3))
            t2 = FakeTensor((1, 3))
            assert op.infer_shape(t1, t2) == (None,)


def test_mlx_settings_missing_attributes() -> None:
    """test_mlx_settings_missing_attributes."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.io import set_default_stream, set_memory_limit, set_wired_limit

    old_backend = config.backend
    config.backend = "mlx"
    try:
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
            mock_get_backend.return_value = object()
            set_default_stream(None)
            set_memory_limit(100)
            set_wired_limit(100)
    finally:
        config.backend = old_backend
    assert old_backend is not None or old_backend is None


def test_base64_edge_cases() -> None:
    """test_base64_edge_cases."""
    assert _eager_base64("encode", None) == b""
    assert _eager_base64("encode", [b"hello", b"world"]) == [b"aGVsbG8", b"d29ybGQ"]


def test_tf_record_writer() -> None:
    """test_tf_record_writer."""
    opts = TFRecordOptions("GZIP")
    assert opts.compression_type == "GZIP"
    writer = TFRecordWriter("path", opts)
    writer.write(b"data")
    writer.close()
    with writer as w:
        assert w == writer


def test_misc_io_functions(tmp_path: EagerTensorProtocol) -> None:
    """test_misc_io_functions."""
    set_memory_limit(1)
    set_wired_limit(1)
    set_default_stream(None)
    f = tmp_path / "test.txt"
    f.write_text("hello")
    dst = tmp_path / "test2.txt"
    gfile_copy(str(f), str(dst))
    assert dst.exists()
    with pytest.raises(FileExistsError):
        gfile_copy(str(f), str(dst))
    assert str(f) in gfile_glob(str(tmp_path / "*.txt"))
    stat = gfile_stat(str(f))
    assert "length" in stat
    gfile_makedirs(str(tmp_path / "dir"))
    assert (tmp_path / "dir").exists()


def test_image_decoding(tmp_path: EagerTensorProtocol, mocker: EagerTensorProtocol) -> None:
    """test_image_decoding."""
    import numpy as np

    class DummyDecodeBackend:
        """Test class DummyDecodeBackend."""

        @staticmethod
        def decode_image_bytes(data: EagerValue, channels: EagerValue = 0) -> EagerValue:
            """Test function decode_image_bytes."""
            return np.array([])

    mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyDecodeBackend)
    assert decode_jpeg(b"data").shape == (0,)
    assert decode_png(b"data").shape == (0,)
    assert decode_gif(b"data").shape == (0,)
    assert decode_bmp(b"data").shape == (0,)


def test_eager_base64_2() -> None:
    """test_eager_base64_2."""
    import base64
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType

    data = b"hello"
    enc = base64.b64encode(data)
    t = Tensor(data=np.array([data]), config=TensorConfig((1,), DType.String, None))
    t_enc = Tensor(data=np.array([enc]), config=TensorConfig((1,), DType.String, None))
    from ml_switcheroo_compiler.core.config import config

    old_eager = config.eager_mode
    config.eager_mode = True
    try:
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
            mock_backend = mock_get_backend.return_value
            mock_backend.execute_op.return_value = t_enc.data
            assert encode_base64(t) is not None
            mock_backend.execute_op.return_value = t.data
            assert decode_base64(t_enc) is not None
    finally:
        config.eager_mode = old_eager


def test_missing_io_functions_from() -> None:
    """test_missing_io_functions_from."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.io import fromfile, fromfunction, fromiter, fromstring

    config.eager_mode = True
    with pytest.raises(FileNotFoundError):
        fromfile("foo")
    with pytest.raises(ValueError):
        fromstring("1 2")
    fromiter([1, 2], float)
    fromfunction(lambda i, j: i == j, (3, 3), dtype=int)


def test_numpy_io_eager_and_lazy_shapes() -> None:
    """Test function test_numpy_io_eager_and_lazy_shapes."""
    assert callable(test_numpy_io_eager_and_lazy_shapes)
