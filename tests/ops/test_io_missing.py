import pytest

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.io import (
    DecodeBase64,
    DecodeCsv,
    DecodeImage,
    EncodeBase64,
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


class MockArray:
    def __init__(self, shape):
        self.shape = tuple(shape)


def test_io_ops_infer_shape():
    ops = [Load(), Save(), SaveGguf(), Savez(), SavezCompressed(), ReadFile(), WriteFile(), DecodeImage(), DecodeCsv(), ParseExample(), SerializeTensor(), ParseTensor(), EncodeBase64(), DecodeBase64(), ParseSequenceExample(), SparsePlus(), SparseSigmoid()]

    for op in ops:
        assert op.infer_shape(MockArray((2, 2))) == (2, 2)
        assert op.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == (2, 2)
        assert op.infer_shape(MockArray((2, 1)), MockArray((2, 2))) == (2, 2)
        assert op.infer_shape() == ()


def test_fallback_load():
    from unittest.mock import patch

    assert _fallback_load(123) is None

    with patch("ml_switcheroo_compiler.serialization.formats.safetensors.SafetensorsWeightFormat.load", return_value="safetensors"):
        assert _fallback_load("test.safetensors") == "safetensors"

    with patch("ml_switcheroo_compiler.ops.io.load_npz", return_value="npz"):
        assert _fallback_load("test.npz") == "npz"

    with patch("ml_switcheroo_compiler.serialization.formats.h5.H5WeightFormat.load", return_value="h5"):
        assert _fallback_load("test.h5") == "h5"

    assert _fallback_load("test.unknown") is None


def test_io_functions_eager():
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
        assert save(t) == "eager_result"
        assert save_gguf(t) == "eager_result"
        assert savez(t) == "eager_result"
        assert savez_compressed(t) == "eager_result"
        assert read_file(t) == "eager_result"
        assert write_file(t, t) == "eager_result"
        assert decode_image(t) == "eager_result"
        assert decode_csv(t, []) == "eager_result"
        assert parse_example(t, {}) == "eager_result"
        assert serialize_tensor(t) == "eager_result"
        assert parse_tensor(t, DType.Float32) == "eager_result"

        # Test encode/decode base64 fallback properly
        with patch("ml_switcheroo_compiler.ops.io._eager_base64", return_value="eager_result"):
            assert encode_base64(t) == "eager_result"
            assert decode_base64(t) == "eager_result"

        assert parse_sequence_example(t) == "eager_result"
        assert sparse_plus(t) == "eager_result"
        assert sparse_sigmoid(t) == "eager_result"

    config.eager_mode = False


def test_io_functions_tracing():
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType

    config.eager_mode = False

    class FakeData:
        shape = (2,)

    t = Tensor(data=FakeData(), config=TensorConfig((2,), DType.Float32, None))

    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="dummy_node") as mock_emit:
        assert load(t) == "dummy_node"
        assert save(t) == "dummy_node"
        assert save_gguf(t) == "dummy_node"
        assert savez(t) == "dummy_node"
        assert savez_compressed(t) == "dummy_node"
        assert read_file(t) == "dummy_node"
        assert write_file(t, t) == "dummy_node"
        assert decode_image(t) == "dummy_node"
        assert decode_csv(t, []) == "dummy_node"
        assert parse_example(t, {}) == "dummy_node"
        assert serialize_tensor(t) == "dummy_node"
        assert parse_tensor(t, DType.Float32) == "dummy_node"
        assert encode_base64(t) == "dummy_node"
        assert decode_base64(t) == "dummy_node"
        assert parse_sequence_example(t) == "dummy_node"
        assert sparse_plus(t) == "dummy_node"
        assert sparse_sigmoid(t) == "dummy_node"

    config.eager_mode = True


def test_missing_io_ops_classes():
    from ml_switcheroo_compiler.ops.io import Fromfile, Fromfunction, Fromiter, Fromstring

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

    for op_cls in [Fromfile, Fromstring, Fromiter, Fromfunction]:
        op = op_cls()
        # Empty args
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


def test_mlx_settings_missing_attributes():
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.io import set_default_stream, set_memory_limit, set_wired_limit

    old_backend = config.backend
    config.backend = "mlx"
    try:
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
            # Provide an object without the attributes
            mock_get_backend.return_value = object()
            set_default_stream(None)
            set_memory_limit(100)
            set_wired_limit(100)
    finally:
        config.backend = old_backend


def test_base64_edge_cases():

    # test None
    assert _eager_base64("encode", None) == b""
    # test sequence
    assert _eager_base64("encode", [b"hello", b"world"]) == [b"aGVsbG8", b"d29ybGQ"]


def test_tf_record_writer():
    opts = TFRecordOptions("GZIP")
    assert opts.compression_type == "GZIP"

    writer = TFRecordWriter("path", opts)
    writer.write(b"data")
    writer.close()

    with writer as w:
        assert w == writer


def test_misc_io_functions(tmp_path):
    set_memory_limit(1)
    set_wired_limit(1)
    set_default_stream(None)

    # gfile
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


def test_image_decoding(tmp_path):

    # The default mock we can hit is the fallback returning a Tensor with empty shape ()
    # since no backend implements it by default in this test scope
    assert decode_jpeg(b"data").shape == ()
    assert decode_png(b"data").shape == ()
    assert decode_gif(b"data").shape == ()
    assert decode_bmp(b"data").shape == ()


def test_eager_base64():
    import base64
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType

    data = b"hello"
    enc = base64.b64encode(data)

    t = Tensor(data=np.array([data]), config=TensorConfig((1,), DType.String, None))
    t_enc = Tensor(data=np.array([enc]), config=TensorConfig((1,), DType.String, None))

    from ml_switcheroo_compiler.core.config import config

    # We can just change the property directly instead of mocking the object
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
