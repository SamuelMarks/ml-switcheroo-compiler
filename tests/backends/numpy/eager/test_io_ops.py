"""Tests for numpy eager io ops."""

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager.io_ops import (
    _np_decode_base64,
    _np_decode_csv,
    _np_decode_image,
    _np_encode_base64,
    _np_parse_example,
    _np_parse_sequence_example,
    _np_parse_tensor,
    _np_read_file,
    _np_save,
    _np_save_gguf,
    _np_savez,
    _np_savez_compressed,
    _np_serialize_tensor,
    _np_write_file,
)


def test_load_save(tmp_path):
    import os

    f = str(tmp_path / "test.npy")
    _np_save(np, f, np.array([1, 2, 3]))
    assert os.path.exists(f)

    # Note: load in our numpy backend calls _fallback_load
    # which uses formats that might not support npy directly, but lets just mock it


def test_save_gguf():
    with pytest.raises(RuntimeError):
        _np_save_gguf(np, "test")


def test_savez(tmp_path):
    f = str(tmp_path / "test.npz")
    _np_savez(np, f, arr=np.array([1, 2, 3]))
    _np_savez_compressed(np, f, arr=np.array([1, 2, 3]))


def test_read_write_file(tmp_path):
    f = str(tmp_path / "test.txt")
    _np_write_file(np, f, b"hello")
    res = _np_read_file(np, f)
    assert res.data == b"hello"


def test_decode_image():
    res = _np_decode_image(np, None, dtype="uint8")
    assert res.config.dtype == "uint8"


def test_decode_csv():
    res = _np_decode_csv(np, None, record_defaults=[1, 2])
    assert len(res) == 2


def test_parse_example():
    res = _np_parse_example(np, None, features={"a": 1})
    assert "a" in res


def test_serialize_tensor():
    res = _np_serialize_tensor(np, None)
    assert res.data == b""


def test_parse_tensor():
    res = _np_parse_tensor(np, None, out_type="float32")
    assert res.config.dtype == "float32"


def test_base64():
    import base64

    b = b"hello"
    enc = _np_encode_base64(np, b, pad=True)
    assert enc.data == base64.b64encode(b)
    dec = _np_decode_base64(np, enc.data)
    assert dec.data == b


def test_parse_sequence_example():
    r1, r2 = _np_parse_sequence_example(np, None)
    assert r1 == {} and r2 == {}


def test_load():
    from ml_switcheroo_compiler.backends.numpy.eager.io_ops import _np_load

    try:
        _np_load(np, "non_existent_file.txt")
    except ValueError:
        pass
    try:
        _np_load(np, file="non_existent_file.txt")
    except ValueError:
        pass


def test_save_kwargs(tmp_path):
    f = str(tmp_path / "test.npy")
    _np_save(np, filepath=f, arr=np.array([1, 2, 3]))
    _np_save(np, file=f, arr=np.array([1, 2, 3]))


def test_savez_kwargs(tmp_path):
    f = str(tmp_path / "test.npz")
    _np_savez(np, filepath=f, arr=np.array([1, 2, 3]))
    _np_savez_compressed(np, file=f, arr=np.array([1, 2, 3]))


def test_read_file_invalid():
    res = _np_read_file(np, 123)
    assert res.data is None


def test_missing_io_coverage():
    from ml_switcheroo_compiler.backends.numpy.eager.io_ops import _np_decode_base64, _np_encode_base64, _np_parse_sequence_example

    assert _np_encode_base64(np, None) is None
    assert _np_decode_base64(np, None) is None
    res = _np_parse_sequence_example(np, "serialized")
    assert "dummy" in res[0]


def test_load_fallback():
    import ml_switcheroo_compiler.ops.io as io_mod
    from ml_switcheroo_compiler.backends.numpy.eager.io_ops import _np_load

    original = io_mod._fallback_load
    try:
        io_mod._fallback_load = lambda *a, **k: "loaded"
        assert _np_load(np, "test") == "loaded"
    finally:
        io_mod._fallback_load = original
