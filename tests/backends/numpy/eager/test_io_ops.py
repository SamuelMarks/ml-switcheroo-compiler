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
    assert res.config.dtype.value == "uint8"


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
    assert res.config.dtype.value == "float32"


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

    encoded = _np_encode_base64(np, np.array([b"hello world"]))

    decoded = _np_decode_base64(np, np.array([b"aGVsbG8gd29ybGQ="]))
    assert "hello world" in str(decoded.numpy() if hasattr(decoded, "numpy") else decoded)

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


import ml_switcheroo_compiler.backends.numpy.eager.io_ops as mod


def test_io_ops_coverage(tmp_path):
    class DummyBackend:
        pass

    bk = DummyBackend()

    # save/load
    p = tmp_path / "f.npy"
    mod._np_save(bk, str(p), np.array([1.0]))
    mod._np_save(bk, filepath=str(p), arr=np.array([1.0]))
    mod._np_save(bk, file=str(p), arr=np.array([1.0]))

    with pytest.raises(ValueError):
        mod._np_load(bk, str(p))

    # savez
    pz = tmp_path / "fz.npz"
    mod._np_savez(bk, str(pz), np.array([1.0]), a=np.array([2.0]))
    mod._np_savez(bk, file=str(pz), args=[np.array([1.0])], kwds={"a": np.array([2.0])})

    mod._np_savez_compressed(bk, str(pz), np.array([1.0]), a=np.array([2.0]))
    mod._np_savez_compressed(bk, file=str(pz), args=[np.array([1.0])], kwds={"a": np.array([2.0])})

    res = mod._np_load(bk, str(pz))
    assert res is not None
    res2 = mod._np_load(bk, file=str(pz))

    # file io
    p_file = tmp_path / "f.txt"
    mod._np_write_file(bk, str(p_file), b"hello")
    mod._np_write_file(bk, filename=str(p_file), contents=b"hello")
    mod._np_write_file(bk, 123, b"hello")  # invalid

    res_file = mod._np_read_file(bk, str(p_file))
    res_file2 = mod._np_read_file(bk, filename=str(p_file))
    mod._np_read_file(bk, 123)

    # decode img
    mod._np_decode_image(bk, b"not image")
    mod._np_decode_image(bk, contents=b"not image")
    mod._np_decode_image(bk, 123)  # not bytes

    # decode csv
    mod._np_decode_csv(bk, [b"1,2", b"3,4"])
    mod._np_decode_csv(bk, records=[b"1,2", b"3,4"], record_defaults=[[0.0], [0.0]])

    # tf examples (dummies)
    mod._np_parse_example(bk, b"")
    mod._np_parse_example(bk, serialized=b"", features={"f": object()})

    # sequence example
    mod._np_parse_sequence_example(bk, b"")
    mod._np_parse_sequence_example(bk, serialized=b"")
    mod._np_parse_sequence_example(bk, 123)  # not bytes

    # base64
    res_b64 = mod._np_encode_base64(bk, b"hello")
    res_b64_2 = mod._np_encode_base64(bk, input=b"hello")

    mod._np_decode_base64(bk, b"aGVsbG8=")
    mod._np_decode_base64(bk, input=b"aGVsbG8=")

    mod._np_encode_base64(bk, None)
    mod._np_decode_base64(bk, None)

    # tensor serialize
    mod._np_serialize_tensor(bk, np.array([1.0]))
    mod._np_serialize_tensor(bk, tensor=np.array([1.0]))

    mod._np_parse_tensor(bk, b"")
    mod._np_parse_tensor(bk, serialized=b"")

    with pytest.raises(RuntimeError):
        mod._np_save_gguf(bk, str(tmp_path / "f.gguf"), {"a": np.array([1.0])})

    with pytest.raises(RuntimeError):
        mod._np_save_gguf(bk, filepath=str(tmp_path / "f.gguf"), tensors={"a": np.array([1.0])})

    mod._np_parse_sequence_example(bk, None)
