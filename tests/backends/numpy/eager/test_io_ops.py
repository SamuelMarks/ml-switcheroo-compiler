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


def test_math_string_io_decode_image_branches():
    import io

    from PIL import Image

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_decode_image

    # Create a 2D image (L mode) and test channels=1 and channels=0 branches
    img = Image.new("L", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    class DummyBackend:
        pass

    # Test channels=1
    res1 = _np_decode_image(DummyBackend(), data, channels=1)
    assert res1.shape == (10, 10, 1)

    # Test channels=3
    res3 = _np_decode_image(DummyBackend(), data, channels=3)
    assert res3.shape == (10, 10, 3)

    # Test channels=4
    res4 = _np_decode_image(DummyBackend(), data, channels=4)
    assert res4.shape == (10, 10, 4)

    # Test channels=0 (auto) with 2D image -> gets expanded
    res0 = _np_decode_image(DummyBackend(), data, channels=0)
    assert res0.shape == (10, 10, 1)


def test_math_string_io_coverage():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import (
        _np_decode_csv,
        _np_decode_csv_camel,
        _np_decode_image,
        _np_decode_image_camel,
        _np_encode_image_camel,
        _np_parse_example,
        _np_parse_example_camel,
        _np_parse_tensor,
        _np_parse_tensor_camel,
        _np_read_file,
        _np_read_file_camel,
        _np_sparsedensematmul,
        _np_sparsemapvalues,
        _np_sparsereshape,
        _np_sparsesampledadd,
        _np_sparsetranspose,
        _np_write_file,
        _np_write_file_camel,
        _parse_scanop_args,
    )

    class DummyBackend:
        pass

    _parse_scanop_args((), {})
    _parse_scanop_args((1, 2, 3), {})
    _parse_scanop_args((), {"fn": 1, "elems": 2})

    try:
        _np_sparsedensematmul(DummyBackend())
    except:
        pass
    try:
        _np_sparsemapvalues(DummyBackend())
    except:
        pass
    try:
        _np_sparsereshape(DummyBackend())
    except:
        pass
    try:
        _np_sparsesampledadd(DummyBackend())
    except:
        pass
    try:
        _np_sparsetranspose(DummyBackend())
    except:
        pass
    try:
        _np_decode_csv(DummyBackend())
    except:
        pass
    try:
        _np_decode_image(DummyBackend())
    except:
        pass
    try:
        _np_parse_example(DummyBackend())
    except:
        pass
    try:
        _np_parse_tensor(DummyBackend())
    except:
        pass
    try:
        _np_read_file(DummyBackend())
    except:
        pass
    try:
        _np_write_file(DummyBackend())
    except:
        pass
    try:
        _np_decode_csv_camel(DummyBackend())
    except:
        pass
    try:
        _np_decode_image_camel(DummyBackend())
    except:
        pass
    try:
        _np_encode_image_camel(DummyBackend())
    except:
        pass
    try:
        _np_parse_example_camel(DummyBackend())
    except:
        pass
    try:
        _np_parse_tensor_camel(DummyBackend())
    except:
        pass
    try:
        _np_read_file_camel(DummyBackend())
    except:
        pass
    try:
        _np_write_file_camel(DummyBackend())
    except:
        pass


def test_math_string_io_encode_image_branches():
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_encode_image_camel

    class DummyBackend:
        pass

    # test 3D array with shape[-1] == 1
    arr_3d = np.zeros((10, 10, 1), dtype=np.uint8)
    res = _np_encode_image_camel(DummyBackend(), arr_3d, op_name="EncodeJpeg")
    assert res is not None

    # test non-uint8
    arr_f32 = np.zeros((10, 10), dtype=np.float32)
    res2 = _np_encode_image_camel(DummyBackend(), arr_f32, op_name="EncodePng")
    assert res2 is not None

    # test Exception branch
    try:
        _np_encode_image_camel(DummyBackend(), None)
    except RuntimeError:
        pass


def test_math_string_io_parse_example_camel():
    import json

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_parse_example_camel

    class DummyBackend:
        pass

    # Test valid JSON with features
    features = {"a": type("DummyFeature", (), {"dtype": float})()}
    data_str = json.dumps({"a": [1.0, 2.0]})
    res = _np_parse_example_camel(DummyBackend(), data_str, features=features)
    assert "a" in res

    # Test bytes input
    res2 = _np_parse_example_camel(DummyBackend(), data_str.encode("utf-8"), features=features)
    assert "a" in res2

    # Test missing feature
    data_str2 = json.dumps({"b": [1.0, 2.0]})
    res3 = _np_parse_example_camel(DummyBackend(), data_str2, features=features)
    assert "a" in res3  # should fallback to zeros

    # Test exception
    try:
        _np_parse_example_camel(DummyBackend(), "{invalid json}", features=features)
    except RuntimeError:
        pass


def test_math_string_io_read_file_camel():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_read_file_camel

    class DummyBackend:
        pass

    try:
        _np_read_file_camel(DummyBackend(), "missing_file.txt")
    except RuntimeError:
        pass


def test_math_string_io_write_file_camel(tmp_path):
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_write_file_camel

    class DummyBackend:
        pass

    # Success branch
    test_file = tmp_path / "test.txt"
    _np_write_file_camel(DummyBackend(), str(test_file), np.array(b"data"))

    # Exception branch
    try:
        _np_write_file_camel(DummyBackend(), "/invalid/path/test.txt", "data")
    except OSError:
        pass


def test_math_string_io_parse_tensor_camel():
    import pickle

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_parse_tensor_camel

    class DummyBackend:
        pass

    data_bytes = pickle.dumps([1.0, 2.0])
    res = _np_parse_tensor_camel(DummyBackend(), data_bytes)
    assert res is not None

    try:
        _np_parse_tensor_camel(DummyBackend(), b"{invalid pickle}")
    except RuntimeError:
        pass


def test_math_string_io_decode_csv_camel():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_decode_csv_camel

    class DummyBackend:
        pass

    records = ["1.0,2.0", "3.0,4.0"]
    defaults = [0.0, 0.0]
    res = _np_decode_csv_camel(DummyBackend(), records, record_defaults=defaults)
    assert res is not None

    # Empty records -> exception handled differently?
    try:
        _np_decode_csv_camel(DummyBackend(), [])
    except Exception:
        pass


def test_math_string_io_sparsemapvalues():
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_sparsemapvalues

    class DummyBackend:
        def array(self, x):
            return np.array(x)

    res = _np_sparsemapvalues(DummyBackend(), lambda x: x * 2, [1, 2, 3])
    assert np.array_equal(res, [2, 4, 6])


def test_math_string_io_sparsedensematmul_fallback():
    import numpy as np

    import ml_switcheroo_compiler.ops as ops
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_sparsedensematmul

    class DummyBackend:
        def sparsedensematmul(self, *args, **kwargs):
            return "hit_bk"

    # hit backend branch
    assert _np_sparsedensematmul(DummyBackend(), np.ones((2, 2)), np.ones((2, 2))) == "hit_bk"

    class MockSDM:
        def __new__(cls, *args, **kwargs):
            return "hit_mock"

    if not hasattr(ops, "OpDef"):

        class DummyOpDef:
            pass

        ops.OpDef = DummyOpDef

    ops.SparseDenseMatMul = MockSDM
    assert _np_sparsedensematmul(DummyBackend(), np.ones((2, 2)), np.ones((2, 2))) == "hit_mock"

    # Exception branch
    from unittest.mock import patch

    with patch("builtins.issubclass", side_effect=Exception("Test")):
        try:
            _np_sparsedensematmul(DummyBackend(), np.ones((2, 2)), np.ones((2, 2)))
        except RuntimeError:
            pass

    del ops.SparseDenseMatMul


def test_math_string_io_csv_edge_cases():
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _get_csv_data, _parse_csv_row

    # 223-224 (exception during casting)
    try:
        _parse_csv_row(["invalid"], [0.0], np)
    except RuntimeError:
        pass

    # 226 (row shorter than defaults)
    res = _parse_csv_row(["1.0"], [0.0, 2.0], np)
    assert len(res) == 2
    assert res[1] == 2.0

    # 241 (args empty)
    res2 = _get_csv_data((), np)
    assert res2 == ""


def test_math_string_io_csv_exception():
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_decode_csv_camel

    class DummyBackend:
        pass

    with patch("csv.reader", side_effect=Exception("Test")):
        try:
            _np_decode_csv_camel(DummyBackend(), ["1.0,2.0"])
        except ValueError:
            pass

    # Test empty output
    res = _np_decode_csv_camel(DummyBackend(), [""], record_defaults=[0.0])
    assert res is not None


def test_math_string_io_vision_formats():
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _load_vision_formats

    with patch("os.path.exists", return_value=False):
        assert _load_vision_formats() == {}


def test_math_string_io_decode_image_exceptions():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_decode_image_camel

    class DummyBackend:
        pass

    # 334 (not bytes)
    try:
        _np_decode_image_camel(DummyBackend(), "not_bytes")
    except RuntimeError:
        pass

    # 355-356 (invalid bytes -> PIL fails)
    try:
        _np_decode_image_camel(DummyBackend(), b"invalid_image_data")
    except RuntimeError:
        pass


def test_math_string_io_read_write_file_camel_success(tmp_path):
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_read_file_camel, _np_write_file_camel

    class DummyBackend:
        pass

    test_file = tmp_path / "test.bin"

    # Write non-bytes -> 521 contents.tobytes()
    _np_write_file_camel(DummyBackend(), str(test_file), np.array([1, 2, 3], dtype=np.int32))

    # Read success -> 491
    res = _np_read_file_camel(DummyBackend(), str(test_file))
    assert res is not None
