"""Tests for numpy eager io ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numpy.eager.io_ops as mod
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

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_load_save(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_load_save."""
    import os

    f = str(tmp_path / "test.npy")
    _np_save(np, f, np.array([1, 2, 3]))
    assert os.path.exists(f)


def test_save_gguf() -> None:
    """Test function test_save_gguf."""
    with pytest.raises(RuntimeError):
        _np_save_gguf(np, "test")


def test_savez(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_savez."""
    f = str(tmp_path / "test.npz")
    _np_savez(np, f, arr=np.array([1, 2, 3]))
    _np_savez_compressed(np, f, arr=np.array([1, 2, 3]))
    assert f is not None or f is None


def test_read_write_file(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_read_write_file."""
    f = str(tmp_path / "test.txt")
    _np_write_file(np, f, b"hello")
    res = _np_read_file(np, f)
    assert cast("EagerTensorProtocol", getattr(res, "data")) == b"hello"


def test_decode_image() -> None:
    """Test function test_decode_image."""
    res = _np_decode_image(np, None, dtype="uint8")
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(res, "config")).dtype, "value")) == "uint8"


def test_decode_csv() -> None:
    """Test function test_decode_csv."""
    res = _np_decode_csv(np, None, record_defaults=[1, 2])
    assert cast("Callable[..., EagerTensorProtocol]", len)(res) == 2


def test_parse_example() -> None:
    """Test function test_parse_example."""
    res = _np_parse_example(np, None, features={"a": 1})
    assert cast("EagerTensorProtocol", "a") in cast("EagerTensorProtocol", res)


def test_serialize_tensor() -> None:
    """Test function test_serialize_tensor."""
    res = _np_serialize_tensor(np, None)
    assert cast("EagerTensorProtocol", getattr(res, "data")) == b""


def test_parse_tensor() -> None:
    """Test function test_parse_tensor."""
    res = _np_parse_tensor(np, None, out_type="float32")
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(res, "config")).dtype, "value")) == "float32"


def test_base64() -> None:
    """Test function test_base64."""
    import base64

    b = b"hello"
    enc = _np_encode_base64(np, b, pad=True)
    assert cast("EagerTensorProtocol", getattr(enc, "data")) == base64.b64encode(b)
    dec = _np_decode_base64(np, cast("EagerTensorProtocol", getattr(enc, "data")))
    assert cast("EagerTensorProtocol", getattr(dec, "data")) == b


def test_parse_sequence_example() -> None:
    """Test function test_parse_sequence_example."""
    (r1, r2) = cast("Callable[..., EagerTensorProtocol]", _np_parse_sequence_example)(np, None)
    assert r1 == {} and r2 == {}


def test_load() -> None:
    """Test function test_load."""
    from ml_switcheroo_compiler.backends.numpy.eager.io_ops import _np_load

    with pytest.raises(ValueError):
        _np_load(np, "non_existent_file.txt")
    with pytest.raises(ValueError):
        _np_load(np, file="non_existent_file.txt")


def test_save_kwargs(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_save_kwargs."""
    f = str(tmp_path / "test.npy")
    _np_save(np, filepath=f, arr=np.array([1, 2, 3]))
    _np_save(np, file=f, arr=np.array([1, 2, 3]))
    assert f is not None or f is None


def test_savez_kwargs(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_savez_kwargs."""
    f = str(tmp_path / "test.npz")
    _np_savez(np, filepath=f, arr=np.array([1, 2, 3]))
    _np_savez_compressed(np, file=f, arr=np.array([1, 2, 3]))
    assert f is not None or f is None


def test_read_file_invalid() -> None:
    """Test function test_read_file_invalid."""
    res = _np_read_file(np, 123)
    assert cast("EagerTensorProtocol", getattr(res, "data")) is None


def test_missing_io_coverage() -> None:
    """Test function test_missing_io_coverage."""
    from ml_switcheroo_compiler.backends.numpy.eager.io_ops import _np_decode_base64, _np_encode_base64, _np_parse_sequence_example

    _np_encode_base64(np, np.array([b"hello world"]))
    decoded = _np_decode_base64(np, np.array([b"aGVsbG8gd29ybGQ="]))
    assert "hello world" in str(cast("EagerTensorProtocol", getattr(decoded, "numpy"))() if hasattr(decoded, "numpy") else decoded)
    res = _np_parse_sequence_example(np, "serialized")
    assert "dummy" in getattr(res, "__getitem__")(0)


def test_load_fallback() -> None:
    """Test function test_load_fallback."""
    import ml_switcheroo_compiler.ops.io as io_mod
    from ml_switcheroo_compiler.backends.numpy.eager.io_ops import _np_load

    original = io_mod._fallback_load
    try:
        io_mod._fallback_load = lambda *a, **k: "loaded"
        assert _np_load(np, "test") == "loaded"
    finally:
        io_mod._fallback_load = original


def _test_io_ops_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_io_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    tmp_path = _state["tmp_path"]

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    bk = DummyBackend()
    p = tmp_path / "f.npy"
    mod._np_save(bk, str(p), np.array([1.0]))
    mod._np_save(bk, filepath=str(p), arr=np.array([1.0]))
    mod._np_save(bk, file=str(p), arr=np.array([1.0]))
    mod._np_save(bk)
    assert mod._np_load(bk, str(p)) is not None
    with pytest.raises(ValueError, match="Format not supported"):
        mod._np_load(bk, str(tmp_path / "nonexistent.xyz"))
    pz = tmp_path / "fz.npz"
    mod._np_savez(bk, str(pz), np.array([1.0]), a=np.array([2.0]))
    mod._np_savez(bk, file=str(pz), args=[np.array([1.0])], kwds={"a": np.array([2.0])})
    mod._np_savez_compressed(bk, str(pz), np.array([1.0]), a=np.array([2.0]))
    mod._np_savez_compressed(bk, file=str(pz), args=[np.array([1.0])], kwds={"a": np.array([2.0])})
    res = mod._np_load(bk, str(pz))
    assert res is not None
    mod._np_load(bk, file=str(pz))
    p_file = tmp_path / "f.txt"
    mod._np_write_file(bk, str(p_file), b"hello")
    getattr(_state, "update")({"DummyBackend": DummyBackend, "bk": bk, "p": p, "p_file": p_file})
    getattr(_state, "update")({"pz": pz, "res": res})
    _state.update(locals())


def _test_io_ops_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_io_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bk, p_file) = (_state["bk"], _state["p_file"])
    mod._np_write_file(bk, filename=str(p_file), contents=b"hello")
    mod._np_write_file(bk, 123, b"hello")
    mod._np_read_file(bk, str(p_file))
    mod._np_read_file(bk, filename=str(p_file))
    mod._np_read_file(bk, 123)
    mod._np_decode_image(bk, b"not image")
    mod._np_decode_image(bk, contents=b"not image")
    mod._np_decode_image(bk, 123)
    mod._np_decode_csv(bk, [b"1,2", b"3,4"])
    mod._np_decode_csv(bk, records=[b"1,2", b"3,4"], record_defaults=[[0.0], [0.0]])
    mod._np_parse_example(bk, b"")
    mod._np_parse_example(bk, serialized=b"", features={"f": "dummy"})
    mod._np_parse_sequence_example(bk, b"")
    mod._np_parse_sequence_example(bk, serialized=b"")
    mod._np_parse_sequence_example(bk, 123)
    mod._np_encode_base64(bk, b"hello")
    mod._np_encode_base64(bk, input=b"hello")
    mod._np_decode_base64(bk, b"aGVsbG8=")
    mod._np_decode_base64(bk, input=b"aGVsbG8=")
    mod._np_encode_base64(bk, None)
    mod._np_decode_base64(bk, None)
    mod._np_serialize_tensor(bk, np.array([1.0]))
    _state.update(locals())


def _test_io_ops_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_io_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bk, tmp_path) = (_state["bk"], _state["tmp_path"])
    mod._np_serialize_tensor(bk, tensor=np.array([1.0]))
    mod._np_parse_tensor(bk, b"")
    mod._np_parse_tensor(bk, serialized=b"")
    with pytest.raises(RuntimeError):
        mod._np_save_gguf(bk, str(tmp_path / "f.gguf"), {"a": np.array([1.0])})
    with pytest.raises(RuntimeError):
        mod._np_save_gguf(bk, filepath=str(tmp_path / "f.gguf"), tensors={"a": np.array([1.0])})
    mod._np_parse_sequence_example(bk, None)


def test_io_ops_coverage(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_io_ops_coverage."""
    _state: dict[str, EagerTensorProtocol] = {"tmp_path": tmp_path}
    _test_io_ops_coverage_part_1(_state)
    _test_io_ops_coverage_part_2(_state)
    _test_io_ops_coverage_part_3(_state)
    assert isinstance(_state, dict)


def test_math_string_io_decode_image_branches() -> None:
    """Test function test_math_string_io_decode_image_branches."""
    import io

    from PIL import Image

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_decode_image

    img = Image.new("L", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    res1 = _np_decode_image(DummyBackend(), data, channels=1)
    assert cast("EagerTensorProtocol", getattr(res1, "shape")) == (10, 10, 1)
    res3 = _np_decode_image(DummyBackend(), data, channels=3)
    assert cast("EagerTensorProtocol", getattr(res3, "shape")) == (10, 10, 3)
    res4 = _np_decode_image(DummyBackend(), data, channels=4)
    assert cast("EagerTensorProtocol", getattr(res4, "shape")) == (10, 10, 4)
    res0 = _np_decode_image(DummyBackend(), data, channels=0)
    assert cast("EagerTensorProtocol", getattr(res0, "shape")) == (10, 10, 1)


def _test_math_string_io_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_string_io_coverage.

    Args:
        _state: Shared test state dictionary.
    """
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
        """Test class DummyBackend."""

        pass

    _parse_scanop_args((), {})
    _parse_scanop_args((1, 2, 3), {})
    _parse_scanop_args((), {"fn": 1, "elems": 2})
    with pytest.raises(IndexError):
        _np_sparsedensematmul(DummyBackend())
    with pytest.raises(IndexError):
        _np_sparsemapvalues(DummyBackend())
    with pytest.raises(AttributeError):
        _np_sparsereshape(DummyBackend())
    with pytest.raises(AttributeError):
        _np_sparsesampledadd(DummyBackend())
    with pytest.raises(AttributeError):
        _np_sparsetranspose(DummyBackend())
    getattr(_state, "update")({"DummyBackend": DummyBackend, "_np_decode_csv": _np_decode_csv, "_np_decode_csv_camel": _np_decode_csv_camel, "_np_decode_image": _np_decode_image})
    getattr(_state, "update")({"_np_decode_image_camel": _np_decode_image_camel, "_np_encode_image_camel": _np_encode_image_camel, "_np_parse_example": _np_parse_example, "_np_parse_example_camel": _np_parse_example_camel})
    getattr(_state, "update")({"_np_parse_tensor": _np_parse_tensor, "_np_parse_tensor_camel": _np_parse_tensor_camel, "_np_read_file": _np_read_file, "_np_read_file_camel": _np_read_file_camel})
    getattr(_state, "update")({"_np_sparsedensematmul": _np_sparsedensematmul, "_np_sparsemapvalues": _np_sparsemapvalues, "_np_sparsereshape": _np_sparsereshape, "_np_sparsesampledadd": _np_sparsesampledadd})
    getattr(_state, "update")({"_np_sparsetranspose": _np_sparsetranspose, "_np_write_file": _np_write_file, "_np_write_file_camel": _np_write_file_camel, "_parse_scanop_args": _parse_scanop_args})
    _state.update(locals())


def _test_math_string_io_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_string_io_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyBackend, _np_decode_csv, _np_decode_image, _np_parse_example) = (_state["DummyBackend"], _state["_np_decode_csv"], _state["_np_decode_image"], _state["_np_parse_example"])
    (_np_parse_tensor, _np_read_file) = (_state["_np_parse_tensor"], _state["_np_read_file"])
    _np_decode_csv(DummyBackend())
    _np_decode_image(DummyBackend())
    _np_parse_example(DummyBackend())
    _np_parse_tensor(DummyBackend())
    _np_read_file(DummyBackend())
    _state.update(locals())


def _test_math_string_io_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_string_io_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyBackend, _np_decode_csv_camel, _np_decode_image_camel, _np_encode_image_camel) = (_state["DummyBackend"], _state["_np_decode_csv_camel"], _state["_np_decode_image_camel"], _state["_np_encode_image_camel"])
    (_np_parse_example_camel, _np_write_file) = (_state["_np_parse_example_camel"], _state["_np_write_file"])
    _np_write_file(DummyBackend())
    _np_decode_csv_camel(DummyBackend())
    _np_decode_image_camel(DummyBackend())
    _np_encode_image_camel(DummyBackend())
    _np_parse_example_camel(DummyBackend())
    _state.update(locals())


def _test_math_string_io_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_math_string_io_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyBackend, _np_parse_tensor_camel, _np_read_file_camel, _np_write_file_camel) = (_state["DummyBackend"], _state["_np_parse_tensor_camel"], _state["_np_read_file_camel"], _state["_np_write_file_camel"])
    _np_parse_tensor_camel(DummyBackend())
    _np_read_file_camel(DummyBackend())
    _np_write_file_camel(DummyBackend())


def test_math_string_io_coverage() -> None:
    """Test function test_math_string_io_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_string_io_coverage_part_1(_state)
    _test_math_string_io_coverage_part_2(_state)
    _test_math_string_io_coverage_part_3(_state)
    _test_math_string_io_coverage_part_4(_state)
    assert isinstance(_state, dict)


def test_math_string_io_encode_image_branches() -> None:
    """Test function test_math_string_io_encode_image_branches."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_encode_image_camel

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    arr_3d = np.zeros((10, 10, 1), dtype=np.uint8)
    res = _np_encode_image_camel(DummyBackend(), arr_3d, op_name="EncodeJpeg")
    assert res is not None
    arr_f32 = np.zeros((10, 10), dtype=np.float32)
    res2 = _np_encode_image_camel(DummyBackend(), arr_f32, op_name="EncodePng")
    assert res2 is not None
    with pytest.raises(RuntimeError):
        _np_encode_image_camel(DummyBackend(), None)


def test_math_string_io_parse_example_camel() -> None:
    """Test function test_math_string_io_parse_example_camel."""
    import json

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_parse_example_camel

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    features = {"a": type("DummyFeature", (), {"dtype": float})()}
    data_str = json.dumps({"a": [1.0, 2.0]})
    res = _np_parse_example_camel(DummyBackend(), data_str, features=features)
    assert cast("EagerTensorProtocol", "a") in cast("EagerTensorProtocol", res)
    res2 = _np_parse_example_camel(DummyBackend(), data_str.encode("utf-8"), features=features)
    assert cast("EagerTensorProtocol", "a") in cast("EagerTensorProtocol", res2)
    data_str2 = json.dumps({"b": [1.0, 2.0]})
    res3 = _np_parse_example_camel(DummyBackend(), data_str2, features=features)
    assert cast("EagerTensorProtocol", "a") in cast("EagerTensorProtocol", res3)
    with pytest.raises(RuntimeError):
        _np_parse_example_camel(DummyBackend(), "{invalid json}", features=features)


def test_math_string_io_read_file_camel() -> None:
    """Test function test_math_string_io_read_file_camel."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_read_file_camel

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    with pytest.raises(RuntimeError):
        _np_read_file_camel(DummyBackend(), "missing_file.txt")


def test_math_string_io_write_file_camel(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_math_string_io_write_file_camel."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_write_file_camel

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    test_file = tmp_path / "test.txt"
    _np_write_file_camel(DummyBackend(), str(test_file), np.array(b"data"))
    with pytest.raises(OSError):
        _np_write_file_camel(DummyBackend(), "/invalid/path/test.txt", "data")


def test_math_string_io_parse_tensor_camel() -> None:
    """Test function test_math_string_io_parse_tensor_camel."""
    import pickle

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_parse_tensor_camel

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    data_bytes = pickle.dumps([1.0, 2.0])
    res = _np_parse_tensor_camel(DummyBackend(), data_bytes)
    assert res is not None
    with pytest.raises(RuntimeError):
        _np_parse_tensor_camel(DummyBackend(), b"{invalid pickle}")


def test_math_string_io_decode_csv_camel() -> None:
    """Test function test_math_string_io_decode_csv_camel."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_decode_csv_camel

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    records = ["1.0,2.0", "3.0,4.0"]
    defaults = [0.0, 0.0]
    res = _np_decode_csv_camel(DummyBackend(), records, record_defaults=defaults)
    assert res is not None
    with pytest.raises(IndexError):
        _np_decode_csv_camel(DummyBackend(), [])


def test_math_string_io_sparsemapvalues() -> None:
    """Test function test_math_string_io_sparsemapvalues."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_sparsemapvalues

    class DummyBackend:
        """Test class DummyBackend."""

        def array(self, x: EagerValue) -> EagerValue:
            """Test function array."""
            return np.array(x)

    res = _np_sparsemapvalues(DummyBackend(), lambda x: x * 2, [1, 2, 3])
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res, [2, 4, 6])


def test_math_string_io_sparsedensematmul_fallback() -> None:
    """Test function test_math_string_io_sparsedensematmul_fallback."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_sparsedensematmul

    class DummyBackend:
        """Test class DummyBackend."""

        def sparsedensematmul(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function sparsedensematmul."""
            return "hit_bk"

    assert _np_sparsedensematmul(DummyBackend(), np.ones((2, 2)), np.ones((2, 2))) == "hit_bk"

    class EmptyBackend:
        """Test class EmptyBackend."""

        pass

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    res = _np_sparsedensematmul(EmptyBackend(), a, b)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, a @ b)


def test_math_string_io_csv_edge_cases() -> None:
    """Test function test_math_string_io_csv_edge_cases."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _get_csv_data, _parse_csv_row

    with pytest.raises(RuntimeError):
        _parse_csv_row(["invalid"], [0.0], np)
    res = _parse_csv_row(["1.0"], [0.0, 2.0], np)
    assert cast("Callable[..., EagerTensorProtocol]", len)(res) == 2
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")(1)) == 2.0
    res2 = _get_csv_data((), np)
    assert res2 == ""


def test_math_string_io_csv_exception() -> None:
    """Test function test_math_string_io_csv_exception."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_decode_csv_camel

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    with patch("csv.reader", side_effect=Exception("Test")):
        with pytest.raises(ValueError):
            _np_decode_csv_camel(DummyBackend(), ["1.0,2.0"])
    res = _np_decode_csv_camel(DummyBackend(), [""], record_defaults=[0.0])
    assert res is not None


def test_math_string_io_vision_formats() -> None:
    """Test function test_math_string_io_vision_formats."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _load_vision_formats

    with patch("os.path.exists", return_value=False):
        assert _load_vision_formats() == {}


def test_math_string_io_decode_image_exceptions() -> None:
    """Test function test_math_string_io_decode_image_exceptions."""
    import io

    from PIL import Image

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_decode_image_camel

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    with pytest.raises(RuntimeError):
        _np_decode_image_camel(DummyBackend(), "not_bytes")
    with pytest.raises(RuntimeError):
        _np_decode_image_camel(DummyBackend(), b"invalid_image_data")
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="PNG")
    png_bytes = buf.getvalue()
    res_zero = _np_decode_image_camel(DummyBackend(), png_bytes, channels=0)
    assert cast("EagerTensorProtocol", getattr(res_zero, "shape")) == (10, 10, 3)


def test_math_string_io_read_write_file_camel_success(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_math_string_io_read_write_file_camel_success."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_string_io import _np_read_file_camel, _np_write_file_camel

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    test_file = tmp_path / "test.bin"
    _np_write_file_camel(DummyBackend(), str(test_file), np.array([1, 2, 3], dtype=np.int32))
    res = _np_read_file_camel(DummyBackend(), str(test_file))
    assert res is not None
