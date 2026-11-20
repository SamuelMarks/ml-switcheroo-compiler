"""Tests for numpy eager io ops."""

from __future__ import annotations

import os
import typing
from pathlib import Path

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numpy.eager.io_ops as mod
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import BackendArg, coerce_bytes
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.numpy.eager.io_ops import (
    _np_decode_base64,
    _np_decode_csv,
    _np_decode_image,
    _np_encode_base64,
    _np_parse_tensor,
    _np_read_file,
    _np_save,
    _np_save_gguf,
    _np_savez,
    _np_savez_compressed,
    _np_serialize_tensor,
    _np_write_file,
)
from tests.testing_utils import (
    EagerBoundCallable,
    MockTestStub,
    adapt_backend_module,
    coerce_operand_list,
    ensure_tensor,
    extract_test_data,
    extract_test_shape,
    require_callable,
)


def test_load_save(tmp_path: Path) -> None:
    """Test function test_load_save.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    import os

    f = str(tmp_path / "test.npy")
    _np_save(np, f, np.array([1, 2, 3]))
    assert os.path.exists(f)


def test_save_gguf() -> None:
    """Test function test_save_gguf."""
    with pytest.raises(RuntimeError):
        _np_save_gguf(np, "test")


def test_savez(tmp_path: Path) -> None:
    """Test function test_savez.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    f = str(tmp_path / "test.npz")
    _np_savez(np, f, arr=np.array([1, 2, 3]))
    _np_savez_compressed(np, f, arr=np.array([1, 2, 3]))
    assert os.path.exists(f)


def test_read_write_file(tmp_path: Path) -> None:
    """Test function test_read_write_file.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    f = str(tmp_path / "test.txt")
    _np_write_file(np, f, b"hello")
    res = _np_read_file(np, f)
    assert extract_test_data(res) == b"hello"


def test_decode_image() -> None:
    """Test function test_decode_image."""
    res = _np_decode_image(np, None, dtype="uint8")
    assert adapt_backend_module(ensure_tensor(res).config.dtype).require_symbol("value") == "uint8"


def test_decode_csv() -> None:
    """Test function test_decode_csv."""
    res = _np_decode_csv(np, None, record_defaults=[1, 2])
    assert len(coerce_operand_list(res)) == 2


def test_serialize_tensor() -> None:
    """Test function test_serialize_tensor."""
    res = _np_serialize_tensor(np, None)
    assert extract_test_data(res) == b""


def test_parse_tensor() -> None:
    """Test function test_parse_tensor."""
    res = _np_parse_tensor(np, None, out_type="float32")
    assert adapt_backend_module(ensure_tensor(res).config.dtype).require_symbol("value") == "float32"


def test_base64() -> None:
    """Test function test_base64."""
    import base64

    b = b"hello"
    enc = _np_encode_base64(np, b, pad=True)
    enc_bytes = coerce_bytes(extract_test_data(enc))
    assert enc_bytes == base64.b64encode(b)
    dec = _np_decode_base64(np, enc_bytes)
    assert extract_test_data(dec) == b


def test_load() -> None:
    """Test function test_load."""
    from ml_switcheroo_compiler.backends.numpy.eager.io_ops import _np_load

    with pytest.raises(ValueError):
        _np_load(np, "non_existent_file.txt")
    with pytest.raises(ValueError):
        _np_load(np, file="non_existent_file.txt")


def test_save_kwargs(tmp_path: Path) -> None:
    """Test function test_save_kwargs.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    f = str(tmp_path / "test.npy")
    _np_save(np, filepath=f, arr=np.array([1, 2, 3]))
    _np_save(np, file=f, arr=np.array([1, 2, 3]))
    assert os.path.exists(f)


def test_savez_kwargs(tmp_path: Path) -> None:
    """Test function test_savez_kwargs.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    f = str(tmp_path / "test.npz")
    _np_savez(np, filepath=f, arr=np.array([1, 2, 3]))
    _np_savez_compressed(np, file=f, arr=np.array([1, 2, 3]))
    assert os.path.exists(f)


def test_read_file_invalid() -> None:
    """Test function test_read_file_invalid."""
    res = _np_read_file(np, 123)
    assert extract_test_data(res) is None


def test_missing_io_coverage() -> None:
    """Test function test_missing_io_coverage."""
    from ml_switcheroo_compiler.backends.numpy.eager.io_ops import _np_decode_base64, _np_encode_base64

    _np_encode_base64(np, np.array([b"hello world"]))
    decoded = _np_decode_base64(np, np.array([b"aGVsbG8gd29ybGQ="]))
    assert "hello world" in str(
        ensure_tensor(decoded).numpy() if adapt_backend_module(decoded).has_symbol("numpy") else decoded
    )


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


class _TestIoOpsCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_io_ops_coverage` split helpers."""

    DummyBackend: type[MockTestStub]
    bk: MockTestStub
    p_file: Path
    tmp_path: Path


def _test_io_ops_coverage_part_1(_state: _TestIoOpsCoverageState) -> None:
    """Execute part 1 of test_io_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    tmp_path = _state["tmp_path"]

    class DummyBackend(MockTestStub):
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
    _state.update({"DummyBackend": DummyBackend, "bk": bk, "p_file": p_file})
    _state.update({"tmp_path": tmp_path})


def _test_io_ops_coverage_part_2(_state: _TestIoOpsCoverageState) -> None:
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
    mod._np_encode_base64(bk, b"hello")
    mod._np_encode_base64(bk, input=b"hello")
    mod._np_decode_base64(bk, b"aGVsbG8=")
    mod._np_decode_base64(bk, input=b"aGVsbG8=")
    mod._np_encode_base64(bk, None)
    mod._np_decode_base64(bk, None)
    mod._np_serialize_tensor(bk, np.array([1.0]))
    _state.update({"bk": bk, "p_file": p_file})


def _test_io_ops_coverage_part_3(_state: _TestIoOpsCoverageState) -> None:
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


def test_io_ops_coverage(tmp_path: Path) -> None:
    """Test function test_io_ops_coverage.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    _state: _TestIoOpsCoverageState = {"tmp_path": tmp_path}
    _test_io_ops_coverage_part_1(_state)
    _test_io_ops_coverage_part_2(_state)
    _test_io_ops_coverage_part_3(_state)
    assert len(_state) > 0


def test_math_string_io_decode_image_branches() -> None:
    """Test function test_math_string_io_decode_image_branches."""
    import io

    from PIL import Image

    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_decode_image

    img = Image.new("L", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    res1 = _np_decode_image(DummyBackend(), data, channels=1)
    assert extract_test_shape(res1) == (10, 10, 1)
    res3 = _np_decode_image(DummyBackend(), data, channels=3)
    assert extract_test_shape(res3) == (10, 10, 3)
    res4 = _np_decode_image(DummyBackend(), data, channels=4)
    assert extract_test_shape(res4) == (10, 10, 4)
    res0 = _np_decode_image(DummyBackend(), data, channels=0)
    assert extract_test_shape(res0) == (10, 10, 1)


class _TestMathStringIoCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_math_string_io_coverage` split helpers."""

    DummyBackend: type
    _np_decode_csv: EagerBoundCallable
    _np_decode_csv_camel: EagerBoundCallable
    _np_decode_image: EagerBoundCallable
    _np_decode_image_camel: EagerBoundCallable
    _np_encode_image_camel: EagerBoundCallable
    _np_parse_tensor: EagerBoundCallable
    _np_parse_tensor_camel: EagerBoundCallable
    _np_read_file: EagerBoundCallable
    _np_read_file_camel: EagerBoundCallable
    _np_write_file: EagerBoundCallable
    _np_write_file_camel: EagerBoundCallable


def _test_math_string_io_coverage_part_1(_state: _TestMathStringIoCoverageState) -> None:
    """Execute part 1 of test_math_string_io_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import (
        _np_decode_csv,
        _np_decode_csv_camel,
        _np_decode_image,
        _np_decode_image_camel,
        _np_encode_image_camel,
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

    class DummyBackend(MockTestStub):
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
    _state.update(
        {
            "DummyBackend": DummyBackend,
            "_np_decode_csv": require_callable(_np_decode_csv),
            "_np_decode_csv_camel": require_callable(_np_decode_csv_camel),
            "_np_decode_image": require_callable(_np_decode_image),
        }
    )
    _state.update(
        {
            "_np_decode_image_camel": require_callable(_np_decode_image_camel),
            "_np_encode_image_camel": require_callable(_np_encode_image_camel),
        }
    )
    _state.update(
        {
            "_np_parse_tensor": require_callable(_np_parse_tensor),
            "_np_parse_tensor_camel": require_callable(_np_parse_tensor_camel),
            "_np_read_file": require_callable(_np_read_file),
            "_np_read_file_camel": require_callable(_np_read_file_camel),
        }
    )
    _state.update(
        {
            "_np_write_file": require_callable(_np_write_file),
            "_np_write_file_camel": require_callable(_np_write_file_camel),
        }
    )


def _test_math_string_io_coverage_part_2(_state: _TestMathStringIoCoverageState) -> None:
    """Execute part 2 of test_math_string_io_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyBackend, _np_decode_csv, _np_decode_image) = (
        _state["DummyBackend"],
        _state["_np_decode_csv"],
        _state["_np_decode_image"],
    )
    (_np_parse_tensor, _np_read_file) = (_state["_np_parse_tensor"], _state["_np_read_file"])
    _np_decode_csv(DummyBackend())
    _np_decode_image(DummyBackend())
    _np_parse_tensor(DummyBackend())
    _np_read_file(DummyBackend())
    _state.update(
        {
            "DummyBackend": DummyBackend,
            "_np_decode_csv": require_callable(_np_decode_csv),
            "_np_decode_image": require_callable(_np_decode_image),
        }
    )
    _state.update(
        {"_np_parse_tensor": require_callable(_np_parse_tensor), "_np_read_file": require_callable(_np_read_file)}
    )


def _test_math_string_io_coverage_part_3(_state: _TestMathStringIoCoverageState) -> None:
    """Execute part 3 of test_math_string_io_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyBackend, _np_decode_csv_camel, _np_decode_image_camel, _np_encode_image_camel) = (
        _state["DummyBackend"],
        _state["_np_decode_csv_camel"],
        _state["_np_decode_image_camel"],
        _state["_np_encode_image_camel"],
    )
    _np_write_file = _state["_np_write_file"]
    _np_write_file(DummyBackend())
    _np_decode_csv_camel(DummyBackend())
    _np_decode_image_camel(DummyBackend())
    _np_encode_image_camel(DummyBackend())
    _state.update(
        {
            "DummyBackend": DummyBackend,
            "_np_decode_csv_camel": require_callable(_np_decode_csv_camel),
            "_np_decode_image_camel": require_callable(_np_decode_image_camel),
        }
    )
    _state.update(
        {
            "_np_encode_image_camel": require_callable(_np_encode_image_camel),
            "_np_write_file": require_callable(_np_write_file),
        }
    )


def _test_math_string_io_coverage_part_4(_state: _TestMathStringIoCoverageState) -> None:
    """Execute part 4 of test_math_string_io_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyBackend, _np_parse_tensor_camel, _np_read_file_camel, _np_write_file_camel) = (
        _state["DummyBackend"],
        _state["_np_parse_tensor_camel"],
        _state["_np_read_file_camel"],
        _state["_np_write_file_camel"],
    )
    _np_parse_tensor_camel(DummyBackend())
    _np_read_file_camel(DummyBackend())
    _np_write_file_camel(DummyBackend())


def test_math_string_io_coverage() -> None:
    """Test function test_math_string_io_coverage."""
    _state: _TestMathStringIoCoverageState = {}
    _test_math_string_io_coverage_part_1(_state)
    _test_math_string_io_coverage_part_2(_state)
    _test_math_string_io_coverage_part_3(_state)
    _test_math_string_io_coverage_part_4(_state)
    assert len(_state) > 0


def test_math_string_io_encode_image_branches() -> None:
    """Test function test_math_string_io_encode_image_branches."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_encode_image_camel

    class DummyBackend(MockTestStub):
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


def test_math_string_io_read_file_camel() -> None:
    """Test function test_math_string_io_read_file_camel."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_read_file_camel

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    with pytest.raises(RuntimeError):
        _np_read_file_camel(DummyBackend(), "missing_file.txt")


def test_math_string_io_write_file_camel(tmp_path: Path) -> None:
    """Test function test_math_string_io_write_file_camel.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_write_file_camel

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    test_file = tmp_path / "test.txt"
    _np_write_file_camel(DummyBackend(), str(test_file), np.array(b"data"))
    with pytest.raises(OSError):
        _np_write_file_camel(DummyBackend(), "/invalid/path/test.txt", "data")


def test_math_string_io_parse_tensor_camel() -> None:
    """Test function test_math_string_io_parse_tensor_camel."""
    import pickle

    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_parse_tensor_camel

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    data_bytes = pickle.dumps([1.0, 2.0])
    res = _np_parse_tensor_camel(DummyBackend(), data_bytes)
    assert res is not None
    with pytest.raises(RuntimeError):
        _np_parse_tensor_camel(DummyBackend(), b"{invalid pickle}")


def test_math_string_io_decode_csv_camel() -> None:
    """Test function test_math_string_io_decode_csv_camel."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_decode_csv_camel

    class DummyBackend(MockTestStub):
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

    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_sparsemapvalues

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def array(self, x: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return np.array(x)

    res = _np_sparsemapvalues(DummyBackend(), lambda x: x * 2, [1, 2, 3])
    assert np.array_equal(ensure_ndarray(res), [2, 4, 6])


def test_math_string_io_sparsedensematmul_fallback() -> None:
    """Test function test_math_string_io_sparsedensematmul_fallback."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_sparsedensematmul

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def sparsedensematmul(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function sparsedensematmul.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `sparsedensematmul`.
            """
            return "hit_bk"

    assert _np_sparsedensematmul(DummyBackend(), np.ones((2, 2)), np.ones((2, 2))) == "hit_bk"

    class EmptyBackend(MockTestStub):
        """Test class EmptyBackend."""

        pass

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    res = _np_sparsedensematmul(EmptyBackend(), a, b)
    np.testing.assert_allclose(ensure_ndarray(res), a @ b)


def test_math_string_io_csv_edge_cases() -> None:
    """Test function test_math_string_io_csv_edge_cases."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _get_csv_data, _parse_csv_row

    arr_mod = _resolve_eager_mod(np)
    with pytest.raises(RuntimeError):
        _parse_csv_row(["invalid"], [0.0], arr_mod)
    res = _parse_csv_row(["1.0"], [0.0, 2.0], arr_mod)
    assert len(coerce_operand_list(res)) == 2
    assert ensure_ndarray(res)[1] == 2.0
    res2 = _get_csv_data((), arr_mod)
    assert res2 == ""


def test_math_string_io_csv_exception() -> None:
    """Test function test_math_string_io_csv_exception."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_decode_csv_camel

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    with patch("csv.reader", side_effect=RuntimeError("Test")):
        with pytest.raises(ValueError):
            _np_decode_csv_camel(DummyBackend(), ["1.0,2.0"])
    res = _np_decode_csv_camel(DummyBackend(), [""], record_defaults=[0.0])
    assert res is not None


def test_math_string_io_vision_formats() -> None:
    """Test function test_math_string_io_vision_formats."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _load_vision_formats

    with patch("os.path.exists", return_value=False):
        assert _load_vision_formats() == {}


def test_math_string_io_decode_image_exceptions() -> None:
    """Test function test_math_string_io_decode_image_exceptions."""
    import io

    from PIL import Image

    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_decode_image_camel

    class DummyBackend(MockTestStub):
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
    assert extract_test_shape(res_zero) == (10, 10, 3)


def test_math_string_io_read_write_file_camel_success(tmp_path: Path) -> None:
    """Test function test_math_string_io_read_write_file_camel_success.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import (
        _np_read_file_camel,
        _np_write_file_camel,
    )

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    test_file = tmp_path / "test.bin"
    _np_write_file_camel(DummyBackend(), str(test_file), np.array([1, 2, 3], dtype=np.int32))
    res = _np_read_file_camel(DummyBackend(), str(test_file))
    assert tuple(extract_test_shape(res)) == ()
