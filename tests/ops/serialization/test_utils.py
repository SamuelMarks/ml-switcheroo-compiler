"""Tests for test_utils."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.serialization.utils import (
    _dtype_to_descr,
    _extract_arr_bytes,
    _extract_arr_shape_dtype,
    _extract_numpy_weights,
    _get_data_bytes,
    _get_shape_and_dtype,
    _tensor_to_npy_bytes,
    concatenate_arrays,
    get_npz_bytes,
    is_numpy_array,
    load_ir_graph,
    load_npz,
    parse_npz,
    save_ir_graph,
    to_numpy,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, data: EagerValue = None) -> None:
        """Test function __init__."""
        self._data = data

    def numpy(self) -> EagerValue:
        """Test function numpy."""
        return self._data


class MockTensorWithData:
    """Test class MockTensorWithData."""

    def __init__(self, data: EagerValue) -> None:
        """Test function __init__."""
        self.data = MockTensor(data)


class MockListTensor:
    """Test class MockListTensor."""

    def tolist(self) -> EagerValue:
        """Test function tolist."""
        return [1, 2]


class MockBadTensor:
    """Test class MockBadTensor."""

    def tolist(self) -> None:
        """Test function tolist."""
        raise ValueError("bad")


def test_extract_numpy_weights() -> None:
    """Test function test_extract_numpy_weights."""
    w = {"a": MockTensor("a_data"), "b": MockTensorWithData("b_data"), "c": MockListTensor(), "d": MockBadTensor()}
    res = cast("Callable[..., EagerTensorProtocol]", _extract_numpy_weights)(w)
    assert res["a"] == "a_data"
    assert res["b"] == "b_data"
    assert res["c"] == [1, 2]


def test_to_numpy() -> None:
    """Test function test_to_numpy."""
    assert cast("Callable[..., EagerTensorProtocol]", to_numpy)(cast("Callable[..., EagerTensorProtocol]", MockTensor)("a_data")) == "a_data"
    assert cast("Callable[..., EagerTensorProtocol]", to_numpy)(cast("Callable[..., EagerTensorProtocol]", MockTensorWithData)("b_data")) == "b_data"
    assert cast("Callable[..., EagerTensorProtocol]", to_numpy)(cast("Callable[..., EagerTensorProtocol]", MockListTensor)()) == [1, 2]
    assert isinstance(cast("Callable[..., EagerTensorProtocol]", to_numpy)(cast("Callable[..., EagerTensorProtocol]", MockBadTensor)()), MockBadTensor)


def test_concatenate_arrays(mocker: EagerTensorProtocol) -> None:
    """Test function test_concatenate_arrays."""
    mocker.patch("ml_switcheroo_compiler.ops.concatenate", return_value="concat")
    assert cast("Callable[..., EagerTensorProtocol]", concatenate_arrays)([1, 2]) == "concat"


def test_is_numpy_array() -> None:
    """Test function test_is_numpy_array."""
    import numpy as np

    assert is_numpy_array(np.array([1]))
    assert cast("Callable[..., EagerTensorProtocol]", is_numpy_array)(cast("Callable[..., EagerTensorProtocol]", MockTensor)())

    class Arr:
        """Test class Arr."""

        def __array__(self) -> EagerValue:
            """Test function __array__."""
            return np.array([1])

    assert cast("Callable[..., EagerTensorProtocol]", is_numpy_array)(cast("Callable[..., EagerTensorProtocol]", Arr)())
    assert not is_numpy_array(1)


def test_dtype_to_descr() -> None:
    """Test function test_dtype_to_descr."""
    assert _dtype_to_descr(DType.Float32) == "<f4"

    class FakeDtype:
        """Test class FakeDtype."""

        value = "unknown"

    assert cast("Callable[..., EagerTensorProtocol]", _dtype_to_descr)(cast("Callable[..., EagerTensorProtocol]", FakeDtype)()) == "<f4"
    assert _dtype_to_descr(DType.Int64) == "<i8"
    assert _dtype_to_descr("float32") == "<f4"
    assert _dtype_to_descr("unknown") == "<f4"


def test_extract_arr_shape_dtype() -> None:
    """Test function test_extract_arr_shape_dtype."""

    class Arr:
        """Test class Arr."""

        shape = (2, 3)
        dtype = type("D", (), {"name": "float32"})()

    assert cast("Callable[..., EagerTensorProtocol]", _extract_arr_shape_dtype)(cast("Callable[..., EagerTensorProtocol]", Arr)()) == ((2, 3), "float32")

    class Arr2:
        """Test class Arr2."""

        pass

    assert cast("Callable[..., EagerTensorProtocol]", _extract_arr_shape_dtype)(cast("Callable[..., EagerTensorProtocol]", Arr2)()) == ((), "<f4")


def test_get_shape_and_dtype() -> None:
    """Test function test_get_shape_and_dtype."""

    class EvalTensor:
        """Test class EvalTensor."""

        def eval(self) -> EagerValue:
            """Test function eval."""

            class Arr:
                """Test class Arr."""

                shape = (2,)
                dtype = type("D", (), {"name": "float32"})()

            return Arr()

    assert cast("Callable[..., EagerTensorProtocol]", _get_shape_and_dtype)(cast("Callable[..., EagerTensorProtocol]", EvalTensor)()) == ((2,), "float32")

    class NpyTensor:
        """Test class NpyTensor."""

        def numpy(self) -> EagerValue:
            """Test function numpy."""

            class Arr:
                """Test class Arr."""

                shape = (3,)
                dtype = type("D", (), {"name": "int32"})()

            return Arr()

    assert cast("Callable[..., EagerTensorProtocol]", _get_shape_and_dtype)(cast("Callable[..., EagerTensorProtocol]", NpyTensor)()) == ((3,), "int32")

    class DataNpyTensor:
        """Test class DataNpyTensor."""

        data = NpyTensor()

    assert cast("Callable[..., EagerTensorProtocol]", _get_shape_and_dtype)(cast("Callable[..., EagerTensorProtocol]", DataNpyTensor)()) == ((3,), "int32")

    class SimpleTensor:
        """Test class SimpleTensor."""

        shape = (4,)
        dtype = "float32"

    assert cast("Callable[..., EagerTensorProtocol]", _get_shape_and_dtype)(cast("Callable[..., EagerTensorProtocol]", SimpleTensor)()) == ((4,), "float32")


def test_extract_arr_bytes() -> None:
    """Test function test_extract_arr_bytes."""

    class T1:
        """Test class T1."""

        def tobytes(self) -> EagerValue:
            """Test function tobytes."""
            return b"t1"

    assert cast("Callable[..., EagerTensorProtocol]", _extract_arr_bytes)(cast("Callable[..., EagerTensorProtocol]", T1)()) == b"t1"

    class T2:
        """Test class T2."""

        data = T1()

    assert cast("Callable[..., EagerTensorProtocol]", _extract_arr_bytes)(cast("Callable[..., EagerTensorProtocol]", T2)()) == b"t1"
    assert cast("Callable[..., EagerTensorProtocol]", _extract_arr_bytes)(cast("Callable[..., EagerTensorProtocol]", object)()) == b""


def test_get_data_bytes() -> None:
    """Test function test_get_data_bytes."""

    class EvalT:
        """Test class EvalT."""

        def eval(self) -> EagerValue:
            """Test function eval."""

            class Arr:
                """Test class Arr."""

                def tobytes(self) -> EagerValue:
                    """Test function tobytes."""
                    return b"eval"

            return cast("EagerTensorProtocol", Arr())

    assert cast("Callable[..., EagerTensorProtocol]", _get_data_bytes)(cast("Callable[..., EagerTensorProtocol]", EvalT)()) == b"eval"

    class NumpyT:
        """Test class NumpyT."""

        def numpy(self) -> EagerValue:
            """Test function numpy."""

            class Arr:
                """Test class Arr."""

                def tobytes(self) -> EagerValue:
                    """Test function tobytes."""
                    return b"npy"

            return cast("EagerTensorProtocol", Arr())

    assert cast("Callable[..., EagerTensorProtocol]", _get_data_bytes)(cast("Callable[..., EagerTensorProtocol]", NumpyT)()) == b"npy"

    class DataNpyT:
        """Test class DataNpyT."""

        data = NumpyT()

    assert cast("Callable[..., EagerTensorProtocol]", _get_data_bytes)(cast("Callable[..., EagerTensorProtocol]", DataNpyT)()) == b"npy"
    assert cast("Callable[..., EagerTensorProtocol]", _get_data_bytes)(cast("Callable[..., EagerTensorProtocol]", object)()) == b""


def test_tensor_to_npy_bytes() -> None:
    """Test function test_tensor_to_npy_bytes."""

    class MockArr:
        """Test class MockArr."""

        shape = (2,)
        dtype = type("D", (), {"name": "float32"})()

        def tobytes(self) -> EagerValue:
            """Test function tobytes."""
            return b"12345678"

    class MockT:
        """Test class MockT."""

        def numpy(self) -> EagerValue:
            """Test function numpy."""
            return MockArr()

    b = cast("Callable[..., EagerTensorProtocol]", _tensor_to_npy_bytes)(cast("Callable[..., EagerTensorProtocol]", MockT)())
    assert b.startswith(b"\x93NUMPY")


def test_get_npz_bytes(mocker: EagerTensorProtocol) -> None:
    """Test function test_get_npz_bytes."""
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    setattr(mock_backend.get_npz_bytes, "return_value", b"npz_bytes")
    assert get_npz_bytes({}) == b"npz_bytes"
    del mock_backend.get_npz_bytes

    class MockArr:
        """Test class MockArr."""

        shape = (2,)
        dtype = type("D", (), {"name": "float32"})()

        def tobytes(self) -> EagerValue:
            """Test function tobytes."""
            return b"12345678"

    class MockT:
        """Test class MockT."""

        def numpy(self) -> EagerValue:
            """Test function numpy."""
            return MockArr()

    b = cast("Callable[..., EagerTensorProtocol]", get_npz_bytes)({"a": cast("Callable[..., EagerTensorProtocol]", MockT)(), "b.npy": cast("Callable[..., EagerTensorProtocol]", MockT)()})
    assert len(b) > 0


def test_load_npz(mocker: EagerTensorProtocol) -> None:
    """Test function test_load_npz."""
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    setattr(mock_backend.load_npz, "return_value", [1, 2])
    assert cast("EagerTensorProtocol", load_npz("file")) == cast("EagerTensorProtocol", [1, 2])
    setattr(mock_backend.load_npz, "side_effect", NotImplementedError)
    mocker.patch("ml_switcheroo_compiler.serialization.utils.parse_npz", return_value={"a": 1})
    assert cast("EagerTensorProtocol", load_npz("file")) == cast("EagerTensorProtocol", [1])


def test_save_load_ir_graph(mocker: EagerTensorProtocol) -> None:
    """Test function test_save_load_ir_graph."""
    mocker.patch("ml_switcheroo_compiler.serialization.ir_format.graph_to_json", return_value="json_graph")
    mocker.patch("ml_switcheroo_compiler.serialization.ir_format.json_to_graph", return_value="graph_obj")
    import tempfile

    with tempfile.NamedTemporaryFile() as f:
        cast("Callable[..., EagerTensorProtocol]", save_ir_graph)("graph", f.name)
        assert cast("EagerTensorProtocol", load_ir_graph(f.name)) == cast("EagerTensorProtocol", "graph_obj")


def test_load_npz_extra(mocker: EagerTensorProtocol) -> None:
    """Test function test_load_npz_extra."""
    assert parse_npz("file") == {}
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    del mock_backend.load_npz
    assert load_npz("file") == []


def test_dtype_to_descr_extra() -> None:
    """Test function test_dtype_to_descr_extra."""
    assert _dtype_to_descr(DType.Float32) == "<f4"

    class FakeDtype:
        """Test class FakeDtype."""

        value = "unknown"

    assert cast("Callable[..., EagerTensorProtocol]", _dtype_to_descr)(cast("Callable[..., EagerTensorProtocol]", FakeDtype)()) == "<f4"
    assert _dtype_to_descr("int8") == "|i1"

    class FakeDtype2:
        """Test class FakeDtype2."""

        value = "float32"

    assert _dtype_to_descr(FakeDtype2) == "<f4"
    assert _dtype_to_descr(DType.Float64) == "<f8"
    assert _dtype_to_descr("int32") == "<i4"
    assert _dtype_to_descr("something_else") == "<f4"


def test_dtype_fallback_no_value() -> None:
    """Test function test_dtype_fallback_no_value."""
    from ml_switcheroo_compiler.serialization.utils import _dtype_to_descr

    assert _dtype_to_descr(None) == "<f4"
    assert _dtype_to_descr(123) == "<f4"
