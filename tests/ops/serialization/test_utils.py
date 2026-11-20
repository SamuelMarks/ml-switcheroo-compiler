"""Tests for test_utils."""

from __future__ import annotations

import typing

import pytest
from ml_switcheroo_ir import LogicalGraph
from pytest_mock import MockerFixture

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
from tests.testing_utils import MockTestStub

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockTensor(MockTestStub):
    """Test class MockTensor."""

    def __init__(self, data: BackendArg = None) -> None:
        """Test function __init__.

        Args:
            data: Test parameter `data`.
        """
        self._data = data

    def numpy(self) -> BackendArg:
        """Test function numpy.

        Returns:
            BackendArg: Computed output value from `numpy`.
        """
        return self._data


class MockTensorWithData(MockTestStub):
    """Test class MockTensorWithData."""

    def __init__(self, data: BackendArg) -> None:
        """Test function __init__.

        Args:
            data: Test parameter `data`.
        """
        self.data = MockTensor(data)


class MockListTensor(MockTestStub):
    """Test class MockListTensor."""

    def tolist(self) -> BackendArg:
        """Test function tolist.

        Returns:
            BackendArg: Computed output value from `tolist`.
        """
        return [1, 2]


class MockBadTensor(MockTestStub):
    """Test class MockBadTensor."""

    def tolist(self) -> None:
        """Test function tolist.

        Raises:
            ValueError: When raised during `tolist` (bad).
        """
        raise ValueError("bad")


def test_extract_numpy_weights() -> None:
    """Test function test_extract_numpy_weights."""
    w = {"a": MockTensor("a_data"), "b": MockTensorWithData("b_data"), "c": MockListTensor(), "d": MockBadTensor()}
    res = _extract_numpy_weights(w)
    assert res["a"] == "a_data"
    assert res["b"] == "b_data"
    assert res["c"] == [1, 2]


def test_to_numpy() -> None:
    """Test function test_to_numpy."""
    assert to_numpy(MockTensor("a_data")) == "a_data"
    assert to_numpy(MockTensorWithData("b_data")) == "b_data"
    assert to_numpy(MockListTensor()) == [1, 2]
    assert isinstance(to_numpy(MockBadTensor()), MockBadTensor)


def test_concatenate_arrays(mocker: MockerFixture) -> None:
    """Test function test_concatenate_arrays.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.ops.concatenate", return_value="concat")
    assert concatenate_arrays([1, 2]) == "concat"


def test_is_numpy_array() -> None:
    """Test function test_is_numpy_array."""
    import numpy as np

    assert is_numpy_array(np.array([1]))
    assert is_numpy_array(MockTensor())

    class Arr(MockTestStub):
        """Test class Arr."""

        def __array__(self) -> BackendArg:
            """Test function __array__.

            Returns:
                BackendArg: Computed output value from `__array__`.
            """
            return np.array([1])

    assert is_numpy_array(Arr())
    assert not is_numpy_array(1)


def test_dtype_to_descr() -> None:
    """Test function test_dtype_to_descr."""
    assert _dtype_to_descr(DType.Float32) == "<f4"

    class FakeDtype(MockTestStub):
        """Test class FakeDtype."""

        value = "unknown"

    assert _dtype_to_descr(FakeDtype()) == "<f4"
    assert _dtype_to_descr(DType.Int64) == "<i8"
    assert _dtype_to_descr("float32") == "<f4"
    assert _dtype_to_descr("unknown") == "<f4"


def test_extract_arr_shape_dtype() -> None:
    """Test function test_extract_arr_shape_dtype."""

    class Arr(MockTestStub):
        """Test class Arr."""

        shape = (2, 3)
        dtype = type("D", (), {"name": "float32"})()

    assert _extract_arr_shape_dtype(Arr()) == ((2, 3), "float32")

    class Arr2(MockTestStub):
        """Test class Arr2."""

        pass

    assert _extract_arr_shape_dtype(Arr2()) == ((), "<f4")


def test_get_shape_and_dtype() -> None:
    """Test function test_get_shape_and_dtype."""

    class EvalTensor:
        """Test class EvalTensor."""

        def eval(self) -> BackendArg:
            """Test function eval.

            Returns:
                BackendArg: Computed output value from `eval`.
            """

            class Arr(MockTestStub):
                """Test class Arr."""

                shape = (2,)
                dtype = type("D", (), {"name": "float32"})()

            return Arr()

    assert _get_shape_and_dtype(EvalTensor()) == ((2,), "float32")

    class NpyTensor:
        """Test class NpyTensor."""

        def numpy(self) -> BackendArg:
            """Test function numpy.

            Returns:
                BackendArg: Computed output value from `numpy`.
            """

            class Arr(MockTestStub):
                """Test class Arr."""

                shape = (3,)
                dtype = type("D", (), {"name": "int32"})()

            return Arr()

    assert _get_shape_and_dtype(NpyTensor()) == ((3,), "int32")

    class DataNpyTensor(MockTestStub):
        """Test class DataNpyTensor."""

        data = NpyTensor()

    assert _get_shape_and_dtype(DataNpyTensor()) == ((3,), "int32")

    class SimpleTensor:
        """Test class SimpleTensor."""

        shape = (4,)
        dtype = "float32"

    assert _get_shape_and_dtype(SimpleTensor()) == ((4,), "float32")


def test_extract_arr_bytes() -> None:
    """Test function test_extract_arr_bytes."""

    class T1(MockTestStub):
        """Test class T1."""

        def tobytes(self) -> BackendArg:
            """Test function tobytes.

            Returns:
                BackendArg: Computed output value from `tobytes`.
            """
            return b"t1"

    assert _extract_arr_bytes(T1()) == b"t1"

    class T2(MockTestStub):
        """Test class T2."""

        data = T1()

    class _BareDummy(MockTestStub):
        """Empty dummy class."""

    assert _extract_arr_bytes(T2()) == b"t1"
    assert _extract_arr_bytes(_BareDummy()) == b""


def test_get_data_bytes() -> None:
    """Test function test_get_data_bytes."""

    class EvalT:
        """Test class EvalT."""

        def eval(self) -> BackendArg:
            """Test function eval.

            Returns:
                BackendArg: Computed output value from `eval`.
            """

            class Arr(MockTestStub):
                """Test class Arr."""

                def tobytes(self) -> BackendArg:
                    """Test function tobytes.

                    Returns:
                        BackendArg: Computed output value from `tobytes`.
                    """
                    return b"eval"

            return Arr()

    assert _get_data_bytes(EvalT()) == b"eval"

    class NumpyT:
        """Test class NumpyT."""

        def numpy(self) -> BackendArg:
            """Test function numpy.

            Returns:
                BackendArg: Computed output value from `numpy`.
            """

            class Arr(MockTestStub):
                """Test class Arr."""

                def tobytes(self) -> BackendArg:
                    """Test function tobytes.

                    Returns:
                        BackendArg: Computed output value from `tobytes`.
                    """
                    return b"npy"

            return Arr()

    assert _get_data_bytes(NumpyT()) == b"npy"

    class DataNpyT(MockTestStub):
        """Test class DataNpyT."""

        data = NumpyT()

    class _BareDummy(MockTestStub):
        """Empty dummy class."""

    assert _get_data_bytes(DataNpyT()) == b"npy"
    assert _get_data_bytes(_BareDummy()) == b""


def test_tensor_to_npy_bytes() -> None:
    """Test function test_tensor_to_npy_bytes."""

    class MockArr:
        """Test class MockArr."""

        shape = (2,)
        dtype = type("D", (), {"name": "float32"})()

        def tobytes(self) -> BackendArg:
            """Test function tobytes.

            Returns:
                BackendArg: Computed output value from `tobytes`.
            """
            return b"12345678"

    class MockT(MockTestStub):
        """Test class MockT."""

        def numpy(self) -> BackendArg:
            """Test function numpy.

            Returns:
                BackendArg: Computed output value from `numpy`.
            """
            return MockArr()

    b = _tensor_to_npy_bytes(MockT())
    assert b.startswith(b"\x93NUMPY")


def test_get_npz_bytes(mocker: MockerFixture) -> None:
    """Test function test_get_npz_bytes.

    Args:
        mocker: Test parameter `mocker`.
    """
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    pytest.MonkeyPatch().setattr(mock_backend.get_npz_bytes, "return_value", b"npz_bytes", raising=False)
    assert get_npz_bytes({}) == b"npz_bytes"
    del mock_backend.get_npz_bytes

    class MockArr:
        """Test class MockArr."""

        shape = (2,)
        dtype = type("D", (), {"name": "float32"})()

        def tobytes(self) -> BackendArg:
            """Test function tobytes.

            Returns:
                BackendArg: Computed output value from `tobytes`.
            """
            return b"12345678"

    class MockT(MockTestStub):
        """Test class MockT."""

        def numpy(self) -> BackendArg:
            """Test function numpy.

            Returns:
                BackendArg: Computed output value from `numpy`.
            """
            return MockArr()

    b = get_npz_bytes({"a": MockT(), "b.npy": MockT()})
    assert len(b) > 0


def test_load_npz(mocker: MockerFixture) -> None:
    """Test function test_load_npz.

    Args:
        mocker: Test parameter `mocker`.
    """
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    pytest.MonkeyPatch().setattr(mock_backend.load_npz, "return_value", [1, 2], raising=False)
    assert load_npz("file") == [1, 2]
    pytest.MonkeyPatch().setattr(mock_backend.load_npz, "side_effect", NotImplementedError, raising=False)
    mocker.patch("ml_switcheroo_compiler.serialization.utils.parse_npz", return_value={"a": 1})
    assert load_npz("file") == [1]


def test_save_load_ir_graph(mocker: MockerFixture) -> None:
    """Test function test_save_load_ir_graph.

    Args:
        mocker (MockerFixture): Test parameter `mocker`.
    """
    graph_obj = LogicalGraph(name="test_graph")
    mocker.patch("ml_switcheroo_compiler.serialization.ir_format.graph_to_json", return_value="json_graph")
    mocker.patch("ml_switcheroo_compiler.serialization.ir_format.json_to_graph", return_value=graph_obj)
    import tempfile

    with tempfile.NamedTemporaryFile() as f:
        save_ir_graph(graph_obj, f.name)
        assert load_ir_graph(f.name) == graph_obj


def test_load_npz_extra(mocker: MockerFixture) -> None:
    """Test function test_load_npz_extra.

    Args:
        mocker: Test parameter `mocker`.
    """
    assert parse_npz("file") == {}
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    del mock_backend.load_npz
    assert load_npz("file") == []


def test_dtype_to_descr_extra() -> None:
    """Test function test_dtype_to_descr_extra."""
    assert _dtype_to_descr(DType.Float32) == "<f4"

    class FakeDtype(MockTestStub):
        """Test class FakeDtype."""

        value = "unknown"

    assert _dtype_to_descr(FakeDtype()) == "<f4"
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
    import numpy as np

    from ml_switcheroo_compiler.serialization.utils import _dtype_to_descr

    assert _dtype_to_descr(None) == "<f4"
    assert _dtype_to_descr(123) == "<f4"
    npz_raw = get_npz_bytes({"w": np.array([1.0, 2.0], dtype=np.float32)})
    parsed_raw = parse_npz(npz_raw)
    assert "w" in parsed_raw
