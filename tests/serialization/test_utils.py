"""Tests for test_utils."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, cast
from unittest import mock

import numpy as np

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
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


def test_extract_numpy_weights() -> None:
    """Test function test_extract_numpy_weights."""

    class TensorWithNumpy:
        """Test class TensorWithNumpy."""

        def numpy(self) -> EagerValue:
            """Test function numpy."""
            return np.array([1])

    class TensorWithDataNumpy:
        """Test class TensorWithDataNumpy."""

        class Data:
            """Test class Data."""

            def numpy(self) -> EagerValue:
                """Test function numpy."""
                return np.array([2])

        data = Data()

    class TensorWithToList:
        """Test class TensorWithToList."""

        def tolist(self) -> EagerValue:
            """Test function tolist."""
            return [3]

    class ErrorTensor:
        """Test class ErrorTensor."""

        def tolist(self) -> None:
            """Test function tolist."""
            raise ValueError()

    weights = {"a": TensorWithNumpy(), "b": TensorWithDataNumpy(), "c": TensorWithToList(), "d": ErrorTensor(), "e": 5}
    res = cast("EagerCallableProtocol", _extract_numpy_weights)(weights)
    assert cast("EagerCallableProtocol", np.array_equal)(res["a"], cast("EagerCallableProtocol", np.array)([1]))
    assert cast("EagerCallableProtocol", np.array_equal)(res["b"], cast("EagerCallableProtocol", np.array)([2]))
    assert res["c"] == [3]
    assert isinstance(res["d"], ErrorTensor)
    assert res["e"] == 5


def test_to_numpy() -> None:
    """Test function test_to_numpy."""

    class TensorWithNumpy:
        """Test class TensorWithNumpy."""

        def numpy(self) -> EagerValue:
            """Test function numpy."""
            return np.array([1])

    class TensorWithDataNumpy:
        """Test class TensorWithDataNumpy."""

        class Data:
            """Test class Data."""

            def numpy(self) -> EagerValue:
                """Test function numpy."""
                return np.array([2])

        data = Data()

    class TensorWithToList:
        """Test class TensorWithToList."""

        def tolist(self) -> EagerValue:
            """Test function tolist."""
            return [3]

    class ErrorTensor:
        """Test class ErrorTensor."""

        def tolist(self) -> None:
            """Test function tolist."""
            raise ValueError()

    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", to_numpy)(cast("EagerCallableProtocol", TensorWithNumpy)()), cast("EagerCallableProtocol", np.array)([1]))
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", to_numpy)(cast("EagerCallableProtocol", TensorWithDataNumpy)()), cast("EagerCallableProtocol", np.array)([2]))
    assert cast("EagerCallableProtocol", to_numpy)(cast("EagerCallableProtocol", TensorWithToList)()) == [3]
    assert isinstance(cast("EagerCallableProtocol", to_numpy)(cast("EagerCallableProtocol", ErrorTensor)()), ErrorTensor)
    assert cast("EagerCallableProtocol", to_numpy)(5) == 5


def test_concatenate_arrays() -> None:
    """Test function test_concatenate_arrays."""
    with mock.patch("ml_switcheroo_compiler.ops.concatenate") as mock_concat:
        mock_concat.return_value = "concatenated"
        res = cast("EagerCallableProtocol", concatenate_arrays)([cast("EagerCallableProtocol", np.array)([1]), cast("EagerCallableProtocol", np.array)([2])])
        assert res == "concatenated"


def test_is_numpy_array() -> None:
    """Test function test_is_numpy_array."""
    assert is_numpy_array(np.array([1]))

    class FakeArray:
        """Test class FakeArray."""

        def __array__(self) -> EagerValue:
            """Return numpy array."""
            return np.array([1])

    assert cast("EagerCallableProtocol", is_numpy_array)(cast("EagerCallableProtocol", FakeArray)())

    class FakeNumpy:
        """Test class FakeNumpy."""

        def numpy(self) -> EagerValue:
            """Test function numpy."""
            return np.array([1])

    assert cast("EagerCallableProtocol", is_numpy_array)(cast("EagerCallableProtocol", FakeNumpy)())


def test_dtype_to_descr() -> None:
    """Test function test_dtype_to_descr."""
    assert _dtype_to_descr(DType.Float32) == "<f4"
    assert _dtype_to_descr("float32") == "<f4"
    assert _dtype_to_descr("invalid") == "<f4"

    class FakeDType:
        """Test class FakeDType."""

        value = "int32"

    assert cast("EagerCallableProtocol", _dtype_to_descr)(cast("EagerCallableProtocol", FakeDType)()) == "<f4"


def test_get_shape_and_dtype() -> None:
    """Test function test_get_shape_and_dtype."""
    arr = np.array([1, 2], dtype=np.float32)
    assert cast("EagerCallableProtocol", _extract_arr_shape_dtype)(arr) == ((2,), "float32")

    class BadArr:
        """Test class BadArr."""

        pass

    assert cast("EagerCallableProtocol", _extract_arr_shape_dtype)(cast("EagerCallableProtocol", BadArr)()) == ((), "<f4")

    class TensorWithEval:
        """Test class TensorWithEval."""

        def eval(self) -> EagerValue:
            """Test function eval."""
            return np.array([1, 2], dtype=np.float32)

    assert cast("EagerCallableProtocol", _get_shape_and_dtype)(cast("EagerCallableProtocol", TensorWithEval)()) == ((2,), "float32")

    class TensorWithNumpy:
        """Test class TensorWithNumpy."""

        def numpy(self) -> EagerValue:
            """Test function numpy."""
            return np.array([1], dtype=np.float64)

    assert cast("EagerCallableProtocol", _get_shape_and_dtype)(cast("EagerCallableProtocol", TensorWithNumpy)()) == ((1,), "float64")

    class TensorWithDataNumpy:
        """Test class TensorWithDataNumpy."""

        class Data:
            """Test class Data."""

            def numpy(self) -> EagerValue:
                """Test function numpy."""
                return np.array([1], dtype=np.int32)

        data = Data()

    assert cast("EagerCallableProtocol", _get_shape_and_dtype)(cast("EagerCallableProtocol", TensorWithDataNumpy)()) == ((1,), "int32")

    assert cast("EagerCallableProtocol", _get_shape_and_dtype)(cast("EagerCallableProtocol", BadArr)()) == ((), "<f4")


def test_get_data_bytes() -> None:
    """Test function test_get_data_bytes."""
    arr = np.array([1.0], dtype=np.float32)
    assert _extract_arr_bytes(arr) == arr.tobytes()

    class TensorWithEval:
        """Test class TensorWithEval."""

        def eval(self) -> EagerValue:
            """Test function eval."""
            return arr

    assert cast("EagerCallableProtocol", _get_data_bytes)(cast("EagerCallableProtocol", TensorWithEval)()) == cast("EagerCallableProtocol", arr.tobytes)()

    class TensorWithNumpy:
        """Test class TensorWithNumpy."""

        def numpy(self) -> EagerValue:
            """Test function numpy."""
            return arr

    assert cast("EagerCallableProtocol", _get_data_bytes)(cast("EagerCallableProtocol", TensorWithNumpy)()) == cast("EagerCallableProtocol", arr.tobytes)()

    class TensorWithDataNumpy:
        """Test class TensorWithDataNumpy."""

        class Data:
            """Test class Data."""

            def numpy(self) -> EagerValue:
                """Test function numpy."""
                return arr

        data = Data()

    assert cast("EagerCallableProtocol", _get_data_bytes)(cast("EagerCallableProtocol", TensorWithDataNumpy)()) == cast("EagerCallableProtocol", arr.tobytes)()

    class BadArr:
        """Test class BadArr."""

        pass

    assert cast("EagerCallableProtocol", _get_data_bytes)(cast("EagerCallableProtocol", BadArr)()) == b""
    assert cast("EagerCallableProtocol", _extract_arr_bytes)(cast("EagerCallableProtocol", BadArr)()) == b""

    class ArrWithDataTobytes:
        """Test class ArrWithDataTobytes."""

        class Data:
            """Test class Data."""

            def tobytes(self) -> EagerValue:
                """Test function tobytes."""
                return b"test"

        data = Data()

    assert cast("EagerCallableProtocol", _extract_arr_bytes)(cast("EagerCallableProtocol", ArrWithDataTobytes)()) == b"test"


def test_tensor_to_npy_bytes() -> None:
    """Test function test_tensor_to_npy_bytes."""
    arr = np.array([1.0], dtype=np.float32)
    res = _tensor_to_npy_bytes(arr)
    assert b"NUMPY" in res

    # 2D shape
    arr2d = np.array([[1.0, 2.0]], dtype=np.float32)
    res2d = _tensor_to_npy_bytes(arr2d)
    assert b"NUMPY" in res2d


def test_get_npz_bytes() -> None:
    """Test function test_get_npz_bytes."""
    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value = mock.Mock()
        mock_backend.return_value.get_npz_bytes = mock.Mock(return_value=b"mocked_bytes")
        assert get_npz_bytes({"a": np.array([1])}) == b"mocked_bytes"

        # Test fallback when get_npz_bytes is not available
        del mock_backend.return_value.get_npz_bytes
        res = get_npz_bytes({"a.npy": np.array([1])})
        assert len(res) > 0


def test_parse_npz() -> None:
    """Test function test_parse_npz."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test.npz")
        np.savez(filepath, a=np.array([1]))
        with open(filepath, "rb") as f:
            res = cast("EagerCallableProtocol", parse_npz)(f)
            assert "a" in res

    # test fallback exception
    res = cast("EagerCallableProtocol", parse_npz)(None)
    assert res == {}


def test_load_npz() -> None:
    """Test function test_load_npz."""
    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value = mock.Mock()
        mock_backend.return_value.load_npz = mock.Mock(return_value=["mocked_weights"])
        assert cast("EagerCallableProtocol", load_npz)(None) == ["mocked_weights"]

        mock_backend.return_value.load_npz.side_effect = Exception("failed")

        with mock.patch("ml_switcheroo_compiler.serialization.utils.parse_npz") as mock_parse:
            mock_parse.return_value = {"a": "parsed_weight"}
            assert cast("EagerCallableProtocol", load_npz)(None) == ["parsed_weight"]


def test_ir_graph_utils() -> None:
    """Test function test_ir_graph_utils."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "graph.json")
        from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

        graph = IRGraph()
        graph.nodes["test"] = IRNode(id="test", op_type="test", inputs=[], shape_metadata=())
        save_ir_graph(graph, filepath)

        loaded = load_ir_graph(filepath)
        assert "test" in loaded.nodes


def test_load_npz_no_hasattr() -> None:
    """Test function test_load_npz_no_hasattr."""
    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value = mock.Mock(spec=[])  # No load_npz attribute
        with mock.patch("ml_switcheroo_compiler.serialization.utils.parse_npz") as mock_parse:
            mock_parse.return_value = {"a": "parsed_no_hasattr"}
            assert cast("EagerCallableProtocol", load_npz)(None) == ["parsed_no_hasattr"]
