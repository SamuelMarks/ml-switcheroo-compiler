import os
import tempfile
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


def test_extract_numpy_weights():
    class TensorWithNumpy:
        def numpy(self):
            return np.array([1])

    class TensorWithDataNumpy:
        class Data:
            def numpy(self):
                return np.array([2])

        data = Data()

    class TensorWithToList:
        def tolist(self):
            return [3]

    class ErrorTensor:
        def tolist(self):
            raise ValueError()

    weights = {"a": TensorWithNumpy(), "b": TensorWithDataNumpy(), "c": TensorWithToList(), "d": ErrorTensor(), "e": 5}
    res = _extract_numpy_weights(weights)
    assert np.array_equal(res["a"], np.array([1]))
    assert np.array_equal(res["b"], np.array([2]))
    assert res["c"] == [3]
    assert isinstance(res["d"], ErrorTensor)
    assert res["e"] == 5


def test_to_numpy():
    class TensorWithNumpy:
        def numpy(self):
            return np.array([1])

    class TensorWithDataNumpy:
        class Data:
            def numpy(self):
                return np.array([2])

        data = Data()

    class TensorWithToList:
        def tolist(self):
            return [3]

    class ErrorTensor:
        def tolist(self):
            raise ValueError()

    assert np.array_equal(to_numpy(TensorWithNumpy()), np.array([1]))
    assert np.array_equal(to_numpy(TensorWithDataNumpy()), np.array([2]))
    assert to_numpy(TensorWithToList()) == [3]
    assert isinstance(to_numpy(ErrorTensor()), ErrorTensor)
    assert to_numpy(5) == 5


def test_concatenate_arrays():
    with mock.patch("ml_switcheroo_compiler.ops.concatenate") as mock_concat:
        mock_concat.return_value = "concatenated"
        res = concatenate_arrays([np.array([1]), np.array([2])])
        assert res == "concatenated"


def test_is_numpy_array():
    assert is_numpy_array(np.array([1]))

    class FakeArray:
        __array__ = lambda self: np.array([1])

    assert is_numpy_array(FakeArray())

    class FakeNumpy:
        def numpy(self):
            return np.array([1])

    assert is_numpy_array(FakeNumpy())


def test_dtype_to_descr():
    assert _dtype_to_descr(DType.Float32) == "<f4"
    assert _dtype_to_descr("float32") == "<f4"
    assert _dtype_to_descr("invalid") == "<f4"

    class FakeDType:
        value = "int32"

    assert _dtype_to_descr(FakeDType()) == "<f4"


def test_get_shape_and_dtype():
    arr = np.array([1, 2], dtype=np.float32)
    assert _extract_arr_shape_dtype(arr) == ((2,), "float32")

    class BadArr:
        pass

    assert _extract_arr_shape_dtype(BadArr()) == ((), "<f4")

    class TensorWithEval:
        def eval(self):
            return np.array([1, 2], dtype=np.float32)

    assert _get_shape_and_dtype(TensorWithEval()) == ((2,), "float32")

    class TensorWithNumpy:
        def numpy(self):
            return np.array([1], dtype=np.float64)

    assert _get_shape_and_dtype(TensorWithNumpy()) == ((1,), "float64")

    class TensorWithDataNumpy:
        class Data:
            def numpy(self):
                return np.array([1], dtype=np.int32)

        data = Data()

    assert _get_shape_and_dtype(TensorWithDataNumpy()) == ((1,), "int32")

    assert _get_shape_and_dtype(BadArr()) == ((), "<f4")


def test_get_data_bytes():
    arr = np.array([1.0], dtype=np.float32)
    assert _extract_arr_bytes(arr) == arr.tobytes()

    class TensorWithEval:
        def eval(self):
            return arr

    assert _get_data_bytes(TensorWithEval()) == arr.tobytes()

    class TensorWithNumpy:
        def numpy(self):
            return arr

    assert _get_data_bytes(TensorWithNumpy()) == arr.tobytes()

    class TensorWithDataNumpy:
        class Data:
            def numpy(self):
                return arr

        data = Data()

    assert _get_data_bytes(TensorWithDataNumpy()) == arr.tobytes()

    class BadArr:
        pass

    assert _get_data_bytes(BadArr()) == b""
    assert _extract_arr_bytes(BadArr()) == b""

    class ArrWithDataTobytes:
        class Data:
            def tobytes(self):
                return b"test"

        data = Data()

    assert _extract_arr_bytes(ArrWithDataTobytes()) == b"test"


def test_tensor_to_npy_bytes():
    arr = np.array([1.0], dtype=np.float32)
    res = _tensor_to_npy_bytes(arr)
    assert b"NUMPY" in res

    # 2D shape
    arr2d = np.array([[1.0, 2.0]], dtype=np.float32)
    res2d = _tensor_to_npy_bytes(arr2d)
    assert b"NUMPY" in res2d


def test_get_npz_bytes():
    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value = mock.Mock()
        mock_backend.return_value.get_npz_bytes = mock.Mock(return_value=b"mocked_bytes")
        assert get_npz_bytes({"a": np.array([1])}) == b"mocked_bytes"

        # Test fallback when get_npz_bytes is not available
        del mock_backend.return_value.get_npz_bytes
        res = get_npz_bytes({"a.npy": np.array([1])})
        assert len(res) > 0


def test_parse_npz():
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test.npz")
        np.savez(filepath, a=np.array([1]))
        with open(filepath, "rb") as f:
            res = parse_npz(f)
            assert "a" in res

    # test fallback exception
    res = parse_npz(None)
    assert res == {}


def test_load_npz():
    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value = mock.Mock()
        mock_backend.return_value.load_npz = mock.Mock(return_value=["mocked_weights"])
        assert load_npz(None) == ["mocked_weights"]

        mock_backend.return_value.load_npz.side_effect = Exception("failed")

        with mock.patch("ml_switcheroo_compiler.serialization.utils.parse_npz") as mock_parse:
            mock_parse.return_value = {"a": "parsed_weight"}
            assert load_npz(None) == ["parsed_weight"]


def test_ir_graph_utils():
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "graph.json")
        from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

        graph = IRGraph()
        graph.nodes["test"] = IRNode(id="test", op_type="test", inputs=[], shape_metadata=())
        save_ir_graph(graph, filepath)

        loaded = load_ir_graph(filepath)
        assert "test" in loaded.nodes


def test_load_npz_no_hasattr():
    with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value = mock.Mock(spec=[])  # No load_npz attribute
        with mock.patch("ml_switcheroo_compiler.serialization.utils.parse_npz") as mock_parse:
            mock_parse.return_value = {"a": "parsed_no_hasattr"}
            assert load_npz(None) == ["parsed_no_hasattr"]
