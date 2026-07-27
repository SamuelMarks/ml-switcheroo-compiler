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


class MockTensor:
    def __init__(self, data=None):
        self._data = data

    def numpy(self):
        return self._data


class MockTensorWithData:
    def __init__(self, data):
        self.data = MockTensor(data)


class MockListTensor:
    def tolist(self):
        return [1, 2]


class MockBadTensor:
    def tolist(self):
        raise ValueError("bad")


def test_extract_numpy_weights():
    w = {"a": MockTensor("a_data"), "b": MockTensorWithData("b_data"), "c": MockListTensor(), "d": MockBadTensor()}
    res = _extract_numpy_weights(w)
    assert res["a"] == "a_data"
    assert res["b"] == "b_data"
    assert res["c"] == [1, 2]


def test_to_numpy():
    assert to_numpy(MockTensor("a_data")) == "a_data"
    assert to_numpy(MockTensorWithData("b_data")) == "b_data"
    assert to_numpy(MockListTensor()) == [1, 2]
    assert isinstance(to_numpy(MockBadTensor()), MockBadTensor)


def test_concatenate_arrays(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.concatenate", return_value="concat")
    assert concatenate_arrays([1, 2]) == "concat"


def test_is_numpy_array():
    import numpy as np

    assert is_numpy_array(np.array([1]))
    assert is_numpy_array(MockTensor())

    class Arr:
        def __array__(self):
            return np.array([1])

    assert is_numpy_array(Arr())
    assert not is_numpy_array(1)


def test_dtype_to_descr():
    assert _dtype_to_descr(DType.Float32) == "<f4"

    class FakeDtype:
        value = "unknown"

    assert _dtype_to_descr(FakeDtype()) == "<f4"
    assert _dtype_to_descr(DType.Int64) == "<i8"
    assert _dtype_to_descr("float32") == "<f4"
    assert _dtype_to_descr("unknown") == "<f4"


def test_extract_arr_shape_dtype():

    class Arr:
        shape = (2, 3)
        dtype = type("D", (), {"name": "float32"})()

    assert _extract_arr_shape_dtype(Arr()) == ((2, 3), "float32")

    class Arr2:
        pass

    assert _extract_arr_shape_dtype(Arr2()) == ((), "<f4")


def test_get_shape_and_dtype():

    class EvalTensor:
        def eval(self):

            class Arr:
                shape = (2,)
                dtype = type("D", (), {"name": "float32"})()

            return Arr()

    assert _get_shape_and_dtype(EvalTensor()) == ((2,), "float32")

    class NpyTensor:
        def numpy(self):

            class Arr:
                shape = (3,)
                dtype = type("D", (), {"name": "int32"})()

            return Arr()

    assert _get_shape_and_dtype(NpyTensor()) == ((3,), "int32")

    class DataNpyTensor:
        data = NpyTensor()

    assert _get_shape_and_dtype(DataNpyTensor()) == ((3,), "int32")

    class SimpleTensor:
        shape = (4,)
        dtype = "float32"

    assert _get_shape_and_dtype(SimpleTensor()) == ((4,), "float32")


def test_extract_arr_bytes():

    class T1:
        def tobytes(self):
            return b"t1"

    assert _extract_arr_bytes(T1()) == b"t1"

    class T2:
        data = T1()

    assert _extract_arr_bytes(T2()) == b"t1"
    assert _extract_arr_bytes(object()) == b""


def test_get_data_bytes():

    class EvalT:
        def eval(self):

            class Arr:
                def tobytes(self):
                    return b"eval"

            return Arr()

    assert _get_data_bytes(EvalT()) == b"eval"

    class NumpyT:
        def numpy(self):

            class Arr:
                def tobytes(self):
                    return b"npy"

            return Arr()

    assert _get_data_bytes(NumpyT()) == b"npy"

    class DataNpyT:
        data = NumpyT()

    assert _get_data_bytes(DataNpyT()) == b"npy"
    assert _get_data_bytes(object()) == b""


def test_tensor_to_npy_bytes():

    class MockArr:
        shape = (2,)
        dtype = type("D", (), {"name": "float32"})()

        def tobytes(self):
            return b"12345678"

    class MockT:
        def numpy(self):
            return MockArr()

    b = _tensor_to_npy_bytes(MockT())
    assert b.startswith(b"\x93NUMPY")


def test_get_npz_bytes(mocker):
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    mock_backend.get_npz_bytes.return_value = b"npz_bytes"
    assert get_npz_bytes({}) == b"npz_bytes"
    del mock_backend.get_npz_bytes

    class MockArr:
        shape = (2,)
        dtype = type("D", (), {"name": "float32"})()

        def tobytes(self):
            return b"12345678"

    class MockT:
        def numpy(self):
            return MockArr()

    b = get_npz_bytes({"a": MockT(), "b.npy": MockT()})
    assert len(b) > 0


def test_load_npz(mocker):
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    mock_backend.load_npz.return_value = [1, 2]
    assert load_npz("file") == [1, 2]
    mock_backend.load_npz.side_effect = NotImplementedError
    mocker.patch("ml_switcheroo_compiler.serialization.utils.parse_npz", return_value={"a": 1})
    assert load_npz("file") == [1]


def test_save_load_ir_graph(mocker):
    mocker.patch("ml_switcheroo_compiler.serialization.ir_format.graph_to_json", return_value="json_graph")
    mocker.patch("ml_switcheroo_compiler.serialization.ir_format.json_to_graph", return_value="graph_obj")
    import tempfile

    with tempfile.NamedTemporaryFile() as f:
        save_ir_graph("graph", f.name)
        assert load_ir_graph(f.name) == "graph_obj"


def test_load_npz_extra(mocker):
    assert parse_npz("file") == {}
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    del mock_backend.load_npz
    assert load_npz("file") == []


def test_dtype_to_descr_extra():
    assert _dtype_to_descr(DType.Float32) == "<f4"

    class FakeDtype:
        value = "unknown"

    assert _dtype_to_descr(FakeDtype()) == "<f4"
    assert _dtype_to_descr("int8") == "|i1"

    class FakeDtype2:
        value = "float32"

    assert _dtype_to_descr(FakeDtype2) == "<f4"
    assert _dtype_to_descr(DType.Float64) == "<f8"
    assert _dtype_to_descr("int32") == "<i4"
    assert _dtype_to_descr("something_else") == "<f4"


def test_dtype_fallback_no_value():
    from ml_switcheroo_compiler.serialization.utils import _dtype_to_descr

    assert _dtype_to_descr(None) == "<f4"
    assert _dtype_to_descr(123) == "<f4"
