# ruff: noqa: E501
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.type_inference import resolve_dtype, resolve_output_dtype_and_device


class MockTensor:
    def __init__(self, dtype):
        self.dtype = dtype


class MockData:
    def __init__(self, dtype_str):
        self.dtype = dtype_str


def test_resolve_dtype():
    assert resolve_dtype(MockData("dtype('int32')"), None) == DType.Int32
    assert resolve_dtype(MockData("dtype_something"), None) == DType.Float32
    assert resolve_dtype(MockData("dtype=float32"), None) == DType.Float32
    assert resolve_dtype(MockData("invalid"), None) == DType.Float32
    assert resolve_dtype(None, MockTensor(DType.Int64)) == DType.Int64
    assert resolve_dtype(None, None) == DType.Float32


def test_resolve_output_dtype_and_device():

    class MockTensorWithDevice:
        dtype = DType.Int32
        device = "cpu"

    t = MockTensorWithDevice()
    assert resolve_output_dtype_and_device(t, {"dtype": DType.Float64}) == (DType.Float64, "cpu")
    assert resolve_output_dtype_and_device(t, {}) == (DType.Int32, "cpu")
    assert resolve_output_dtype_and_device(None, {}) == (DType.Float32, None)
