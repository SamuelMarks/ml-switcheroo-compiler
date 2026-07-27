# ruff: noqa: E501
from ml_switcheroo_compiler.ops.tensor_array import TensorArrayRead, TensorArrayStack, TensorArrayWrite


class MockHandle:
    def __init__(self, elem_shape=(), size=None):
        self.element_shape = elem_shape
        self.size = size


def test_tensor_array_infer_shape():
    h1 = MockHandle((2, 3), size=4)
    h2 = MockHandle((2, 3), size=None)
    assert TensorArrayRead().infer_shape(h1, 0) == (2, 3)
    assert TensorArrayWrite().infer_shape(h1, 0, None) == ()
    assert TensorArrayStack().infer_shape(h1) == (4, 2, 3)
    assert TensorArrayStack().infer_shape(h2) == (None, 2, 3)
