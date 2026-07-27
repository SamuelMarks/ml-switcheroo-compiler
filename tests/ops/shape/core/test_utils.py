# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_emit_shape_node(mocker):
    from ml_switcheroo_compiler.core.dtype import DType

    config.eager_mode = False
    mock_add_node = mocker.patch("ml_switcheroo_compiler.ops.shape.utils.global_tracing_state.add_node")
    mocker.patch("ml_switcheroo_compiler.ops.shape.utils.global_tracing_state.is_tracing", True)
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    _emit_shape_node("Op", [t], {}, (2, 3), DType.Float32)

    class FakeDtype1:
        def __init__(self):
            self.value = "int32"

    _emit_shape_node("Op", [t], {}, (2, 3), FakeDtype1())

    class FakeDtype2:
        def __init__(self):
            self.name = "int64"

        def __str__(self):
            return "int64"

    _emit_shape_node("Op", [t], {}, (2, 3), FakeDtype2())

    class FakeDtype3:
        pass

    _emit_shape_node("Op", [t], {}, (2, 3), FakeDtype3())

    class FakeNumpyDtype:
        def __init__(self):
            self.name = "float64"

    FakeNumpyDtype.__name__ = "dtype"

    class dtype:
        def __init__(self):
            self.name = "float64"

    _emit_shape_node("Op", [t], {}, (2, 3), dtype())


def test_compute_reduction_shape():
    from ml_switcheroo_compiler.ops.shape.utils import compute_reduction_shape

    assert compute_reduction_shape((2, 3, 4), (1,), True) == (2, 1, 4)
    assert compute_reduction_shape((2, 3, 4), (1,), False) == (2, 4)
