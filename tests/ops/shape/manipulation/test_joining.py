from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.joining import append, column_stack, concatenate, dstack, hstack, stack, vstack


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_concatenate(mocker):
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((4, 3)).data, TensorConfig((4, 3), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", return_value="concat")
    assert concatenate([t1, t2], 0) == "concat"

    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.joining.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((6, 3))
    assert concatenate([t1, t2], 0).config.shape == (6, 3)


def test_stack(mocker):
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", return_value="stack")
    assert stack([t1, t2], 0) == "stack"

    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.joining.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 2, 3))
    assert stack([t1, t2], 0).config.shape == (2, 2, 3)


def test_vstack(mocker):
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((4, 3)).data, TensorConfig((4, 3), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", return_value="vstack")
    assert vstack([t1, t2]) == "vstack"

    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.joining.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((6, 3))
    assert vstack([t1, t2]).config.shape == (6, 3)


def test_hstack(mocker):
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((2, 4)).data, TensorConfig((2, 4), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", return_value="hstack")
    assert hstack([t1, t2]) == "hstack"

    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.joining.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 7))
    assert hstack([t1, t2]).config.shape == (2, 7)


def test_dstack(mocker):
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", return_value="dstack")
    assert dstack([t1, t2]) == "dstack"

    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.joining.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 3, 2))
    assert dstack([t1, t2]).config.shape == (2, 3, 2)


def test_append(mocker):
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((4, 3)).data, TensorConfig((4, 3), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining.concatenate", return_value="concat")
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining.reshape", side_effect=lambda x, shape: x)
    assert append(t1, t2, axis=0) == "concat"
    assert append(t1, t2, axis=None) == "concat"

    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.joining.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((6, 3))
    assert append(t1, t2, axis=0).config.shape == (6, 3)


def test_column_stack(mocker):
    t1 = Tensor(MockTensor((3,)).data, TensorConfig((3,), "float32", "cpu"))
    t2 = Tensor(MockTensor((3, 2)).data, TensorConfig((3, 2), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining.hstack", return_value="hstack")
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining.reshape", side_effect=lambda x, shape: x)
    assert column_stack([t1, t2]) == "hstack"


def test_stack_dtype_branches(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", return_value="stack")
    config.eager_mode = False

    class FakeDTypeName:
        name = "int32"

    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), FakeDTypeName(), "cpu"))
    stack([t1, t1], 0)

    class FakeDTypeDunderName:
        __name__ = "int32"

    t2 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), FakeDTypeDunderName(), "cpu"))
    stack([t2, t2], 0)

    class FakeDTypeValue:
        value = "int32"

    t3 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), FakeDTypeValue(), "cpu"))
    stack([t3, t3], 0)

    class FakeDTypeInvalid:
        pass

    t4 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), FakeDTypeInvalid(), "cpu"))
    stack([t4, t4], 0)
