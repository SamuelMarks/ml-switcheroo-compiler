"""Extra tests for tensor coverage."""

from unittest import mock

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import ArrayAt, ArrayAtIndexer, Parameter, Tensor, TensorConfig, Variable


def test_array_at() -> None:
    """Test ArrayAt and ArrayAtIndexer."""
    t_mock = mock.Mock(spec=Tensor)
    indexer = ArrayAtIndexer(t_mock)

    arr_at = indexer[1]
    assert isinstance(arr_at, ArrayAt)
    assert arr_at.tensor == t_mock
    assert arr_at.indices == 1

    # Test methods return the tensor
    assert arr_at.add(5) == t_mock
    assert arr_at.multiply(5) == t_mock
    assert arr_at.set(5) == t_mock
    assert arr_at.maximum(5) == t_mock
    assert arr_at.minimum(5) == t_mock


def test_variable_assign_eager() -> None:
    """Test Variable.assign operations in eager mode."""
    t_cfg = TensorConfig((2, 2), "float32", "cpu", False, True)
    var = Variable("data1", t_cfg)
    val = Variable("data2", t_cfg)

    orig = config.eager_mode
    try:
        config.eager_mode = True
        with mock.patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
            mock_backend = mock.Mock()
            mock_get_backend.return_value = mock_backend
            mock_backend.execute_op.return_value = "new_data"

            res = var.assign(val)
            assert res is var
            assert var._data == "new_data"
            mock_backend.execute_op.assert_called_with("Assign", "data1", "data2")

            var._data = "data1"
            res = var.assign_add(val)
            assert res is var
            assert var._data == "new_data"
            mock_backend.execute_op.assert_called_with("AssignAdd", "data1", "data2")

            var._data = "data1"
            res = var.assign_sub(val)
            assert res is var
            assert var._data == "new_data"
            mock_backend.execute_op.assert_called_with("AssignSub", "data1", "data2")
    finally:
        config.eager_mode = orig


def test_variable_assign_non_eager() -> None:
    """Test Variable.assign operations in non-eager mode."""
    t_cfg = TensorConfig((2, 2), "float32", "cpu", False, True)
    var = Variable("data1", t_cfg)
    val = Variable("data2", t_cfg)

    orig = config.eager_mode
    try:
        config.eager_mode = False
        with mock.patch("ml_switcheroo_compiler.ops.registry.get_util") as mock_get_util:
            mock_emit = mock.Mock()
            mock_get_util.return_value = mock_emit

            res = var.assign(val)
            assert res is var
            mock_emit.assert_called_with("Assign", [var, val], {}, var.shape, var.dtype)

            res = var.assign_add(val)
            assert res is var
            mock_emit.assert_called_with("AssignAdd", [var, val], {}, var.shape, var.dtype)

            res = var.assign_sub(val)
            assert res is var
            mock_emit.assert_called_with("AssignSub", [var, val], {}, var.shape, var.dtype)
    finally:
        config.eager_mode = orig


def test_parameter_index() -> None:
    """Test Parameter.__index__."""
    t_cfg = TensorConfig((), "int32", "cpu", False, True)

    class MockData:
        def numpy(self):
            return 42

    param = Parameter(MockData(), t_cfg)
    # The __index__ calls self.numpy(), we need to mock numpy() for Parameter since it inherits from Tensor
    with mock.patch.object(param, "numpy", return_value=42):
        assert param.__index__() == 42


def test_parameter_init() -> None:
    """Test parameter initialization always sets trainable=True."""
    t_cfg = TensorConfig((), "int32", "cpu", False, False)
    param = Parameter("data", t_cfg)
    assert param.config.trainable is True


def test_tensor_eval_not_tracing():
    """Test eval when not tracing but has proxy data."""
    t_cfg = TensorConfig((2, 2), "float32", "cpu", False, True)

    class MockData:
        id = "mock_id"

    t = Tensor(MockData(), t_cfg)

    orig = config.eager_mode
    try:
        config.eager_mode = False
        import ml_switcheroo_compiler.tracing.state as state

        state.global_tracing_state.is_tracing = False

        # Test eval
        assert t.eval() is t
    finally:
        config.eager_mode = orig
