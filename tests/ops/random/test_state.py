import pytest

# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random.state import _dispatch_random, _get_numpy_rng, bits, clone, fold_in, key, key_data, key_impl, rng_bit_generator, rng_uniform, split, wrap_key_data


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = DType.Float32
        self.device = "cpu"
        self.data = type("M", (), {"id": "1"})()


@pytest.mark.skip(reason="PRNGKey removed")
def test_prngkey(mocker):
    config.eager_mode = False
    mock_add_node = mocker.patch("ml_switcheroo_compiler.random.state.global_tracing_state.add_node")
    mocker.patch("ml_switcheroo_compiler.random.state.global_tracing_state.is_tracing", True)
    assert (42).config.shape == (2,)
    mocker.patch("ml_switcheroo_compiler.random.state.global_tracing_state.is_tracing", False)
    assert (42).config.shape == (2,)
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.random.state.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2,))
    pass
    assert (42).config.shape == (2,)
    mocker.patch("ml_switcheroo_compiler.random.state.global_tracing_state.is_tracing", False)
    assert (42).config.shape == (2,)


def test_split_fold_in(mocker):
    t = Tensor(MockTensor((2,)).data, TensorConfig((2,), DType.UInt32, "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.random.state._emit_random_node", return_value="node")
    assert split(t, 2) == "node"
    assert fold_in(t, 42) == "node"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.random.state.get_active_backend").return_value
    mock_backend.execute_op.side_effect = ["res1", "res2"]
    assert split(t, 2).config.shape == (2, 2)
    assert fold_in(t, 42).config.shape == (2,)


def test_dispatch_random(mocker):
    config.eager_mode = False
    mock_get_op = mocker.patch("ml_switcheroo_compiler.random.state.get_op")
    mock_op = mocker.MagicMock()
    mock_get_op.return_value = mocker.MagicMock(return_value=mock_op)
    assert _dispatch_random("my_func") == mock_op()
    mock_get_op.return_value = None
    mocker.patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="shape_node")
    assert _dispatch_random("my_func2") == "shape_node"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.random.state.get_active_backend").return_value
    mock_backend.execute_op.return_value = "res"
    assert _dispatch_random("my_func3") == "res"


def test_random_state_wrappers(mocker):
    mocker.patch("ml_switcheroo_compiler.random.state._dispatch_random", return_value="dispatch")
    assert key() == "dispatch"
    assert key_data() == "dispatch"
    assert key_impl() == "dispatch"
    assert wrap_key_data() == "dispatch"
    assert clone() == "dispatch"
    assert bits() == "dispatch"
    assert rng_bit_generator("key", ()) == "dispatch"
    assert rng_uniform("a", "b", ()) == "dispatch"


def test_get_numpy_rng():
    assert _get_numpy_rng() is not None


def test_emit_random_node_extra(mocker):
    from ml_switcheroo_compiler.random.state import _emit_random_node

    t = Tensor(MockTensor((2,)).data, TensorConfig((2,), DType.UInt32, "cpu"))
    config.eager_mode = False
    mock_add_node = mocker.patch("ml_switcheroo_compiler.random.state.global_tracing_state.add_node")
    mocker.patch("ml_switcheroo_compiler.random.state.global_tracing_state.is_tracing", True)
    assert _emit_random_node("TestOp", [t], (2,), DType.Float32, attributes=None).config.shape == (2,)
    assert _emit_random_node("TestOp", [t], (2,), DType.Float32, attributes={"shape": (2,), "dtype": DType.Float32}).config.shape == (2,)
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.random.state.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2,))
    pass
    assert _emit_random_node("TestOp", [t], (2,), DType.Float32, attributes=None).config.shape == (2,)
    assert _emit_random_node("TestOp", [t], (2,), DType.Float32, attributes={"shape": (2,), "dtype": DType.Float32}).config.shape == (2,)


def test_emit_random_node_eager(mocker):
    from ml_switcheroo_compiler.random.state import _emit_random_node

    t = Tensor(MockTensor((2,)).data, TensorConfig((2,), DType.UInt32, "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.random.state.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2,))
    assert _emit_random_node("TestOp", [t], (2,), DType.Float32, attributes=None).config.shape == (2,)


def test_emit_random_node_eager_attrs(mocker):
    from ml_switcheroo_compiler.random.state import _emit_random_node

    t = Tensor(MockTensor((2,)).data, TensorConfig((2,), DType.UInt32, "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.random.state.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2,))
    assert _emit_random_node("TestOp", [t], (2,), DType.Float32, attributes={"shape": (2,), "dtype": DType.Float32}).config.shape == (2,)
