"""Module test_activations.py."""

from ml_switcheroo_compiler.core.dtype import DType

"""Test nn activations coverage."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.activations import isotonic_regression
from ml_switcheroo_compiler.tracing import global_tracing_state


def test_isotonic_regression_tracing():
    """Test isotonic_regression in tracing mode."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        y = Tensor(None, TensorConfig((5,), "float32", "cpu"))
        out1, out2 = isotonic_regression(y)
        assert out1.shape == (5,)
        assert out2.shape == (5,)
        assert out2.dtype == DType.Int32
    finally:
        global_tracing_state.stop_tracing()
        config.eager_mode = True


def test_activations_dispatcher():
    """test_activations_dispatcher."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.nn.activations import hard_silu, hard_swish, mish, squareplus

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "mock_result"

        assert hard_silu(1, kw=2) == "mock_result"
        mock_dispatch.assert_called_with("HardSilu", 1, kw=2)

        assert hard_swish(1, kw=2) == "mock_result"
        mock_dispatch.assert_called_with("HardSwish", 1, kw=2)

        assert mish(1, kw=2) == "mock_result"
        mock_dispatch.assert_called_with("Mish", 1, kw=2)

        assert squareplus(1, kw=2) == "mock_result"
        mock_dispatch.assert_called_with("Squareplus", 1, kw=2)


def test_isotonic_regression_eager():
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.nn.activations import isotonic_regression

    config.eager_mode = True
    try:
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
            backend = MagicMock()
            backend.execute_op.return_value = "eager_iso"
            mock_backend.return_value = backend

            res = isotonic_regression("y", sample_weights="w")
            assert res == "eager_iso"
    finally:
        config.eager_mode = False


from ml_switcheroo_compiler.ops.nn.activations import LogSoftmax, OneHot, Rrelu, Sigmoid, Softmax, rrelu, softmax


def test_activations_eager_mode(mocker):
    """Test function."""
    config.eager_mode = True

    class DummyBackend:
        def execute_op(self, op_name, *args, **kwargs):
            return op_name

    mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackend())

    assert softmax("x", axis=1) == "Softmax"

    config.eager_mode = False


def test_rrelu_non_eager_mode():
    """Test function."""
    config.eager_mode = False

    class DummyNode:
        id = "n"

    res = rrelu(DummyNode())
    assert res is not None


def test_activations_infer_shape_broadcast():
    """Test function."""

    class DummyTensor:
        def __init__(self, shape):
            self.shape = shape

    t1 = DummyTensor((2, 3))
    t2 = DummyTensor((2, 1))

    op = Softmax()
    res = op.infer_shape(t1, t2)
    assert res == (2, 3)

    op2 = LogSoftmax()
    res2 = op2.infer_shape(t1, t2)
    assert res2 == (2, 3)

    op3 = Sigmoid()
    res3 = op3.infer_shape(t1, t2)
    assert res3 == (2, 3)

    op4 = Rrelu()
    res4 = op4.infer_shape(t1, t2)
    assert res4 == (2, 3)


def test_onehot_infer_shape():
    """Test function."""
    op = OneHot()

    class DummyTensor:
        def __init__(self, shape):
            self.shape = shape

    t1 = DummyTensor((2, 3))

    res = op.infer_shape(t1, 5, axis=1)
    assert res == (2, 5, 3)


from ml_switcheroo_compiler.ops.nn.activations import HardSilu, HardSwish, Squareplus, log_softmax, one_hot, sigmoid


def test_activations_non_eager_mode():
    """Test function."""
    config.eager_mode = False

    class DummyNode:
        id = "n"

        class DummyShape:
            shape = (1,)

        shape_metadata = DummyShape()

    res = softmax(DummyNode())
    assert res is not None

    res = log_softmax(DummyNode())
    assert res is not None

    res = sigmoid(DummyNode())
    assert res is not None

    res = one_hot(DummyNode(), 5)
    assert res is not None

    res = rrelu(DummyNode())
    assert res is not None


def test_activations_infer_shape_missing():
    """Test function."""
    # Rrelu missing shapes
    assert Rrelu().infer_shape() == ()

    class DummyTensor:
        def __init__(self, shape):
            self.shape = shape

    t = DummyTensor((2, 2))

    assert HardSilu().infer_shape(t) == (2, 2)
    assert HardSilu().infer_shape() == ()

    assert HardSwish().infer_shape(t) == (2, 2)
    assert HardSwish().infer_shape() == ()

    assert Squareplus().infer_shape(t) == (2, 2)
    assert Squareplus().infer_shape() == ()
