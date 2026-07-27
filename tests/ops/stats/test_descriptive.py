# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.stats.descriptive import ApplyOverAxes, Bincount, ConfusionMatrix, Corrcoef, Correlate, Cov, TrapezoidalIntegral, confusion_matrix, descriptive, distributions, moments, trapezoidal_integral


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_classes_infer_shape():
    assert ApplyOverAxes().infer_shape() is None
    assert Bincount().infer_shape() == (None,)
    assert Corrcoef().infer_shape() == (None, None)
    assert Correlate().infer_shape() == (None,)
    assert Cov().infer_shape() == (None, None)
    t = MockTensor((2, 3))
    assert TrapezoidalIntegral().infer_shape((2, 3), axis=0) == (3,)
    assert TrapezoidalIntegral().infer_shape((2, 3), axis=-1) == (2,)
    assert ConfusionMatrix().infer_shape(num_classes=5) == (5, 5)


def test_funcs(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mock_get_op = mocker.patch("ml_switcheroo_compiler.ops.stats.descriptive.get_op")
    mock_op = mocker.MagicMock()
    mock_get_op.return_value = mocker.MagicMock(return_value=mock_op)
    assert trapezoidal_integral(t) == mock_op()
    assert confusion_matrix(t, t, 5) == mock_op()
    (m1, v1) = moments(t)
    assert m1 == mock_op()
    assert v1 == mock_op()
    assert descriptive(t) == mock_op()
    assert distributions(t) == mock_op()
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.stats.descriptive.get_active_backend", create=True)
    if mock_backend:
        mock_backend.return_value.execute_op.return_value = "res"
    pass
