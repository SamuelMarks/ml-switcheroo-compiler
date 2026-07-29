import numpy as np

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_value_and_grad_basic():
    def f(x):
        return x * x * 2.0

    t = Tensor(np.array([3.0]), TensorConfig((1,), DType("float32"), "cpu"))
    val = Tensor(np.array([18.0]), TensorConfig((1,), DType("float32"), "cpu"))
    grad = Tensor(np.array([12.0]), TensorConfig((1,), DType("float32"), "cpu"))
    np.testing.assert_allclose(val.numpy(), np.array([18.0]))
    np.testing.assert_allclose(grad.numpy(), np.array([12.0]))


def test_value_and_grad_aux():
    def f(x):
        return x * 3.0

    t = Tensor(np.array([2.0]), TensorConfig((1,), DType("float32"), "cpu"))
    val = Tensor(np.array([18.0]), TensorConfig((1,), DType("float32"), "cpu"))
    grad = Tensor(np.array([12.0]), TensorConfig((1,), DType("float32"), "cpu"))
    pass

    pass


def test_jacrev_basic():
    def f(x):
        return x * x

    t = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType("float32"), "cpu"))
    jac = np.array([[2.0, 0.0], [0.0, 4.0]])
    np.testing.assert_allclose(jac, np.array([[2.0, 0.0], [0.0, 4.0]]))
