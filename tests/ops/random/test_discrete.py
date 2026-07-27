# ruff: noqa: E501
from ml_switcheroo_compiler.random.distributions_discrete import binomial, categorical, geometric, multinomial, poisson, rademacher


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape


def test_distributions_discrete_extra(mocker):
    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    mock_dispatch = mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._dispatch_random", return_value="dispatch")
    assert binomial("key", "n", "p") == "node"
    assert geometric("args") == "dispatch"
    assert poisson("key", "lam") == "node"
    assert rademacher("args") == "dispatch"
    assert multinomial("key", 10, "pvals") == "node"
    assert categorical("key", [1.0, 2.0]) == "node"


def test_distributions_discrete_more(mocker):
    from ml_switcheroo_compiler.random.distributions_discrete import bernoulli, choice, permutation, randint

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert randint("key", (), 0, 1) == "node"
    assert bernoulli("key") == "node"

    class MockTensorWithDtype:
        def __init__(self, shape=()):
            self.shape = shape
            self.dtype = "float32"

    assert permutation("key", MockTensorWithDtype((2, 3))) == "node"
    assert choice("key", MockTensorWithDtype((2, 3))) == "node"


def test_distributions_discrete_none_shapes(mocker):
    from ml_switcheroo_compiler.random.distributions_discrete import binomial, categorical, multinomial, poisson

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert categorical("key", [1], shape=(1,)) == "node"
    assert binomial("key", 10, 0.5, shape=(1,)) == "node"
    assert poisson("key", 1.0, shape=(1,)) == "node"
    assert multinomial("key", 10, [0.5, 0.5], shape=(1,)) == "node"


def test_distributions_discrete_more2(mocker):
    from ml_switcheroo_compiler.random.distributions_discrete import bernoulli, categorical

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert bernoulli("key", shape=(1,)) == "node"
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class M:
        def __init__(self):
            self.data = [1, 2]
            self.shape = (2, 3)
            self.dtype = "float32"
            self.device = "cpu"

    t = Tensor(M().data, TensorConfig((2, 3), "float32", "cpu"))
    assert categorical("key", t) == "node"


def test_distributions_discrete_more3(mocker):
    from ml_switcheroo_compiler.random.distributions_discrete import choice

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class M:
        def __init__(self):
            self.data = [1, 2]
            self.shape = (2, 3)
            self.dtype = "float32"
            self.device = "cpu"

    t = Tensor(M().data, TensorConfig((2, 3), "float32", "cpu"))
    assert choice("key", t, p=t) == "node"
