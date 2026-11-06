"""Tests for test_discrete."""

from __future__ import annotations

import typing

# ruff: noqa: E501
from ml_switcheroo_compiler.random.distributions_discrete import binomial, categorical, geometric, multinomial, poisson, rademacher
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape


def test_distributions_discrete_extra(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_extra.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._dispatch_random", return_value="dispatch")
    assert ensure_callable(binomial)("key", "n", "p") == "node"
    assert ensure_eager_tensor(geometric("args")) == ensure_eager_tensor("dispatch")
    assert ensure_callable(poisson)("key", "lam") == "node"
    assert ensure_eager_tensor(rademacher("args")) == ensure_eager_tensor("dispatch")
    assert ensure_callable(multinomial)("key", 10, "pvals") == "node"
    assert ensure_eager_tensor(categorical("key", [1.0, 2.0])) == ensure_eager_tensor("node")


def test_distributions_discrete_more(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_more.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import bernoulli, choice, permutation, randint

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert ensure_eager_tensor(randint("key", (), 0, 1)) == ensure_eager_tensor("node")
    assert ensure_eager_tensor(bernoulli("key")) == ensure_eager_tensor("node")

    class MockTensorWithDtype:
        """Test class MockTensorWithDtype."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape
            self.dtype = "float32"

    assert ensure_callable(permutation)("key", ensure_callable(MockTensorWithDtype)((2, 3))) == "node"
    assert ensure_callable(choice)("key", ensure_callable(MockTensorWithDtype)((2, 3))) == "node"


def test_distributions_discrete_none_shapes(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_none_shapes.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import binomial, categorical, multinomial, poisson

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert ensure_eager_tensor(categorical("key", [1], shape=(1,))) == ensure_eager_tensor("node")
    assert ensure_eager_tensor(binomial("key", 10, 0.5, shape=(1,))) == ensure_eager_tensor("node")
    assert ensure_eager_tensor(poisson("key", 1.0, shape=(1,))) == ensure_eager_tensor("node")
    assert ensure_callable(multinomial)("key", 10, [0.5, 0.5], shape=(1,)) == "node"


def test_distributions_discrete_more2(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_more2.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import bernoulli, categorical

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert ensure_eager_tensor(bernoulli("key", shape=(1,))) == ensure_eager_tensor("node")
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class M:
        """Test class M."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.data = [1, 2]
            self.shape = (2, 3)
            self.dtype = "float32"
            self.device = "cpu"

    t = Tensor(M().data, TensorConfig((2, 3), "float32", "cpu"))
    assert ensure_eager_tensor(categorical("key", t)) == ensure_eager_tensor("node")


def test_distributions_discrete_more3(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_more3.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import choice

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class M:
        """Test class M."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.data = [1, 2]
            self.shape = (2, 3)
            self.dtype = "float32"
            self.device = "cpu"

    t = Tensor(M().data, TensorConfig((2, 3), "float32", "cpu"))
    assert ensure_eager_tensor(choice("key", t, p=t)) == ensure_eager_tensor("node")
