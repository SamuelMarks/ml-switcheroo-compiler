"""Tests for test_discrete."""

from __future__ import annotations

import typing

# ruff: noqa: E501
from ml_switcheroo_compiler.random.distributions_discrete import binomial, categorical, geometric, multinomial, poisson, rademacher
from tests.testing_utils import as_callable, as_tensor

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
    assert as_callable(binomial)("key", "n", "p") == "node"
    assert as_tensor(geometric("args")) == as_tensor("dispatch")
    assert as_callable(poisson)("key", "lam") == "node"
    assert as_tensor(rademacher("args")) == as_tensor("dispatch")
    assert as_callable(multinomial)("key", 10, "pvals") == "node"
    assert as_tensor(categorical("key", [1.0, 2.0])) == as_tensor("node")


def test_distributions_discrete_more(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_more.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import bernoulli, choice, permutation, randint

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert as_tensor(randint("key", (), 0, 1)) == as_tensor("node")
    assert as_tensor(bernoulli("key")) == as_tensor("node")

    class MockTensorWithDtype:
        """Test class MockTensorWithDtype."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape
            self.dtype = "float32"

    assert as_callable(permutation)("key", as_callable(MockTensorWithDtype)((2, 3))) == "node"
    assert as_callable(choice)("key", as_callable(MockTensorWithDtype)((2, 3))) == "node"


def test_distributions_discrete_none_shapes(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_none_shapes.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import binomial, categorical, multinomial, poisson

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert as_tensor(categorical("key", [1], shape=(1,))) == as_tensor("node")
    assert as_tensor(binomial("key", 10, 0.5, shape=(1,))) == as_tensor("node")
    assert as_tensor(poisson("key", 1.0, shape=(1,))) == as_tensor("node")
    assert as_callable(multinomial)("key", 10, [0.5, 0.5], shape=(1,)) == "node"


def test_distributions_discrete_more2(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_more2.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import bernoulli, categorical

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert as_tensor(bernoulli("key", shape=(1,))) == as_tensor("node")
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
    assert as_tensor(categorical("key", t)) == as_tensor("node")


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
    assert as_tensor(choice("key", t, p=t)) == as_tensor("node")
