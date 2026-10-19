"""Tests for test_discrete."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.random.distributions_discrete import binomial, categorical, geometric, multinomial, poisson, rademacher

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


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
    assert cast("EagerCallableProtocol", binomial)("key", "n", "p") == "node"
    assert cast("EagerTensorProtocol", geometric("args")) == cast("EagerTensorProtocol", "dispatch")
    assert cast("EagerCallableProtocol", poisson)("key", "lam") == "node"
    assert cast("EagerTensorProtocol", rademacher("args")) == cast("EagerTensorProtocol", "dispatch")
    assert cast("EagerCallableProtocol", multinomial)("key", 10, "pvals") == "node"
    assert cast("EagerTensorProtocol", categorical("key", [1.0, 2.0])) == cast("EagerTensorProtocol", "node")


def test_distributions_discrete_more(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_more.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import bernoulli, choice, permutation, randint

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert cast("EagerTensorProtocol", randint("key", (), 0, 1)) == cast("EagerTensorProtocol", "node")
    assert cast("EagerTensorProtocol", bernoulli("key")) == cast("EagerTensorProtocol", "node")

    class MockTensorWithDtype:
        """Test class MockTensorWithDtype."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape
            self.dtype = "float32"

    assert cast("EagerCallableProtocol", permutation)("key", cast("EagerCallableProtocol", MockTensorWithDtype)((2, 3))) == "node"
    assert cast("EagerCallableProtocol", choice)("key", cast("EagerCallableProtocol", MockTensorWithDtype)((2, 3))) == "node"


def test_distributions_discrete_none_shapes(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_none_shapes.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import binomial, categorical, multinomial, poisson

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert cast("EagerTensorProtocol", categorical("key", [1], shape=(1,))) == cast("EagerTensorProtocol", "node")
    assert cast("EagerTensorProtocol", binomial("key", 10, 0.5, shape=(1,))) == cast("EagerTensorProtocol", "node")
    assert cast("EagerTensorProtocol", poisson("key", 1.0, shape=(1,))) == cast("EagerTensorProtocol", "node")
    assert cast("EagerCallableProtocol", multinomial)("key", 10, [0.5, 0.5], shape=(1,)) == "node"


def test_distributions_discrete_more2(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_more2.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import bernoulli, categorical

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert cast("EagerTensorProtocol", bernoulli("key", shape=(1,))) == cast("EagerTensorProtocol", "node")
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
    assert cast("EagerTensorProtocol", categorical("key", t)) == cast("EagerTensorProtocol", "node")


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
    assert cast("EagerTensorProtocol", choice("key", t, p=t)) == cast("EagerTensorProtocol", "node")
