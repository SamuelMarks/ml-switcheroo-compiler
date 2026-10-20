"""Tests for test_discrete."""

from __future__ import annotations

import typing

# ruff: noqa: E501
from ml_switcheroo_compiler.random.distributions_discrete import binomial, categorical, geometric, multinomial, poisson, rademacher

if typing.TYPE_CHECKING:
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
    assert typing.cast("EagerCallableProtocol", binomial)("key", "n", "p") == "node"
    assert typing.cast("EagerTensorProtocol", geometric("args")) == typing.cast("EagerTensorProtocol", "dispatch")
    assert typing.cast("EagerCallableProtocol", poisson)("key", "lam") == "node"
    assert typing.cast("EagerTensorProtocol", rademacher("args")) == typing.cast("EagerTensorProtocol", "dispatch")
    assert typing.cast("EagerCallableProtocol", multinomial)("key", 10, "pvals") == "node"
    assert typing.cast("EagerTensorProtocol", categorical("key", [1.0, 2.0])) == typing.cast("EagerTensorProtocol", "node")


def test_distributions_discrete_more(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_more.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import bernoulli, choice, permutation, randint

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert typing.cast("EagerTensorProtocol", randint("key", (), 0, 1)) == typing.cast("EagerTensorProtocol", "node")
    assert typing.cast("EagerTensorProtocol", bernoulli("key")) == typing.cast("EagerTensorProtocol", "node")

    class MockTensorWithDtype:
        """Test class MockTensorWithDtype."""

        def __init__(self, shape: EagerValue = ()) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape
            self.dtype = "float32"

    assert typing.cast("EagerCallableProtocol", permutation)("key", typing.cast("EagerCallableProtocol", MockTensorWithDtype)((2, 3))) == "node"
    assert typing.cast("EagerCallableProtocol", choice)("key", typing.cast("EagerCallableProtocol", MockTensorWithDtype)((2, 3))) == "node"


def test_distributions_discrete_none_shapes(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_none_shapes.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import binomial, categorical, multinomial, poisson

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert typing.cast("EagerTensorProtocol", categorical("key", [1], shape=(1,))) == typing.cast("EagerTensorProtocol", "node")
    assert typing.cast("EagerTensorProtocol", binomial("key", 10, 0.5, shape=(1,))) == typing.cast("EagerTensorProtocol", "node")
    assert typing.cast("EagerTensorProtocol", poisson("key", 1.0, shape=(1,))) == typing.cast("EagerTensorProtocol", "node")
    assert typing.cast("EagerCallableProtocol", multinomial)("key", 10, [0.5, 0.5], shape=(1,)) == "node"


def test_distributions_discrete_more2(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_discrete_more2.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import bernoulli, categorical

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert typing.cast("EagerTensorProtocol", bernoulli("key", shape=(1,))) == typing.cast("EagerTensorProtocol", "node")
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
    assert typing.cast("EagerTensorProtocol", categorical("key", t)) == typing.cast("EagerTensorProtocol", "node")


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
    assert typing.cast("EagerTensorProtocol", choice("key", t, p=t)) == typing.cast("EagerTensorProtocol", "node")
