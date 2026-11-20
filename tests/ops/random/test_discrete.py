"""Tests for test_discrete."""

from __future__ import annotations

import typing

from pytest_mock import MockerFixture

# ruff: noqa: E501
from ml_switcheroo_compiler.random.distributions_discrete import (
    binomial,
    categorical,
    geometric,
    multinomial,
    poisson,
    rademacher,
)
from tests.testing_utils import MockTensorStub, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: BackendArg = ()) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape


def test_distributions_discrete_extra(mocker: MockerFixture) -> None:
    """Test function test_distributions_discrete_extra.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._dispatch_random", return_value="dispatch")
    assert require_callable(binomial)("key", "n", "p") == "node"
    assert geometric("args") == "dispatch"
    assert require_callable(poisson)("key", "lam") == "node"
    assert rademacher("args") == "dispatch"
    assert require_callable(multinomial)("key", 10, "pvals") == "node"
    assert categorical("key", [1.0, 2.0]) == "node"


def test_distributions_discrete_more(mocker: MockerFixture) -> None:
    """Test function test_distributions_discrete_more.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import bernoulli, choice, permutation, randint

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert randint("key", (), 0, 1) == "node"
    assert bernoulli("key") == "node"

    class MockTensorWithDtype(MockTensorStub):
        """Test class MockTensorWithDtype."""

        def __init__(self, shape: BackendArg = ()) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape
            self.dtype = "float32"

    assert permutation("key", MockTensorWithDtype((2, 3))) == "node"
    assert choice("key", MockTensorWithDtype((2, 3))) == "node"


def test_distributions_discrete_none_shapes(mocker: MockerFixture) -> None:
    """Test function test_distributions_discrete_none_shapes.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import binomial, categorical, multinomial, poisson

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert categorical("key", [1], shape=(1,)) == "node"
    assert binomial("key", 10, 0.5, shape=(1,)) == "node"
    assert poisson("key", 1.0, shape=(1,)) == "node"
    assert require_callable(multinomial)("key", 10, [0.5, 0.5], shape=(1,)) == "node"


def test_distributions_discrete_more2(mocker: MockerFixture) -> None:
    """Test function test_distributions_discrete_more2.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.random.distributions_discrete import bernoulli, categorical

    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete._emit_random_node", return_value="node")
    assert bernoulli("key", shape=(1,)) == "node"
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
    assert categorical("key", t) == "node"


def test_distributions_discrete_more3(mocker: MockerFixture) -> None:
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
    assert choice("key", t, p=t) == "node"
