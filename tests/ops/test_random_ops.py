"""Test random ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.random_ops import binomial, categorical, choice, dirichlet, permutation, truncated_normal
from ml_switcheroo_compiler.ops.random_ops.core import Rademacher, rademacher

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


class DummyShape:
    """Mock shape container."""

    shape: tuple[int, int] = (2, 2)


def test_random_ops_infer_shape() -> None:
    """Test shape inference for random operations.

    Returns:
        None.
    """
    t = Tensor(None, TensorConfig((2, 2), "float32", "cpu"))
    assert cast("EagerCallableProtocol", categorical)().infer_shape(t) == (2, 2)
    assert cast("EagerCallableProtocol", dirichlet)().infer_shape(t) == (2, 2)
    assert cast("EagerCallableProtocol", binomial)().infer_shape(t) == (2, 2)
    assert cast("EagerCallableProtocol", truncated_normal)().infer_shape(t) == (2, 2)
    assert cast("EagerCallableProtocol", permutation)().infer_shape(t) == (2, 2)
    assert cast("EagerCallableProtocol", choice)().infer_shape(t) == (2, 2)

    assert Rademacher().infer_shape(DummyShape().shape) == (2, 2)
    assert Rademacher().infer_shape() == ()
    assert Rademacher().infer_shape(size=5) == (5,)
    assert Rademacher().infer_shape(size=(2, 3)) == (2, 3)
    assert Rademacher().infer_shape(shape=(3, 4)) == (3, 4)

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="rademacher_val") as mock_dispatch:
        res = rademacher(shape=(2, 2))
        assert res == "rademacher_val"
        mock_dispatch.assert_called_once_with("Rademacher", shape=(2, 2))
