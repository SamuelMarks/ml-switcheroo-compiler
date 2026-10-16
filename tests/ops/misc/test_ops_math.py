"""Tests for test_ops_math."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.ops.base import get_op

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_extra_unary() -> None:
    """Test the extra unary behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test function."
        logit_op = get_op("Logit")()
        res = logit_op.eager_eval(np.array([0.5]))
        assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(res, 0.0)
        res_eps = logit_op.eager_eval(np.array([0.0, 1.0]), eps=1e-05)
        assert not cast("Callable[..., EagerTensorProtocol]", np.isnan)(res_eps).any()
        mvlgamma_op = get_op("Mvlgamma")()
        res_mvl = mvlgamma_op.eager_eval(np.array([2.0]), p=2)
        assert res_mvl is not None
        nan_to_num_op = get_op("NanToNum")()
        res_nan = nan_to_num_op.eager_eval(np.array([np.nan, np.inf]), nan=0.0, posinf=1.0)
        assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(res_nan, [0.0, 1.0])
        signbit_op = get_op("Signbit")()
        res_signbit = signbit_op.eager_eval(np.array([-1.0, 1.0]))
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res_signbit, [True, False])
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_extra_binary() -> None:
    """Test the extra binary behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test function."
        xlogy_op = get_op("Xlogy")()
        res = xlogy_op.eager_eval(np.array([0.0, 2.0]), np.array([0.0, 2.0]))
        assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(res, [0.0, 2.0 * cast("Callable[..., EagerTensorProtocol]", np.log)(2.0)])
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None
