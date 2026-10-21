"""Tests for test_ops_math."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.ops.base import get_op
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


def test_extra_unary() -> None:
    """Test the extra unary behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(AttributeError):
        logit_op = get_op("Logit")()
        res = logit_op.eager_eval(np.array([0.5]))
        assert as_callable(np.allclose)(res, 0.0)
        res_eps = logit_op.eager_eval(np.array([0.0, 1.0]), eps=1e-05)
        assert not as_callable(np.isnan)(res_eps).any()
        mvlgamma_op = get_op("Mvlgamma")()
        res_mvl = mvlgamma_op.eager_eval(np.array([2.0]), p=2)
        assert res_mvl is not None
        nan_to_num_op = get_op("NanToNum")()
        res_nan = nan_to_num_op.eager_eval(np.array([np.nan, np.inf]), nan=0.0, posinf=1.0)
        assert as_callable(np.allclose)(res_nan, [0.0, 1.0])
        signbit_op = get_op("Signbit")()
        res_signbit = signbit_op.eager_eval(np.array([-1.0, 1.0]))
        assert as_callable(np.array_equal)(res_signbit, [True, False])


def test_extra_binary() -> None:
    """Test the extra binary behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    xlogy_op = get_op("Xlogy")()
    res = xlogy_op.eager_eval(np.array([0.0, 2.0]), np.array([0.0, 2.0]))
    assert as_callable(np.allclose)(res, [0.0, 2.0 * as_callable(np.log)(2.0)])
