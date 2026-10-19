"""Tests for test_distributed_unimplemented."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops import _pmean, _psum
from ml_switcheroo_compiler.backends.numpy.eager.distributed import _np_all_gather, _np_all_reduce, _np_all_to_all, set_np_distributed_context
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _np_psum

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_math_misc_distributed() -> None:
    """Test function test_math_misc_distributed."""
    set_np_distributed_context(world_size=1, rank=0, addr="127.0.0.1", port=40001)

    t = np.array([1, 2])

    res = _np_all_gather(np, t, axis=0)

    _np_all_reduce(np, t, op_type="sum")

    _np_psum(np, t)

    _np_all_to_all(np, t)
    assert res is not None


def test_core_math_ops_distributed() -> None:
    """Test function test_core_math_ops_distributed."""
    t = np.array([1, 2])
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _psum)(np, t), t)
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _pmean)(np, t), t)
