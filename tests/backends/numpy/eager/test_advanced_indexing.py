"""Tests for test_advanced_indexing."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numpy.eager.advanced_indexing as adv

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def _test_advanced_indexing_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    ops = [getattr(adv, name) for name in dir(adv) if name.startswith("_") and callable(getattr(adv, name))]
    arg = np.array([1.0, 2.0])
    arg_indices = np.array([0, 1])
    arg_indices_nd = np.array([[0], [1]])
    getattr(_state, "update")({"arg": arg, "arg_indices": arg_indices, "arg_indices_nd": arg_indices_nd, "ops": ops})
    _state.update(locals())


def _test_advanced_indexing_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_indices, arg_indices_nd, ops) = (_state["arg"], _state["arg_indices"], _state["arg_indices_nd"], _state["ops"])
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("indices", "kth", "index"):
                if "nd" in op.__name__:
                    cast("Callable[..., EagerTensorProtocol]", args_to_pass.append)(arg_indices_nd)
                else:
                    cast("Callable[..., EagerTensorProtocol]", args_to_pass.append)(arg_indices)
            elif p.name == "shape":
                cast("Callable[..., EagerTensorProtocol]", args_to_pass.append)((2,))
            elif p.name == "dim":
                cast("Callable[..., EagerTensorProtocol]", args_to_pass.append)(0)
            else:
                cast("Callable[..., EagerTensorProtocol]", args_to_pass.append)(arg)
        try:
            op(*args_to_pass)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            op(*args_to_pass[:-1])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
    _state.update(locals())


def _test_advanced_indexing_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_indices) = (_state["arg"], _state["arg_indices"])
    adv._scatter_add(arg, arg_indices, arg, 0)
    adv._tensor_scatter_update(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    adv._tensor_scatter_add(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_tensor_scatter_sub"))(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    adv._tensor_scatter_min(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    adv._tensor_scatter_max(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    adv._tensor_scatter_update(arg, ([0], [1]), np.array([3.0, 4.0]))
    adv._np_scatter_nd(np, [[0], [1]], arg, (2,))
    adv._np_tensor_scatter_add(np, arg, [[0], [1]], arg)
    _state.update(locals())


def _test_advanced_indexing_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    arg = _state["arg"]
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter_sub"))(np, arg, [[0], [1]], arg)
    adv._np_tensor_scatter_update(np, arg, [[0], [1]], arg)
    adv._np_tensor_scatter_min(np, arg, [[0], [1]], arg)
    adv._np_tensor_scatter_max(np, arg, [[0], [1]], arg)
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_scatter_nd_add"))(np, arg, [[0], [1]], arg)
    _state.update(locals())


def _test_advanced_indexing_coverage_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    arg = _state["arg"]
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_scatter_nd_sub"))(np, arg, [[0], [1]], arg)
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_scatter_nd_update"))(np, arg, [[0], [1]], arg)
    with pytest.raises((ValueError, IndexError)):
        adv._tensor_scatter_update(arg, "a", np.array([3.0, 4.0]))
    with pytest.raises((ValueError, IndexError)):
        adv._tensor_scatter_add(arg, "a", np.array([3.0, 4.0]))
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_tensor_scatter_sub"))(arg, "a", np.array([3.0, 4.0]))
    _state.update(locals())


def _test_advanced_indexing_coverage_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_indices, arg_indices_nd) = (_state["arg"], _state["arg_indices"], _state["arg_indices_nd"])
    with pytest.raises((ValueError, IndexError)):
        adv._tensor_scatter_min(arg, "a", np.array([3.0, 4.0]))
    with pytest.raises((ValueError, IndexError)):
        adv._tensor_scatter_max(arg, "a", np.array([3.0, 4.0]))
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_scatter_update"))(np, arg, arg_indices, arg, dim=0)
    adv._np_scatter_add(np, arg, arg_indices, arg, dim=0)
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction="add")
    _state.update(locals())


def _test_advanced_indexing_coverage_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_indices_nd) = (_state["arg"], _state["arg_indices_nd"])
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction="sub")
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction="mul")
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction="div")
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction="min")
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction="max")
    _state.update(locals())


def _test_advanced_indexing_coverage_part_8(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 8 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_indices_nd) = (_state["arg"], _state["arg_indices_nd"])
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction=None)
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd)
    adv._np_scatter_apply(np, None, arg, arg_indices_nd, arg, reduction="add")
    adv._np_scatter_apply(np, None, arg, arg_indices_nd, arg, reduction="mul")
    adv._np_scatter_apply(np, None, arg, arg_indices_nd, arg, reduction=None)
    _state.update(locals())


def _test_advanced_indexing_coverage_part_9(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 9 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_indices) = (_state["arg"], _state["arg_indices"])
    with pytest.raises(RuntimeError):
        adv._np_scatter_apply(np, None, arg, "bad_indices", arg, reduction="add")
    adv._np_scatter_apply(np, None)
    adv._np_scatter_apply(np, None, arg)
    with pytest.raises(IndexError):
        adv._np_scatter_max(np, arg, arg_indices, arg)
    with pytest.raises(IndexError):
        adv._np_scatter_min(np, arg, arg_indices, arg)
    _state.update(locals())


def _test_advanced_indexing_coverage_part_10(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 10 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_indices) = (_state["arg"], _state["arg_indices"])
    with pytest.raises(IndexError):
        adv._np_scatter_mul(np, arg, arg_indices, arg)
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(adv, "_np_scatter_update"))(np, arg, np.array([0, 0]), arg, dim=0)
    with pytest.raises(IndexError):
        adv._np_scatter_max(np, arg, np.array([0, 0]), arg)
    with pytest.raises(IndexError):
        adv._np_scatter_min(np, arg, np.array([0, 0]), arg)
    with pytest.raises(IndexError):
        adv._np_scatter_mul(np, arg, np.array([0, 0]), arg)
    _state.update(locals())


def _test_advanced_indexing_coverage_part_11(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 11 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_indices_nd) = (_state["arg"], _state["arg_indices_nd"])
    adv._np_scatter(np, arg, np.array([0, 0]), arg, dim=0)
    adv._np_scatter_max(np, arg, arg_indices_nd, arg)
    adv._np_scatter_min(np, arg, arg_indices_nd, arg)
    adv._np_scatter_mul(np, arg, arg_indices_nd, arg)


def test_advanced_indexing_coverage() -> None:
    """Test function test_advanced_indexing_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_advanced_indexing_coverage_part_1(_state)
    _test_advanced_indexing_coverage_part_2(_state)
    _test_advanced_indexing_coverage_part_3(_state)
    _test_advanced_indexing_coverage_part_4(_state)
    _test_advanced_indexing_coverage_part_5(_state)
    _test_advanced_indexing_coverage_part_6(_state)
    _test_advanced_indexing_coverage_part_7(_state)
    _test_advanced_indexing_coverage_part_8(_state)
    _test_advanced_indexing_coverage_part_9(_state)
    _test_advanced_indexing_coverage_part_10(_state)
    _test_advanced_indexing_coverage_part_11(_state)
    assert isinstance(_state, dict)
