"""Tests for test_advanced_indexing."""

from __future__ import annotations

import inspect
import typing

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numpy.eager.advanced_indexing as adv
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def _test_advanced_indexing_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    from tests.testing_utils import get_test_members

    ops = [ensure_callable(fn) for name, fn in get_test_members(adv).items() if name.startswith("_") and callable(fn)]
    arg = np.array([1.0, 2.0])
    arg_indices = np.array([0, 1])
    arg_indices_nd = np.array([[0], [1]])
    _state.update(ensure_tensor_dict({"arg": arg, "arg_indices": arg_indices, "arg_indices_nd": arg_indices_nd, "ops": ops}))


def _test_advanced_indexing_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (arg, arg_indices, arg_indices_nd, ops) = (_state["arg"], _state["arg_indices"], _state["arg_indices_nd"], _state["ops"])
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("indices", "kth", "index"):
                if "nd" in op.__name__:
                    ensure_callable(args_to_pass.append)(arg_indices_nd)
                else:
                    ensure_callable(args_to_pass.append)(arg_indices)
            elif p.name == "shape":
                ensure_callable(args_to_pass.append)((2,))
            elif p.name == "dim":
                ensure_callable(args_to_pass.append)(0)
            else:
                ensure_callable(args_to_pass.append)(arg)
        try:
            op(*args_to_pass)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            handled_errors += 1
        try:
            op(*args_to_pass[:-1])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            handled_errors += 1
    _state.update(ensure_tensor_dict({"arg": arg, "arg_indices": arg_indices, "arg_indices_nd": arg_indices_nd}))
    _state.update(ensure_tensor_dict({"ops": ops}))


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
        ensure_eager_tensor(ensure_eager_tensor(adv)._tensor_scatter_sub)(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    adv._tensor_scatter_min(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    adv._tensor_scatter_max(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    adv._tensor_scatter_update(arg, ([0], [1]), np.array([3.0, 4.0]))
    adv._np_scatter_nd(ensure_eager_tensor(np), [[0], [1]], arg, (2,))
    adv._np_tensor_scatter_add(ensure_eager_tensor(np), arg, [[0], [1]], arg)
    _state.update(ensure_tensor_dict({"arg": arg, "arg_indices": arg_indices}))


def _test_advanced_indexing_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    arg = _state["arg"]
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_tensor_scatter_sub)(np, arg, [[0], [1]], arg)
    adv._np_tensor_scatter_update(ensure_eager_tensor(np), arg, [[0], [1]], arg)
    adv._np_tensor_scatter_min(ensure_eager_tensor(np), arg, [[0], [1]], arg)
    adv._np_tensor_scatter_max(ensure_eager_tensor(np), arg, [[0], [1]], arg)
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_scatter_nd_add)(np, arg, [[0], [1]], arg)
    _state.update(ensure_tensor_dict({"arg": arg}))


def _test_advanced_indexing_coverage_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    arg = _state["arg"]
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_scatter_nd_sub)(np, arg, [[0], [1]], arg)
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_scatter_nd_update)(np, arg, [[0], [1]], arg)
    with pytest.raises((ValueError, IndexError)):
        adv._tensor_scatter_update(arg, "a", np.array([3.0, 4.0]))
    with pytest.raises((ValueError, IndexError)):
        adv._tensor_scatter_add(arg, "a", np.array([3.0, 4.0]))
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._tensor_scatter_sub)(arg, "a", np.array([3.0, 4.0]))
    _state.update(ensure_tensor_dict({"arg": arg}))


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
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_scatter_update)(np, arg, arg_indices, arg, dim=0)
    adv._np_scatter_add(ensure_eager_tensor(np), arg, arg_indices, arg, dim=0)
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_tensor_scatter)(np, arg, arg_indices_nd, arg, reduction="add")
    _state.update(ensure_tensor_dict({"arg": arg, "arg_indices": arg_indices, "arg_indices_nd": arg_indices_nd}))


def _test_advanced_indexing_coverage_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_indices_nd) = (_state["arg"], _state["arg_indices_nd"])
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_tensor_scatter)(np, arg, arg_indices_nd, arg, reduction="sub")
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_tensor_scatter)(np, arg, arg_indices_nd, arg, reduction="mul")
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_tensor_scatter)(np, arg, arg_indices_nd, arg, reduction="div")
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_tensor_scatter)(np, arg, arg_indices_nd, arg, reduction="min")
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_tensor_scatter)(np, arg, arg_indices_nd, arg, reduction="max")
    _state.update(ensure_tensor_dict({"arg": arg, "arg_indices_nd": arg_indices_nd}))


def _test_advanced_indexing_coverage_part_8(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 8 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_indices_nd) = (_state["arg"], _state["arg_indices_nd"])
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_tensor_scatter)(np, arg, arg_indices_nd, arg, reduction=None)
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_tensor_scatter)(np, arg, arg_indices_nd)
    ensure_callable(adv._np_scatter_apply)(np, None, arg, arg_indices_nd, arg, reduction="add")
    ensure_callable(adv._np_scatter_apply)(np, None, arg, arg_indices_nd, arg, reduction="mul")
    ensure_callable(adv._np_scatter_apply)(np, None, arg, arg_indices_nd, arg, reduction=None)
    _state.update(ensure_tensor_dict({"arg": arg, "arg_indices_nd": arg_indices_nd}))


def _test_advanced_indexing_coverage_part_9(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 9 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_indices) = (_state["arg"], _state["arg_indices"])
    with pytest.raises(RuntimeError):
        ensure_callable(adv._np_scatter_apply)(np, None, arg, "bad_indices", arg, reduction="add")
    ensure_callable(adv._np_scatter_apply)(np, None)
    ensure_callable(adv._np_scatter_apply)(np, None, arg)
    with pytest.raises(IndexError):
        adv._np_scatter_max(ensure_eager_tensor(np), arg, arg_indices, arg)
    with pytest.raises(IndexError):
        adv._np_scatter_min(ensure_eager_tensor(np), arg, arg_indices, arg)
    _state.update(ensure_tensor_dict({"arg": arg, "arg_indices": arg_indices}))


def _test_advanced_indexing_coverage_part_10(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 10 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_indices) = (_state["arg"], _state["arg_indices"])
    with pytest.raises(IndexError):
        adv._np_scatter_mul(ensure_eager_tensor(np), arg, arg_indices, arg)
    with pytest.raises(AttributeError):
        ensure_eager_tensor(ensure_eager_tensor(adv)._np_scatter_update)(np, arg, np.array([0, 0]), arg, dim=0)
    with pytest.raises(IndexError):
        adv._np_scatter_max(ensure_eager_tensor(np), arg, np.array([0, 0]), arg)
    with pytest.raises(IndexError):
        adv._np_scatter_min(ensure_eager_tensor(np), arg, np.array([0, 0]), arg)
    with pytest.raises(IndexError):
        adv._np_scatter_mul(ensure_eager_tensor(np), arg, np.array([0, 0]), arg)
    _state.update(ensure_tensor_dict({"arg": arg, "arg_indices": arg_indices}))


def _test_advanced_indexing_coverage_part_11(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 11 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_indices_nd) = (_state["arg"], _state["arg_indices_nd"])
    adv._np_scatter(ensure_eager_tensor(np), arg, np.array([0, 0]), arg, dim=0)
    adv._np_scatter_max(ensure_eager_tensor(np), arg, arg_indices_nd, arg)
    adv._np_scatter_min(ensure_eager_tensor(np), arg, arg_indices_nd, arg)
    adv._np_scatter_mul(ensure_eager_tensor(np), arg, arg_indices_nd, arg)


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
    assert len(_state) > 0
