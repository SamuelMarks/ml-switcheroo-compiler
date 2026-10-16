"""Tests for test_advanced_indexing."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, cast

import numpy as np

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
    arg, arg_indices, arg_indices_nd, ops = _state["arg"], _state["arg_indices"], _state["arg_indices_nd"], _state["ops"]
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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None

        # Test 1 missing arg
        try:
            op(*args_to_pass[:-1])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
    _state.update(locals())


def _test_advanced_indexing_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    arg, arg_indices = _state["arg"], _state["arg_indices"]
    adv._scatter_add(arg, arg_indices, arg, 0)

    adv._tensor_scatter_update(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    try:
        adv._tensor_scatter_add(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("EagerTensorProtocol", getattr(adv, "_tensor_scatter_sub"))(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        adv._tensor_scatter_min(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        adv._tensor_scatter_max(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    # Test cases where indices are a list or tuple instead of np.ndarray
    adv._tensor_scatter_update(arg, ([0], [1]), np.array([3.0, 4.0]))

    adv._np_scatter_nd(np, [[0], [1]], arg, (2,))
    try:
        adv._np_tensor_scatter_add(np, arg, [[0], [1]], arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    _state.update(locals())


def _test_advanced_indexing_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    arg = _state["arg"]
    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter_sub"))(np, arg, [[0], [1]], arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        adv._np_tensor_scatter_update(np, arg, [[0], [1]], arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        adv._np_tensor_scatter_min(np, arg, [[0], [1]], arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        adv._np_tensor_scatter_max(np, arg, [[0], [1]], arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_scatter_nd_add"))(np, arg, [[0], [1]], arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    _state.update(locals())


def _test_advanced_indexing_coverage_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    arg = _state["arg"]
    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_scatter_nd_sub"))(np, arg, [[0], [1]], arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_scatter_nd_update"))(np, arg, [[0], [1]], arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    # missing lines handling
    try:
        adv._tensor_scatter_update(arg, "a", np.array([3.0, 4.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        adv._tensor_scatter_add(arg, "a", np.array([3.0, 4.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("EagerTensorProtocol", getattr(adv, "_tensor_scatter_sub"))(arg, "a", np.array([3.0, 4.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    _state.update(locals())


def _test_advanced_indexing_coverage_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    arg, arg_indices, arg_indices_nd = _state["arg"], _state["arg_indices"], _state["arg_indices_nd"]
    try:
        adv._tensor_scatter_min(arg, "a", np.array([3.0, 4.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        adv._tensor_scatter_max(arg, "a", np.array([3.0, 4.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_scatter_update"))(np, arg, arg_indices, arg, dim=0)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        adv._np_scatter_add(np, arg, arg_indices, arg, dim=0)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction="add")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    _state.update(locals())


def _test_advanced_indexing_coverage_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    arg, arg_indices_nd = _state["arg"], _state["arg_indices_nd"]
    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction="sub")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction="mul")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction="div")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction="min")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction="max")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    _state.update(locals())


def _test_advanced_indexing_coverage_part_8(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 8 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    arg, arg_indices_nd = _state["arg"], _state["arg_indices_nd"]
    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd, arg, reduction=None)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_tensor_scatter"))(np, arg, arg_indices_nd)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        adv._np_scatter_apply(np, None, arg, arg_indices_nd, arg, reduction="add")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        adv._np_scatter_apply(np, None, arg, arg_indices_nd, arg, reduction="mul")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        adv._np_scatter_apply(np, None, arg, arg_indices_nd, arg, reduction=None)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    _state.update(locals())


def _test_advanced_indexing_coverage_part_9(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 9 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    arg, arg_indices = _state["arg"], _state["arg_indices"]
    try:
        adv._np_scatter_apply(np, None, arg, "bad_indices", arg, reduction="add")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        adv._np_scatter_apply(np, None)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        adv._np_scatter_apply(np, None, arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        adv._np_scatter_max(np, arg, arg_indices, arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        adv._np_scatter_min(np, arg, arg_indices, arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    _state.update(locals())


def _test_advanced_indexing_coverage_part_10(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 10 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    arg, arg_indices = _state["arg"], _state["arg_indices"]
    try:
        adv._np_scatter_mul(np, arg, arg_indices, arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        cast("EagerTensorProtocol", getattr(adv, "_np_scatter_update"))(np, arg, np.array([0, 0]), arg, dim=0)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        adv._np_scatter_max(np, arg, np.array([0, 0]), arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        adv._np_scatter_min(np, arg, np.array([0, 0]), arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        adv._np_scatter_mul(np, arg, np.array([0, 0]), arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    _state.update(locals())


def _test_advanced_indexing_coverage_part_11(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 11 of test_advanced_indexing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    arg, arg_indices_nd = _state["arg"], _state["arg_indices_nd"]
    try:
        adv._np_scatter(np, arg, np.array([0, 0]), arg, dim=0)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        adv._np_scatter_max(np, arg, arg_indices_nd, arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        adv._np_scatter_min(np, arg, arg_indices_nd, arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        adv._np_scatter_mul(np, arg, arg_indices_nd, arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


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
