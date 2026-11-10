"""Unit tests for math_internal module advanced math and configuration functions."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_internal as mi
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mod, ensure_ndarray, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


class MockBackend:
    """Mock backend module for testing math_internal functions."""

    def get_printoptions(self) -> dict[str, int]:
        """Mock get_printoptions."""
        return {"precision": 8}

    def rawmatmul(self, a: np.ndarray[tuple[int, ...], np.dtype[np.float64]], b: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Mock rawmatmul.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.
        """
        return ensure_ndarray((ensure_eager_tensor(a) @ ensure_eager_tensor(b)) + ensure_eager_tensor(1))


def _test_math_internal_operations_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_internal_operations.

    Args:
        _state: Shared test state dictionary.
    """
    bk = MockBackend()

    # 1. _np_population_count
    pop = mi._np_population_count(ensure_mod(bk), [0, 1, 3, 7, 15])
    ensure_callable(np.testing.assert_array_equal)(pop, [0, 1, 2, 3, 4])

    # 2. _np_get_printoptions_
    assert mi._np_get_printoptions_(ensure_mod(bk)) == {"precision": 8}

    # 3. _np_affineconfig
    assert mi._np_affineconfig(ensure_mod(bk), scale=2.0) == {"scale": 2.0}

    # 4. _np_blurconfig
    blur = mi._np_blurconfig(ensure_mod(bk), kernel_size=(3, 3), sigma=1.0)
    assert blur is not None

    # 5. _np_customroot (with and without solve)
    res_root = mi._np_customroot(ensure_mod(bk), lambda x: x, 5.0, lambda f, g: 42.0)
    assert res_root == 42.0
    res_root_no_solve = mi._np_customroot(ensure_mod(bk), lambda x: x, 5.0)
    assert res_root_no_solve == 5.0

    # 6. _np_elasticconfig
    elastic = mi._np_elasticconfig(ensure_mod(bk), interpolation="nearest")
    assert elastic is not None

    # 7. LinearOperator configurations
    assert mi._np_linearoperator(ensure_mod(bk)) is not None
    assert mi._np_linearoperatoradjoint(ensure_mod(bk)) is not None
    assert mi._np_linearoperatorcirculant(ensure_mod(bk)) is not None
    assert mi._np_linearoperatorcirculant2d(ensure_mod(bk)) is not None
    assert mi._np_linearoperatorcirculant3d(ensure_mod(bk)) is not None
    assert mi._np_linearoperatorcomposition(ensure_mod(bk)) is not None
    assert mi._np_linearoperatorhouseholder(ensure_mod(bk)) is not None
    assert mi._np_linearoperatorkronecker(ensure_mod(bk)) is not None
    assert mi._np_linearoperatorlowrankupdate(ensure_mod(bk)) is not None
    _state.update(ensure_tensor_dict({"bk": bk, "blur": blur, "elastic": elastic, "pop": pop}))
    _state.update(ensure_tensor_dict({"res_root": res_root, "res_root_no_solve": res_root_no_solve}))


def _test_math_internal_operations_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_internal_operations.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    assert mi._np_linearoperatortoeplitz(bk) is not None

    # 8. _np_perspectiveconfig
    persp = mi._np_perspectiveconfig(bk, interpolation="nearest")
    assert persp is not None
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[2.0, 0.0], [1.0, 2.0]])
    res_bkm = mi._np_rawmatmul(bk, a, b)
    ensure_callable(np.testing.assert_allclose)(res_bkm, a @ b + 1)

    class EmptyBackend:
        """Test class EmptyBackend."""

        pass

    res_default_mm = mi._np_rawmatmul(ensure_mod(EmptyBackend()), a, b)
    ensure_callable(np.testing.assert_allclose)(res_default_mm, a @ b)

    # 10. _np_rawmerge
    merge_list = mi._np_rawmerge(bk, [10, 20])
    assert ensure_eager_tensor(merge_list)[0] == 10
    assert ensure_eager_tensor(merge_list)[1] == 0

    merge_args = mi._np_rawmerge(bk, 10, 20)
    assert ensure_eager_tensor(merge_args)[0] == 10

    merge_empty = mi._np_rawmerge(bk)
    assert ensure_eager_tensor(merge_empty)[0] is None
    assert ensure_eager_tensor(merge_empty)[1] == -1

    # 11. _np_rawop
    assert mi._np_rawop(bk, 999) == 999
    assert mi._np_rawop(bk) is None
    _state.update(ensure_tensor_dict({"EmptyBackend": EmptyBackend, "a": a, "b": b, "merge_args": merge_args}))
    _state.update(ensure_tensor_dict({"merge_empty": merge_empty, "merge_list": merge_list, "persp": persp, "res_bkm": res_bkm}))
    _state.update(ensure_tensor_dict({"res_default_mm": res_default_mm}))
    _state.update(ensure_tensor_dict({"bk": bk}))


def _test_math_internal_operations_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_internal_operations.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    sw_true = mi._np_rawswitch(bk, "data", True)
    assert sw_true == (None, "data")
    sw_false = mi._np_rawswitch(bk, "data", False)
    assert sw_false == ("data", None)

    sw_op_true = mi._np_switchop(bk, "data", pred=True)
    assert sw_op_true == (None, "data")
    sw_op_false = mi._np_switchop(bk, "data", pred=False)
    assert sw_op_false == ("data", None)

    # 13. _np_scanop (invalid fn, empty elems, with acc, without acc)
    scan_invalid = mi._np_scanop(bk, "not_callable")
    assert scan_invalid == "not_callable"

    scan_empty = mi._np_scanop(bk, elems=np.array([]), fn=lambda x, y: x + y)
    assert ensure_eager_tensor(ensure_eager_tensor(scan_empty).size) == 0

    scan_no_acc = mi._np_scanop(bk, elems=np.array([1, 2, 3]), fn=lambda x, y: x + y)
    ensure_callable(np.testing.assert_array_equal)(scan_no_acc, [1, 3, 6])

    scan_with_acc = mi._np_scanop(bk, lambda x, y: x + y, np.array([1, 2, 3]), 10)
    ensure_callable(np.testing.assert_array_equal)(scan_with_acc, [11, 13, 16])

    # 14. _np_tensorarrayread and _np_tensorarraywrite (both within bounds and extending)
    handle = [10, 20]
    assert mi._np_tensorarrayread(bk, handle, 1) == 20

    h_in = mi._np_tensorarraywrite(bk, handle, 0, 99)
    assert ensure_eager_tensor(h_in)[0] == 99

    h_ext = mi._np_tensorarraywrite(bk, handle, 4, 88)
    assert ensure_callable(len)(h_ext) == 5
    _state.update(ensure_tensor_dict({"h_ext": h_ext, "h_in": h_in, "handle": handle, "scan_empty": scan_empty}))
    _state.update(ensure_tensor_dict({"scan_invalid": scan_invalid, "scan_no_acc": scan_no_acc, "scan_with_acc": scan_with_acc, "sw_false": sw_false}))
    _state.update(ensure_tensor_dict({"sw_op_false": sw_op_false, "sw_op_true": sw_op_true, "sw_true": sw_true}))
    _state.update(ensure_tensor_dict({"bk": bk}))


def _test_math_internal_operations_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_math_internal_operations.

    Args:
        _state: Shared test state dictionary.
    """
    bk, h_ext = _state["bk"], _state["h_ext"]
    assert h_ext[4] == 88

    # 15. _np_tensorconfig
    tcfg = mi._np_tensorconfig(bk, shape=(2, 3), dtype="float32", device="cpu")
    assert tcfg is not None
    _state.update(ensure_tensor_dict({"tcfg": tcfg}))


def test_math_internal_operations() -> None:
    """Verify operations and configuration functions in math_internal.py."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_internal_operations_part_1(_state)
    _test_math_internal_operations_part_2(_state)
    _test_math_internal_operations_part_3(_state)
    _test_math_internal_operations_part_4(_state)
    assert len(_state) > 0
