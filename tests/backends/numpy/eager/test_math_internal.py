"""Unit tests for math_internal module advanced math and configuration functions."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.shape import OpValue
from tests.backends.numpy.eager._math_advanced_test_ns import math_internal as mi
from tests.testing_utils import MockTestStub, NumpyTestArray, coerce_operand_list


class MockBackend(MockTestStub):
    """Mock backend module for testing math_internal functions."""

    def get_printoptions(self) -> dict[str, int]:
        """Mock get_printoptions.

        Returns:
            dict[str, int]: Computed output value from `get_printoptions`.
        """
        return {"precision": 8}

    def rawmatmul(
        self,
        a: NumpyTestArray,
        b: NumpyTestArray,
    ) -> NumpyTestArray:
        """Mock rawmatmul.

        Args:
            a (NumpyTestArray): Test parameter `a`.
            b (NumpyTestArray): Test parameter `b`.

        Returns:
            NumpyTestArray: Computed output value from `rawmatmul`.
        """
        return ensure_ndarray((a @ b) + 1)


class _TestMathInternalOperationsState(typing.TypedDict, total=False):
    """Typed shared state for `_test_math_internal_operations` split helpers."""

    bk: MockBackend
    h_ext: OpValue


def _test_math_internal_operations_part_1(_state: _TestMathInternalOperationsState) -> None:
    """Execute part 1 of test_math_internal_operations.

    Args:
        _state: Shared test state dictionary.
    """
    bk = MockBackend()

    # 1. _np_population_count
    pop = mi._np_population_count(bk, [0, 1, 3, 7, 15])
    np.testing.assert_array_equal(ensure_ndarray(pop), [0, 1, 2, 3, 4])

    # 2. _np_get_printoptions_
    assert mi._np_get_printoptions_(bk) == {"precision": 8}

    # 3. _np_affineconfig
    assert mi._np_affineconfig(bk, scale=2.0) == {"scale": 2.0}

    # 4. _np_blurconfig
    blur = mi._np_blurconfig(bk, kernel_size=(3, 3), sigma=1.0)
    assert blur is not None

    # 5. _np_customroot (with and without solve)
    res_root = mi._np_customroot(bk, lambda x: x, 5.0, lambda f, g: 42.0)
    assert res_root == 42.0
    res_root_no_solve = mi._np_customroot(bk, lambda x: x, 5.0)
    assert res_root_no_solve == 5.0

    # 6. _np_elasticconfig
    elastic = mi._np_elasticconfig(bk, interpolation="nearest")
    assert elastic is not None

    # 7. LinearOperator configurations
    assert mi._np_linearoperator(bk) is not None
    assert mi._np_linearoperatoradjoint(bk) is not None
    assert mi._np_linearoperatorcirculant(bk) is not None
    assert mi._np_linearoperatorcirculant2d(bk) is not None
    assert mi._np_linearoperatorcirculant3d(bk) is not None
    assert mi._np_linearoperatorcomposition(bk) is not None
    assert mi._np_linearoperatorhouseholder(bk) is not None
    assert mi._np_linearoperatorkronecker(bk) is not None
    assert mi._np_linearoperatorlowrankupdate(bk) is not None
    _state.update({"bk": bk})


def _test_math_internal_operations_part_2(_state: _TestMathInternalOperationsState) -> None:
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
    np.testing.assert_allclose(ensure_ndarray(res_bkm), a @ b + 1)

    class EmptyBackend(MockTestStub):
        """Test class EmptyBackend."""

        pass

    res_default_mm = mi._np_rawmatmul(EmptyBackend(), a, b)
    np.testing.assert_allclose(ensure_ndarray(res_default_mm), a @ b)

    # 10. _np_rawmerge
    merge_list = mi._np_rawmerge(bk, [10, 20])
    assert ensure_ndarray(merge_list)[0] == 10
    assert ensure_ndarray(merge_list)[1] == 0

    merge_args = mi._np_rawmerge(bk, 10, 20)
    assert ensure_ndarray(merge_args)[0] == 10

    merge_empty = mi._np_rawmerge(bk)
    assert ensure_ndarray(merge_empty)[0] is None
    assert ensure_ndarray(merge_empty)[1] == -1

    # 11. _np_rawop
    assert mi._np_rawop(bk, 999) == 999
    assert mi._np_rawop(bk) is None
    _state.update({"bk": bk})


def _test_math_internal_operations_part_3(_state: _TestMathInternalOperationsState) -> None:
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
    assert ensure_ndarray(scan_empty).size == 0

    scan_no_acc = mi._np_scanop(bk, elems=np.array([1, 2, 3]), fn=lambda x, y: x + y)
    np.testing.assert_array_equal(ensure_ndarray(scan_no_acc), [1, 3, 6])

    scan_with_acc = mi._np_scanop(bk, lambda x, y: x + y, np.array([1, 2, 3]), 10)
    np.testing.assert_array_equal(ensure_ndarray(scan_with_acc), [11, 13, 16])

    # 14. _np_tensorarrayread and _np_tensorarraywrite (both within bounds and extending)
    handle = [10, 20]
    assert mi._np_tensorarrayread(bk, handle, 1) == 20

    h_in = mi._np_tensorarraywrite(bk, handle, 0, 99)
    assert ensure_ndarray(h_in)[0] == 99

    h_ext = mi._np_tensorarraywrite(bk, handle, 4, 88)
    assert len(coerce_operand_list(h_ext)) == 5
    _state.update({"h_ext": h_ext})
    _state.update({"bk": bk})


def _test_math_internal_operations_part_4(_state: _TestMathInternalOperationsState) -> None:
    """Execute part 4 of test_math_internal_operations.

    Args:
        _state: Shared test state dictionary.
    """
    bk, h_ext = _state["bk"], _state["h_ext"]
    assert coerce_operand_list(h_ext)[4] == 88

    # 15. _np_tensorconfig
    tcfg = mi._np_tensorconfig(bk, shape=(2, 3), dtype="float32", device="cpu")
    assert tcfg is not None


def test_math_internal_operations() -> None:
    """Verify operations and configuration functions in math_internal.py."""
    _state: _TestMathInternalOperationsState = {}
    _test_math_internal_operations_part_1(_state)
    _test_math_internal_operations_part_2(_state)
    _test_math_internal_operations_part_3(_state)
    _test_math_internal_operations_part_4(_state)
    assert len(_state) > 0
