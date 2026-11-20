"""Tests for test_core_math_trig."""

from __future__ import annotations

import math
import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_trig import (
    _acos,
    _acosh,
    _asin,
    _asinh,
    _atan,
    _atan2,
    _atanh,
    _isin,
    _isinf,
    _isposinf,
    _sinc,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import MockTestStub, coerce_operand_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class _EmptyBackend(MockTestStub):
    """Mock backend without any attributes."""

    pass


class _TestMathTrigAllBranchesState(typing.TypedDict, total=False):
    """Typed shared state for `_test_math_trig_all_branches` split helpers."""

    empty_backend: _EmptyBackend


def _test_math_trig_all_branches_part_1(_state: _TestMathTrigAllBranchesState) -> None:
    """Execute part 1 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _EmptyBackend()

    # 1. _acos
    # Case A: backend has arccos
    class BackendArccos(MockTestStub):
        """Backend with arccos."""

        def arccos(self, x: float) -> str:
            """Arccos.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `arccos`.
            """
            return "arccos_val"

    assert _acos(BackendArccos(), 0.5) == "arccos_val"

    # Case B: backend has acos (no arccos)
    class BackendAcos(MockTestStub):
        """Backend with acos."""

        def acos(self, x: float) -> str:
            """Acos.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `acos`.
            """
            return "acos_val"

    assert _acos(BackendAcos(), 0.5) == "acos_val"

    # Case C: backend has neither -> fallback to np.arccos
    assert np.isclose(ensure_ndarray(_acos(empty_backend, 0.5)), math.acos(0.5))

    # 2. _acosh
    # Case A: backend has arccosh
    class BackendArccosh(MockTestStub):
        """Backend with arccosh."""

        def arccosh(self, x: float) -> str:
            """Arccosh.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `arccosh`.
            """
            return "arccosh_val"

    assert _acosh(BackendArccosh(), 1.5) == "arccosh_val"
    _state.update({"empty_backend": empty_backend})


def _test_math_trig_all_branches_part_2(_state: _TestMathTrigAllBranchesState) -> None:
    """Execute part 2 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]

    class BackendAcosh(MockTestStub):
        """Backend with acosh."""

        def acosh(self, x: float) -> str:
            """Acosh.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `acosh`.
            """
            return "acosh_val"

    assert _acosh(BackendAcosh(), 1.5) == "acosh_val"

    # Case C: fallback
    assert np.isclose(ensure_ndarray(_acosh(empty_backend, 0.5)), math.acos(0.5))

    # 3. _asin
    # Case A: backend has arcsin
    class BackendArcsin(MockTestStub):
        """Backend with arcsin."""

        def arcsin(self, x: float) -> str:
            """Arcsin.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `arcsin`.
            """
            return "arcsin_val"

    assert _asin(BackendArcsin(), 0.5) == "arcsin_val"

    # Case B: backend has asin
    class BackendAsin(MockTestStub):
        """Backend with asin."""

        def asin(self, x: float) -> str:
            """Asin.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `asin`.
            """
            return "asin_val"

    assert _asin(BackendAsin(), 0.5) == "asin_val"

    # Case C: fallback
    assert np.isclose(ensure_ndarray(_asin(empty_backend, 0.5)), math.asin(0.5))
    _state.update({"empty_backend": empty_backend})


def _test_math_trig_all_branches_part_3(_state: _TestMathTrigAllBranchesState) -> None:
    """Execute part 3 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]

    class BackendArcsinh(MockTestStub):
        """Backend with arcsinh."""

        def arcsinh(self, x: float) -> str:
            """Arcsinh.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `arcsinh`.
            """
            return "arcsinh_val"

    assert _asinh(BackendArcsinh(), 0.5) == "arcsinh_val"

    class BackendAsin(MockTestStub):
        """Backend with asinh."""

        def asinh(self, x: float) -> str:
            """Asinh.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `asinh`.
            """
            return "asinh_val"

    assert _asinh(BackendAsin(), 0.5) == "asinh_val"

    # Case C: fallback
    assert np.isclose(ensure_ndarray(_asinh(empty_backend, 0.5)), math.asin(0.5))

    # 5. _atan
    # Case A: backend has arctan
    class BackendArctan(MockTestStub):
        """Backend with arctan."""

        def arctan(self, x: float) -> str:
            """Arctan.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `arctan`.
            """
            return "arctan_val"

    assert _atan(BackendArctan(), 0.5) == "arctan_val"
    _state.update({"empty_backend": empty_backend})


def _test_math_trig_all_branches_part_4(_state: _TestMathTrigAllBranchesState) -> None:
    """Execute part 4 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]

    class BackendAtan(MockTestStub):
        """Backend with atan."""

        def atan(self, x: float) -> str:
            """Atan.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `atan`.
            """
            return "atan_val"

    assert _atan(BackendAtan(), 0.5) == "atan_val"

    # Case C: fallback
    assert np.isclose(ensure_ndarray(_atan(empty_backend, 0.5)), math.atan(0.5))

    # 6. _atanh
    # Case A: backend has arctanh
    class BackendArctanh(MockTestStub):
        """Backend with arctanh."""

        def arctanh(self, x: float) -> str:
            """Arctanh.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `arctanh`.
            """
            return "arctanh_val"

    assert _atanh(BackendArctanh(), 0.5) == "arctanh_val"

    # Case B: backend has atanh
    class BackendAtanh(MockTestStub):
        """Backend with atanh."""

        def atanh(self, x: float) -> str:
            """Atanh.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `atanh`.
            """
            return "atanh_val"

    assert _atanh(BackendAtanh(), 0.5) == "atanh_val"

    # Case C: fallback
    assert np.isclose(ensure_ndarray(_atanh(empty_backend, 0.5)), math.atan(0.5))
    _state.update({"empty_backend": empty_backend})


def _test_math_trig_all_branches_part_5(_state: _TestMathTrigAllBranchesState) -> None:
    """Execute part 5 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]

    class BackendArctan2(MockTestStub):
        """Backend with arctan2."""

        def arctan2(self, y: float, x: float) -> str:
            """Arctan2.

            Args:
                y: Test parameter `y`.
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `arctan2`.
            """
            return "arctan2_val"

    assert _atan2(BackendArctan2(), 1.0, 1.0) == "arctan2_val"

    # Case B: backend has atan2
    class BackendAtan2(MockTestStub):
        """Backend with atan2."""

        def atan2(self, y: float, x: float) -> str:
            """Atan2.

            Args:
                y: Test parameter `y`.
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `atan2`.
            """
            return "atan2_val"

    assert _atan2(BackendAtan2(), 1.0, 1.0) == "atan2_val"

    # Case C: fallback
    assert np.isclose(ensure_ndarray(_atan2(empty_backend, 1.0, 1.0)), math.atan2(1.0, 1.0))

    class BackendSinc(MockTestStub):
        """Backend with sinc."""

        def sinc(self, x: float) -> str:
            """Sinc.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `sinc`.
            """
            return "sinc_val"

    assert _sinc(BackendSinc(), 0.5) == "sinc_val"

    # Case B: fallback
    assert np.isclose(ensure_ndarray(_sinc(empty_backend, 0.0)), 1.0)
    _state.update({"empty_backend": empty_backend})


def _test_math_trig_all_branches_part_6(_state: _TestMathTrigAllBranchesState) -> None:
    """Execute part 6 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]

    class BackendIsin(MockTestStub):
        """Backend with isin."""

        def isin(self, element: BackendArg, test_elements: BackendArg) -> str:
            """Isin.

            Args:
                element: Test parameter `element`.
                test_elements: Test parameter `test_elements`.

            Returns:
                str: Computed output value from `isin`.
            """
            return "isin_val"

    assert _isin(BackendIsin(), [1], [1, 2]) == "isin_val"

    # 10. _isinf
    # Case A: backend has isinf and args are passed directly
    class BackendIsinfDirect(MockTestStub):
        """Backend with isinf."""

        def isinf(self, x: BackendArg) -> str:
            """Isinf.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `isinf`.
            """
            return "isinf_direct"

    assert _isinf(BackendIsinfDirect(), np.array([float("inf")])) == "isinf_direct"

    # Case B: backend lacks isinf -> fallback to np.isinf
    inf_res = _isinf(empty_backend, np.array([float("inf"), 1.0]))
    assert coerce_operand_list(inf_res) == [True, False]

    class BackendIsposinfDirect(MockTestStub):
        """Backend with isposinf."""

        def isposinf(self, x: BackendArg) -> str:
            """Isposinf.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `isposinf`.
            """
            return "isposinf_direct"

    assert _isposinf(BackendIsposinfDirect(), np.array([float("inf")])) == "isposinf_direct"

    # Case B: backend lacks isposinf -> fallback to np.isposinf
    posinf_res = _isposinf(empty_backend, np.array([float("inf"), float("-inf"), 1.0]))
    assert coerce_operand_list(posinf_res) == [True, False, False]
    _state.update({"empty_backend": empty_backend})


def _test_math_trig_all_branches_part_7(_state: _TestMathTrigAllBranchesState) -> None:
    """Execute part 7 of test_math_trig_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    for op_name in [
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "Atan2",
        "Sinc",
        "Isin",
        "Isinf",
        "Isposinf",
    ]:
        assert global_eager_registry.get(op_name) is not None


def test_math_trig_all_branches() -> None:
    """Test 100% lines and branches of math_trig.py.

    Returns:
        None
    """
    _state: _TestMathTrigAllBranchesState = {}
    _test_math_trig_all_branches_part_1(_state)
    _test_math_trig_all_branches_part_2(_state)
    _test_math_trig_all_branches_part_3(_state)
    _test_math_trig_all_branches_part_4(_state)
    _test_math_trig_all_branches_part_5(_state)
    _test_math_trig_all_branches_part_6(_state)
    _test_math_trig_all_branches_part_7(_state)
    assert len(_state) > 0
