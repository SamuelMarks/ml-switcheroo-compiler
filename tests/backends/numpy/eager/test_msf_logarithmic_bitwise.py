# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager logarithmic, bitwise, and matrix utility special functions."""

from __future__ import annotations

import typing

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState
from tests.testing_utils import MockTestStub, extract_test_shape, require_callable

_EqT = typing.TypeVar("_EqT")

mod = _mod_raw


def test_math_matrix_utils_coverage() -> None:
    """Test function test_math_matrix_utils_coverage."""
    import ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils as mat_utils

    assert mat_utils._np_confusion_matrix_cap(np) is None
    res = mat_utils._np_confusion_matrix_cap(np, np.array([1]), np.array([1]))
    assert extract_test_shape(res) == (2, 2)

    class MockBkDist(MockTestStub):
        """Test class MockBkDist."""

        def distributions(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function distributions.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `distributions`.
            """
            return "hit"

    assert mat_utils._np_distributions(MockBkDist(), [1]) == "hit"
    import sys

    class MockOps:
        """Test class MockOps."""

        class distributions:
            """Test class distributions."""

            def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
                """Test function __init__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                self.val = "hit2"

            def __eq__(self, other: _EqT) -> bool:
                """Test function __eq__.

                Args:
                    other (_EqT): Test parameter `other`.

                Returns:
                    bool: Computed output value from `__eq__`.
                """
                _typed_ret_7: bool = self.val == other
                return _typed_ret_7

            def mean(self) -> BackendArg:
                """Test function mean.

                Returns:
                    BackendArg: Computed output value from `mean`.
                """
                return 0

    from unittest.mock import patch

    import ml_switcheroo_compiler.ops as _real_ops

    assert _real_ops is not None
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.ops": MockOps()}):
        res2 = mat_utils._np_distributions(np, [1])
        np.testing.assert_array_equal(ensure_ndarray(res2), np.array([1.0, 0.0]))


def test_math_general_coverage() -> None:
    """Test function test_math_general_coverage."""
    import ml_switcheroo_compiler.backends.eager.reference_math.math_general as misc_ext

    class MockBkDesc(MockTestStub):
        """Test class MockBkDesc."""

        def descriptive(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function descriptive.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `descriptive`.
            """
            return "hit"

    assert misc_ext._np_descriptive_2(MockBkDesc(), [1]) == "hit"
    import sys

    class MockOpsDesc:
        """Test class MockOpsDesc."""

        class descriptive:
            """Test class descriptive."""

            def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
                """Test function __init__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                self.val = "hit2"

            def __eq__(self, other: _EqT) -> bool:
                """Test function __eq__.

                Args:
                    other (_EqT): Test parameter `other`.

                Returns:
                    bool: Computed output value from `__eq__`.
                """
                _typed_ret_8: bool = self.val == other
                return _typed_ret_8

    from unittest.mock import patch

    import ml_switcheroo_compiler.ops as _real_ops

    assert _real_ops is not None
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.ops": MockOpsDesc()}):
        res2 = misc_ext._np_descriptive_2(np, [1])
        np.testing.assert_array_equal(ensure_ndarray(res2), np.array([1.0, 0.0, 0.0]))
    res = misc_ext._np_descriptive_2(np, [1, 2, 3])
    np.testing.assert_allclose(ensure_ndarray(res), [2.0, 0.66666667, 0.81649658], rtol=1e-05)


class _TestMathMiscCoverageState(BaseMathMiscState, total=False):
    """Typed shared state for `_test_math_misc_coverage` split helpers."""


def _test_math_misc_coverage_part_1(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 1 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class DummyBk(MockTestStub):
        """Test class DummyBk."""

        @staticmethod
        def any(x: BackendArg) -> BackendArg:
            """Test function any.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `any`.
            """
            return bool(np.any(ensure_ndarray(x)))

        @staticmethod
        def isinf(x: BackendArg) -> BackendArg:
            """Test function isinf.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `isinf`.
            """
            return np.isinf(ensure_ndarray(x))

        @staticmethod
        def isnan(x: BackendArg) -> BackendArg:
            """Test function isnan.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `isnan`.
            """
            return np.isnan(ensure_ndarray(x))

    dummy_bk = DummyBk()
    bk = np
    arg1 = np.array([1.0, 2.0])
    arg2 = np.array([2.0, 3.0])
    arg_bool = np.array([True, False])
    arg_int = np.array([1, 2])
    with pytest.raises(TypeError):
        mod._np_xlogy(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_2(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 2 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_xlogy(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_xlogy(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_xlogy(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_xlogy(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_xlogy(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_3(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 3 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_xlogy(dummy_bk, arg1, arg2)
    with pytest.raises(AttributeError):
        mod._np_mvlgamma(bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_mvlgamma(bk, arg1, arg2)
    with pytest.raises(AttributeError):
        mod._np_mvlgamma(bk, arg1, arg1, arg2)
    with pytest.raises(AttributeError):
        mod._np_mvlgamma(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_4(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 4 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_mvlgamma(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_mvlgamma(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_mvlgamma(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_pmean)(bk, arg1)
    require_callable(mod._np_pmean)(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_5(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 5 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    require_callable(mod._np_pmean)(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_pmean)(bk, arg_bool)
    with pytest.raises(TypeError):
        require_callable(mod._np_pmean)(bk, arg_int)
    with pytest.raises(TypeError):
        require_callable(mod._np_pmean)(dummy_bk, arg1)
    require_callable(mod._np_pmean)(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_6(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 6 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_logsumexp(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_logsumexp(bk, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_logsumexp)(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_logsumexp(bk, arg_bool)
    mod._np_logsumexp(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_7(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 7 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_logsumexp(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_logsumexp(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_segment_sum)(bk, arg1)
    with pytest.raises(IndexError):
        mod._np_segment_sum(bk, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_segment_sum)(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_8(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 8 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        require_callable(mod._np_segment_sum)(bk, arg_bool)
    with pytest.raises(TypeError):
        require_callable(mod._np_segment_sum)(bk, arg_int)
    with pytest.raises(TypeError):
        require_callable(mod._np_segment_sum)(dummy_bk, arg1)
    with pytest.raises(IndexError):
        mod._np_segment_sum(dummy_bk, arg1, arg2)
    mod._np_psum(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_9(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 9 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_psum(bk, arg1, arg2)
    mod._np_psum(bk, arg1, arg1, arg2)
    mod._np_psum(bk, arg_bool)
    mod._np_psum(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_psum(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_10(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 10 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_psum(dummy_bk, arg1, arg2)
    mod._np_log1p2(bk, arg1)
    mod._np_log1p2(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_log1p2(bk, arg1, arg1, arg2)
    mod._np_log1p2(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_11(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 11 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_log1p2(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_log1p2(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_log1p2(dummy_bk, arg1, arg2)
    mod._np_rsqrt(bk, arg1)
    mod._np_rsqrt(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_12(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 12 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_rsqrt(bk, arg1, arg1, arg2)
    mod._np_rsqrt(bk, arg_bool)
    mod._np_rsqrt(bk, arg_int)
    mod._np_rsqrt(dummy_bk, arg1)
    mod._np_rsqrt(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_13(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 13 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(IndexError):
        mod._np_truncate_div(bk, arg1)
    mod._np_truncate_div(bk, arg1, arg2)
    mod._np_truncate_div(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_truncate_div(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_truncate_div(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_14(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 14 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_truncate_div(dummy_bk, arg1)
    mod._np_truncate_div(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_truncate_mod(bk, arg1)
    mod._np_truncate_mod(bk, arg1, arg2)
    mod._np_truncate_mod(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_15(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 15 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_truncate_mod(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_truncate_mod(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_truncate_mod(dummy_bk, arg1)
    mod._np_truncate_mod(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_betainc(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_16(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 16 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_betainc(bk, arg1, arg2)
    mod._np_betainc(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_betainc(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_betainc(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_betainc(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_17(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 17 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_betainc(dummy_bk, arg1, arg2)
    mod._np_bessel_i0e(bk, arg1)
    mod._np_bessel_i0e(bk, arg1, arg2)
    mod._np_bessel_i0e(bk, arg1, arg1, arg2)
    mod._np_bessel_i0e(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_18(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 18 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_bessel_i0e(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_bessel_i0e(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_bessel_i0e(dummy_bk, arg1, arg2)
    mod._np_bessel_i1e(bk, arg1)
    mod._np_bessel_i1e(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_19(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 19 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_bessel_i1e(bk, arg1, arg1, arg2)
    mod._np_bessel_i1e(bk, arg_bool)
    mod._np_bessel_i1e(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_bessel_i1e(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_bessel_i1e(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_20(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 20 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_clz(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_clz(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_clz(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_clz(bk, arg_bool)
    mod._np_clz(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_21(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 21 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_clz(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_clz(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_population_count(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_population_count(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_population_count(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_22(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 22 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_population_count(bk, arg_bool)
    mod._np_population_count(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_population_count(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_population_count(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_bitcast_convert_type)(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def test_msf_logarithmic_bitwise() -> None:
    """Tests for NumPy eager logarithmic, bitwise, and matrix utility special functions runner."""
    _state: _TestMathMiscCoverageState = {}
    for _part_fn in (
        _test_math_misc_coverage_part_1,
        _test_math_misc_coverage_part_2,
        _test_math_misc_coverage_part_3,
        _test_math_misc_coverage_part_4,
        _test_math_misc_coverage_part_5,
        _test_math_misc_coverage_part_6,
        _test_math_misc_coverage_part_7,
        _test_math_misc_coverage_part_8,
        _test_math_misc_coverage_part_9,
        _test_math_misc_coverage_part_10,
        _test_math_misc_coverage_part_11,
        _test_math_misc_coverage_part_12,
        _test_math_misc_coverage_part_13,
        _test_math_misc_coverage_part_14,
        _test_math_misc_coverage_part_15,
        _test_math_misc_coverage_part_16,
        _test_math_misc_coverage_part_17,
        _test_math_misc_coverage_part_18,
        _test_math_misc_coverage_part_19,
        _test_math_misc_coverage_part_20,
        _test_math_misc_coverage_part_21,
        _test_math_misc_coverage_part_22,
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([1.0, 2.0]))
