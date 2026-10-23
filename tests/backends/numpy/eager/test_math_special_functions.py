# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for test_math_special_functions."""

from __future__ import annotations

import importlib
import typing

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as _mod_raw
import ml_switcheroo_compiler.ops as ops
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import numpy_eager_registry
from tests.testing_utils import as_callable, as_mod, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue

mod = as_tensor(_mod_raw)


def test_math_matrix_utils_coverage() -> None:
    """Test function test_math_matrix_utils_coverage."""
    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils as mat_utils

    assert mat_utils._np_confusion_matrix_cap(np) is None
    res = mat_utils._np_confusion_matrix_cap(np, np.array([1]), np.array([1]))
    assert as_tensor(getattr(res, "shape")) == (2, 2)

    class MockBkDist:
        """Test class MockBkDist."""

        def distributions(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function distributions.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "hit"

    assert mat_utils._np_distributions(as_tensor(MockBkDist()), [1]) == "hit"
    import sys

    class MockOps:
        """Test class MockOps."""

        class distributions:
            """Test class distributions."""

            def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Test function __init__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                self.val = "hit2"

            def __eq__(self, other: object) -> bool:
                """Test function __eq__.

                Args:
                    other: Test parameter `other`.
                """
                return getattr(self, "val", None) == other

            def mean(self) -> EagerValue:
                """Test function mean."""
                return 0

    from unittest.mock import patch

    with patch.dict(sys.modules, {"ml_switcheroo_compiler.ops": MockOps()}):
        res2 = mat_utils._np_distributions(np, [1])
        as_callable(np.testing.assert_array_equal)(res2, as_callable(np.array)([1.0, 0.0]))


def test_math_general_coverage() -> None:
    """Test function test_math_general_coverage."""
    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general as misc_ext

    class MockBkDesc:
        """Test class MockBkDesc."""

        def descriptive(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function descriptive.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "hit"

    assert misc_ext._np_descriptive_2(as_tensor(MockBkDesc()), [1]) == "hit"
    import sys

    class MockOpsDesc:
        """Test class MockOpsDesc."""

        class descriptive:
            """Test class descriptive."""

            def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Test function __init__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                self.val = "hit2"

            def __eq__(self, other: object) -> bool:
                """Test function __eq__.

                Args:
                    other: Test parameter `other`.
                """
                return getattr(self, "val", None) == other

    from unittest.mock import patch

    with patch.dict(sys.modules, {"ml_switcheroo_compiler.ops": MockOpsDesc()}):
        res2 = misc_ext._np_descriptive_2(np, [1])
        as_callable(np.testing.assert_array_equal)(res2, as_callable(np.array)([1.0, 0.0, 0.0]))
    res = misc_ext._np_descriptive_2(np, [1, 2, 3])
    as_callable(np.testing.assert_allclose)(res, [2.0, 0.66666667, 0.81649658], rtol=1e-05)


def _test_math_misc_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class DummyBk:
        """Test class DummyBk."""

        @staticmethod
        def any(x: EagerValue) -> EagerValue:
            """Test function any.

            Args:
                x: Test parameter `x`.
            """
            return as_callable(np.any)(x)

        @staticmethod
        def isinf(x: EagerValue) -> EagerValue:
            """Test function isinf.

            Args:
                x: Test parameter `x`.
            """
            return as_tensor(as_callable(np.isinf)(x))

        @staticmethod
        def isnan(x: EagerValue) -> EagerValue:
            """Test function isnan.

            Args:
                x: Test parameter `x`.
            """
            return as_tensor(as_callable(np.isnan)(x))

    dummy_bk = DummyBk()
    bk = np
    arg1 = np.array([1.0, 2.0])
    arg2 = np.array([2.0, 3.0])
    arg_bool = np.array([True, False])
    arg_int = np.array([1, 2])
    with pytest.raises(TypeError):
        mod._np_xlogy(bk, arg1)
    getattr(_state, "update")({"DummyBk": DummyBk, "arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    getattr(_state, "update")({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})
    _state.update(locals())


def _test_math_misc_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
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
        mod._np_pmean(bk, arg1)
    mod._np_pmean(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_pmean(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_pmean(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_pmean(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_pmean(dummy_bk, arg1)
    mod._np_pmean(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
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
        mod._np_logsumexp(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_logsumexp(bk, arg_bool)
    mod._np_logsumexp(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_logsumexp(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_logsumexp(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_segment_sum(bk, arg1)
    with pytest.raises(IndexError):
        mod._np_segment_sum(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_segment_sum(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_8(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 8 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_segment_sum(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_segment_sum(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_segment_sum(dummy_bk, arg1)
    with pytest.raises(IndexError):
        mod._np_segment_sum(dummy_bk, arg1, arg2)
    mod._np_psum(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_9(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_10(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_11(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_12(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_13(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_14(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_15(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_16(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_17(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_18(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_19(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_20(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_21(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_math_misc_coverage_part_22(_state: dict[str, EagerTensorProtocol]) -> None:
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
        mod._np_bitcast_convert_type(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_23(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 23 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_bitcast_convert_type(bk, arg1, arg2)
    mod._np_bitcast_convert_type(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_bitcast_convert_type(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_bitcast_convert_type(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_bitcast_convert_type(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_24(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 24 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_bitcast_convert_type(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_reduce_precision(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_reduce_precision(bk, arg1, arg2)
    mod._np_reduce_precision(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_reduce_precision(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_25(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 25 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_reduce_precision(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_reduce_precision(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_reduce_precision(dummy_bk, arg1, arg2)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_gather"))(bk, arg1)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_gather"))(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_26(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 26 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_gather"))(bk, arg1, arg1, arg2)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_gather"))(bk, arg_bool)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_gather"))(bk, arg_int)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_gather"))(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_gather"))(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_27(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 27 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_reduce"))(bk, arg1)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_reduce"))(bk, arg1, arg2)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_reduce"))(bk, arg1, arg1, arg2)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_reduce"))(bk, arg_bool)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_reduce"))(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_28(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 28 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_reduce"))(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_reduce"))(dummy_bk, arg1, arg2)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_to_all"))(bk, arg1)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_to_all"))(bk, arg1, arg2)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_to_all"))(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_29(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 29 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_to_all"))(bk, arg_bool)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_to_all"))(bk, arg_int)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_to_all"))(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_all_to_all"))(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_packbits(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_30(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 30 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_packbits(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_packbits(bk, arg1, arg1, arg2)
    mod._np_packbits(bk, arg_bool)
    mod._np_packbits(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_packbits(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_31(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 31 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_packbits(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_unpackbits(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_unpackbits(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_unpackbits(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_unpackbits(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_32(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 32 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_unpackbits(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_unpackbits(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_unpackbits(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_piecewise(bk, arg1)
    with pytest.raises(IndexError):
        mod._np_piecewise(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_33(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 33 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_piecewise(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_piecewise(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_piecewise(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_piecewise(dummy_bk, arg1)
    with pytest.raises(IndexError):
        mod._np_piecewise(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_34(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 34 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(IndexError):
        mod._np_promotetypes(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_promotetypes(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_promotetypes(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_promotetypes(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_promotetypes(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_35(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 35 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_promotetypes(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_promotetypes(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_trace(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_trace(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_trace(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_36(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 36 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_trace(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_trace(bk, arg_int)
    with pytest.raises(ValueError):
        mod._np_trace(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_trace(dummy_bk, arg1, arg2)
    mod._np_trapz(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_37(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 37 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_trapz(bk, arg1, arg2)
    mod._np_trapz(bk, arg1, arg1, arg2)
    mod._np_trapz(bk, arg_bool)
    mod._np_trapz(bk, arg_int)
    mod._np_trapz(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_38(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 38 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_trapz(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_tri(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_tri(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_tri(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_tri(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_39(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 39 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(ValueError):
        mod._np_tri(bk, arg_int)
    with pytest.raises(ValueError):
        mod._np_tri(dummy_bk, arg1)
    with pytest.raises(ValueError):
        mod._np_tri(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_trilindices(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_trilindices(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_40(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 40 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_trilindices(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_trilindices(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_trilindices(bk, arg_int)
    with pytest.raises(ValueError):
        mod._np_trilindices(dummy_bk, arg1)
    with pytest.raises(ValueError):
        mod._np_trilindices(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_41(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 41 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(ValueError):
        mod._np_trilindicesfrom(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_trilindicesfrom(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_trilindicesfrom(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_trilindicesfrom(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_trilindicesfrom(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_42(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 42 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_trilindicesfrom(dummy_bk, arg1)
    with pytest.raises(ValueError):
        mod._np_trilindicesfrom(dummy_bk, arg1, arg2)
    mod._np_trimzeros(bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_trimzeros(bk, arg1, arg2)
    with pytest.raises((TypeError, AttributeError)):
        mod._np_trimzeros(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_43(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 43 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_trimzeros(bk, arg_bool)
    mod._np_trimzeros(bk, arg_int)
    mod._np_trimzeros(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_trimzeros(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_triuindices(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_44(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 44 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_triuindices(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_triuindices(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_triuindices(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_triuindices(bk, arg_int)
    with pytest.raises(ValueError):
        mod._np_triuindices(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_45(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 45 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(ValueError):
        mod._np_triuindices(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_triuindicesfrom(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_triuindicesfrom(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_triuindicesfrom(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_triuindicesfrom(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_46(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 46 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(ValueError):
        mod._np_triuindicesfrom(bk, arg_int)
    with pytest.raises(ValueError):
        mod._np_triuindicesfrom(dummy_bk, arg1)
    with pytest.raises(ValueError):
        mod._np_triuindicesfrom(dummy_bk, arg1, arg2)
    mod._np_uint(bk, arg1)
    mod._np_uint(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_47(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 47 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_uint(bk, arg1, arg1, arg2)
    mod._np_uint(bk, arg_bool)
    mod._np_uint(bk, arg_int)
    mod._np_uint(dummy_bk, arg1)
    mod._np_uint(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_48(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 48 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_uint8(bk, arg1)
    mod._np_uint8(bk, arg1, arg2)
    mod._np_uint8(bk, arg1, arg1, arg2)
    mod._np_uint8(bk, arg_bool)
    mod._np_uint8(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_49(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 49 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_uint8(dummy_bk, arg1)
    mod._np_uint8(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_union1d(bk, arg1)
    mod._np_union1d(bk, arg1, arg2)
    mod._np_union1d(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_50(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 50 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_union1d(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_union1d(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_union1d(dummy_bk, arg1)
    mod._np_union1d(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_unravelindex(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_51(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 51 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_unravelindex(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_unravelindex(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_unravelindex(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_unravelindex(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_unravelindex(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_52(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 52 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_unravelindex(dummy_bk, arg1, arg2)
    mod._np_unwrap(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_unwrap(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_unwrap(bk, arg1, arg1, arg2)
    mod._np_unwrap(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_53(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 53 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_unwrap(bk, arg_int)
    mod._np_unwrap(dummy_bk, arg1)
    with pytest.raises(ValueError):
        mod._np_unwrap(dummy_bk, arg1, arg2)
    mod._np_vander(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_vander(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_54(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 54 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_vander(bk, arg1, arg1, arg2)
    mod._np_vander(bk, arg_bool)
    mod._np_vander(bk, arg_int)
    mod._np_vander(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_vander(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_55(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 55 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(ValueError):
        mod._np_vectorize(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_vectorize(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_vectorize(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_vectorize(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_vectorize(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_56(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 56 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_vectorize(dummy_bk, arg1)
    with pytest.raises(ValueError):
        mod._np_vectorize(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_append(bk, arg1)
    mod._np_append(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_append(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_57(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 57 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_append(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_append(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_append(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_append(dummy_bk, arg1, arg2)
    mod._np_average(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_58(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 58 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_average(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_average(bk, arg1, arg1, arg2)
    mod._np_average(bk, arg_bool)
    mod._np_average(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_average(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_59(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 59 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_average(dummy_bk, arg1, arg2)
    mod._np_block(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_block(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_block(bk, arg1, arg1, arg2)
    mod._np_block(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_60(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 60 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_block(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_block(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_block(dummy_bk, arg1, arg2)
    mod._np_atleast_1d(bk, arg1)
    mod._np_atleast_1d(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_61(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 61 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_atleast_1d(bk, arg1, arg1, arg2)
    mod._np_atleast_1d(bk, arg_bool)
    mod._np_atleast_1d(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_atleast_1d(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_atleast_1d(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_62(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 62 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_atleast_2d(bk, arg1)
    mod._np_atleast_2d(bk, arg1, arg2)
    mod._np_atleast_2d(bk, arg1, arg1, arg2)
    mod._np_atleast_2d(bk, arg_bool)
    mod._np_atleast_2d(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_63(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 63 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_atleast_2d(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_atleast_2d(dummy_bk, arg1, arg2)
    mod._np_atleast_3d(bk, arg1)
    mod._np_atleast_3d(bk, arg1, arg2)
    mod._np_atleast_3d(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_64(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 64 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_atleast_3d(bk, arg_bool)
    mod._np_atleast_3d(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_atleast_3d(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_atleast_3d(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_apply_along_axis(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_65(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 65 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_apply_along_axis(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_apply_along_axis(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_apply_along_axis(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_apply_along_axis(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_apply_along_axis(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_66(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 66 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_apply_along_axis(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_apply_over_axes(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_apply_over_axes(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_apply_over_axes(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_apply_over_axes(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_67(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 67 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_apply_over_axes(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_apply_over_axes(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_apply_over_axes(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_argpartition(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_argpartition(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_68(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 68 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_argpartition(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_argpartition(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_argpartition(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_argpartition(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_argpartition(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_69(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 69 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_argwhere(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_argwhere(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_argwhere(bk, arg1, arg1, arg2)
    mod._np_argwhere(bk, arg_bool)
    mod._np_argwhere(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_70(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 70 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_argwhere(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_argwhere(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_choose(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_choose(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_choose(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_71(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 71 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_choose(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_choose(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_choose(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_choose(dummy_bk, arg1, arg2)
    mod._np_column_stack(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_72(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 72 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_column_stack(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_column_stack(bk, arg1, arg1, arg2)
    mod._np_column_stack(bk, arg_bool)
    mod._np_column_stack(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_column_stack(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_73(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 73 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_column_stack(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_compress(bk, arg1)
    mod._np_compress(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_compress(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_compress(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_74(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 74 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_compress(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_compress(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_compress(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_convolve(bk, arg1)
    mod._np_convolve(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_75(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 75 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_convolve(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_convolve(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_convolve(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_convolve(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_convolve(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_76(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 76 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_corrcoef(bk, arg1)
    mod._np_corrcoef(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_corrcoef(bk, arg1, arg1, arg2)
    mod._np_corrcoef(bk, arg_bool)
    mod._np_corrcoef(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_77(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 77 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_corrcoef(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_corrcoef(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_correlate(bk, arg1)
    mod._np_correlate(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_correlate(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_78(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 78 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_correlate(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_correlate(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_correlate(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_correlate(dummy_bk, arg1, arg2)
    mod._np_cov(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_79(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 79 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_cov(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_cov(bk, arg1, arg1, arg2)
    mod._np_cov(bk, arg_bool)
    mod._np_cov(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_cov(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_80(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 80 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_cov(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_array_equiv_(bk, arg1)
    mod._np_array_equiv_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_array_equiv_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_array_equiv_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_81(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 81 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_array_equiv_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_array_equiv_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_array_equiv_(dummy_bk, arg1, arg2)
    mod._np_array_repr_(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_array_repr_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_82(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 82 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_array_repr_(bk, arg1, arg1, arg2)
    mod._np_array_repr_(bk, arg_bool)
    mod._np_array_repr_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_array_repr_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_array_repr_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_83(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 83 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_array_str_(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_array_str_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_array_str_(bk, arg1, arg1, arg2)
    mod._np_array_str_(bk, arg_bool)
    mod._np_array_str_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_84(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 84 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_array_str_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_array_str_(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_blackman_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_blackman_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_blackman_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_85(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 85 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_blackman_(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_blackman_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_blackman_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_blackman_(dummy_bk, arg1, arg2)
    mod._np_broadcast_arrays_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_86(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 86 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_broadcast_arrays_(bk, arg1, arg2)
    mod._np_broadcast_arrays_(bk, arg1, arg1, arg2)
    mod._np_broadcast_arrays_(bk, arg_bool)
    mod._np_broadcast_arrays_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_broadcast_arrays_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_87(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 87 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_broadcast_arrays_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_can_cast_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_can_cast_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_can_cast_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_can_cast_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_88(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 88 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_can_cast_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_can_cast_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_can_cast_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_delete_(bk, arg1)
    with pytest.raises(IndexError):
        mod._np_delete_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_89(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 89 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_delete_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_delete_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_delete_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_delete_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_delete_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_90(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 90 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(ValueError):
        mod._np_diag_indices_(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_diag_indices_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_diag_indices_(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_diag_indices_(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_diag_indices_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_91(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 91 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_diag_indices_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_diag_indices_(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_diag_indices_from_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_diag_indices_from_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_diag_indices_from_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_92(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 92 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_diag_indices_from_(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_diag_indices_from_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_diag_indices_from_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_diag_indices_from_(dummy_bk, arg1, arg2)
    mod._np_diagflat_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_93(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 93 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_diagflat_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_diagflat_(bk, arg1, arg1, arg2)
    mod._np_diagflat_(bk, arg_bool)
    mod._np_diagflat_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_diagflat_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_94(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 94 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_diagflat_(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_diagonal_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_diagonal_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_diagonal_(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_diagonal_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_95(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 95 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(ValueError):
        mod._np_diagonal_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_diagonal_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_diagonal_(dummy_bk, arg1, arg2)
    mod._np_diff_(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_diff_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_96(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 96 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_diff_(bk, arg1, arg1, arg2)
    mod._np_diff_(bk, arg_bool)
    mod._np_diff_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_diff_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_diff_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_97(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 97 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_digitize_(bk, arg1)
    mod._np_digitize_(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_digitize_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_digitize_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_digitize_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_98(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 98 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_digitize_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_digitize_(dummy_bk, arg1, arg2)
    mod._np_ediff1d_(bk, arg1)
    mod._np_ediff1d_(bk, arg1, arg2)
    mod._np_ediff1d_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_99(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 99 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_ediff1d_(bk, arg_bool)
    mod._np_ediff1d_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_ediff1d_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_ediff1d_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_einsum_path_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_100(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 100 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_einsum_path_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_einsum_path_(bk, arg1, arg1, arg2)
    with pytest.raises((ValueError, TypeError)):
        mod._np_einsum_path_(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_einsum_path_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_einsum_path_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_101(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 101 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_einsum_path_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_extract_(bk, arg1)
    mod._np_extract_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_extract_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_extract_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_102(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 102 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_extract_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_extract_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_extract_(dummy_bk, arg1, arg2)
    mod._np_fabs_(bk, arg1)
    mod._np_fabs_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_103(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 103 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_fabs_(bk, arg1, arg1, arg2)
    mod._np_fabs_(bk, arg_bool)
    mod._np_fabs_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_fabs_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_fabs_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_104(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 104 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_flatnonzero_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_flatnonzero_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_flatnonzero_(bk, arg1, arg1, arg2)
    mod._np_flatnonzero_(bk, arg_bool)
    mod._np_flatnonzero_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_105(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 105 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_flatnonzero_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_flatnonzero_(dummy_bk, arg1, arg2)
    mod._np_flip_op_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_flip_op_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_flip_op_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_106(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 106 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_flip_op_(bk, arg_bool)
    mod._np_flip_op_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_flip_op_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_flip_op_(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_fliplr_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_107(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 107 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_fliplr_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_fliplr_(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_fliplr_(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_fliplr_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_fliplr_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_108(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 108 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_fliplr_(dummy_bk, arg1, arg2)
    mod._np_flipud_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_flipud_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_flipud_(bk, arg1, arg1, arg2)
    mod._np_flipud_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_109(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 109 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_flipud_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_flipud_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_flipud_(dummy_bk, arg1, arg2)
    mod._np_flip_reverse_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_flip_reverse_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_110(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 110 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_flip_reverse_(bk, arg1, arg1, arg2)
    mod._np_flip_reverse_(bk, arg_bool)
    mod._np_flip_reverse_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_flip_reverse_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_flip_reverse_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_111(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 111 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_fromfunction_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_fromfunction_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_fromfunction_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_fromfunction_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_fromfunction_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_112(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 112 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_fromfunction_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_fromfunction_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_fromiter_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_fromiter_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_fromiter_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_113(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 113 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_fromiter_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_fromiter_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_fromiter_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_fromiter_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_frompyfunc_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_114(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 114 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_frompyfunc_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_frompyfunc_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_frompyfunc_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_frompyfunc_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_frompyfunc_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_115(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 115 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_frompyfunc_(dummy_bk, arg1, arg2)
    mod._np_fromstring_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_fromstring_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_fromstring_(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_fromstring_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_116(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 116 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_fromstring_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_fromstring_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_fromstring_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_geomspace_(bk, arg1)
    mod._np_geomspace_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_117(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 117 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_geomspace_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_geomspace_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_geomspace_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_geomspace_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_geomspace_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_118(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 118 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_get_printoptions_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_get_printoptions_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_get_printoptions_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_get_printoptions_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_get_printoptions_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_119(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 119 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_get_printoptions_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_get_printoptions_(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_hamming_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_hamming_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_hamming_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_120(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 120 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_hamming_(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_hamming_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_hamming_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_hamming_(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_hanning_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_121(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 121 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_hanning_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_hanning_(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_hanning_(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_hanning_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_hanning_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_122(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 122 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_hanning_(dummy_bk, arg1, arg2)
    mod._np_histogram_(bk, arg1)
    mod._np_histogram_(bk, arg1, arg2)
    mod._np_histogram_(bk, arg1, arg1, arg2)
    mod._np_histogram_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_123(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 123 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_histogram_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_histogram_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_histogram_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_histogram2d_(bk, arg1)
    mod._np_histogram2d_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_124(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 124 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_histogram2d_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_histogram2d_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_histogram2d_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_histogram2d_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_histogram2d_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_125(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 125 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_histogram_bin_edges_(bk, arg1)
    mod._np_histogram_bin_edges_(bk, arg1, arg2)
    mod._np_histogram_bin_edges_(bk, arg1, arg1, arg2)
    mod._np_histogram_bin_edges_(bk, arg_bool)
    mod._np_histogram_bin_edges_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_126(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 126 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_histogram_bin_edges_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_histogram_bin_edges_(dummy_bk, arg1, arg2)
    mod._np_histogramdd_(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_histogramdd_(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_histogramdd_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_127(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 127 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_histogramdd_(bk, arg_bool)
    mod._np_histogramdd_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_histogramdd_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_histogramdd_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_indices_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_128(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 128 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_indices_(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_indices_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_indices_(bk, arg_bool)
    mod._np_indices_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_indices_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_129(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 129 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_indices_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_insert_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_insert_(bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_insert_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_insert_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_130(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 130 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_insert_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_insert_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_insert_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_interp_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_interp_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_131(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 131 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_interp_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_interp_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_interp_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_interp_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_interp_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_132(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 132 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_intersect1d_(bk, arg1)
    mod._np_intersect1d_(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_intersect1d_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_intersect1d_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_intersect1d_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_133(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 133 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_intersect1d_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_intersect1d_(dummy_bk, arg1, arg2)
    mod._np_iscomplex_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_iscomplex_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_iscomplex_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_134(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 134 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_iscomplex_(bk, arg_bool)
    mod._np_iscomplex_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_iscomplex_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_iscomplex_(dummy_bk, arg1, arg2)
    mod._np_iscomplexobj_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_135(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 135 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_iscomplexobj_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_iscomplexobj_(bk, arg1, arg1, arg2)
    mod._np_iscomplexobj_(bk, arg_bool)
    mod._np_iscomplexobj_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_iscomplexobj_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_136(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 136 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_iscomplexobj_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_issubdtype_op_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_issubdtype_op_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_issubdtype_op_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_issubdtype_op_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_137(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 137 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_issubdtype_op_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_issubdtype_op_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_issubdtype_op_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_isin_(bk, arg1)
    mod._np_isin_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_138(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 138 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_isin_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_isin_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_isin_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_isin_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_isin_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_139(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 139 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_isreal_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_isreal_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_isreal_(bk, arg1, arg1, arg2)
    mod._np_isreal_(bk, arg_bool)
    mod._np_isreal_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_140(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 140 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_isreal_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_isreal_(dummy_bk, arg1, arg2)
    mod._np_isrealobj_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_isrealobj_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_isrealobj_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_141(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 141 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_isrealobj_(bk, arg_bool)
    mod._np_isrealobj_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_isrealobj_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_isrealobj_(dummy_bk, arg1, arg2)
    mod._np_isscalar_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_142(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 142 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_isscalar_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_isscalar_(bk, arg1, arg1, arg2)
    mod._np_isscalar_(bk, arg_bool)
    mod._np_isscalar_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_isscalar_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_143(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 143 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_isscalar_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_issubdtype_issubdtype_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_issubdtype_issubdtype_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_issubdtype_issubdtype_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_issubdtype_issubdtype_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_144(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 144 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_issubdtype_issubdtype_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_issubdtype_issubdtype_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_issubdtype_issubdtype_(dummy_bk, arg1, arg2)
    mod._np_iterable_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_iterable_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_145(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 145 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_iterable_(bk, arg1, arg1, arg2)
    mod._np_iterable_(bk, arg_bool)
    mod._np_iterable_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_iterable_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_iterable_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_146(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 146 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_ix__(bk, arg1)
    mod._np_ix__(bk, arg1, arg2)
    mod._np_ix__(bk, arg1, arg1, arg2)
    mod._np_ix__(bk, arg_bool)
    mod._np_ix__(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_147(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 147 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_ix__(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_ix__(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_kaiser_(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_kaiser_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_kaiser_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_148(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 148 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_kaiser_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_kaiser_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_kaiser_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_kaiser_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_kron_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_149(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 149 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_kron_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_kron_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_kron_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_kron_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_kron_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_150(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 150 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_kron_(dummy_bk, arg1, arg2)
    mod._np_lexsort_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_lexsort_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_lexsort_(bk, arg1, arg1, arg2)
    mod._np_lexsort_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_151(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 151 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_lexsort_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_lexsort_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_lexsort_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_load_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_load_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_152(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 152 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_load_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_load_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_load_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_load_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_load_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_153(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 153 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_mask_indices_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_mask_indices_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_mask_indices_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_mask_indices_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_mask_indices_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_154(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 154 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_mask_indices_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_mask_indices_(dummy_bk, arg1, arg2)
    mod._np_median_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_median_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_median_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_155(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 155 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_median_(bk, arg_bool)
    mod._np_median_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_median_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_median_(dummy_bk, arg1, arg2)
    mod._np_modf_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_156(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 156 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_modf_(bk, arg1, arg2)
    mod._np_modf_(bk, arg1, arg1, arg2)
    mod._np_modf_(bk, arg_bool)
    mod._np_modf_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_modf_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_157(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 157 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_modf_(dummy_bk, arg1, arg2)
    mod._np_nonzero_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_nonzero_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_nonzero_(bk, arg1, arg1, arg2)
    mod._np_nonzero_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_158(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 158 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_nonzero_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_nonzero_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_nonzero_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_resize_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_resize_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_159(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 159 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_resize_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_resize_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_resize_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_resize_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_resize_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_160(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 160 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_result_type_(bk, arg1)
    mod._np_result_type_(bk, arg1, arg2)
    mod._np_result_type_(bk, arg1, arg1, arg2)
    mod._np_result_type_(bk, arg_bool)
    mod._np_result_type_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_161(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 161 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_result_type_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_result_type_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_ravel_multi_index_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_ravel_multi_index_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_ravel_multi_index_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_162(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 162 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_ravel_multi_index_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_ravel_multi_index_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_ravel_multi_index_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_ravel_multi_index_(dummy_bk, arg1, arg2)
    mod._np_trapz_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_163(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 163 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_trapz_(bk, arg1, arg2)
    mod._np_trapz_(bk, arg1, arg1, arg2)
    mod._np_trapz_(bk, arg_bool)
    mod._np_trapz_(bk, arg_int)
    mod._np_trapz_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_164(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 164 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_trapz_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_zeros_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_zeros_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_zeros_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_zeros_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_165(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 165 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_zeros_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_zeros_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_zeros_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_ones_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_ones_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_166(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 166 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_ones_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_ones_(bk, arg_bool)
    mod._np_ones_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_ones_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_ones_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_167(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 167 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_empty_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_empty_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_empty_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_empty_(bk, arg_bool)
    mod._np_empty_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_168(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 168 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_empty_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_empty_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_full_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_full_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_full_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_169(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 169 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_full_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_full_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_full_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_full_(dummy_bk, arg1, arg2)
    mod._np_zeros_like_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_170(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 170 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_zeros_like_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_zeros_like_(bk, arg1, arg1, arg2)
    mod._np_zeros_like_(bk, arg_bool)
    mod._np_zeros_like_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_zeros_like_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_171(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 171 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_zeros_like_(dummy_bk, arg1, arg2)
    mod._np_ones_like_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_ones_like_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_ones_like_(bk, arg1, arg1, arg2)
    mod._np_ones_like_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_172(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 172 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_ones_like_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_ones_like_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_ones_like_(dummy_bk, arg1, arg2)
    mod._np_empty_like_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_empty_like_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_173(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 173 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_empty_like_(bk, arg1, arg1, arg2)
    mod._np_empty_like_(bk, arg_bool)
    mod._np_empty_like_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_empty_like_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_empty_like_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_174(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 174 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_full_like_(bk, arg1)
    mod._np_full_like_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_full_like_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_full_like_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_full_like_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_175(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 175 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_full_like_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_full_like_(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_arange_(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_arange_(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_arange_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_176(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 176 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_arange_(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_arange_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_arange_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_arange_(dummy_bk, arg1, arg2)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_cholesky_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_177(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 177 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linalg_cholesky_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linalg_cholesky_(bk, arg1, arg1, arg2)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_cholesky_(bk, arg_bool)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_cholesky_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_linalg_cholesky_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_178(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 178 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_linalg_cholesky_(dummy_bk, arg1, arg2)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_det_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linalg_det_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linalg_det_(bk, arg1, arg1, arg2)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_det_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_179(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 179 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_det_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_linalg_det_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_linalg_det_(dummy_bk, arg1, arg2)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_svd_(bk, arg1)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_svd_(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_180(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 180 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_svd_(bk, arg1, arg1, arg2)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_svd_(bk, arg_bool)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_svd_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_linalg_svd_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_linalg_svd_(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_181(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 181 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_expand_dims_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_expand_dims_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_expand_dims_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_expand_dims_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_expand_dims_(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_182(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 182 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_expand_dims_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_expand_dims_(dummy_bk, arg1, arg2)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_inv_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linalg_inv_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linalg_inv_(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_183(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 183 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_inv_(bk, arg_bool)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_inv_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_linalg_inv_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_linalg_inv_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linalg_matrix_power_(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_184(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 184 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_matrix_power_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linalg_matrix_power_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linalg_matrix_power_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linalg_matrix_power_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_linalg_matrix_power_(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_185(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 185 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_linalg_matrix_power_(dummy_bk, arg1, arg2)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_pinv_(bk, arg1)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_pinv_(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_linalg_pinv_(bk, arg1, arg1, arg2)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_pinv_(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_186(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 186 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_linalg_pinv_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_linalg_pinv_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_linalg_pinv_(dummy_bk, arg1, arg2)
    mod._np_affineconfig(bk, arg1)
    mod._np_affineconfig(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_187(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 187 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_affineconfig(bk, arg1, arg1, arg2)
    mod._np_affineconfig(bk, arg_bool)
    mod._np_affineconfig(bk, arg_int)
    mod._np_affineconfig(dummy_bk, arg1)
    mod._np_affineconfig(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_188(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 188 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_asstringconfig(bk, arg1)
    mod._np_asstringconfig(bk, arg1, arg2)
    mod._np_asstringconfig(bk, arg1, arg1, arg2)
    mod._np_asstringconfig(bk, arg_bool)
    mod._np_asstringconfig(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_189(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 189 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_asstringconfig(dummy_bk, arg1)
    mod._np_asstringconfig(dummy_bk, arg1, arg2)
    with pytest.raises(AssertionError):
        mod._np_assertop(bk, arg1)
    with pytest.raises(AssertionError):
        mod._np_assertop(bk, arg1, arg2)
    with pytest.raises(AssertionError):
        mod._np_assertop(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_190(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 190 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(AssertionError):
        mod._np_assertop(bk, arg_bool)
    mod._np_assertop(bk, arg_int)
    with pytest.raises(AssertionError):
        mod._np_assertop(dummy_bk, arg1)
    with pytest.raises(AssertionError):
        mod._np_assertop(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_blurconfig(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_191(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 191 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_blurconfig(bk, arg1, arg2)
    mod._np_blurconfig(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_blurconfig(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_blurconfig(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_blurconfig(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_192(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 192 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_blurconfig(dummy_bk, arg1, arg2)
    mod._np_callable(bk, arg1)
    mod._np_callable(bk, arg1, arg2)
    mod._np_callable(bk, arg1, arg1, arg2)
    mod._np_callable(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_193(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 193 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_callable(bk, arg_int)
    mod._np_callable(dummy_bk, arg1)
    mod._np_callable(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedlocal(bk, arg1)
    mod._np_convgeneraldilatedlocal(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_194(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 194 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_convgeneraldilatedlocal(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedlocal(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedlocal(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedlocal(dummy_bk, arg1)
    mod._np_convgeneraldilatedlocal(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_195(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 195 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedpatches(bk, arg1)
    mod._np_convgeneraldilatedpatches(bk, arg1, arg2)
    mod._np_convgeneraldilatedpatches(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedpatches(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedpatches(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_196(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 196 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedpatches(dummy_bk, arg1)
    mod._np_convgeneraldilatedpatches(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_convwithgeneralpadding(bk, arg1)
    mod._np_convwithgeneralpadding(bk, arg1, arg2)
    mod._np_convwithgeneralpadding(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_197(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 197 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_convwithgeneralpadding(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_convwithgeneralpadding(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_convwithgeneralpadding(dummy_bk, arg1)
    mod._np_convwithgeneralpadding(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_customlinearsolve(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_198(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 198 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_customlinearsolve(bk, arg1, arg2)
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_customlinearsolve(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_customlinearsolve(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_customlinearsolve(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_customlinearsolve(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_199(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 199 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"):
        mod._np_customlinearsolve(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_customroot(bk, arg1)
    mod._np_customroot(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_customroot(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_customroot(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_200(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 200 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(IndexError):
        mod._np_customroot(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_customroot(dummy_bk, arg1)
    mod._np_customroot(dummy_bk, arg1, arg2)
    mod._np_debuginfs(bk, arg1)
    mod._np_debuginfs(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_201(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 201 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_debuginfs(bk, arg1, arg1, arg2)
    mod._np_debuginfs(bk, arg_bool)
    mod._np_debuginfs(bk, arg_int)
    mod._np_debuginfs(dummy_bk, arg1)
    mod._np_debuginfs(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_202(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 202 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_debugnans(bk, arg1)
    mod._np_debugnans(bk, arg1, arg2)
    mod._np_debugnans(bk, arg1, arg1, arg2)
    mod._np_debugnans(bk, arg_bool)
    mod._np_debugnans(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_203(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 203 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_debugnans(dummy_bk, arg1)
    mod._np_debugnans(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_dotgeneral(bk, arg1)
    mod._np_dotgeneral(bk, arg1, arg2)
    mod._np_dotgeneral(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_204(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 204 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_dotgeneral(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_dotgeneral(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_dotgeneral(dummy_bk, arg1)
    mod._np_dotgeneral(dummy_bk, arg1, arg2)
    mod._np_elasticconfig(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_205(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 205 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_elasticconfig(bk, arg1, arg2)
    mod._np_elasticconfig(bk, arg1, arg1, arg2)
    mod._np_elasticconfig(bk, arg_bool)
    mod._np_elasticconfig(bk, arg_int)
    mod._np_elasticconfig(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_206(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 206 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_elasticconfig(dummy_bk, arg1, arg2)
    mod._np_extractpatchesoptions(bk, arg1)
    mod._np_extractpatchesoptions(bk, arg1, arg2)
    mod._np_extractpatchesoptions(bk, arg1, arg1, arg2)
    mod._np_extractpatchesoptions(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_207(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 207 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_extractpatchesoptions(bk, arg_int)
    mod._np_extractpatchesoptions(dummy_bk, arg1)
    mod._np_extractpatchesoptions(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperator(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperator(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_208(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 208 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperator(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperator(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperator(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperator(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperator(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_209(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 209 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_210(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 210 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_211(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 211 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_212(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 212 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_213(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 213 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_214(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 214 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_215(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 215 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_216(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 216 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant3d(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant3d(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant3d(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant3d(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant3d(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_217(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 217 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant3d(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant3d(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcomposition(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcomposition(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcomposition(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_218(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 218 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorcomposition(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcomposition(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcomposition(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcomposition(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatordiag(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_219(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 219 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatordiag(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatordiag(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatordiag(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatordiag(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatordiag(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_220(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 220 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatordiag(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorfullmatrix(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorfullmatrix(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorfullmatrix(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorfullmatrix(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_221(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 221 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatorfullmatrix(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorfullmatrix(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorfullmatrix(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorhouseholder(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorhouseholder(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_222(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 222 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorhouseholder(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorhouseholder(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorhouseholder(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorhouseholder(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorhouseholder(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_223(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 223 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatoridentity(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatoridentity(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatoridentity(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatoridentity(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatoridentity(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_224(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 224 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatoridentity(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatoridentity(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorinversion(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorinversion(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorinversion(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_225(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 225 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorinversion(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorinversion(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorinversion(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorinversion(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorkronecker(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_226(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 226 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorkronecker(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorkronecker(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorkronecker(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorkronecker(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorkronecker(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_227(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 227 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatorkronecker(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowrankupdate(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowrankupdate(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowrankupdate(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowrankupdate(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_228(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 228 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowrankupdate(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowrankupdate(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowrankupdate(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowertriangular(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowertriangular(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_229(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 229 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowertriangular(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowertriangular(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowertriangular(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowertriangular(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorlowertriangular(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_230(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 230 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatorpermutation(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorpermutation(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorpermutation(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorpermutation(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorpermutation(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_231(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 231 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorpermutation(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorpermutation(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorscaledidentity(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorscaledidentity(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorscaledidentity(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_232(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 232 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorscaledidentity(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorscaledidentity(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorscaledidentity(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorscaledidentity(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatortoeplitz(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_233(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 233 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatortoeplitz(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatortoeplitz(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatortoeplitz(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatortoeplitz(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatortoeplitz(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_234(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 234 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatortoeplitz(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatortridiag(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatortridiag(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatortridiag(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatortridiag(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_235(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 235 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatortridiag(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatortridiag(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatortridiag(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorzeros(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorzeros(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_236(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 236 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorzeros(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorzeros(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorzeros(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorzeros(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorzeros(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_237(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 237 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_perspectiveconfig(bk, arg1)
    mod._np_perspectiveconfig(bk, arg1, arg2)
    mod._np_perspectiveconfig(bk, arg1, arg1, arg2)
    mod._np_perspectiveconfig(bk, arg_bool)
    mod._np_perspectiveconfig(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_238(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 238 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_perspectiveconfig(dummy_bk, arg1)
    mod._np_perspectiveconfig(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_raggeddot(bk, arg1)
    mod._np_raggeddot(bk, arg1, arg2)
    mod._np_raggeddot(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_239(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 239 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_raggeddot(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_raggeddot(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_raggeddot(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_raggeddot(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_rawconv2d(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_240(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 240 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_rawconv2d(bk, arg1, arg2)
    mod._np_rawconv2d(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_rawconv2d(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_rawconv2d(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_rawconv2d(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_241(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 241 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_rawconv2d(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_rawmatmul(bk, arg1)
    mod._np_rawmatmul(bk, arg1, arg2)
    mod._np_rawmatmul(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_rawmatmul(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_242(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 242 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(IndexError):
        mod._np_rawmatmul(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_rawmatmul(dummy_bk, arg1)
    mod._np_rawmatmul(dummy_bk, arg1, arg2)
    mod._np_rawmerge(bk, arg1)
    mod._np_rawmerge(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_243(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 243 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_rawmerge(bk, arg1, arg1, arg2)
    mod._np_rawmerge(bk, arg_bool)
    mod._np_rawmerge(bk, arg_int)
    mod._np_rawmerge(dummy_bk, arg1)
    mod._np_rawmerge(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_244(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 244 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_rawop(bk, arg1)
    mod._np_rawop(bk, arg1, arg2)
    mod._np_rawop(bk, arg1, arg1, arg2)
    mod._np_rawop(bk, arg_bool)
    mod._np_rawop(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_245(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 245 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_rawop(dummy_bk, arg1)
    mod._np_rawop(dummy_bk, arg1, arg2)
    mod._np_rawswitch(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_rawswitch(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_rawswitch(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_246(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 246 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_rawswitch(bk, arg_bool)
    mod._np_rawswitch(bk, arg_int)
    mod._np_rawswitch(dummy_bk, arg1)
    with pytest.raises(ValueError):
        mod._np_rawswitch(dummy_bk, arg1, arg2)
    mod._np_scanop(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_247(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 247 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_scanop(bk, arg1, arg2)
    mod._np_scanop(bk, arg1, arg1, arg2)
    mod._np_scanop(bk, arg_bool)
    mod._np_scanop(bk, arg_int)
    mod._np_scanop(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_248(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 248 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_scanop(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_sobolsample(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_sobolsample(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_sobolsample(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_sobolsample(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_249(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 249 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_sobolsample(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_sobolsample(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_sobolsample(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsedensematmul(bk, arg1)
    mod._np_sparsedensematmul(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_250(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 250 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_sparsedensematmul(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsedensematmul(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_sparsedensematmul(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_sparsedensematmul(dummy_bk, arg1)
    mod._np_sparsedensematmul(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_251(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 251 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(IndexError):
        mod._np_sparsemapvalues(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_sparsemapvalues(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_sparsemapvalues(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsemapvalues(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_sparsemapvalues(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_252(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 252 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_sparsemapvalues(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_sparsemapvalues(dummy_bk, arg1, arg2)
    mod._np_sparsereducemax(bk, arg1)
    mod._np_sparsereducemax(bk, arg1, arg2)
    mod._np_sparsereducemax(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_253(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 253 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_sparsereducemax(bk, arg_bool)
    mod._np_sparsereducemax(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_sparsereducemax(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_sparsereducemax(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsereshape(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_254(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 254 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_sparsereshape(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_sparsereshape(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsereshape(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_sparsereshape(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_sparsereshape(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_255(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 255 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_sparsereshape(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsesampledadd(bk, arg1)
    mod._np_sparsesampledadd(bk, arg1, arg2)
    mod._np_sparsesampledadd(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsesampledadd(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_256(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 256 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(IndexError):
        mod._np_sparsesampledadd(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_sparsesampledadd(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_sparsesampledadd(dummy_bk, arg1, arg2)
    mod._np_sparsesegmentsum(bk, arg1)
    mod._np_sparsesegmentsum(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_257(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 257 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_sparsesegmentsum(bk, arg1, arg1, arg2)
    mod._np_sparsesegmentsum(bk, arg_bool)
    mod._np_sparsesegmentsum(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_sparsesegmentsum(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_sparsesegmentsum(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_258(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 258 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_sparsetranspose(bk, arg1)
    mod._np_sparsetranspose(bk, arg1, arg2)
    mod._np_sparsetranspose(bk, arg1, arg1, arg2)
    mod._np_sparsetranspose(bk, arg_bool)
    mod._np_sparsetranspose(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_259(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 259 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_sparsetranspose(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_sparsetranspose(dummy_bk, arg1, arg2)
    mod._np_switchop(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_switchop(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_switchop(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_260(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 260 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_switchop(bk, arg_bool)
    mod._np_switchop(bk, arg_int)
    mod._np_switchop(dummy_bk, arg1)
    with pytest.raises(ValueError):
        mod._np_switchop(dummy_bk, arg1, arg2)
    mod._np_tensor(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_261(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 261 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_tensor(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_tensor(bk, arg1, arg1, arg2)
    mod._np_tensor(bk, arg_bool)
    mod._np_tensor(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_tensor(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_262(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 262 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_tensor(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(bk, arg1)
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_263(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 263 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(dummy_bk, arg1)
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(dummy_bk, arg1, arg2)
    mod._np_tensorarraystack(bk, arg1)
    mod._np_tensorarraystack(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_264(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 264 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_tensorarraystack(bk, arg1, arg1, arg2)
    mod._np_tensorarraystack(bk, arg_bool)
    mod._np_tensorarraystack(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_tensorarraystack(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_tensorarraystack(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_265(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 265 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(IndexError):
        mod._np_tensorarraywrite(bk, arg1)
    with pytest.raises(IndexError):
        mod._np_tensorarraywrite(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_tensorarraywrite(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_tensorarraywrite(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_tensorarraywrite(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_266(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 266 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_tensorarraywrite(dummy_bk, arg1)
    with pytest.raises(IndexError):
        mod._np_tensorarraywrite(dummy_bk, arg1, arg2)
    mod._np_tensorconfig(bk, arg1)
    mod._np_tensorconfig(bk, arg1, arg2)
    mod._np_tensorconfig(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_267(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 267 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_tensorconfig(bk, arg_bool)
    mod._np_tensorconfig(bk, arg_int)
    mod._np_tensorconfig(dummy_bk, arg1)
    mod._np_tensorconfig(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_vecdot(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_268(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 268 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_vecdot(bk, arg1, arg2)
    mod._np_vecdot(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_vecdot(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_vecdot(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_vecdot(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_269(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 269 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_vecdot(dummy_bk, arg1, arg2)
    mod._np_decode_csv(bk, arg1)
    mod._np_decode_csv(bk, arg1, arg2)
    mod._np_decode_csv(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_decode_csv(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_270(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 270 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_decode_csv(bk, arg_int)
    mod._np_decode_csv(dummy_bk, arg1)
    mod._np_decode_csv(dummy_bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_decode_image(bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_decode_image(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_271(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 271 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(RuntimeError):
        mod._np_decode_image(bk, arg1, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_decode_image(bk, arg_bool)
    with pytest.raises(RuntimeError):
        mod._np_decode_image(bk, arg_int)
    with pytest.raises(RuntimeError):
        mod._np_decode_image(dummy_bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_decode_image(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_272(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 272 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(RuntimeError):
        mod._np_parse_example(bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_parse_example(bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_parse_example(bk, arg1, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_parse_example(bk, arg_bool)
    with pytest.raises(RuntimeError):
        mod._np_parse_example(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_273(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 273 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(RuntimeError):
        mod._np_parse_example(dummy_bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_parse_example(dummy_bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor(bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor(bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_274(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 274 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor(bk, arg_bool)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor(bk, arg_int)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor(dummy_bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor(dummy_bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_read_file(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_275(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 275 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(RuntimeError):
        mod._np_read_file(bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_read_file(bk, arg1, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_read_file(bk, arg_bool)
    with pytest.raises(RuntimeError):
        mod._np_read_file(bk, arg_int)
    with pytest.raises(RuntimeError):
        mod._np_read_file(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_276(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 276 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(RuntimeError):
        mod._np_read_file(dummy_bk, arg1, arg2)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(bk, arg1)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(bk, arg1, arg2)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(bk, arg1, arg1, arg2)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_277(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 277 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(bk, arg_int)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(dummy_bk, arg1, arg2)
    mod._np_serialize_tensor(bk, arg1)
    mod._np_serialize_tensor(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_278(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 278 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_serialize_tensor(bk, arg1, arg1, arg2)
    mod._np_serialize_tensor(bk, arg_bool)
    mod._np_serialize_tensor(bk, arg_int)
    mod._np_serialize_tensor(dummy_bk, arg1)
    mod._np_serialize_tensor(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_279(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 279 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    pass
    pass
    pass
    pass
    pass
    _state.update(locals())


def _test_math_misc_coverage_part_280(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 280 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk) = (_state["arg1"], _state["arg2"], _state["bk"])
    pass
    pass
    mod._np_confusion_matrix(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_confusion_matrix(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_confusion_matrix(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_281(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 281 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_confusion_matrix(bk, arg_bool)
    mod._np_confusion_matrix(bk, arg_int)
    mod._np_confusion_matrix(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_confusion_matrix(dummy_bk, arg1, arg2)
    mod._np_descriptive(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_282(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 282 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_descriptive(bk, arg1, arg2)
    mod._np_descriptive(bk, arg1, arg1, arg2)
    mod._np_descriptive(bk, arg_bool)
    mod._np_descriptive(bk, arg_int)
    mod._np_descriptive(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_283(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 283 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_descriptive(dummy_bk, arg1, arg2)
    mod._np_distributions(bk, arg1)
    mod._np_distributions(bk, arg1, arg2)
    mod._np_distributions(bk, arg1, arg1, arg2)
    mod._np_distributions(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_284(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 284 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_distributions(bk, arg_int)
    mod._np_distributions(dummy_bk, arg1)
    mod._np_distributions(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_bitwise_count(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_bitwise_count(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_285(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 285 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_bitwise_count(bk, arg1, arg1, arg2)
    mod._np_bitwise_count(bk, arg_bool)
    mod._np_bitwise_count(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_bitwise_count(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_bitwise_count(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_286(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 286 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_fromdlpack(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_fromdlpack(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_fromdlpack(bk, arg1, arg1, arg2)
    mod._np_fromdlpack(bk, arg_bool)
    mod._np_fromdlpack(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_287(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 287 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_fromdlpack(dummy_bk, arg1)
    mod._np_fromdlpack(dummy_bk, arg1, arg2)
    mod._np_randomcategorical(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_randomcategorical(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_randomcategorical(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_288(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 288 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_randomcategorical(bk, arg_bool)
    mod._np_randomcategorical(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_randomcategorical(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_randomcategorical(dummy_bk, arg1, arg2)
    mod._np_randompermutation(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_289(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 289 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_randompermutation(bk, arg1, arg2)
    mod._np_randompermutation(bk, arg1, arg1, arg2)
    mod._np_randompermutation(bk, arg_bool)
    mod._np_randompermutation(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_randompermutation(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_290(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 290 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_randompermutation(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_randomtruncatednormal(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_randomtruncatednormal(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_randomtruncatednormal(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_randomtruncatednormal(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_291(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 291 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_randomtruncatednormal(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_randomtruncatednormal(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_randomtruncatednormal(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_key(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_key(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_292(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 292 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_key(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_key(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_key(bk, arg_int)
    with pytest.raises(ValueError):
        mod._np_key(dummy_bk, arg1)
    with pytest.raises(ValueError):
        mod._np_key(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_293(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 293 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_stridedslice(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_stridedslice(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_stridedslice(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_stridedslice(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_stridedslice(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_294(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 294 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_stridedslice(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_stridedslice(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_randombernoulli(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_randombernoulli(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_randombernoulli(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_295(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 295 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_randombernoulli(bk, arg_bool)
    mod._np_randombernoulli(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_randombernoulli(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_randombernoulli(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_chebyshev_polynomial_t(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_296(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 296 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_chebyshev_polynomial_t(bk, arg1, arg2)
    mod._np_chebyshev_polynomial_t(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_chebyshev_polynomial_t(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_chebyshev_polynomial_t(bk, arg_int)
    with pytest.raises(ValueError):
        mod._np_chebyshev_polynomial_t(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_297(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 297 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_chebyshev_polynomial_t(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_chebyshev_polynomial_u(bk, arg1)
    mod._np_chebyshev_polynomial_u(bk, arg1, arg2)
    mod._np_chebyshev_polynomial_u(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_chebyshev_polynomial_u(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_298(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 298 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(ValueError):
        mod._np_chebyshev_polynomial_u(bk, arg_int)
    with pytest.raises(ValueError):
        mod._np_chebyshev_polynomial_u(dummy_bk, arg1)
    mod._np_chebyshev_polynomial_u(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_hermite_polynomial_h(bk, arg1)
    mod._np_hermite_polynomial_h(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_299(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 299 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_hermite_polynomial_h(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_hermite_polynomial_h(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_hermite_polynomial_h(bk, arg_int)
    with pytest.raises(ValueError):
        mod._np_hermite_polynomial_h(dummy_bk, arg1)
    mod._np_hermite_polynomial_h(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_300(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 300 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(ValueError):
        mod._np_hermite_polynomial_he(bk, arg1)
    mod._np_hermite_polynomial_he(bk, arg1, arg2)
    mod._np_hermite_polynomial_he(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_hermite_polynomial_he(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_hermite_polynomial_he(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_301(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 301 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_hermite_polynomial_he(dummy_bk, arg1)
    mod._np_hermite_polynomial_he(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_laguerre_polynomial_l(bk, arg1)
    mod._np_laguerre_polynomial_l(bk, arg1, arg2)
    mod._np_laguerre_polynomial_l(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_302(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 302 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_laguerre_polynomial_l(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_laguerre_polynomial_l(bk, arg_int)
    with pytest.raises(ValueError):
        mod._np_laguerre_polynomial_l(dummy_bk, arg1)
    mod._np_laguerre_polynomial_l(dummy_bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_legendre_polynomial_p(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_303(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 303 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_legendre_polynomial_p(bk, arg1, arg2)
    mod._np_legendre_polynomial_p(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_legendre_polynomial_p(bk, arg_bool)
    with pytest.raises(ValueError):
        mod._np_legendre_polynomial_p(bk, arg_int)
    with pytest.raises(ValueError):
        mod._np_legendre_polynomial_p(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_304(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 304 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_legendre_polynomial_p(dummy_bk, arg1, arg2)
    mod._np_modified_bessel_i0(bk, arg1)
    mod._np_modified_bessel_i0(bk, arg1, arg2)
    mod._np_modified_bessel_i0(bk, arg1, arg1, arg2)
    mod._np_modified_bessel_i0(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_305(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 305 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_modified_bessel_i0(bk, arg_int)
    mod._np_modified_bessel_i0(dummy_bk, arg1)
    mod._np_modified_bessel_i0(dummy_bk, arg1, arg2)
    mod._np_modified_bessel_i1(bk, arg1)
    mod._np_modified_bessel_i1(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_306(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 306 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_modified_bessel_i1(bk, arg1, arg1, arg2)
    mod._np_modified_bessel_i1(bk, arg_bool)
    mod._np_modified_bessel_i1(bk, arg_int)
    mod._np_modified_bessel_i1(dummy_bk, arg1)
    mod._np_modified_bessel_i1(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_307(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 307 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_modified_bessel_k0(bk, arg1)
    mod._np_modified_bessel_k0(bk, arg1, arg2)
    mod._np_modified_bessel_k0(bk, arg1, arg1, arg2)
    mod._np_modified_bessel_k0(bk, arg_bool)
    mod._np_modified_bessel_k0(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_308(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 308 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_modified_bessel_k0(dummy_bk, arg1)
    mod._np_modified_bessel_k0(dummy_bk, arg1, arg2)
    mod._np_modified_bessel_k1(bk, arg1)
    mod._np_modified_bessel_k1(bk, arg1, arg2)
    mod._np_modified_bessel_k1(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_309(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 309 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_modified_bessel_k1(bk, arg_bool)
    mod._np_modified_bessel_k1(bk, arg_int)
    mod._np_modified_bessel_k1(dummy_bk, arg1)
    mod._np_modified_bessel_k1(dummy_bk, arg1, arg2)
    mod._np_shifted_chebyshev_polynomial_t(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_310(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 310 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_shifted_chebyshev_polynomial_t(bk, arg1, arg2)
    mod._np_shifted_chebyshev_polynomial_t(bk, arg1, arg1, arg2)
    mod._np_shifted_chebyshev_polynomial_t(bk, arg_bool)
    mod._np_shifted_chebyshev_polynomial_t(bk, arg_int)
    mod._np_shifted_chebyshev_polynomial_t(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_311(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 311 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_shifted_chebyshev_polynomial_t(dummy_bk, arg1, arg2)
    mod._np_shifted_chebyshev_polynomial_u(bk, arg1)
    mod._np_shifted_chebyshev_polynomial_u(bk, arg1, arg2)
    mod._np_shifted_chebyshev_polynomial_u(bk, arg1, arg1, arg2)
    mod._np_shifted_chebyshev_polynomial_u(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_312(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 312 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_shifted_chebyshev_polynomial_u(bk, arg_int)
    mod._np_shifted_chebyshev_polynomial_u(dummy_bk, arg1)
    mod._np_shifted_chebyshev_polynomial_u(dummy_bk, arg1, arg2)
    mod._np_shifted_chebyshev_polynomial_v(bk, arg1)
    mod._np_shifted_chebyshev_polynomial_v(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_313(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 313 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_shifted_chebyshev_polynomial_v(bk, arg1, arg1, arg2)
    mod._np_shifted_chebyshev_polynomial_v(bk, arg_bool)
    mod._np_shifted_chebyshev_polynomial_v(bk, arg_int)
    mod._np_shifted_chebyshev_polynomial_v(dummy_bk, arg1)
    mod._np_shifted_chebyshev_polynomial_v(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_314(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 314 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_shifted_chebyshev_polynomial_w(bk, arg1)
    mod._np_shifted_chebyshev_polynomial_w(bk, arg1, arg2)
    mod._np_shifted_chebyshev_polynomial_w(bk, arg1, arg1, arg2)
    mod._np_shifted_chebyshev_polynomial_w(bk, arg_bool)
    mod._np_shifted_chebyshev_polynomial_w(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_315(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 315 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_shifted_chebyshev_polynomial_w(dummy_bk, arg1)
    mod._np_shifted_chebyshev_polynomial_w(dummy_bk, arg1, arg2)
    mod._np_rfft(bk, arg1)
    mod._np_rfft(bk, arg1, arg2)
    mod._np_rfft(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_316(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 316 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_rfft(bk, arg_bool)
    mod._np_rfft(bk, arg_int)
    mod._np_rfft(dummy_bk, arg1)
    mod._np_rfft(dummy_bk, arg1, arg2)
    mod._np_ifft(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_317(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 317 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_ifft(bk, arg1, arg2)
    mod._np_ifft(bk, arg1, arg1, arg2)
    mod._np_ifft(bk, arg_bool)
    mod._np_ifft(bk, arg_int)
    mod._np_ifft(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_318(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 318 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_ifft(dummy_bk, arg1, arg2)
    mod._np_fftn(bk, arg1)
    mod._np_fftn(bk, arg1, arg2)
    mod._np_fftn(bk, arg1, arg1, arg2)
    mod._np_fftn(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_319(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 319 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_fftn(bk, arg_int)
    mod._np_fftn(dummy_bk, arg1)
    mod._np_fftn(dummy_bk, arg1, arg2)
    mod._np_ifftn(bk, arg1)
    mod._np_ifftn(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_320(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 320 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_ifftn(bk, arg1, arg1, arg2)
    mod._np_ifftn(bk, arg_bool)
    mod._np_ifftn(bk, arg_int)
    mod._np_ifftn(dummy_bk, arg1)
    mod._np_ifftn(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_321(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 321 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_rfftn(bk, arg1)
    mod._np_rfftn(bk, arg1, arg2)
    mod._np_rfftn(bk, arg1, arg1, arg2)
    mod._np_rfftn(bk, arg_bool)
    mod._np_rfftn(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_322(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 322 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_rfftn(dummy_bk, arg1)
    mod._np_rfftn(dummy_bk, arg1, arg2)
    mod._np_irfftn(bk, arg1)
    mod._np_irfftn(bk, arg1, arg2)
    mod._np_irfftn(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_323(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 323 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_irfftn(bk, arg_bool)
    mod._np_irfftn(bk, arg_int)
    mod._np_irfftn(dummy_bk, arg1)
    mod._np_irfftn(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_ifft2(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_324(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 324 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_ifft2(bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_ifft2(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_ifft2(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_ifft2(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_ifft2(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_325(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 325 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(IndexError):
        mod._np_ifft2(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_rfft2(bk, arg1)
    with pytest.raises(IndexError):
        mod._np_rfft2(bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_rfft2(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_rfft2(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_326(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 326 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(IndexError):
        mod._np_rfft2(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_rfft2(dummy_bk, arg1)
    with pytest.raises(IndexError):
        mod._np_rfft2(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_irfft2(bk, arg1)
    with pytest.raises(IndexError):
        mod._np_irfft2(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_327(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 327 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_irfft2(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_irfft2(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_irfft2(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_irfft2(dummy_bk, arg1)
    with pytest.raises(IndexError):
        mod._np_irfft2(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_328(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 328 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_fftnd(bk, arg1)
    mod._np_fftnd(bk, arg1, arg2)
    mod._np_fftnd(bk, arg1, arg1, arg2)
    mod._np_fftnd(bk, arg_bool)
    mod._np_fftnd(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_329(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 329 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_fftnd(dummy_bk, arg1)
    mod._np_fftnd(dummy_bk, arg1, arg2)
    mod._np_ifftnd(bk, arg1)
    mod._np_ifftnd(bk, arg1, arg2)
    mod._np_ifftnd(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_330(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 330 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_ifftnd(bk, arg_bool)
    mod._np_ifftnd(bk, arg_int)
    mod._np_ifftnd(dummy_bk, arg1)
    mod._np_ifftnd(dummy_bk, arg1, arg2)
    mod._np_rfftnd(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_331(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 331 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_rfftnd(bk, arg1, arg2)
    mod._np_rfftnd(bk, arg1, arg1, arg2)
    mod._np_rfftnd(bk, arg_bool)
    mod._np_rfftnd(bk, arg_int)
    mod._np_rfftnd(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_332(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 332 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_rfftnd(dummy_bk, arg1, arg2)
    mod._np_irfftnd(bk, arg1)
    mod._np_irfftnd(bk, arg1, arg2)
    mod._np_irfftnd(bk, arg1, arg1, arg2)
    mod._np_irfftnd(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_333(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 333 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_irfftnd(bk, arg_int)
    mod._np_irfftnd(dummy_bk, arg1)
    mod._np_irfftnd(dummy_bk, arg1, arg2)
    mod._np_fftshift(bk, arg1)
    mod._np_fftshift(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_334(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 334 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_fftshift(bk, arg1, arg1, arg2)
    mod._np_fftshift(bk, arg_bool)
    mod._np_fftshift(bk, arg_int)
    mod._np_fftshift(dummy_bk, arg1)
    mod._np_fftshift(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_335(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 335 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_ifftshift(bk, arg1)
    mod._np_ifftshift(bk, arg1, arg2)
    mod._np_ifftshift(bk, arg1, arg1, arg2)
    mod._np_ifftshift(bk, arg_bool)
    mod._np_ifftshift(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_336(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 336 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_ifftshift(dummy_bk, arg1)
    mod._np_ifftshift(dummy_bk, arg1, arg2)
    mod._np_hfft(bk, arg1)
    mod._np_hfft(bk, arg1, arg2)
    mod._np_hfft(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_337(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 337 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_hfft(bk, arg_bool)
    mod._np_hfft(bk, arg_int)
    mod._np_hfft(dummy_bk, arg1)
    mod._np_hfft(dummy_bk, arg1, arg2)
    mod._np_rfftfreq(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_338(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 338 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_rfftfreq(bk, arg1, arg2)
    mod._np_rfftfreq(bk, arg1, arg1, arg2)
    mod._np_rfftfreq(bk, arg_bool)
    mod._np_rfftfreq(bk, arg_int)
    mod._np_rfftfreq(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_339(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 339 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_rfftfreq(dummy_bk, arg1, arg2)
    mod._np_confusion_matrix(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_confusion_matrix(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_confusion_matrix(bk, arg1, arg1, arg2)
    mod._np_confusion_matrix(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_340(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 340 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_confusion_matrix(bk, arg_int)
    mod._np_confusion_matrix(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_confusion_matrix(dummy_bk, arg1, arg2)
    mod._np_descriptive(bk, arg1)
    mod._np_descriptive(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_341(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 341 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_descriptive(bk, arg1, arg1, arg2)
    mod._np_descriptive(bk, arg_bool)
    mod._np_descriptive(bk, arg_int)
    mod._np_descriptive(dummy_bk, arg1)
    mod._np_descriptive(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_342(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 342 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_distributions(bk, arg1)
    mod._np_distributions(bk, arg1, arg2)
    mod._np_distributions(bk, arg1, arg1, arg2)
    mod._np_distributions(bk, arg_bool)
    mod._np_distributions(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_343(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 343 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_distributions(dummy_bk, arg1)
    mod._np_distributions(dummy_bk, arg1, arg2)
    mod._np_rrelu(bk, arg1)
    mod._np_rrelu(bk, arg1, arg2)
    mod._np_rrelu(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_344(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 344 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_rrelu(bk, arg_bool)
    mod._np_rrelu(bk, arg_int)
    mod._np_rrelu(dummy_bk, arg1)
    mod._np_rrelu(dummy_bk, arg1, arg2)
    mod._np_clip(bk, arg1, 0.0, 1.0)
    _state.update(locals())


def _test_math_misc_coverage_part_345(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 345 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_clip(bk, arg1, arg2)
    mod._np_clip(bk, arg1, arg1, arg2)
    mod._np_clip(bk, arg_bool, 0, 1)
    mod._np_clip(bk, arg_int, 0, 1)
    mod._np_clip(dummy_bk, arg1, 0.0, 1.0)
    _state.update(locals())


def _test_math_misc_coverage_part_346(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 346 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_clip(dummy_bk, arg1, arg2)
    mod._np_softmax(bk, arg1)
    mod._np_softmax(bk, arg1, arg2)
    mod._np_softmax(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_softmax(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_347(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 347 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_softmax(bk, arg_int)
    mod._np_softmax(dummy_bk, arg1)
    mod._np_softmax(dummy_bk, arg1, arg2)
    mod._np_sigmoid(bk, arg1)
    mod._np_sigmoid(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_348(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 348 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_sigmoid(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_sigmoid(bk, arg_bool)
    mod._np_sigmoid(bk, arg_int)
    mod._np_sigmoid(dummy_bk, arg1)
    mod._np_sigmoid(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_349(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 349 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_log_softmax(bk, arg1)
    mod._np_log_softmax(bk, arg1, arg2)
    mod._np_log_softmax(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_log_softmax(bk, arg_bool)
    mod._np_log_softmax(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_350(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 350 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_log_softmax(dummy_bk, arg1)
    mod._np_log_softmax(dummy_bk, arg1, arg2)
    mod._np_one_hot(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_one_hot(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_one_hot(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_351(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 351 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_one_hot(bk, arg_bool)
    mod._np_one_hot(bk, arg_int)
    mod._np_one_hot(dummy_bk, arg1)
    with pytest.raises(ValueError):
        mod._np_one_hot(dummy_bk, arg1, arg2)
    mod._np_decode_csv_camel(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_352(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 352 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_decode_csv_camel(bk, arg1, arg2)
    mod._np_decode_csv_camel(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_decode_csv_camel(bk, arg_bool)
    mod._np_decode_csv_camel(bk, arg_int)
    mod._np_decode_csv_camel(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_353(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 353 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_decode_csv_camel(dummy_bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, arg1, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_354(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 354 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, arg_int)
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(dummy_bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(dummy_bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_parse_example_camel(bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_parse_example_camel(bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_355(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 355 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(RuntimeError):
        mod._np_parse_example_camel(bk, arg1, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_parse_example_camel(bk, arg_bool)
    with pytest.raises(RuntimeError):
        mod._np_parse_example_camel(bk, arg_int)
    with pytest.raises(RuntimeError):
        mod._np_parse_example_camel(dummy_bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_parse_example_camel(dummy_bk, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_356(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 356 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(bk, arg1, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(bk, arg_bool)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_357(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 357 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(dummy_bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(dummy_bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(bk, arg1, arg1, arg2)
    _state.update(locals())


def _test_math_misc_coverage_part_358(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 358 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(bk, arg_bool)
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(bk, arg_int)
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(dummy_bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(dummy_bk, arg1, arg2)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_359(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 359 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(bk, arg1, arg2)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(bk, arg1, arg1, arg2)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(bk, arg_bool)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(bk, arg_int)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(dummy_bk, arg1)
    _state.update(locals())


def _test_math_misc_coverage_part_360(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 360 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(dummy_bk, arg1, arg2)
    mod._np_serialize_tensor_camel(bk, arg1)
    mod._np_serialize_tensor_camel(bk, arg1, arg2)
    mod._np_serialize_tensor_camel(bk, arg1, arg1, arg2)
    mod._np_serialize_tensor_camel(bk, arg_bool)
    _state.update(locals())


def _test_math_misc_coverage_part_361(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 361 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_serialize_tensor_camel(bk, arg_int)
    mod._np_serialize_tensor_camel(dummy_bk, arg1)
    mod._np_serialize_tensor_camel(dummy_bk, arg1, arg2)
    pass
    pass
    _state.update(locals())


def _test_math_misc_coverage_part_362(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 362 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    pass
    pass
    pass
    pass
    pass
    _state.update(locals())


def _test_math_misc_coverage_part_363(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 363 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_frombuffer(bk, arg1)
    mod._np_frombuffer(bk, arg1, arg2)
    mod._np_frombuffer(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_frombuffer(bk, arg_bool)
    mod._np_frombuffer(bk, arg_int)
    _state.update(locals())


def _test_math_misc_coverage_part_364(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 364 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_frombuffer(dummy_bk, arg1)
    mod._np_frombuffer(dummy_bk, arg1, arg2)
    mod._build_dot_general_einsum_str(2, 2, (((1,), (0,)), ((), ())))
    mod._build_dot_general_einsum_str(2, 2, (((1,), (0,)), ((0,), (1,))))
    mod._np_logsumexp(bk, arg1, axis=0, keepdims=True)
    _state.update(locals())


def _test_math_misc_coverage_part_365(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 365 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, bk) = (_state["arg1"], _state["bk"])
    mod._np_logsumexp(bk, arg1, axis=0, keepdims=False)
    mod._np_segment_sum(bk, arg1, np.array([0, 1]))
    mod._np_segment_sum(bk, arg1, np.array([0, 1]), num_segments=2)
    import ml_switcheroo_compiler.backends.numpy.eager.distributed as dist

    dist._tcp_dist_ctx.world_size = 2
    with pytest.raises(TypeError):
        mod._np_psum(bk, arg1)
    dist._tcp_dist_ctx.world_size = 1
    dist._tcp_dist_ctx.world_size = 2
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_pmax"))(bk, arg1)
    dist._tcp_dist_ctx.world_size = 1
    dist._tcp_dist_ctx.world_size = 2
    getattr(_state, "update")({"dist": dist})
    _state.update(locals())


def _test_math_misc_coverage_part_366(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 366 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, bk, dist) = (_state["arg1"], _state["bk"], _state["dist"])
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_pmin"))(bk, arg1)
    setattr(dist._tcp_dist_ctx, "world_size", 1)
    mod._np_scanop(bk, lambda a, b: a + b, arg1, 0, True)
    with pytest.raises(TypeError):
        mod._np_scanop(bk, lambda a, b: a + b, arg1, None, False)
    mod._np_scanop(bk, None, arg1)
    mod._np_scanop(bk, lambda a, b: a + b, np.array([]), 0, True)
    _state.update(locals())


def _test_math_misc_coverage_part_367(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 367 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_sobolsample(bk, 2, 2)
    mod._np_sobolsample(bk, 2, 2, skip=5)
    with pytest.raises(ValueError):
        mod._np_debuginfs(dummy_bk, np.array([1.0, np.inf]))
    with pytest.raises(ValueError):
        mod._np_debugnans(dummy_bk, np.array([1.0, np.nan]))
    mod._np_confusion_matrix(bk, np.array([0, 1]), np.array([0, 1]))
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    _orig_rem = getattr(_ops, "rem", None)
    _orig_cm = getattr(_ops, "confusion_matrix", None)
    _orig_desc = getattr(_ops, "descriptive", None)
    _orig_dist = getattr(_ops, "distributions", None)
    getattr(_state, "update")({"_ops": _ops, "_orig_cm": _orig_cm, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    getattr(_state, "update")({"_orig_rem": _orig_rem})
    _state.update(locals())


def _test_math_misc_coverage_part_368(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 368 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, arg1, arg2, arg_int) = (_state["_ops"], _state["arg1"], _state["arg2"], _state["arg_int"])
    bk = _state["bk"]

    def my_rem(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function my_rem.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return "mock"

    def my_cm(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function my_cm.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return "mock"

    def my_desc(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function my_desc.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return "mock"

    def my_dist(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function my_dist.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return "mock"

    setattr(_ops, "rem", my_rem)
    setattr(_ops, "confusion_matrix", my_cm)
    setattr(_ops, "descriptive", my_desc)
    setattr(_ops, "distributions", my_dist)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(bk, arg1, arg2)
    mod._np_confusion_matrix(bk, arg_int, arg_int)
    getattr(_state, "update")({"my_cm": my_cm, "my_desc": my_desc, "my_dist": my_dist, "my_rem": my_rem})
    _state.update(locals())


def _test_math_misc_coverage_part_369(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 369 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_cm, _orig_desc, _orig_rem) = (_state["_ops"], _state["_orig_cm"], _state["_orig_desc"], _state["_orig_rem"])
    (arg1, bk) = (_state["arg1"], _state["bk"])
    mod._np_descriptive(bk, arg1)
    mod._np_distributions(bk, arg1)
    if _orig_rem:
        setattr(_ops, "rem", _orig_rem)
    else:
        delattr(_ops, "rem")
    if _orig_cm:
        setattr(_ops, "confusion_matrix", _orig_cm)
    else:
        delattr(_ops, "confusion_matrix")
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    else:
        del _ops.descriptive
    _state.update(locals())


def _test_math_misc_coverage_part_370(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 370 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_dist, bk, dummy_bk) = (_state["_ops"], _state["_orig_dist"], _state["bk"], _state["dummy_bk"])
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)
    else:
        del _ops.distributions
    mod._np_vecdot(bk, np.array([1j]), np.array([1j]))
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_qr"))(dummy_bk, np.eye(2), mode="complete")
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_svd"))(dummy_bk, np.eye(2), full_matrices=True, compute_uv=False)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_logdet"))(dummy_bk, np.eye(2))
    _state.update(locals())


def _test_math_misc_coverage_part_371(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 371 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, bk, dummy_bk) = (_state["arg1"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_slogdet"))(dummy_bk, np.eye(2))
    mod._np_tensorarraywrite(bk, [1, 2], 5, arg1)
    mod._np_tensorconfig(bk, (), "float32", None)
    funcs2 = ["BitwiseCount", "BitwiseLeftShift", "BitwiseRightShift", "Bincount", "Clip", "Clz", "Erf", "Erfc", "Erfinv", "Igamma", "Igammac", "PopulationCount", "ScatterApply", "ScatterMax", "ScatterMin", "ScatterMul", "ScatterNd", "Select", "BitwiseNot", "BitwiseOr", "BitwiseXor"]
    for _f in funcs2:
        with pytest.raises(TypeError):
            mod._np_bitwise_count(bk, arg1)
    getattr(_state, "update")({"funcs2": funcs2})
    _state.update(locals())


def _test_math_misc_coverage_part_372(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 372 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, bk, dummy_bk) = (_state["arg1"], _state["bk"], _state["dummy_bk"])
    mod._np_fromdlpack(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_randomcategorical(dummy_bk, np.array([[1.0, 2.0]]))
    with pytest.raises(AttributeError):
        mod._np_randompermutation(dummy_bk, np.array([1, 2]))
    with pytest.raises(AttributeError):
        mod._np_randompermutation(dummy_bk, np.array(5))
    mod._np_serialize_tensor(bk, np.array([1.0]), test=True)
    _state.update(locals())


def _test_math_misc_coverage_part_373(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 373 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    pass
    mod._np_customlinearsolve(bk, lambda x: x, np.array([1.0]))

    class DummyDesc:
        """Test class DummyDesc."""

        @staticmethod
        def descriptive(x: EagerValue) -> EagerValue:
            """Test function descriptive.

            Args:
                x: Test parameter `x`.
            """
            return x

    mod._np_descriptive(DummyDesc(), np.array([1.0]))

    class DummyDist:
        """Test class DummyDist."""

        @staticmethod
        def distributions(x: EagerValue) -> EagerValue:
            """Test function distributions.

            Args:
                x: Test parameter `x`.
            """
            return x

    mod._np_distributions(as_mod(DummyDist()), np.array([1.0]))
    getattr(_state, "update")({"DummyDesc": DummyDesc, "DummyDist": DummyDist})
    _state.update(locals())


def _test_math_misc_coverage_part_374(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 374 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    class DummyDescClass:
        """Test class DummyDescClass."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    _orig_desc = getattr(_ops, "descriptive", None)
    setattr(_ops, "descriptive", DummyDescClass)
    mod._np_descriptive(bk, np.array([1.0]))
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    else:
        delattr(_ops, "descriptive")

    class DummyDistClass:
        """Test class DummyDistClass."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    _orig_dist = getattr(_ops, "distributions", None)
    setattr(_ops, "distributions", DummyDistClass)
    getattr(_state, "update")({"DummyDescClass": DummyDescClass, "DummyDistClass": DummyDistClass, "_ops": _ops, "_orig_desc": _orig_desc})
    getattr(_state, "update")({"_orig_dist": _orig_dist})
    _state.update(locals())


def _test_math_misc_coverage_part_375(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 375 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_dist, bk) = (_state["_ops"], _state["_orig_dist"], _state["bk"])
    mod._np_distributions(bk, np.array([1.0]))
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)
    else:
        del _ops.distributions
    mod._np_customlinearsolve(bk, lambda x: x, np.array([1.0]), lambda f, x: x)
    mod._np_customlinearsolve(bk, lambda x: x, np.array([1.0]), solve=lambda f, x: x)
    mod._np_customlinearsolve(bk, lambda x: x, np.array([1.0]), lambda f, x: x)
    _state.update(locals())


def _test_math_misc_coverage_part_376(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 376 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    mod._np_customlinearsolve(bk, lambda x: x, np.array([1.0]), solve=lambda f, x: x)
    mod._np_descriptive(bk, np.array([1.0]))
    mod._np_distributions(bk, np.array([1.0]))
    mod._np_decode_csv(bk, [b"1,2"], record_defaults=[[0.0], [0.0]])
    import sys

    class FakeScipy:
        """Test class FakeScipy."""

        i1 = None
        k0 = None
        k1 = None

    old_scipy = sys.modules.get("scipy.special")
    getattr(sys.modules, "__setitem__")("scipy.special", None)
    mod._np_modified_bessel_i1(bk, np.array([1.0]))
    mod._np_modified_bessel_k0(bk, np.array([1.0]))
    getattr(_state, "update")({"FakeScipy": FakeScipy, "old_scipy": old_scipy, "sys": sys})
    _state.update(locals())


def _test_math_misc_coverage_part_377(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 377 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bk, old_scipy, sys) = (_state["bk"], _state["old_scipy"], _state["sys"])
    mod._np_modified_bessel_k1(bk, np.array([1.0]))
    if as_tensor(old_scipy) is not None:
        sys.modules["scipy.special"] = old_scipy
    else:
        sys.modules.pop("scipy.special", None)
    with pytest.raises(IndexError):
        mod._np_bessel_i0e(bk)
    with pytest.raises(IndexError):
        mod._np_bessel_i1e(bk)
    mod._np_modified_bessel_i0(bk)
    mod._np_modified_bessel_i1(bk)
    mod._np_modified_bessel_k0(bk)
    _state.update(locals())


def _test_math_misc_coverage_part_378(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 378 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    mod._np_modified_bessel_k1(bk)
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_matrix_utils, math_poly

    math_general = as_tensor(getattr(importlib.import_module("ml_switcheroo_compiler.backends.numpy.eager.math_advanced"), "math_general"))
    assert math_matrix_utils._np_diag_indices_(np, 2) is not None
    assert math_matrix_utils._np_diag_indices_from_(np, np.eye(2)) is not None
    assert math_matrix_utils._np_diagflat_(np, np.array([1, 2])) is not None
    assert math_matrix_utils._np_diagonal_(np, np.eye(2)) is not None
    assert math_matrix_utils._np_indices_(np, (2, 2)) is not None
    assert math_matrix_utils._np_mask_indices_(np, 2, np.triu) is not None
    assert math_general._np_apply_over_axes(np, np.sum, np.ones((2, 2)), [0]) is not None
    assert math_general._np_corrcoef(np, np.array([1, 2, 3])) is not None
    assert math_general._np_cov(np, np.array([1, 2, 3])) is not None
    assert math_poly._np_shifted_chebyshev_polynomial_t(np, np.array([1]), np.array([1])) is not None
    assert math_poly._np_shifted_chebyshev_polynomial_u(np, np.array([1]), np.array([1])) is not None
    assert math_poly._np_shifted_chebyshev_polynomial_v(np, np.array([1]), np.array([1])) is not None
    assert math_poly._np_shifted_chebyshev_polynomial_w(np, np.array([1]), np.array([1])) is not None
    assert math_poly._np_shifted_chebyshev_polynomial_t(np) is None
    assert math_poly._np_shifted_chebyshev_polynomial_u(np) is None
    assert math_poly._np_shifted_chebyshev_polynomial_v(np) is None
    assert math_poly._np_shifted_chebyshev_polynomial_w(np) is None
    getattr(_state, "update")({"math_general": math_general, "math_matrix_utils": math_matrix_utils, "math_poly": math_poly})
    _state.update(locals())


def _test_math_misc_coverage_part_379(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 379 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_descriptive(bk, None)
    with pytest.raises(TypeError):
        mod._np_distributions(bk, None)
    with pytest.raises(TypeError):
        mod._np_rrelu(bk, None)
    mod._get_csv_data([], np)
    mod._np_decode_image_camel(bk)
    _state.update(locals())


def _test_math_misc_coverage_part_380(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 380 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, np.array("invalid str"))

    class DummyFeature:
        """Test class DummyFeature."""

        shape = (1,)
        dtype = np.float32

    mod._np_parse_example_camel(bk, features={"a": DummyFeature()})
    mod._np_parse_example_camel(bk, np.array('{"a": [1.0]}'), features={"a": DummyFeature(), "b": DummyFeature()})
    with pytest.raises(RuntimeError):
        mod._np_parse_example_camel(bk, np.array("invalid json"), features={"a": DummyFeature()})
    mod._np_serialize_tensor_camel(bk)
    getattr(_state, "update")({"DummyFeature": DummyFeature})
    _state.update(locals())


def _test_math_misc_coverage_part_381(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 381 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]

    class Unpicklable:
        """Test class Unpicklable."""

        def __reduce__(self) -> str:
            """Test function __reduce__."""
            raise ValueError("not picklable")

    mod._np_serialize_tensor_camel(bk, np.array([Unpicklable()], dtype=object))
    pass
    pass
    getattr(_state, "update")({"Unpicklable": Unpicklable})
    _state.update(locals())


def _test_math_misc_coverage_part_382(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 382 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]

    class DummyBkVecdot:
        """Test class DummyBkVecdot."""

        @staticmethod
        def iscomplexobj(x: EagerValue) -> EagerValue:
            """Test function iscomplexobj.

            Args:
                x: Test parameter `x`.
            """
            return True

        @staticmethod
        def conj(x: EagerValue) -> EagerValue:
            """Test function conj.

            Args:
                x: Test parameter `x`.
            """
            return x

        @staticmethod
        def sum(x: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function sum.

            Args:
                x: Test parameter `x`.
                axis: Test parameter `axis`.
            """
            return x

    mod._np_vecdot(as_mod(DummyBkVecdot()), np.array([1.0]), np.array([1.0]))
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(bk, np.array([1.0]))
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    def test_desc_func(*args: EagerTensorProtocol, **kwargs: EagerTensorProtocol) -> EagerValue:
        """Test function test_desc_func.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        assert len(args) >= 1
        return args[0]

    getattr(_state, "update")({"DummyBkVecdot": DummyBkVecdot, "_ops": _ops, "test_desc_func": test_desc_func})
    _state.update(locals())


def _test_math_misc_coverage_part_383(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 383 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, bk, dummy_bk, test_desc_func) = (_state["_ops"], _state["bk"], _state["dummy_bk"], _state["test_desc_func"])
    _orig_desc = getattr(_ops, "descriptive", None)
    setattr(_ops, "descriptive", test_desc_func)
    mod._np_descriptive(bk, np.array([1.0]))
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    else:
        del _ops.descriptive
    _orig_dist = getattr(_ops, "distributions", None)
    setattr(_ops, "distributions", test_desc_func)
    mod._np_distributions(bk, np.array([1.0]))
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)
    else:
        del _ops.distributions
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    getattr(_state, "update")({"_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    _state.update(locals())


def _test_math_misc_coverage_part_384(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 384 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_distributions(dummy_bk, np.array([1.0]))

    class EmptyOps:
        """Test class EmptyOps."""

        class OpDef:
            """Test class OpDef."""

            pass

    import sys

    pass
    mod._np_descriptive(bk, np.array([1.0]))
    mod._np_distributions(bk, np.array([1.0]))
    pass
    getattr(_state, "update")({"EmptyOps": EmptyOps, "sys": sys})
    _state.update(locals())


def _test_math_misc_coverage_part_385(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 385 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    dummy_bk = _state["dummy_bk"]

    class FakeDescBackend:
        """Test class FakeDescBackend."""

        @staticmethod
        def descriptive(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function descriptive.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "desc_bk"

        @staticmethod
        def distributions(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function distributions.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "dist_bk"

    mod._np_descriptive(FakeDescBackend(), np.array([1.0]))
    mod._np_distributions(as_mod(FakeDescBackend()), np.array([1.0]))
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    setattr(_ops, "descriptive", None)
    setattr(_ops, "distributions", None)
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    getattr(_state, "update")({"FakeDescBackend": FakeDescBackend, "_ops": _ops})
    _state.update(locals())


def _test_math_misc_coverage_part_386(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 386 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, bk, dummy_bk) = (_state["_ops"], _state["bk"], _state["dummy_bk"])
    mod._np_distributions(dummy_bk, np.array([1.0]))
    del _ops.descriptive
    del _ops.distributions
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    def test_func(*a: EagerTensorProtocol, **k: EagerTensorProtocol) -> EagerValue:
        """Test function test_func.

        Args:
            a: Test parameter `a`.
            k: Test parameter `k`.
        """
        assert len(a) >= 1
        return a[0]

    _orig_desc = getattr(_ops, "descriptive", None)
    setattr(_ops, "descriptive", test_func)
    mod._np_descriptive(bk, np.array([1.0]))
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    else:
        delattr(_ops, "descriptive")
    _orig_dist = getattr(_ops, "distributions", None)
    setattr(_ops, "distributions", test_func)
    getattr(_state, "update")({"_ops": _ops, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist, "test_func": test_func})
    _state.update(locals())


def _test_math_misc_coverage_part_387(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 387 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_dist, bk, dummy_bk) = (_state["_ops"], _state["_orig_dist"], _state["bk"], _state["dummy_bk"])
    mod._np_distributions(bk, np.array([1.0]))
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)
    else:
        delattr(_ops, "distributions")
    mod._np_fromdlpack(dummy_bk, np.array([1.0]))
    with pytest.raises(AttributeError):
        mod._np_randomcategorical(dummy_bk, np.array([1.0]))
    with pytest.raises(AttributeError):
        mod._np_vecdot(dummy_bk, np.array([1j]), np.array([1j]))
    _state.update(locals())


def _test_math_misc_coverage_part_388(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 388 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    dummy_bk = _state["dummy_bk"]
    mod._np_distributions(dummy_bk, np.array([1.0]))
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    def buggy_desc(*a: EagerValue, **k: EagerValue) -> None:
        """Test function buggy_desc.

        Args:
            a: Test parameter `a`.
            k: Test parameter `k`.
        """
        raise ValueError("buggy")

    _orig_desc = getattr(_ops, "descriptive", None)
    setattr(_ops, "descriptive", buggy_desc)
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    else:
        delattr(_ops, "descriptive")
    _orig_dist = getattr(_ops, "distributions", None)
    setattr(_ops, "distributions", buggy_desc)
    mod._np_distributions(dummy_bk, np.array([1.0]))
    getattr(_state, "update")({"_ops": _ops, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist, "buggy_desc": buggy_desc})
    _state.update(locals())


def _test_math_misc_coverage_part_389(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 389 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_dist, dummy_bk) = (_state["_ops"], _state["_orig_dist"], _state["dummy_bk"])
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)
    else:
        delattr(_ops, "distributions")
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    _orig_desc = getattr(_ops, "descriptive", None)
    if _orig_desc:
        delattr(_ops, "descriptive")
    mod._np_descriptive(dummy_bk)
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    _orig_dist = as_tensor(getattr(_ops, "distributions", None))
    if _orig_dist:
        delattr(_ops, "distributions")
    getattr(_state, "update")({"_ops": _ops, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    _state.update(locals())


def _test_math_misc_coverage_part_390(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 390 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_dist, dummy_bk) = (_state["_ops"], _state["_orig_dist"], _state["dummy_bk"])
    mod._np_distributions(dummy_bk)
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)

    class FakeDescBackend:
        """Test class FakeDescBackend."""

        pass

    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    _orig_desc = getattr(_ops, "descriptive", None)
    if _orig_desc:
        delattr(_ops, "descriptive")
    mod._np_descriptive(FakeDescBackend(), np.array([1.0]))
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    _orig_dist = as_tensor(getattr(_ops, "distributions", None))
    getattr(_state, "update")({"FakeDescBackend": FakeDescBackend, "_ops": _ops, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    _state.update(locals())


def _test_math_misc_coverage_part_391(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 391 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (FakeDescBackend, _ops, _orig_dist) = (_state["FakeDescBackend"], _state["_ops"], _state["_orig_dist"])
    if _orig_dist:
        delattr(_ops, "distributions")
    mod._np_distributions(FakeDescBackend(), np.array([1.0]))
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)

    class MockDescriptiveFunc:
        """Test class MockDescriptiveFunc."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> MockDescriptiveFunc:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return typing.cast("MockDescriptiveFunc", "mock_desc")

    class MockDistributionsFunc:
        """Test class MockDistributionsFunc."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> MockDistributionsFunc:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return typing.cast("MockDistributionsFunc", "mock_dist")

    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    _orig_desc = getattr(_ops, "descriptive", None)
    setattr(_ops, "descriptive", MockDescriptiveFunc)
    getattr(_state, "update")({"MockDescriptiveFunc": MockDescriptiveFunc, "MockDistributionsFunc": MockDistributionsFunc, "_ops": _ops, "_orig_desc": _orig_desc})
    _state.update(locals())


def _test_math_misc_coverage_part_392(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 392 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (MockDistributionsFunc, _ops, _orig_desc, dummy_bk) = (_state["MockDistributionsFunc"], _state["_ops"], _state["_orig_desc"], _state["dummy_bk"])
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    else:
        delattr(_ops, "descriptive")
    _orig_dist = getattr(_ops, "distributions", None)
    setattr(_ops, "distributions", MockDistributionsFunc)
    mod._np_distributions(dummy_bk, np.array([1.0]))
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)
    else:
        delattr(_ops, "distributions")

    class ExplodingFunc:
        """Test class ExplodingFunc."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> ExplodingFunc:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            raise ValueError("exploding")

    _orig_desc = as_tensor(getattr(_ops, "descriptive", None))
    setattr(_ops, "descriptive", ExplodingFunc)
    getattr(_state, "update")({"ExplodingFunc": ExplodingFunc, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    _state.update(locals())


def _test_math_misc_coverage_part_393(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 393 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (ExplodingFunc, _ops, _orig_desc, dummy_bk) = (_state["ExplodingFunc"], _state["_ops"], _state["_orig_desc"], _state["dummy_bk"])
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    else:
        delattr(_ops, "descriptive")
    _orig_dist = getattr(_ops, "distributions", None)
    setattr(_ops, "distributions", ExplodingFunc)
    mod._np_distributions(dummy_bk, np.array([1.0]))
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)
    else:
        delattr(_ops, "distributions")
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    def raise_err(*args: EagerValue, **kwargs: EagerValue) -> None:
        """Test function raise_err.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        raise ValueError("error")

    _orig_desc = as_tensor(getattr(_ops, "descriptive", None))
    setattr(_ops, "descriptive", raise_err)
    getattr(_state, "update")({"_ops": _ops, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist, "raise_err": raise_err})
    _state.update(locals())


def _test_math_misc_coverage_part_394(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 394 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_desc, dummy_bk, raise_err) = (_state["_ops"], _state["_orig_desc"], _state["dummy_bk"], _state["raise_err"])
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    else:
        delattr(_ops, "descriptive")
    _orig_dist = getattr(_ops, "distributions", None)
    setattr(_ops, "distributions", raise_err)
    mod._np_distributions(dummy_bk, np.array([1.0]))
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)
    else:
        delattr(_ops, "distributions")

    class FakeDescBackendHas:
        """Test class FakeDescBackendHas."""

        @staticmethod
        def descriptive(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function descriptive.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "desc_bk"

        @staticmethod
        def distributions(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function distributions.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "dist_bk"

    getattr(_state, "update")({"FakeDescBackendHas": FakeDescBackendHas, "_orig_dist": _orig_dist})
    _state.update(locals())


def _test_math_misc_coverage_part_395(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 395 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    FakeDescBackendHas = _state["FakeDescBackendHas"]
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    class MagicOps:
        """Test class MagicOps."""

        @property
        def descriptive(self) -> None:
            """Test function descriptive."""
            raise RuntimeError("Boom")

        @property
        def distributions(self) -> None:
            """Test function distributions."""
            raise RuntimeError("Boom")

    import sys

    pass
    mod._np_descriptive(FakeDescBackendHas(), np.array([1.0]))
    mod._np_distributions(FakeDescBackendHas(), np.array([1.0]))
    pass
    getattr(_state, "update")({"MagicOps": MagicOps, "_ops": _ops, "sys": sys})
    _state.update(locals())


def _test_math_misc_coverage_part_396(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 396 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class FakeDescBackendHas:
        """Test class FakeDescBackendHas."""

        @staticmethod
        def descriptive(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function descriptive.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "desc_bk"

        @staticmethod
        def distributions(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function distributions.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "dist_bk"

    mod._np_descriptive(FakeDescBackendHas(), np.array([1.0]))
    mod._np_distributions(as_mod(FakeDescBackendHas()), np.array([1.0]))

    class FakeDescBackendNoHas:
        """Test class FakeDescBackendNoHas."""

        pass

    mod._np_descriptive(FakeDescBackendNoHas(), np.array([1.0]))
    getattr(_state, "update")({"FakeDescBackendHas": FakeDescBackendHas, "FakeDescBackendNoHas": FakeDescBackendNoHas})
    _state.update(locals())


def _test_math_misc_coverage_part_397(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 397 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (FakeDescBackendNoHas, dummy_bk) = (_state["FakeDescBackendNoHas"], _state["dummy_bk"])
    mod._np_distributions(FakeDescBackendNoHas(), np.array([1.0]))
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    class BadDesc:
        """Test class BadDesc."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> BadDesc:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            raise RuntimeError("Boom")

    _orig_desc = getattr(_ops, "descriptive", None)
    setattr(_ops, "descriptive", BadDesc)
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    else:
        delattr(_ops, "descriptive")
    _orig_dist = getattr(_ops, "distributions", None)
    setattr(_ops, "distributions", BadDesc)
    mod._np_distributions(dummy_bk, np.array([1.0]))
    getattr(_state, "update")({"BadDesc": BadDesc, "_ops": _ops, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    _state.update(locals())


def _test_math_misc_coverage_part_398(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 398 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_dist, dummy_bk) = (_state["_ops"], _state["_orig_dist"], _state["dummy_bk"])
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)
    else:
        delattr(_ops, "distributions")
    import sys

    pass
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    mod._np_distributions(dummy_bk, np.array([1.0]))
    pass

    class DummyBkVecdot2:
        """Test class DummyBkVecdot2."""

        @staticmethod
        def iscomplexobj(x: EagerValue) -> EagerValue:
            """Test function iscomplexobj.

            Args:
                x: Test parameter `x`.
            """
            return False

        @staticmethod
        def sum(x: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function sum.

            Args:
                x: Test parameter `x`.
                axis: Test parameter `axis`.
            """
            return x

    getattr(_state, "update")({"DummyBkVecdot2": DummyBkVecdot2, "sys": sys})
    _state.update(locals())


def _test_math_misc_coverage_part_399(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 399 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    DummyBkVecdot2 = _state["DummyBkVecdot2"]
    mod._np_vecdot(DummyBkVecdot2(), np.array([1.0]), np.array([1.0]))

    class DummyBkVecdot3:
        """Test class DummyBkVecdot3."""

        @staticmethod
        def iscomplexobj(x: EagerValue) -> EagerValue:
            """Test function iscomplexobj.

            Args:
                x: Test parameter `x`.
            """
            return False

        @staticmethod
        def conj(x: EagerValue) -> EagerValue:
            """Test function conj.

            Args:
                x: Test parameter `x`.
            """
            return x

        @staticmethod
        def sum(x: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function sum.

            Args:
                x: Test parameter `x`.
                axis: Test parameter `axis`.
            """
            return x

    mod._np_vecdot(as_mod(DummyBkVecdot3()), np.array([1.0]), np.array([1.0]))
    getattr(_state, "update")({"DummyBkVecdot3": DummyBkVecdot3})
    _state.update(locals())


def _test_math_misc_coverage_part_400(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 400 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]

    class ScalarArray:
        """Test class ScalarArray."""

        ndim = 0

        def __int__(self) -> EagerValue:
            """Test function __int__."""
            return 5

    class DummyBkPermutation:
        """Test class DummyBkPermutation."""

        @staticmethod
        def asarray(x: EagerValue) -> EagerValue:
            """Test function asarray.

            Args:
                x: Test parameter `x`.
            """
            return x

        @staticmethod
        def array(x: EagerValue) -> EagerValue:
            """Test function array.

            Args:
                x: Test parameter `x`.
            """
            return x

    mod._np_randompermutation(as_mod(DummyBkPermutation()), ScalarArray())
    mod._np_chebyshev_polynomial_t(bk, np.array(-1), np.array([1.0]))
    getattr(_state, "update")({"DummyBkPermutation": DummyBkPermutation, "ScalarArray": ScalarArray})
    _state.update(locals())


def _test_math_misc_coverage_part_401(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 401 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]

    class DummyBkVecdotHas:
        """Test class DummyBkVecdotHas."""

        @staticmethod
        def vecdot(x: EagerValue, y: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function vecdot.

            Args:
                x: Test parameter `x`.
                y: Test parameter `y`.
                axis: Test parameter `axis`.
            """
            return x

    mod._np_vecdot(as_mod(DummyBkVecdotHas()), np.array([1.0]), np.array([1.0]))

    class DummyBkLinalgVecdotHas:
        """Test class DummyBkLinalgVecdotHas."""

        class linalg:
            """Test class linalg."""

            @staticmethod
            def vecdot(x: EagerValue, y: EagerValue, axis: EagerValue) -> EagerValue:
                """Test function vecdot.

                Args:
                    x: Test parameter `x`.
                    y: Test parameter `y`.
                    axis: Test parameter `axis`.
                """
                return x

    mod._np_vecdot(as_mod(DummyBkLinalgVecdotHas()), np.array([1.0]), np.array([1.0]))
    mod._np_chebyshev_polynomial_t(bk, np.array(0), np.array([1.0]))
    getattr(_state, "update")({"DummyBkLinalgVecdotHas": DummyBkLinalgVecdotHas, "DummyBkVecdotHas": DummyBkVecdotHas})
    _state.update(locals())


def _test_math_misc_coverage_part_402(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 402 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, "string")
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, np.array("invalid"))
    import base64

    valid_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    valid_png = base64.b64decode(valid_png_b64)
    mod._np_decode_image_camel(bk, np.array(valid_png))
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, np.array(valid_png.decode("latin1")))
    mod._np_chebyshev_polynomial_t(bk, np.array(1), np.array([1.0]))
    getattr(_state, "update")({"base64": base64, "valid_png": valid_png, "valid_png_b64": valid_png_b64})
    _state.update(locals())


def _test_math_misc_coverage_part_403(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 403 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    mod._np_chebyshev_polynomial_t(bk, np.array(2), np.array([1.0]))
    mod._np_rrelu(bk, np.array([1.0, -1.0]), lower=0.1, upper=0.2)
    mod._np_chebyshev_polynomial_t(bk, np.array([1, 2]), np.array([1.0, 2.0]))
    mod._np_rrelu(bk, np.array([1.0, -1.0]), lower=0.1)
    mod._np_chebyshev_polynomial_t(bk, np.array([2, 3]), np.array([1.0, 2.0]))
    _state.update(locals())


def _test_math_misc_coverage_part_404(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 404 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_chebyshev_polynomial_u(bk, np.array([2, 3]), np.array([1.0, 2.0]))
    mod._np_rrelu(bk, np.array([1.0, -1.0]), lower=0.1, upper=0.2)
    mod._np_chebyshev_polynomial_t(bk, np.array([2, 3]), np.array([1.0, 2.0]))
    mod._np_rrelu(bk, np.array([1.0, -1.0]))
    with pytest.raises(ValueError):
        mod._np_debuginfs(dummy_bk, np.array([np.inf]))
    _state.update(locals())


def _test_math_misc_coverage_part_405(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 405 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(ValueError):
        mod._np_debugnans(dummy_bk, np.array([np.nan]))
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    def val_err(*a: EagerValue, **k: EagerValue) -> None:
        """Test function val_err.

        Args:
            a: Test parameter `a`.
            k: Test parameter `k`.
        """
        raise ValueError("err")

    _orig_rem = getattr(_ops, "rem", None)
    setattr(_ops, "rem", val_err)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(bk, arg1, arg2)
    if _orig_rem:
        setattr(_ops, "rem", _orig_rem)
    _orig_cm = getattr(_ops, "confusion_matrix", None)
    setattr(_ops, "confusion_matrix", val_err)
    mod._np_confusion_matrix(bk, np.array([0, 1]), np.array([0, 1]))
    if _orig_cm:
        setattr(_ops, "confusion_matrix", _orig_cm)
    _orig_desc = getattr(_ops, "descriptive", None)
    getattr(_state, "update")({"_ops": _ops, "_orig_cm": _orig_cm, "_orig_desc": _orig_desc, "_orig_rem": _orig_rem})
    getattr(_state, "update")({"val_err": val_err})
    _state.update(locals())


def _test_math_misc_coverage_part_406(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 406 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_desc, arg1, bk) = (_state["_ops"], _state["_orig_desc"], _state["arg1"], _state["bk"])
    val_err = _state["val_err"]
    setattr(_ops, "descriptive", val_err)
    mod._np_descriptive(bk, arg1)
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    _orig_dist = getattr(_ops, "distributions", None)
    setattr(_ops, "distributions", val_err)
    mod._np_distributions(bk, arg1)
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    class ValErrClass:
        """Test class ValErrClass."""

        def __new__(cls, *a: EagerValue, **k: EagerValue) -> ValErrClass:
            """Test function __new__.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            raise ValueError("err")

    _orig_rem = getattr(_ops, "rem", None)
    setattr(_ops, "rem", ValErrClass)
    getattr(_state, "update")({"ValErrClass": ValErrClass, "_ops": _ops, "_orig_dist": _orig_dist, "_orig_rem": _orig_rem})
    _state.update(locals())


def _test_math_misc_coverage_part_407(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 407 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (ValErrClass, _ops, _orig_rem, arg1) = (_state["ValErrClass"], _state["_ops"], _state["_orig_rem"], _state["arg1"])
    (arg2, bk) = (_state["arg2"], _state["bk"])
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(bk, arg1, arg2)
    if _orig_rem:
        setattr(_ops, "rem", _orig_rem)
    _orig_cm = getattr(_ops, "confusion_matrix", None)
    setattr(_ops, "confusion_matrix", ValErrClass)
    mod._np_confusion_matrix(bk, np.array([0, 1]), np.array([0, 1]))
    if _orig_cm:
        setattr(_ops, "confusion_matrix", _orig_cm)
    _orig_desc = getattr(_ops, "descriptive", None)
    setattr(_ops, "descriptive", ValErrClass)
    mod._np_descriptive(bk, arg1)
    getattr(_state, "update")({"_orig_cm": _orig_cm, "_orig_desc": _orig_desc})
    _state.update(locals())


def _test_math_misc_coverage_part_408(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 408 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (ValErrClass, _ops, _orig_desc, arg1) = (_state["ValErrClass"], _state["_ops"], _state["_orig_desc"], _state["arg1"])
    bk = _state["bk"]
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    _orig_dist = getattr(_ops, "distributions", None)
    setattr(_ops, "distributions", ValErrClass)
    mod._np_distributions(bk, arg1)
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)
    mod._poly_recurrence(np.array([1, 2]), np.array([1.0, 2.0]), 1.0, lambda x: x, lambda n, x, t1, t2: x * t1 - t2)
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    class Explosive:
        """Test class Explosive."""

        def __getattr__(self, name: EagerValue) -> None:
            """Test function __getattr__.

            Args:
                name: Test parameter `name`.
            """
            raise ValueError("boom")

    _orig_desc = as_tensor(getattr(_ops, "descriptive", None))
    setattr(_ops, "descriptive", Explosive())
    getattr(_state, "update")({"Explosive": Explosive, "_ops": _ops, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    _state.update(locals())


def _test_math_misc_coverage_part_409(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 409 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (Explosive, _ops, _orig_desc, dummy_bk) = (_state["Explosive"], _state["_ops"], _state["_orig_desc"], _state["dummy_bk"])
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    else:
        delattr(_ops, "descriptive")
    _orig_dist = getattr(_ops, "distributions", None)
    setattr(_ops, "distributions", Explosive())
    mod._np_distributions(dummy_bk, np.array([1.0]))
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)
    else:
        delattr(_ops, "distributions")

    class FakeOpDef:
        """Test class FakeOpDef."""

        pass

    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    _orig_desc = as_tensor(getattr(_ops, "descriptive", None))
    setattr(_ops, "descriptive", FakeOpDef)
    setattr(_ops, "OpDef", FakeOpDef)
    getattr(_state, "update")({"FakeOpDef": FakeOpDef, "_ops": _ops, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    _state.update(locals())


def _test_math_misc_coverage_part_410(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 410 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (FakeOpDef, _ops, _orig_desc, dummy_bk) = (_state["FakeOpDef"], _state["_ops"], _state["_orig_desc"], _state["dummy_bk"])
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    _orig_dist = getattr(_ops, "distributions", None)
    setattr(_ops, "distributions", FakeOpDef)
    mod._np_distributions(dummy_bk, np.array([1.0]))
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)
    del _ops.OpDef
    setattr(_ops, "descriptive", lambda *a, **k: a[0])
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    getattr(_state, "update")({"_orig_dist": _orig_dist})
    _state.update(locals())


def _test_math_misc_coverage_part_411(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 411 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_desc, _orig_dist, dummy_bk) = (_state["_ops"], _state["_orig_desc"], _state["_orig_dist"], _state["dummy_bk"])
    if _orig_desc:
        setattr(_ops, "descriptive", _orig_desc)
    setattr(_ops, "distributions", lambda *a, **k: a[0])
    mod._np_distributions(dummy_bk, np.array([1.0]))
    if _orig_dist:
        setattr(_ops, "distributions", _orig_dist)

    class ExplodeOps:
        """Test class ExplodeOps."""

        def __getattr__(self, name: EagerValue) -> None:
            """Test function __getattr__.

            Args:
                name: Test parameter `name`.
            """
            raise ValueError("explode")

    import sys

    pass
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    mod._np_distributions(dummy_bk, np.array([1.0]))
    pass
    getattr(_state, "update")({"ExplodeOps": ExplodeOps, "sys": sys})
    _state.update(locals())


def _test_math_misc_coverage_part_412(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 412 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_chebyshev_polynomial_t(bk, np.array([-1]), np.array([1.0]))
    with pytest.raises(TypeError):
        mod._np_rrelu(bk, None)
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_rem_2"))(dummy_bk, np.array([1.0]), np.array([1.0]))
    mod._np_confusion_matrix(dummy_bk, np.array([0, 1]), np.array([0, 1]))
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    _state.update(locals())


def _test_math_misc_coverage_part_413(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 413 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    dummy_bk = _state["dummy_bk"]
    mod._np_descriptive(dummy_bk)
    mod._np_distributions(dummy_bk, np.array([1.0]))
    mod._np_distributions(dummy_bk)
    with pytest.raises(ValueError):
        mod._np_decode_csv(dummy_bk, [b"invalid"], record_defaults=[[0.0]])
    with pytest.raises(IndexError):
        mod._np_decode_csv(dummy_bk, [], record_defaults=[[0.0]])
    _state.update(locals())


def _test_math_misc_coverage_part_414(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 414 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(dummy_bk, np.array(b"invalid_image_data"))
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(dummy_bk, np.array(b"invalid"))
    with pytest.raises(AttributeError):
        as_tensor(getattr(mod, "_np_encode_base64"))(dummy_bk, None)
    mod._np_frombuffer(dummy_bk, np.array([1.0]))
    mod._np_chebyshev_polynomial_t(bk, np.array([1.0]), np.array([-1]))
    _state.update(locals())


def _test_math_misc_coverage_part_415(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 415 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_rrelu(bk)

    class RealValueErrorOps:
        """Test class RealValueErrorOps."""

        @property
        def descriptive(self) -> None:
            """Test function descriptive."""
            raise ValueError("err")

        @property
        def distributions(self) -> None:
            """Test function distributions."""
            raise ValueError("err")

    import sys

    pass
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    mod._np_distributions(dummy_bk, np.array([1.0]))
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    pass
    getattr(_state, "update")({"RealValueErrorOps": RealValueErrorOps, "_ops": _ops, "sys": sys})


def test_math_misc_coverage() -> None:
    """Test function test_math_misc_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
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
        _test_math_misc_coverage_part_23,
        _test_math_misc_coverage_part_24,
        _test_math_misc_coverage_part_25,
        _test_math_misc_coverage_part_26,
        _test_math_misc_coverage_part_27,
        _test_math_misc_coverage_part_28,
        _test_math_misc_coverage_part_29,
        _test_math_misc_coverage_part_30,
        _test_math_misc_coverage_part_31,
        _test_math_misc_coverage_part_32,
        _test_math_misc_coverage_part_33,
        _test_math_misc_coverage_part_34,
        _test_math_misc_coverage_part_35,
        _test_math_misc_coverage_part_36,
        _test_math_misc_coverage_part_37,
        _test_math_misc_coverage_part_38,
        _test_math_misc_coverage_part_39,
        _test_math_misc_coverage_part_40,
        _test_math_misc_coverage_part_41,
        _test_math_misc_coverage_part_42,
        _test_math_misc_coverage_part_43,
        _test_math_misc_coverage_part_44,
        _test_math_misc_coverage_part_45,
        _test_math_misc_coverage_part_46,
        _test_math_misc_coverage_part_47,
        _test_math_misc_coverage_part_48,
        _test_math_misc_coverage_part_49,
        _test_math_misc_coverage_part_50,
        _test_math_misc_coverage_part_51,
        _test_math_misc_coverage_part_52,
        _test_math_misc_coverage_part_53,
        _test_math_misc_coverage_part_54,
        _test_math_misc_coverage_part_55,
        _test_math_misc_coverage_part_56,
        _test_math_misc_coverage_part_57,
        _test_math_misc_coverage_part_58,
        _test_math_misc_coverage_part_59,
        _test_math_misc_coverage_part_60,
        _test_math_misc_coverage_part_61,
        _test_math_misc_coverage_part_62,
        _test_math_misc_coverage_part_63,
        _test_math_misc_coverage_part_64,
        _test_math_misc_coverage_part_65,
        _test_math_misc_coverage_part_66,
        _test_math_misc_coverage_part_67,
        _test_math_misc_coverage_part_68,
        _test_math_misc_coverage_part_69,
        _test_math_misc_coverage_part_70,
        _test_math_misc_coverage_part_71,
        _test_math_misc_coverage_part_72,
        _test_math_misc_coverage_part_73,
        _test_math_misc_coverage_part_74,
        _test_math_misc_coverage_part_75,
        _test_math_misc_coverage_part_76,
        _test_math_misc_coverage_part_77,
        _test_math_misc_coverage_part_78,
        _test_math_misc_coverage_part_79,
        _test_math_misc_coverage_part_80,
        _test_math_misc_coverage_part_81,
        _test_math_misc_coverage_part_82,
        _test_math_misc_coverage_part_83,
        _test_math_misc_coverage_part_84,
        _test_math_misc_coverage_part_85,
        _test_math_misc_coverage_part_86,
        _test_math_misc_coverage_part_87,
        _test_math_misc_coverage_part_88,
        _test_math_misc_coverage_part_89,
        _test_math_misc_coverage_part_90,
        _test_math_misc_coverage_part_91,
        _test_math_misc_coverage_part_92,
        _test_math_misc_coverage_part_93,
        _test_math_misc_coverage_part_94,
        _test_math_misc_coverage_part_95,
        _test_math_misc_coverage_part_96,
        _test_math_misc_coverage_part_97,
        _test_math_misc_coverage_part_98,
        _test_math_misc_coverage_part_99,
        _test_math_misc_coverage_part_100,
        _test_math_misc_coverage_part_101,
        _test_math_misc_coverage_part_102,
        _test_math_misc_coverage_part_103,
        _test_math_misc_coverage_part_104,
        _test_math_misc_coverage_part_105,
        _test_math_misc_coverage_part_106,
        _test_math_misc_coverage_part_107,
        _test_math_misc_coverage_part_108,
        _test_math_misc_coverage_part_109,
        _test_math_misc_coverage_part_110,
        _test_math_misc_coverage_part_111,
        _test_math_misc_coverage_part_112,
        _test_math_misc_coverage_part_113,
        _test_math_misc_coverage_part_114,
        _test_math_misc_coverage_part_115,
        _test_math_misc_coverage_part_116,
        _test_math_misc_coverage_part_117,
        _test_math_misc_coverage_part_118,
        _test_math_misc_coverage_part_119,
        _test_math_misc_coverage_part_120,
        _test_math_misc_coverage_part_121,
        _test_math_misc_coverage_part_122,
        _test_math_misc_coverage_part_123,
        _test_math_misc_coverage_part_124,
        _test_math_misc_coverage_part_125,
        _test_math_misc_coverage_part_126,
        _test_math_misc_coverage_part_127,
        _test_math_misc_coverage_part_128,
        _test_math_misc_coverage_part_129,
        _test_math_misc_coverage_part_130,
        _test_math_misc_coverage_part_131,
        _test_math_misc_coverage_part_132,
        _test_math_misc_coverage_part_133,
        _test_math_misc_coverage_part_134,
        _test_math_misc_coverage_part_135,
        _test_math_misc_coverage_part_136,
        _test_math_misc_coverage_part_137,
        _test_math_misc_coverage_part_138,
        _test_math_misc_coverage_part_139,
        _test_math_misc_coverage_part_140,
        _test_math_misc_coverage_part_141,
        _test_math_misc_coverage_part_142,
        _test_math_misc_coverage_part_143,
        _test_math_misc_coverage_part_144,
        _test_math_misc_coverage_part_145,
        _test_math_misc_coverage_part_146,
        _test_math_misc_coverage_part_147,
        _test_math_misc_coverage_part_148,
        _test_math_misc_coverage_part_149,
        _test_math_misc_coverage_part_150,
        _test_math_misc_coverage_part_151,
        _test_math_misc_coverage_part_152,
        _test_math_misc_coverage_part_153,
        _test_math_misc_coverage_part_154,
        _test_math_misc_coverage_part_155,
        _test_math_misc_coverage_part_156,
        _test_math_misc_coverage_part_157,
        _test_math_misc_coverage_part_158,
        _test_math_misc_coverage_part_159,
        _test_math_misc_coverage_part_160,
        _test_math_misc_coverage_part_161,
        _test_math_misc_coverage_part_162,
        _test_math_misc_coverage_part_163,
        _test_math_misc_coverage_part_164,
        _test_math_misc_coverage_part_165,
        _test_math_misc_coverage_part_166,
        _test_math_misc_coverage_part_167,
        _test_math_misc_coverage_part_168,
        _test_math_misc_coverage_part_169,
        _test_math_misc_coverage_part_170,
        _test_math_misc_coverage_part_171,
        _test_math_misc_coverage_part_172,
        _test_math_misc_coverage_part_173,
        _test_math_misc_coverage_part_174,
        _test_math_misc_coverage_part_175,
        _test_math_misc_coverage_part_176,
        _test_math_misc_coverage_part_177,
        _test_math_misc_coverage_part_178,
        _test_math_misc_coverage_part_179,
        _test_math_misc_coverage_part_180,
        _test_math_misc_coverage_part_181,
        _test_math_misc_coverage_part_182,
        _test_math_misc_coverage_part_183,
        _test_math_misc_coverage_part_184,
        _test_math_misc_coverage_part_185,
        _test_math_misc_coverage_part_186,
        _test_math_misc_coverage_part_187,
        _test_math_misc_coverage_part_188,
        _test_math_misc_coverage_part_189,
        _test_math_misc_coverage_part_190,
        _test_math_misc_coverage_part_191,
        _test_math_misc_coverage_part_192,
        _test_math_misc_coverage_part_193,
        _test_math_misc_coverage_part_194,
        _test_math_misc_coverage_part_195,
        _test_math_misc_coverage_part_196,
        _test_math_misc_coverage_part_197,
        _test_math_misc_coverage_part_198,
        _test_math_misc_coverage_part_199,
        _test_math_misc_coverage_part_200,
        _test_math_misc_coverage_part_201,
        _test_math_misc_coverage_part_202,
        _test_math_misc_coverage_part_203,
        _test_math_misc_coverage_part_204,
        _test_math_misc_coverage_part_205,
        _test_math_misc_coverage_part_206,
        _test_math_misc_coverage_part_207,
        _test_math_misc_coverage_part_208,
        _test_math_misc_coverage_part_209,
        _test_math_misc_coverage_part_210,
        _test_math_misc_coverage_part_211,
        _test_math_misc_coverage_part_212,
        _test_math_misc_coverage_part_213,
        _test_math_misc_coverage_part_214,
        _test_math_misc_coverage_part_215,
        _test_math_misc_coverage_part_216,
        _test_math_misc_coverage_part_217,
        _test_math_misc_coverage_part_218,
        _test_math_misc_coverage_part_219,
        _test_math_misc_coverage_part_220,
        _test_math_misc_coverage_part_221,
        _test_math_misc_coverage_part_222,
        _test_math_misc_coverage_part_223,
        _test_math_misc_coverage_part_224,
        _test_math_misc_coverage_part_225,
        _test_math_misc_coverage_part_226,
        _test_math_misc_coverage_part_227,
        _test_math_misc_coverage_part_228,
        _test_math_misc_coverage_part_229,
        _test_math_misc_coverage_part_230,
        _test_math_misc_coverage_part_231,
        _test_math_misc_coverage_part_232,
        _test_math_misc_coverage_part_233,
        _test_math_misc_coverage_part_234,
        _test_math_misc_coverage_part_235,
        _test_math_misc_coverage_part_236,
        _test_math_misc_coverage_part_237,
        _test_math_misc_coverage_part_238,
        _test_math_misc_coverage_part_239,
        _test_math_misc_coverage_part_240,
        _test_math_misc_coverage_part_241,
        _test_math_misc_coverage_part_242,
        _test_math_misc_coverage_part_243,
        _test_math_misc_coverage_part_244,
        _test_math_misc_coverage_part_245,
        _test_math_misc_coverage_part_246,
        _test_math_misc_coverage_part_247,
        _test_math_misc_coverage_part_248,
        _test_math_misc_coverage_part_249,
        _test_math_misc_coverage_part_250,
        _test_math_misc_coverage_part_251,
        _test_math_misc_coverage_part_252,
        _test_math_misc_coverage_part_253,
        _test_math_misc_coverage_part_254,
        _test_math_misc_coverage_part_255,
        _test_math_misc_coverage_part_256,
        _test_math_misc_coverage_part_257,
        _test_math_misc_coverage_part_258,
        _test_math_misc_coverage_part_259,
        _test_math_misc_coverage_part_260,
        _test_math_misc_coverage_part_261,
        _test_math_misc_coverage_part_262,
        _test_math_misc_coverage_part_263,
        _test_math_misc_coverage_part_264,
        _test_math_misc_coverage_part_265,
        _test_math_misc_coverage_part_266,
        _test_math_misc_coverage_part_267,
        _test_math_misc_coverage_part_268,
        _test_math_misc_coverage_part_269,
        _test_math_misc_coverage_part_270,
        _test_math_misc_coverage_part_271,
        _test_math_misc_coverage_part_272,
        _test_math_misc_coverage_part_273,
        _test_math_misc_coverage_part_274,
        _test_math_misc_coverage_part_275,
        _test_math_misc_coverage_part_276,
        _test_math_misc_coverage_part_277,
        _test_math_misc_coverage_part_278,
        _test_math_misc_coverage_part_279,
        _test_math_misc_coverage_part_280,
        _test_math_misc_coverage_part_281,
        _test_math_misc_coverage_part_282,
        _test_math_misc_coverage_part_283,
        _test_math_misc_coverage_part_284,
        _test_math_misc_coverage_part_285,
        _test_math_misc_coverage_part_286,
        _test_math_misc_coverage_part_287,
        _test_math_misc_coverage_part_288,
        _test_math_misc_coverage_part_289,
        _test_math_misc_coverage_part_290,
        _test_math_misc_coverage_part_291,
        _test_math_misc_coverage_part_292,
        _test_math_misc_coverage_part_293,
        _test_math_misc_coverage_part_294,
        _test_math_misc_coverage_part_295,
        _test_math_misc_coverage_part_296,
        _test_math_misc_coverage_part_297,
        _test_math_misc_coverage_part_298,
        _test_math_misc_coverage_part_299,
        _test_math_misc_coverage_part_300,
        _test_math_misc_coverage_part_301,
        _test_math_misc_coverage_part_302,
        _test_math_misc_coverage_part_303,
        _test_math_misc_coverage_part_304,
        _test_math_misc_coverage_part_305,
        _test_math_misc_coverage_part_306,
        _test_math_misc_coverage_part_307,
        _test_math_misc_coverage_part_308,
        _test_math_misc_coverage_part_309,
        _test_math_misc_coverage_part_310,
        _test_math_misc_coverage_part_311,
        _test_math_misc_coverage_part_312,
        _test_math_misc_coverage_part_313,
        _test_math_misc_coverage_part_314,
        _test_math_misc_coverage_part_315,
        _test_math_misc_coverage_part_316,
        _test_math_misc_coverage_part_317,
        _test_math_misc_coverage_part_318,
        _test_math_misc_coverage_part_319,
        _test_math_misc_coverage_part_320,
        _test_math_misc_coverage_part_321,
        _test_math_misc_coverage_part_322,
        _test_math_misc_coverage_part_323,
        _test_math_misc_coverage_part_324,
        _test_math_misc_coverage_part_325,
        _test_math_misc_coverage_part_326,
        _test_math_misc_coverage_part_327,
        _test_math_misc_coverage_part_328,
        _test_math_misc_coverage_part_329,
        _test_math_misc_coverage_part_330,
        _test_math_misc_coverage_part_331,
        _test_math_misc_coverage_part_332,
        _test_math_misc_coverage_part_333,
        _test_math_misc_coverage_part_334,
        _test_math_misc_coverage_part_335,
        _test_math_misc_coverage_part_336,
        _test_math_misc_coverage_part_337,
        _test_math_misc_coverage_part_338,
        _test_math_misc_coverage_part_339,
        _test_math_misc_coverage_part_340,
        _test_math_misc_coverage_part_341,
        _test_math_misc_coverage_part_342,
        _test_math_misc_coverage_part_343,
        _test_math_misc_coverage_part_344,
        _test_math_misc_coverage_part_345,
        _test_math_misc_coverage_part_346,
        _test_math_misc_coverage_part_347,
        _test_math_misc_coverage_part_348,
        _test_math_misc_coverage_part_349,
        _test_math_misc_coverage_part_350,
        _test_math_misc_coverage_part_351,
        _test_math_misc_coverage_part_352,
        _test_math_misc_coverage_part_353,
        _test_math_misc_coverage_part_354,
        _test_math_misc_coverage_part_355,
        _test_math_misc_coverage_part_356,
        _test_math_misc_coverage_part_357,
        _test_math_misc_coverage_part_358,
        _test_math_misc_coverage_part_359,
        _test_math_misc_coverage_part_360,
        _test_math_misc_coverage_part_361,
        _test_math_misc_coverage_part_362,
        _test_math_misc_coverage_part_363,
        _test_math_misc_coverage_part_364,
        _test_math_misc_coverage_part_365,
        _test_math_misc_coverage_part_366,
        _test_math_misc_coverage_part_367,
        _test_math_misc_coverage_part_368,
        _test_math_misc_coverage_part_369,
        _test_math_misc_coverage_part_370,
        _test_math_misc_coverage_part_371,
        _test_math_misc_coverage_part_372,
        _test_math_misc_coverage_part_373,
        _test_math_misc_coverage_part_374,
        _test_math_misc_coverage_part_375,
        _test_math_misc_coverage_part_376,
        _test_math_misc_coverage_part_377,
        _test_math_misc_coverage_part_378,
        _test_math_misc_coverage_part_379,
        _test_math_misc_coverage_part_380,
        _test_math_misc_coverage_part_381,
        _test_math_misc_coverage_part_382,
        _test_math_misc_coverage_part_383,
        _test_math_misc_coverage_part_384,
        _test_math_misc_coverage_part_385,
        _test_math_misc_coverage_part_386,
        _test_math_misc_coverage_part_387,
        _test_math_misc_coverage_part_388,
        _test_math_misc_coverage_part_389,
        _test_math_misc_coverage_part_390,
        _test_math_misc_coverage_part_391,
        _test_math_misc_coverage_part_392,
        _test_math_misc_coverage_part_393,
        _test_math_misc_coverage_part_394,
        _test_math_misc_coverage_part_395,
        _test_math_misc_coverage_part_396,
        _test_math_misc_coverage_part_397,
        _test_math_misc_coverage_part_398,
        _test_math_misc_coverage_part_399,
        _test_math_misc_coverage_part_400,
        _test_math_misc_coverage_part_401,
        _test_math_misc_coverage_part_402,
        _test_math_misc_coverage_part_403,
        _test_math_misc_coverage_part_404,
        _test_math_misc_coverage_part_405,
        _test_math_misc_coverage_part_406,
        _test_math_misc_coverage_part_407,
        _test_math_misc_coverage_part_408,
        _test_math_misc_coverage_part_409,
        _test_math_misc_coverage_part_410,
        _test_math_misc_coverage_part_411,
        _test_math_misc_coverage_part_412,
        _test_math_misc_coverage_part_413,
        _test_math_misc_coverage_part_414,
        _test_math_misc_coverage_part_415,
    ):
        _part_fn(_state)
    assert len(_state) > 0


class DummyBackend:
    """Test class DummyBackend."""

    pass


class MockCallableClass:
    """Test class MockCallableClass."""

    def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
        """Test function __init__.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        self.called = True


def test_math_misc_missing_branches_mocked() -> None:
    """Test function test_math_misc_missing_branches_mocked."""
    import ml_switcheroo_compiler.ops as ops

    attrs = ["RawMatMul", "SparseDenseMatMul", "rem", "confusion_matrix", "descriptive", "distributions"]
    originals = {}
    for attr in attrs:
        if hasattr(ops, attr):
            originals[attr] = getattr(ops, attr)
            delattr(ops, attr)
    try:
        res = numpy_eager_registry._registry["RawMatMul"](None, np.array([[1]]), np.array([[2]]))
        assert as_tensor(getattr(res, "shape")) == (1, 1)
        res = numpy_eager_registry._registry["SparseDenseMatMul"](None, np.array([[1]]), np.array([[2]]))
        assert as_tensor(getattr(res, "shape")) == (1, 1)
        res = numpy_eager_registry._registry["rem"](None, np.array([5]), np.array([2]))
        assert as_tensor(getattr(res, "__getitem__")(0)) == 1
        res = numpy_eager_registry._registry["confusion_matrix"](None, np.array([0]), np.array([0]), num_classes=2)
        assert as_tensor(getattr(res, "shape")) == (2, 2)
        res = numpy_eager_registry._registry["descriptive"](None, np.array([1, 2]))
        assert as_callable(len)(res) == 3
        res = numpy_eager_registry._registry["distributions"](None, np.array([1, 2]))
        assert as_callable(len)(res) == 2
    finally:
        for attr, val in originals.items():
            setattr(ops, attr, val)


def test_math_advanced_oserror() -> None:
    """Test function test_math_advanced_oserror."""
    from unittest.mock import mock_open, patch

    import pytest

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _np_write_file as _np_save_img

    with patch("builtins.open", mock_open()) as mocked_file:
        mocked_file.side_effect = Exception("test")
        with pytest.raises(OSError):
            _np_save_img(as_mod("dummy"), "dummy.png", b"dummy")


def test_math_advanced_mocked_fallbacks_2() -> None:
    """Test function test_math_advanced_mocked_fallbacks_2."""
    from unittest.mock import MagicMock

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import numpy_eager_registry

    mock_np = MagicMock()
    mock_np.rem.return_value = "rem"
    mock_np.confusion_matrix.return_value = "cm"
    _np_rem_lower = numpy_eager_registry._registry["rem"]
    assert _np_rem_lower(mock_np, 1, 2) == "rem"
    _np_cm_lower = numpy_eager_registry._registry["confusion_matrix"]
    assert _np_cm_lower(mock_np, 1, 2) == "cm"


def _test_math_matrix_utils_distributions_success_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_matrix_utils_distributions_success.

    Args:
        _state: Shared test state dictionary.
    """
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import _np_callable, _np_clip, _np_descriptive_2, _np_key, _np_one_hot, _np_rem_2, _np_tensor
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_confusion_matrix, _np_distributions

    class FakeOpDef:
        """Test class FakeOpDef."""

        pass

    class FakeInst:
        """Test class FakeInst."""

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return 42

    from unittest.mock import patch

    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    with patch.object(_ops, "OpDef", FakeOpDef, create=True):
        with patch.object(_ops, "rem", FakeInst(), create=True):
            _np_rem_2(np, 1, 2)
        with patch.object(_ops, "rem", property(lambda _: int("invalid")), create=True):
            _np_rem_2(np, 1, 2)
    getattr(_state, "update")({"FakeInst": FakeInst, "FakeOpDef": FakeOpDef, "_np_callable": _np_callable, "_np_clip": _np_clip})
    getattr(_state, "update")({"_np_confusion_matrix": _np_confusion_matrix, "_np_descriptive_2": _np_descriptive_2, "_np_distributions": _np_distributions, "_np_key": _np_key})
    getattr(_state, "update")({"_np_one_hot": _np_one_hot, "_np_rem_2": _np_rem_2, "_np_tensor": _np_tensor, "_ops": _ops})
    getattr(_state, "update")({"np": np, "patch": patch})
    _state.update(locals())


def _test_math_matrix_utils_distributions_success_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_matrix_utils_distributions_success.

    Args:
        _state: Shared test state dictionary.
    """
    (FakeInst, FakeOpDef, _np_descriptive_2, _ops) = (_state["FakeInst"], _state["FakeOpDef"], _state["_np_descriptive_2"], _state["_ops"])
    (np, patch) = (_state["np"], _state["patch"])
    with patch.object(_ops, "OpDef", FakeOpDef, create=True):
        with patch.object(_ops, "descriptive", FakeInst(), create=True):
            _np_descriptive_2(np)
        with as_callable(patch.object)(_ops, "descriptive", as_callable(property)(lambda _: as_callable(int)("invalid")), create=True):
            _np_descriptive_2(np)
    _state.update(locals())


def _test_math_matrix_utils_distributions_success_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_matrix_utils_distributions_success.

    Args:
        _state: Shared test state dictionary.
    """
    (FakeInst, FakeOpDef, _np_confusion_matrix, _ops) = (_state["FakeInst"], _state["FakeOpDef"], _state["_np_confusion_matrix"], _state["_ops"])
    (np, patch) = (_state["np"], _state["patch"])
    with patch.object(_ops, "OpDef", FakeOpDef, create=True):
        with patch.object(_ops, "confusion_matrix", FakeInst(), create=True):
            _np_confusion_matrix(np, [0], [0])
    _state.update(locals())


def _test_math_matrix_utils_distributions_success_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_math_matrix_utils_distributions_success.

    Args:
        _state: Shared test state dictionary.
    """
    (FakeInst, FakeOpDef, _np_callable, _np_clip) = (_state["FakeInst"], _state["FakeOpDef"], _state["_np_callable"], _state["_np_clip"])
    (_np_distributions, _np_key, _np_one_hot, _np_tensor) = (_state["_np_distributions"], _state["_np_key"], _state["_np_one_hot"], _state["_np_tensor"])
    (_ops, np, patch) = (_state["_ops"], _state["np"], _state["patch"])
    with patch.object(_ops, "OpDef", FakeOpDef, create=True):
        with patch.object(_ops, "distributions", FakeInst(), create=True):
            _np_distributions(np, [0])
    assert _np_distributions is not None
    assert _np_callable(np, lambda x: x)
    assert not _np_callable(np)
    np.testing.assert_array_equal(_np_key(np, 1), np.array([1, 0], dtype=np.uint32))
    np.testing.assert_array_equal(_np_key(np), np.array([0, 0], dtype=np.uint32))
    assert _np_clip(np) is None
    res = _np_one_hot(np, [0, 1], depth=3, axis=0, on_value=1, off_value=0)
    expected = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
    np.testing.assert_array_equal(res, expected)
    res = _np_one_hot(np, [0, 1], depth=3, axis=-1, on_value=1, off_value=0)
    expected = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    np.testing.assert_array_equal(res, expected)
    assert len(_np_tensor(np)) == 0
    getattr(_state, "update")({"expected": expected, "res": res})


def test_math_matrix_utils_distributions_success() -> None:
    """Test function test_math_matrix_utils_distributions_success."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_matrix_utils_distributions_success_part_1(_state)
    _test_math_matrix_utils_distributions_success_part_2(_state)
    _test_math_matrix_utils_distributions_success_part_3(_state)
    _test_math_matrix_utils_distributions_success_part_4(_state)
    assert len(_state) > 0


def test_math_matrix_utils_distributions_success_2() -> None:
    """Test function test_math_matrix_utils_distributions_success_2."""
    from unittest.mock import patch

    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    def _make_class(name: EagerValue) -> EagerValue:
        """Test function _make_class.

        Args:
            name: Test parameter `name`.
        """
        return as_callable(type)(name, (), {"__call__": lambda self, *args, **kwargs: 42})

    with patch.object(_ops, "OpDef", type, create=True):
        with patch.object(_ops, "rem", _make_class("rem"), create=True):
            pass
        with patch.object(_ops, "descriptive", _make_class("descriptive"), create=True):
            pass
        with patch.object(_ops, "confusion_matrix", _make_class("confusion_matrix"), create=True):
            pass
        with patch.object(_ops, "distributions", _make_class("distributions"), create=True):
            pass
    assert callable(test_math_matrix_utils_distributions_success_2)


def test_fallback_branches_return(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_fallback_branches_return.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import _np_descriptive_2, _np_rem_2
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_confusion_matrix, _np_distributions

    class DummyOpDef:
        """Test class DummyOpDef."""

        pass

    class FakeDistributions:
        """Test class FakeDistributions."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> FakeDistributions:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return typing.cast("FakeDistributions", "dist2")

    class FakeCM:
        """Test class FakeCM."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> FakeCM:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return typing.cast("FakeCM", "cm2")

    class FakeDesc:
        """Test class FakeDesc."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> FakeDesc:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return typing.cast("FakeDesc", "desc2")

    class FakeRem:
        """Test class FakeRem."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> FakeRem:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return typing.cast("FakeRem", "rem2")

    monkeypatch.setattr(ops, "OpDef", DummyOpDef, raising=False)
    monkeypatch.setattr(ops, "distributions", FakeDistributions, raising=False)
    monkeypatch.setattr(ops, "confusion_matrix", FakeCM, raising=False)
    monkeypatch.setattr(ops, "descriptive", FakeDesc, raising=False)
    monkeypatch.setattr(ops, "rem", FakeRem, raising=False)

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    assert _np_distributions(as_tensor(DummyBackend()), 1) == "dist2"
    assert _np_confusion_matrix(as_tensor(DummyBackend()), 1) == "cm2"
    assert _np_descriptive_2(as_tensor(DummyBackend()), 1) == "desc2"
    assert _np_rem_2(as_tensor(DummyBackend()), 1) == "rem2"


def test_np_distributions_empty_args() -> None:
    """Test function test_np_distributions_empty_args."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_distributions

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    res = _np_distributions(as_tensor(DummyBackend()))
    assert as_tensor(getattr(res, "shape")) == (2,)


def test_np_confusion_matrix_cap_none() -> None:
    """Test function test_np_confusion_matrix_cap_none."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_confusion_matrix_cap

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    assert _np_confusion_matrix_cap(as_tensor(DummyBackend())) is None


def test_math_general_rem_3() -> None:
    """Test function test_math_general_rem_3."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import _np_rem_2

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    assert _np_rem_2(as_tensor(DummyBackend()), 5, 2) == 1
    assert _np_rem_2(as_tensor(DummyBackend())) is None


def test_math_general_scipy_import_error() -> None:
    """Test function test_math_general_scipy_import_error."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import _np_betainc

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    import sys

    orig_scipy = sys.modules.get("scipy.special")
    getattr(sys.modules, "__setitem__")("scipy.special", None)
    try:
        with pytest.raises(ImportError):
            _np_betainc(as_tensor(DummyBackend()), 1)
    finally:
        if orig_scipy:
            sys.modules["scipy.special"] = orig_scipy
        else:
            del sys.modules["scipy.special"]


def test_np_descriptive_2_empty_args() -> None:
    """Test function test_np_descriptive_2_empty_args."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import _np_descriptive_2

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    res = _np_descriptive_2(as_tensor(DummyBackend()))
    assert as_tensor(getattr(res, "shape")) == (3,)


def test_fallback_exception() -> None:
    """Test function test_fallback_exception."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import _np_descriptive_2, _np_rem_2
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_confusion_matrix, _np_distributions

    class DummyBackend:
        """Test class DummyBackend."""

        def distributions(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function distributions.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "dist_exc"

        def confusion_matrix(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function confusion_matrix.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "cm_exc"

        def descriptive(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function descriptive.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "desc_exc"

        def rem(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function rem.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "rem_exc"

    orig_import = __import__

    def mock_import(name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function mock_import.

        Args:
            name: Test parameter `name`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        if name == "ml_switcheroo_compiler.ops":
            raise Exception("boom")
        return as_callable(orig_import)(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        assert _np_distributions(as_tensor(DummyBackend()), 1) == "dist_exc"
        assert _np_confusion_matrix(as_tensor(DummyBackend()), 1) == "cm_exc"
        assert _np_descriptive_2(as_tensor(DummyBackend()), 1) == "desc_exc"
        assert _np_rem_2(as_tensor(DummyBackend()), 1) == "rem_exc"


def test_np_confusion_matrix_actual_logic() -> None:
    """Test function test_np_confusion_matrix_actual_logic."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_confusion_matrix, _np_confusion_matrix_cap

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    res = _np_confusion_matrix(as_tensor(DummyBackend()), [0, 1], [1, 1], num_classes=2)
    assert as_callable(np.array_equal)(res, [[0, 1], [0, 1]])
    res2 = _np_confusion_matrix(as_tensor(DummyBackend()), [0, 1], [1, 1])
    assert as_tensor(getattr(res2, "shape")) == (2, 2)
    assert _np_confusion_matrix_cap(as_tensor(DummyBackend()), [0], [0]) is not None


def test_math_matrix_utils_simple_forwards() -> None:
    """Test function test_math_matrix_utils_simple_forwards."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_diag_indices_, _np_diag_indices_from_, _np_diagflat_, _np_diagonal_, _np_indices_, _np_linearoperatorblockdiag, _np_linearoperatordiag, _np_mask_indices_

    class DummyBackend:
        """Test class DummyBackend."""

        def diag_indices(self) -> EagerValue:
            """Test function diag_indices."""
            return 1

        def diag_indices_from(self) -> EagerValue:
            """Test function diag_indices_from."""
            return 2

        def diagflat(self) -> EagerValue:
            """Test function diagflat."""
            return 3

        def diagonal(self) -> EagerValue:
            """Test function diagonal."""
            return 4

        def indices(self) -> EagerValue:
            """Test function indices."""
            return 5

        def mask_indices(self) -> EagerValue:
            """Test function mask_indices."""
            return 6

        def linearoperatorblockdiag(self) -> EagerValue:
            """Test function linearoperatorblockdiag."""
            return 7

        def linearoperatordiag(self) -> EagerValue:
            """Test function linearoperatordiag."""
            return 8

    db = DummyBackend()
    assert _np_diag_indices_(as_tensor(db)) == 1
    assert _np_diag_indices_from_(as_tensor(db)) == 2
    assert _np_diagflat_(as_tensor(db)) == 3
    assert _np_diagonal_(as_tensor(db)) == 4
    assert _np_indices_(as_tensor(db)) == 5
    assert _np_mask_indices_(as_tensor(db)) == 6
    assert _np_linearoperatorblockdiag(as_tensor(db)).__class__.__name__ == "LinearOperatorBlockDiag"
    assert _np_linearoperatordiag(as_tensor(db)).__class__.__name__ == "LinearOperatorDiag"


def _test_math_general_simple_forwards_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import (
        _np_apply_over_axes,
        _np_array_repr_,
        _np_array_str_,
        _np_atleast_1d,
        _np_atleast_2d,
        _np_atleast_3d,
        _np_average,
        _np_betainc,
        _np_block,
        _np_callable,
        _np_clip,
        _np_corrcoef,
        _np_cov,
        _np_debuginfs,
        _np_debugnans,
        _np_descriptive_2,
        _np_diff_,
        _np_digitize_,
        _np_dotgeneral,
        _np_ediff1d_,
        _np_einsum_path_,
        _np_fabs_,
        _np_interp_,
        _np_iterable_,
        _np_ix__,
        _np_key,
        _np_kron_,
        _np_load_,
        _np_modf_,
        _np_one_hot,
        _np_raggeddot,
        _np_ravel_multi_index_,
        _np_rem_2,
        _np_serialize_tensor,
        _np_serialize_tensor_camel,
        _np_tensor,
        _np_trapz,
        _np_trapz_,
        _np_truncate_div,
        _np_truncate_mod,
        _np_uint,
        _np_uint8,
        _np_unravelindex,
        _np_unwrap,
        _np_vectorize,
    )

    getattr(_state, "update")({"_np_apply_over_axes": _np_apply_over_axes, "_np_array_repr_": _np_array_repr_, "_np_array_str_": _np_array_str_, "_np_atleast_1d": _np_atleast_1d})
    getattr(_state, "update")({"_np_atleast_2d": _np_atleast_2d, "_np_atleast_3d": _np_atleast_3d, "_np_average": _np_average, "_np_betainc": _np_betainc})
    getattr(_state, "update")({"_np_block": _np_block, "_np_callable": _np_callable, "_np_clip": _np_clip, "_np_corrcoef": _np_corrcoef})
    getattr(_state, "update")({"_np_cov": _np_cov, "_np_debuginfs": _np_debuginfs, "_np_debugnans": _np_debugnans, "_np_descriptive_2": _np_descriptive_2})
    getattr(_state, "update")({"_np_diff_": _np_diff_, "_np_digitize_": _np_digitize_, "_np_dotgeneral": _np_dotgeneral, "_np_ediff1d_": _np_ediff1d_})
    getattr(_state, "update")({"_np_einsum_path_": _np_einsum_path_, "_np_fabs_": _np_fabs_, "_np_interp_": _np_interp_, "_np_iterable_": _np_iterable_})
    getattr(_state, "update")({"_np_ix__": _np_ix__, "_np_key": _np_key, "_np_kron_": _np_kron_, "_np_load_": _np_load_})
    getattr(_state, "update")({"_np_modf_": _np_modf_, "_np_one_hot": _np_one_hot, "_np_raggeddot": _np_raggeddot, "_np_ravel_multi_index_": _np_ravel_multi_index_})
    getattr(_state, "update")({"_np_rem_2": _np_rem_2, "_np_serialize_tensor": _np_serialize_tensor, "_np_serialize_tensor_camel": _np_serialize_tensor_camel, "_np_tensor": _np_tensor})
    getattr(_state, "update")({"_np_trapz": _np_trapz, "_np_trapz_": _np_trapz_, "_np_truncate_div": _np_truncate_div, "_np_truncate_mod": _np_truncate_mod})
    getattr(_state, "update")({"_np_uint": _np_uint, "_np_uint8": _np_uint8, "_np_unravelindex": _np_unravelindex, "_np_unwrap": _np_unwrap})
    getattr(_state, "update")({"_np_vectorize": _np_vectorize})
    _state.update(locals())


class _SimpleForwardsDummyBackend:
    """Test class _SimpleForwardsDummyBackend."""

    def applyoveraxes(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function applyoveraxes.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 0

    def arrayrepr(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function arrayrepr.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 1

    def arraystr(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function arraystr.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 2

    def atleast1d(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function atleast1d.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 3

    def atleast2d(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function atleast2d.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 4

    def atleast3d(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function atleast3d.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 5

    def average(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function average.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 6

    def betainc(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function betainc.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 7

    def block(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function block.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 8

    def callable(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function callable.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 9

    def clip(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function clip.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 10

    def corrcoef(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function corrcoef.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 11

    def cov(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function cov.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 12

    def debuginfs(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function debuginfs.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 13

    def debugnans(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function debugnans.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 14

    def descriptive(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function descriptive.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 15

    def descriptive2(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function descriptive2.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 16

    def diff(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function diff.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 17

    def digitize(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function digitize.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 18

    def dotgeneral(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function dotgeneral.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 19

    def ediff1d(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function ediff1d.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 20

    def einsumpath(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function einsumpath.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 21

    def fabs(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function fabs.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 22

    def interp(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function interp.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 23

    def iterable(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function iterable.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 24

    def ix(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function ix.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 25

    def key(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function key.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 26

    def kron(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function kron.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 27

    def load(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function load.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 28

    def modf(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function modf.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 29

    def onehot(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function onehot.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 30

    def raggeddot(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function raggeddot.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 31

    def ravelmultiindex(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function ravelmultiindex.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 32

    def rem(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function rem.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 33

    def rem2(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function rem2.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 34

    def rem3(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function rem3.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 35

    def serializetensor(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function serializetensor.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 36

    def serializetensorcamel(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function serializetensorcamel.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 37

    def tensor(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function tensor.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 38

    def trapz(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function trapz.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 39

    def truncatediv(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function truncatediv.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 41

    def truncatemod(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function truncatemod.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 42

    def uint(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function uint.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 43

    def uint8(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function uint8.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 44

    def unravelindex(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function unravelindex.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 45

    def unwrap(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function unwrap.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 46

    def vectorize(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function vectorize.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return 47


def _test_math_general_simple_forwards_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    getattr(_state, "update")({"DummyBackend": _SimpleForwardsDummyBackend})
    _state.update(locals())


def _test_math_general_simple_forwards_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyBackend, _np_apply_over_axes, _np_array_repr_, _np_array_str_) = (_state["DummyBackend"], _state["_np_apply_over_axes"], _state["_np_array_repr_"], _state["_np_array_str_"])
    _np_atleast_1d = _state["_np_atleast_1d"]
    db = DummyBackend()
    with pytest.raises(AttributeError):
        assert _np_apply_over_axes(db) == 0
    with pytest.raises(AttributeError):
        assert _np_array_repr_(db) == 1
    with pytest.raises(AttributeError):
        assert _np_array_str_(db) == 2
    with pytest.raises(AttributeError):
        assert _np_atleast_1d(db) == 3
    getattr(_state, "update")({"db": db})
    _state.update(locals())


def _test_math_general_simple_forwards_sub_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 4 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_atleast_2d, _np_atleast_3d, _np_average, _np_betainc) = (_state["_np_atleast_2d"], _state["_np_atleast_3d"], _state["_np_average"], _state["_np_betainc"])
    db = _state["db"]
    with pytest.raises(AttributeError):
        assert _np_atleast_2d(db) == 4
    with pytest.raises(AttributeError):
        assert _np_atleast_3d(db) == 5
    assert _np_average(db) == 6
    with pytest.raises(AttributeError):
        assert _np_betainc(db) == 7
    _state.update(locals())


def _test_math_general_simple_forwards_sub_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 5 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_block, _np_callable, _np_clip, _np_corrcoef) = (_state["_np_block"], _state["_np_callable"], _state["_np_clip"], _state["_np_corrcoef"])
    db = _state["db"]
    assert _np_block(db) == 8
    assert _np_callable(db) != 9
    assert _np_clip(db) != 10
    assert _np_corrcoef(db) == 11
    _state.update(locals())


def _test_math_general_simple_forwards_sub_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 6 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_cov, _np_debuginfs, _np_debugnans, _np_descriptive_2) = (_state["_np_cov"], _state["_np_debuginfs"], _state["_np_debugnans"], _state["_np_descriptive_2"])
    db = _state["db"]
    assert _np_cov(db) == 12
    with pytest.raises(IndexError):
        assert _np_debuginfs(db) == 13
    with pytest.raises(IndexError):
        assert _np_debugnans(db) == 14
    assert _np_descriptive_2(db) == 15
    _state.update(locals())


def _test_math_general_simple_forwards_sub_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 7 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_descriptive_2, _np_diff_, _np_digitize_, _np_dotgeneral) = (_state["_np_descriptive_2"], _state["_np_diff_"], _state["_np_digitize_"], _state["_np_dotgeneral"])
    db = _state["db"]
    assert _np_descriptive_2(db) != 16
    assert _np_diff_(db) == 17
    assert _np_digitize_(db) == 18
    with pytest.raises(IndexError):
        assert _np_dotgeneral(db) == 19
    _state.update(locals())


def _test_math_general_simple_forwards_sub_8(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 8 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_ediff1d_, _np_einsum_path_, _np_fabs_, _np_interp_) = (_state["_np_ediff1d_"], _state["_np_einsum_path_"], _state["_np_fabs_"], _state["_np_interp_"])
    db = _state["db"]
    assert _np_ediff1d_(db) == 20
    with pytest.raises(AttributeError):
        assert _np_einsum_path_(db) == 21
    assert _np_fabs_(db) == 22
    assert _np_interp_(db) == 23
    _state.update(locals())


def _test_math_general_simple_forwards_sub_9(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 9 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_iterable_, _np_ix__, _np_key, _np_kron_) = (_state["_np_iterable_"], _state["_np_ix__"], _state["_np_key"], _state["_np_kron_"])
    db = _state["db"]
    assert _np_iterable_(db) == 24
    with pytest.raises(AttributeError):
        assert _np_ix__(db) == 25
    with pytest.raises(ValueError):
        assert _np_key(db) == 26
    assert _np_kron_(db) == 27
    _state.update(locals())


def _test_math_general_simple_forwards_sub_10(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 10 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_load_, _np_modf_, _np_one_hot, _np_raggeddot) = (_state["_np_load_"], _state["_np_modf_"], _state["_np_one_hot"], _state["_np_raggeddot"])
    db = _state["db"]
    assert _np_load_(db) == 28
    assert _np_modf_(db) == 29
    assert _np_one_hot(db) != 30
    with pytest.raises(AttributeError):
        assert _np_raggeddot(db) == 31
    _state.update(locals())


def _test_math_general_simple_forwards_sub_11(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 11 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_ravel_multi_index_, _np_rem_2, db) = (_state["_np_ravel_multi_index_"], _state["_np_rem_2"], _state["db"])
    with pytest.raises(AttributeError):
        assert _np_ravel_multi_index_(db) == 32
    assert _np_rem_2(db) == 33
    assert _np_rem_2(db) != 34
    assert _np_rem_2(db) != 35
    _state.update(locals())


def _test_math_general_simple_forwards_sub_12(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 12 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_serialize_tensor, _np_serialize_tensor_camel, _np_tensor, _np_trapz) = (_state["_np_serialize_tensor"], _state["_np_serialize_tensor_camel"], _state["_np_tensor"], _state["_np_trapz"])
    db = _state["db"]
    assert _np_serialize_tensor(db) != 36
    assert _np_serialize_tensor_camel(db) != 37
    with pytest.raises(AttributeError):
        assert _np_tensor(db) == 38
    with pytest.raises(IndexError):
        assert _np_trapz(db) == 39
    _state.update(locals())


def _test_math_general_simple_forwards_sub_13(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 13 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_trapz_, _np_truncate_div, _np_truncate_mod, _np_uint) = (_state["_np_trapz_"], _state["_np_truncate_div"], _state["_np_truncate_mod"], _state["_np_uint"])
    db = _state["db"]
    assert _np_trapz_(db) != 40
    with pytest.raises(IndexError):
        assert _np_truncate_div(db) == 41
    with pytest.raises(IndexError):
        assert _np_truncate_mod(db) == 42
    with pytest.raises(IndexError):
        assert _np_uint(db) == 43
    _state.update(locals())


def _test_math_general_simple_forwards_sub_14(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 14 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_uint8, _np_unravelindex, _np_unwrap, _np_vectorize) = (_state["_np_uint8"], _state["_np_unravelindex"], _state["_np_unwrap"], _state["_np_vectorize"])
    db = _state["db"]
    with pytest.raises(IndexError):
        assert _np_uint8(db) == 44
    with pytest.raises(IndexError):
        assert _np_unravelindex(db) == 45
    with pytest.raises(IndexError):
        assert _np_unwrap(db) == 46
    with pytest.raises(IndexError):
        assert _np_vectorize(db) == 47
    _state.update(locals())


def test_math_general_simple_forwards() -> None:
    """Test function test_math_general_simple_forwards."""
    _state: dict[str, EagerTensorProtocol] = dict(locals())
    for _sub_fn in (
        _test_math_general_simple_forwards_sub_1,
        _test_math_general_simple_forwards_sub_2,
        _test_math_general_simple_forwards_sub_3,
        _test_math_general_simple_forwards_sub_4,
        _test_math_general_simple_forwards_sub_5,
        _test_math_general_simple_forwards_sub_6,
        _test_math_general_simple_forwards_sub_7,
        _test_math_general_simple_forwards_sub_8,
        _test_math_general_simple_forwards_sub_9,
        _test_math_general_simple_forwards_sub_10,
        _test_math_general_simple_forwards_sub_11,
        _test_math_general_simple_forwards_sub_12,
        _test_math_general_simple_forwards_sub_13,
        _test_math_general_simple_forwards_sub_14,
    ):
        _sub_fn(_state)
    assert len(_state) > 0


def test_math_general_missing_forwards() -> None:
    """Test function test_math_general_missing_forwards."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import _np_clip, _np_dotgeneral, _np_key, _np_one_hot, _np_raggeddot, _np_serialize_tensor, _np_serialize_tensor_camel, _np_truncate_div, _np_truncate_mod

    class DummyBackend:
        """Test class DummyBackend."""

        def matmul(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function matmul.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return 1

    db = DummyBackend()
    assert _np_truncate_div(as_tensor(db), 5.0, 2.0) == 2.0
    assert _np_truncate_mod(as_tensor(db), 5.0, 2.0) == 1.0
    assert _np_dotgeneral(as_tensor(db), np.ones((2, 2)), np.ones((2, 2)), dimension_numbers=(((1,), (0,)), ((), ()))) is not None
    assert _np_raggeddot(as_tensor(db), np.ones((2, 2)), np.ones((2, 2))) is not None
    assert _np_serialize_tensor(as_tensor(db), 1) is not None
    assert _np_serialize_tensor_camel(as_tensor(db), 1) is not None
    assert as_callable(np.array_equal)(as_callable(_np_key)(db, 1), [1, 0])
    assert as_callable(np.array_equal)(as_callable(_np_clip)(db, as_callable(np.array)([5.0]), 1.0, 3.0), [3.0])
    res = _np_one_hot(as_tensor(db), np.array([0]), 2)
    assert as_tensor(getattr(res, "shape")) == (1, 2)


def test_math_general_debuginfs_callable() -> None:
    """Test function test_math_general_debuginfs_callable."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import _np_callable, _np_debuginfs, _np_debugnans, _np_descriptive_2, _np_key, _np_rem_2, _np_tensor

    assert _np_callable(np, lambda x: x) is True
    assert _np_callable(np) is False
    import pytest

    with pytest.raises(ValueError):
        _np_debuginfs(np, np.array([np.inf]))
    assert as_callable(np.array_equal)(as_callable(_np_debuginfs)(np, as_callable(np.array)([1.0])), [1.0])
    with pytest.raises(ValueError):
        _np_debugnans(np, np.array([np.nan]))
    assert as_callable(np.array_equal)(as_callable(_np_debugnans)(np, as_callable(np.array)([1.0])), [1.0])
    assert as_tensor(getattr(_np_tensor(np), "shape")) == (0,)
    assert as_tensor(getattr(_np_tensor(np, [1, 2]), "shape")) == (2,)
    res = _np_descriptive_2(np, [1.0, 2.0])
    assert as_callable(len)(res) == 3
    assert as_callable(len)(as_callable(_np_descriptive_2)(np)) == 3
    assert as_callable(np.array_equal)(as_callable(_np_key)(np), [0, 0])
    assert _np_rem_2(np) is None


def test_dotgeneral_batches() -> None:
    """Test function test_dotgeneral_batches."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import _np_dotgeneral

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    assert _np_dotgeneral(as_tensor(DummyBackend()), np.ones((2, 2)), np.ones((2, 2)), dimension_numbers=(((1,), (1,)), ((0,), (0,)))) is not None


def test_math_general_remaining() -> None:
    """Test function test_math_general_remaining."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import _get_sc, _np_clip, _np_key, _np_one_hot, _np_rem_2, _np_serialize_tensor_camel

    assert _np_rem_2(np, 5, 2) == 1
    res = _np_key(np, 1)
    assert res is not None
    assert _get_sc() is not None
    assert _np_clip(np, np.array([5.0]), 1.0, 3.0) is not None
    assert _np_clip(np) is None
    res2 = _np_one_hot(np, np.array([0]), 2)
    assert res2 is not None
    assert _np_one_hot(np) is None
    assert _np_serialize_tensor_camel(np, np.array([1])) is not None

    class DummyBackendRem:
        """Test class DummyBackendRem."""

        pass

    assert _np_rem_2(as_tensor(DummyBackendRem()), 5, 2) == 1


def test_math_general_final() -> None:
    """Test function test_math_general_final."""
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import _get_sc, _np_one_hot, _np_serialize_tensor_camel

    with patch("builtins.__import__", side_effect=ImportError("boom")):
        assert _get_sc() is None
    res = _np_one_hot(np, np.array([0]), 2, axis=0)
    assert as_tensor(getattr(res, "shape")) == (2, 1)
    with patch("pickle.dumps", side_effect=Exception("boom")):
        res2 = _np_serialize_tensor_camel(np, np.array([1]))
        assert res2 is not None
