# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager diagonal indices, flat diagonals, and histogram binning."""

from __future__ import annotations

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state

mod = _mod_raw

_TestMathMiscCoverageState = BaseMathMiscState


def _test_math_misc_coverage_part_79(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_80(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_81(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_82(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_83(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_84(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_85(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_86(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_87(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_88(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_89(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_90(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_91(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_92(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_93(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_94(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_95(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_96(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_97(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_98(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_99(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_100(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_101(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_102(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_103(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_104(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_105(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_106(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def test_msf_diagonal_binning() -> None:
    """Tests for NumPy eager diagonal indices, flat diagonals, and histogram binning runner."""
    _state: _TestMathMiscCoverageState = _make_base_math_state(79)
    for _part_fn in (
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
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([1.0, 2.0]))
