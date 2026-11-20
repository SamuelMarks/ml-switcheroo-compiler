# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager finite differences, spectral windows, and 1D interpolation."""

from __future__ import annotations

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state

mod = _mod_raw

_TestMathMiscCoverageState = BaseMathMiscState


def _test_math_misc_coverage_part_107(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_108(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_109(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_110(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_111(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_112(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_113(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_114(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_115(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_116(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_117(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_118(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_119(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_120(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_121(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_122(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_123(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_124(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_125(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_126(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_127(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_128(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_129(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_130(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_131(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_132(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_133(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_134(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def test_msf_finite_diff_windows() -> None:
    """Tests for NumPy eager finite differences, spectral windows, and 1D interpolation runner."""
    _state: _TestMathMiscCoverageState = _make_base_math_state(107)
    for _part_fn in (
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
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([1.0, 2.0]))
