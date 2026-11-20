# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager block stacking, column stacking, deletion, and insertion ops."""

from __future__ import annotations

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state

mod = _mod_raw

_TestMathMiscCoverageState = BaseMathMiscState


def _test_math_misc_coverage_part_51(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_52(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_53(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_54(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_55(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_56(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_57(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_58(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_59(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_60(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_61(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_62(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_63(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_64(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_65(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_66(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_67(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_68(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_69(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_70(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_71(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_72(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_73(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_74(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_75(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_76(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_77(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_78(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def test_msf_stacking_splitting() -> None:
    """Tests for NumPy eager block stacking, column stacking, deletion, and insertion ops runner."""
    _state: _TestMathMiscCoverageState = _make_base_math_state(51)
    for _part_fn in (
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
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([1.0, 2.0]))
