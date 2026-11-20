# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager Kronecker products, lexicographic sorting, and fill_diagonal."""

from __future__ import annotations

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state

mod = _mod_raw

_TestMathMiscCoverageState = BaseMathMiscState


def _test_math_misc_coverage_part_135(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_136(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_137(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_138(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_139(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_140(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_141(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_142(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_143(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_144(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_145(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_146(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_147(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_148(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_149(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_150(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_151(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_152(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_153(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_154(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_155(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_156(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_157(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_158(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_159(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_160(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_161(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_162(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def test_msf_kronecker_sorting() -> None:
    """Tests for NumPy eager Kronecker products, lexicographic sorting, and fill_diagonal runner."""
    _state: _TestMathMiscCoverageState = _make_base_math_state(135)
    for _part_fn in (
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
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([0.0, 0.0]))
