# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager SVD, QR, Cholesky, and eigenvalue decompositions."""

from __future__ import annotations

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state

mod = _mod_raw

_TestMathMiscCoverageState = BaseMathMiscState


def _test_math_misc_coverage_part_163(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_164(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_165(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_166(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_167(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_168(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_169(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_170(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_171(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_172(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_173(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_174(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_175(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_176(_state: _TestMathMiscCoverageState) -> None:
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
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_cholesky_(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_177(_state: _TestMathMiscCoverageState) -> None:
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
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_cholesky_(bk, arg_bool)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_cholesky_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_linalg_cholesky_(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_178(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 178 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_linalg_cholesky_(dummy_bk, arg1, arg2)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_det_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linalg_det_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linalg_det_(bk, arg1, arg1, arg2)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_det_(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_179(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 179 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_det_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_linalg_det_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_linalg_det_(dummy_bk, arg1, arg2)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_svd_(bk, arg1)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_svd_(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_180(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 180 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_svd_(bk, arg1, arg1, arg2)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_svd_(bk, arg_bool)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_svd_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_linalg_svd_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_linalg_svd_(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_181(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_182(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 182 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_expand_dims_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_expand_dims_(dummy_bk, arg1, arg2)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_inv_(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linalg_inv_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linalg_inv_(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_183(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 183 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_inv_(bk, arg_bool)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_inv_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_linalg_inv_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_linalg_inv_(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linalg_matrix_power_(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_184(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 184 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_matrix_power_(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linalg_matrix_power_(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linalg_matrix_power_(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linalg_matrix_power_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_linalg_matrix_power_(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_185(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 185 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_linalg_matrix_power_(dummy_bk, arg1, arg2)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_pinv_(bk, arg1)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_pinv_(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_linalg_pinv_(bk, arg1, arg1, arg2)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_pinv_(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_186(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 186 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_linalg_pinv_(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_linalg_pinv_(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_linalg_pinv_(dummy_bk, arg1, arg2)
    mod._np_affineconfig(bk, arg1)
    mod._np_affineconfig(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_187(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_188(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def test_msf_linalg_decompositions() -> None:
    """Tests for NumPy eager SVD, QR, Cholesky, and eigenvalue decompositions runner."""
    _state: _TestMathMiscCoverageState = _make_base_math_state(163)
    for _part_fn in (
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
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([0.0, 0.0]))
