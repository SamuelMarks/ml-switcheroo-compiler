# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager circulant, Householder, Kronecker, and Toeplitz LinearOperators."""

from __future__ import annotations

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state

mod = _mod_raw

_TestMathMiscCoverageState = BaseMathMiscState


def _test_math_misc_coverage_part_216(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_217(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_218(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_219(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_220(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_221(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_222(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_223(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_224(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_225(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_226(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_227(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_228(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_229(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_230(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_231(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_232(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_233(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_234(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_235(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_236(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_237(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_238(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_239(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_240(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_241(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_242(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def test_msf_linear_operators_b() -> None:
    """Tests for NumPy eager circulant, Householder, Kronecker, and Toeplitz LinearOperators runner."""
    _state: _TestMathMiscCoverageState = _make_base_math_state(216)
    for _part_fn in (
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
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([0.0, 0.0]))
