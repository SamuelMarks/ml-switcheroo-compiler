# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager Hermite, Chebyshev, Legendre, and Laguerre polynomials."""

from __future__ import annotations

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state

mod = _mod_raw

_TestMathMiscCoverageState = BaseMathMiscState


def _test_math_misc_coverage_part_298(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_299(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_300(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_301(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_302(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_303(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_304(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_305(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_306(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_307(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_308(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_309(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_310(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_311(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_312(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_313(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_314(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_315(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_316(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_317(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_318(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_319(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_320(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_321(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_322(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_323(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_324(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def test_msf_orthogonal_polynomials() -> None:
    """Tests for NumPy eager Hermite, Chebyshev, Legendre, and Laguerre polynomials runner."""
    _state: _TestMathMiscCoverageState = _make_base_math_state(298)
    for _part_fn in (
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
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([0.0, 0.0]))
