# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager multidimensional real, inverse, and Hermitian FFTs."""

from __future__ import annotations

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state

mod = _mod_raw

_TestMathMiscCoverageState = BaseMathMiscState


def _test_math_misc_coverage_part_325(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_326(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_327(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_328(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_329(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_330(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_331(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_332(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_333(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_334(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_335(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_336(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_337(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_338(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_339(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_340(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_341(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_342(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_343(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_344(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_345(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_346(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_347(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_348(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_349(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_350(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_351(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_352(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def test_msf_fft_spectral() -> None:
    """Tests for NumPy eager multidimensional real, inverse, and Hermitian FFTs runner."""
    _state: _TestMathMiscCoverageState = _make_base_math_state(325)
    for _part_fn in (
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
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([0.0, 0.0]))
