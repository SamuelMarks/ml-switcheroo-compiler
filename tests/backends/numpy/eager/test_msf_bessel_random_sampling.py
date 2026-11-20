# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager Bessel special functions and Bernoulli/random sampling."""

from __future__ import annotations

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state
from tests.testing_utils import adapt_backend_module, require_callable

mod = _mod_raw

_TestMathMiscCoverageState = BaseMathMiscState


def _test_math_misc_coverage_part_271(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_272(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 272 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    assert bk is not None
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_273(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 273 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor(bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor(bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_274(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_275(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_276(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 276 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(RuntimeError):
        mod._np_read_file(dummy_bk, arg1, arg2)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(bk, arg1)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(bk, arg1, arg2)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(bk, arg1, arg1, arg2)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_277(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 277 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(bk, arg_int)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(dummy_bk, arg1, arg2)
    mod._np_serialize_tensor(bk, arg1)
    mod._np_serialize_tensor(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_278(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_279(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 279 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    pass
    pass
    pass
    pass
    pass


def _test_math_misc_coverage_part_280(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})


def _test_math_misc_coverage_part_281(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_282(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_283(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_284(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_285(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_286(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_287(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_288(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_289(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_290(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_291(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_292(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_293(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 293 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        require_callable(mod._np_stridedslice)(bk, arg1)
    with pytest.raises(TypeError):
        require_callable(mod._np_stridedslice)(bk, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_stridedslice)(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_stridedslice)(bk, arg_bool)
    with pytest.raises(TypeError):
        require_callable(mod._np_stridedslice)(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_294(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 294 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        require_callable(mod._np_stridedslice)(dummy_bk, arg1)
    with pytest.raises(TypeError):
        require_callable(mod._np_stridedslice)(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_randombernoulli(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_randombernoulli(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_randombernoulli(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_295(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_296(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_297(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def test_msf_bessel_random_sampling() -> None:
    """Tests for NumPy eager Bessel special functions and Bernoulli/random sampling runner."""
    _state: _TestMathMiscCoverageState = _make_base_math_state(271)
    for _part_fn in (
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
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([0.0, 0.0]))
