# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager bit-packing, array creation, and basic concatenation ops."""

from __future__ import annotations

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state
from tests.testing_utils import adapt_backend_module, require_callable

mod = _mod_raw

_TestMathMiscCoverageState = BaseMathMiscState


def _test_math_misc_coverage_part_23(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 23 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_bitcast_convert_type(bk, arg1, arg2)
    mod._np_bitcast_convert_type(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_bitcast_convert_type)(bk, arg_bool)
    with pytest.raises(TypeError):
        require_callable(mod._np_bitcast_convert_type)(bk, arg_int)
    with pytest.raises(TypeError):
        require_callable(mod._np_bitcast_convert_type)(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_24(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 24 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_bitcast_convert_type(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_reduce_precision)(bk, arg1)
    with pytest.raises(TypeError):
        require_callable(mod._np_reduce_precision)(bk, arg1, arg2)
    require_callable(mod._np_reduce_precision)(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_reduce_precision)(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_25(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 25 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        require_callable(mod._np_reduce_precision)(bk, arg_int)
    with pytest.raises(TypeError):
        require_callable(mod._np_reduce_precision)(dummy_bk, arg1)
    with pytest.raises(TypeError):
        require_callable(mod._np_reduce_precision)(dummy_bk, arg1, arg2)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_gather")(bk, arg1)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_gather")(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_26(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 26 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_gather")(bk, arg1, arg1, arg2)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_gather")(bk, arg_bool)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_gather")(bk, arg_int)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_gather")(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_gather")(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_27(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 27 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_reduce")(bk, arg1)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_reduce")(bk, arg1, arg2)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_reduce")(bk, arg1, arg1, arg2)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_reduce")(bk, arg_bool)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_reduce")(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_28(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 28 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_reduce")(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_reduce")(dummy_bk, arg1, arg2)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_to_all")(bk, arg1)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_to_all")(bk, arg1, arg2)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_to_all")(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_29(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 29 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_to_all")(bk, arg_bool)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_to_all")(bk, arg_int)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_to_all")(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_all_to_all")(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_packbits(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_30(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_31(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_32(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_33(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_34(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_35(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_36(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_37(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_38(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_39(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_40(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_41(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_42(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_43(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_44(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_45(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_46(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_47(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_48(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_49(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_50(_state: _TestMathMiscCoverageState) -> None:
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
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def test_msf_packbits_creation() -> None:
    """Tests for NumPy eager bit-packing, array creation, and basic concatenation ops runner."""
    _state: _TestMathMiscCoverageState = _make_base_math_state(23)
    for _part_fn in (
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
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([1.0, 2.0]))
