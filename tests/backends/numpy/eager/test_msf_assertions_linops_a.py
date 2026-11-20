# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager assertion ops and adjoint/block-diagonal LinearOperators."""

from __future__ import annotations

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state

mod = _mod_raw

_TestMathMiscCoverageState = BaseMathMiscState


def _test_math_misc_coverage_part_189(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 189 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_asstringconfig(dummy_bk, arg1)
    mod._np_asstringconfig(dummy_bk, arg1, arg2)
    with pytest.raises(AssertionError):
        mod._np_assertop(bk, arg1)
    with pytest.raises(AssertionError):
        mod._np_assertop(bk, arg1, arg2)
    with pytest.raises(AssertionError):
        mod._np_assertop(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_190(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 190 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(AssertionError):
        mod._np_assertop(bk, arg_bool)
    mod._np_assertop(bk, arg_int)
    with pytest.raises(AssertionError):
        mod._np_assertop(dummy_bk, arg1)
    with pytest.raises(AssertionError):
        mod._np_assertop(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_blurconfig(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_191(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 191 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_blurconfig(bk, arg1, arg2)
    mod._np_blurconfig(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_blurconfig(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_blurconfig(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_blurconfig(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_192(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 192 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_blurconfig(dummy_bk, arg1, arg2)
    mod._np_callable(bk, arg1)
    mod._np_callable(bk, arg1, arg2)
    mod._np_callable(bk, arg1, arg1, arg2)
    mod._np_callable(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_193(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 193 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_callable(bk, arg_int)
    mod._np_callable(dummy_bk, arg1)
    mod._np_callable(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedlocal(bk, arg1)
    mod._np_convgeneraldilatedlocal(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_194(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 194 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_convgeneraldilatedlocal(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedlocal(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedlocal(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedlocal(dummy_bk, arg1)
    mod._np_convgeneraldilatedlocal(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_195(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 195 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedpatches(bk, arg1)
    mod._np_convgeneraldilatedpatches(bk, arg1, arg2)
    mod._np_convgeneraldilatedpatches(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedpatches(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedpatches(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_196(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 196 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_convgeneraldilatedpatches(dummy_bk, arg1)
    mod._np_convgeneraldilatedpatches(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_convwithgeneralpadding(bk, arg1)
    mod._np_convwithgeneralpadding(bk, arg1, arg2)
    mod._np_convwithgeneralpadding(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_197(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 197 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_convwithgeneralpadding(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_convwithgeneralpadding(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_convwithgeneralpadding(dummy_bk, arg1)
    mod._np_convwithgeneralpadding(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_customlinearsolve(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_198(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 198 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_customlinearsolve(bk, arg1, arg2)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_customlinearsolve(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_customlinearsolve(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_customlinearsolve(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_customlinearsolve(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_199(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 199 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        mod._np_customlinearsolve(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_customroot(bk, arg1)
    mod._np_customroot(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_customroot(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_customroot(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_200(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 200 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(IndexError):
        mod._np_customroot(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_customroot(dummy_bk, arg1)
    mod._np_customroot(dummy_bk, arg1, arg2)
    mod._np_debuginfs(bk, arg1)
    mod._np_debuginfs(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_201(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 201 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_debuginfs(bk, arg1, arg1, arg2)
    mod._np_debuginfs(bk, arg_bool)
    mod._np_debuginfs(bk, arg_int)
    mod._np_debuginfs(dummy_bk, arg1)
    mod._np_debuginfs(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_202(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 202 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_debugnans(bk, arg1)
    mod._np_debugnans(bk, arg1, arg2)
    mod._np_debugnans(bk, arg1, arg1, arg2)
    mod._np_debugnans(bk, arg_bool)
    mod._np_debugnans(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_203(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 203 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_debugnans(dummy_bk, arg1)
    mod._np_debugnans(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_dotgeneral(bk, arg1)
    mod._np_dotgeneral(bk, arg1, arg2)
    mod._np_dotgeneral(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_204(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 204 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_dotgeneral(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_dotgeneral(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_dotgeneral(dummy_bk, arg1)
    mod._np_dotgeneral(dummy_bk, arg1, arg2)
    mod._np_elasticconfig(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_205(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 205 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_elasticconfig(bk, arg1, arg2)
    mod._np_elasticconfig(bk, arg1, arg1, arg2)
    mod._np_elasticconfig(bk, arg_bool)
    mod._np_elasticconfig(bk, arg_int)
    mod._np_elasticconfig(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_206(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 206 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_elasticconfig(dummy_bk, arg1, arg2)
    mod._np_extractpatchesoptions(bk, arg1)
    mod._np_extractpatchesoptions(bk, arg1, arg2)
    mod._np_extractpatchesoptions(bk, arg1, arg1, arg2)
    mod._np_extractpatchesoptions(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_207(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 207 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_extractpatchesoptions(bk, arg_int)
    mod._np_extractpatchesoptions(dummy_bk, arg1)
    mod._np_extractpatchesoptions(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperator(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperator(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_208(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 208 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperator(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperator(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperator(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperator(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperator(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_209(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 209 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_210(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 210 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatoradjoint(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_211(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 211 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblockdiag(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_212(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 212 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_213(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 213 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatorblocklowertriangular(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_214(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 214 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_215(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 215 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(bk, arg_bool)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(bk, arg_int)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(dummy_bk, arg1)
    with pytest.raises(TypeError):
        mod._np_linearoperatorcirculant2d(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def test_msf_assertions_linops_a() -> None:
    """Tests for NumPy eager assertion ops and adjoint/block-diagonal LinearOperators runner."""
    _state: _TestMathMiscCoverageState = _make_base_math_state(189)
    for _part_fn in (
        _test_math_misc_coverage_part_189,
        _test_math_misc_coverage_part_190,
        _test_math_misc_coverage_part_191,
        _test_math_misc_coverage_part_192,
        _test_math_misc_coverage_part_193,
        _test_math_misc_coverage_part_194,
        _test_math_misc_coverage_part_195,
        _test_math_misc_coverage_part_196,
        _test_math_misc_coverage_part_197,
        _test_math_misc_coverage_part_198,
        _test_math_misc_coverage_part_199,
        _test_math_misc_coverage_part_200,
        _test_math_misc_coverage_part_201,
        _test_math_misc_coverage_part_202,
        _test_math_misc_coverage_part_203,
        _test_math_misc_coverage_part_204,
        _test_math_misc_coverage_part_205,
        _test_math_misc_coverage_part_206,
        _test_math_misc_coverage_part_207,
        _test_math_misc_coverage_part_208,
        _test_math_misc_coverage_part_209,
        _test_math_misc_coverage_part_210,
        _test_math_misc_coverage_part_211,
        _test_math_misc_coverage_part_212,
        _test_math_misc_coverage_part_213,
        _test_math_misc_coverage_part_214,
        _test_math_misc_coverage_part_215,
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([0.0, 0.0]))
