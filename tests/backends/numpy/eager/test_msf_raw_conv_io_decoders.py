# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager raw convolution, matmul, and CSV/image decoding ops."""

from __future__ import annotations

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state
from tests.testing_utils import require_callable

mod = _mod_raw

_TestMathMiscCoverageState = BaseMathMiscState


def _test_math_misc_coverage_part_243(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 243 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_rawmerge(bk, arg1, arg1, arg2)
    mod._np_rawmerge(bk, arg_bool)
    mod._np_rawmerge(bk, arg_int)
    mod._np_rawmerge(dummy_bk, arg1)
    mod._np_rawmerge(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_244(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 244 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_rawop(bk, arg1)
    mod._np_rawop(bk, arg1, arg2)
    mod._np_rawop(bk, arg1, arg1, arg2)
    mod._np_rawop(bk, arg_bool)
    mod._np_rawop(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_245(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 245 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_rawop(dummy_bk, arg1)
    mod._np_rawop(dummy_bk, arg1, arg2)
    mod._np_rawswitch(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_rawswitch(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_rawswitch(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_246(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 246 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_rawswitch(bk, arg_bool)
    mod._np_rawswitch(bk, arg_int)
    mod._np_rawswitch(dummy_bk, arg1)
    with pytest.raises(ValueError):
        mod._np_rawswitch(dummy_bk, arg1, arg2)
    mod._np_scanop(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_247(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 247 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_scanop(bk, arg1, arg2)
    mod._np_scanop(bk, arg1, arg1, arg2)
    mod._np_scanop(bk, arg_bool)
    mod._np_scanop(bk, arg_int)
    mod._np_scanop(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_248(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 248 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_scanop(dummy_bk, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_sobolsample)(bk, arg1)
    with pytest.raises(TypeError):
        require_callable(mod._np_sobolsample)(bk, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_sobolsample)(bk, arg1, arg1, arg2)
    with pytest.raises(TypeError):
        require_callable(mod._np_sobolsample)(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_249(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 249 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(TypeError):
        require_callable(mod._np_sobolsample)(bk, arg_int)
    with pytest.raises(TypeError):
        require_callable(mod._np_sobolsample)(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        require_callable(mod._np_sobolsample)(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsedensematmul(bk, arg1)
    mod._np_sparsedensematmul(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_250(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 250 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_sparsedensematmul(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsedensematmul(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_sparsedensematmul(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_sparsedensematmul(dummy_bk, arg1)
    mod._np_sparsedensematmul(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_251(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 251 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(IndexError):
        mod._np_sparsemapvalues(bk, arg1)
    with pytest.raises(TypeError):
        mod._np_sparsemapvalues(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_sparsemapvalues(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsemapvalues(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_sparsemapvalues(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_252(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 252 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_sparsemapvalues(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_sparsemapvalues(dummy_bk, arg1, arg2)
    mod._np_sparsereducemax(bk, arg1)
    mod._np_sparsereducemax(bk, arg1, arg2)
    mod._np_sparsereducemax(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_253(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 253 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_sparsereducemax(bk, arg_bool)
    mod._np_sparsereducemax(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_sparsereducemax(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_sparsereducemax(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsereshape(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_254(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 254 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_sparsereshape(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_sparsereshape(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsereshape(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_sparsereshape(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_sparsereshape(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_255(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 255 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_sparsereshape(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsesampledadd(bk, arg1)
    mod._np_sparsesampledadd(bk, arg1, arg2)
    mod._np_sparsesampledadd(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_sparsesampledadd(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_256(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 256 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(IndexError):
        mod._np_sparsesampledadd(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_sparsesampledadd(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_sparsesampledadd(dummy_bk, arg1, arg2)
    mod._np_sparsesegmentsum(bk, arg1)
    mod._np_sparsesegmentsum(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_257(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 257 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_sparsesegmentsum(bk, arg1, arg1, arg2)
    mod._np_sparsesegmentsum(bk, arg_bool)
    mod._np_sparsesegmentsum(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_sparsesegmentsum(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_sparsesegmentsum(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_258(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 258 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_sparsetranspose(bk, arg1)
    mod._np_sparsetranspose(bk, arg1, arg2)
    mod._np_sparsetranspose(bk, arg1, arg1, arg2)
    mod._np_sparsetranspose(bk, arg_bool)
    mod._np_sparsetranspose(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_259(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 259 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        mod._np_sparsetranspose(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_sparsetranspose(dummy_bk, arg1, arg2)
    mod._np_switchop(bk, arg1)
    with pytest.raises(ValueError):
        mod._np_switchop(bk, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_switchop(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_260(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 260 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_switchop(bk, arg_bool)
    mod._np_switchop(bk, arg_int)
    mod._np_switchop(dummy_bk, arg1)
    with pytest.raises(ValueError):
        mod._np_switchop(dummy_bk, arg1, arg2)
    mod._np_tensor(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_261(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 261 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(TypeError):
        mod._np_tensor(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_tensor(bk, arg1, arg1, arg2)
    mod._np_tensor(bk, arg_bool)
    mod._np_tensor(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_tensor(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_262(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 262 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_tensor(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(bk, arg1)
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_263(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 263 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(dummy_bk, arg1)
    with pytest.raises(IndexError):
        mod._np_tensorarrayread(dummy_bk, arg1, arg2)
    mod._np_tensorarraystack(bk, arg1)
    mod._np_tensorarraystack(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_264(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 264 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_tensorarraystack(bk, arg1, arg1, arg2)
    mod._np_tensorarraystack(bk, arg_bool)
    mod._np_tensorarraystack(bk, arg_int)
    with pytest.raises(AttributeError):
        mod._np_tensorarraystack(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_tensorarraystack(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_265(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 265 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(IndexError):
        mod._np_tensorarraywrite(bk, arg1)
    with pytest.raises(IndexError):
        mod._np_tensorarraywrite(bk, arg1, arg2)
    with pytest.raises(TypeError):
        mod._np_tensorarraywrite(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_tensorarraywrite(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_tensorarraywrite(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_266(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 266 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(IndexError):
        mod._np_tensorarraywrite(dummy_bk, arg1)
    with pytest.raises(IndexError):
        mod._np_tensorarraywrite(dummy_bk, arg1, arg2)
    mod._np_tensorconfig(bk, arg1)
    mod._np_tensorconfig(bk, arg1, arg2)
    mod._np_tensorconfig(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_267(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 267 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_tensorconfig(bk, arg_bool)
    mod._np_tensorconfig(bk, arg_int)
    mod._np_tensorconfig(dummy_bk, arg1)
    mod._np_tensorconfig(dummy_bk, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_vecdot(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_268(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 268 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_vecdot(bk, arg1, arg2)
    mod._np_vecdot(bk, arg1, arg1, arg2)
    with pytest.raises(IndexError):
        mod._np_vecdot(bk, arg_bool)
    with pytest.raises(IndexError):
        mod._np_vecdot(bk, arg_int)
    with pytest.raises(IndexError):
        mod._np_vecdot(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_269(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 269 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        mod._np_vecdot(dummy_bk, arg1, arg2)
    mod._np_decode_csv(bk, arg1)
    mod._np_decode_csv(bk, arg1, arg2)
    mod._np_decode_csv(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_decode_csv(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_270(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 270 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_decode_csv(bk, arg_int)
    mod._np_decode_csv(dummy_bk, arg1)
    mod._np_decode_csv(dummy_bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_decode_image(bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_decode_image(bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def test_msf_raw_conv_io_decoders() -> None:
    """Tests for NumPy eager raw convolution, matmul, and CSV/image decoding ops runner."""
    _state: _TestMathMiscCoverageState = _make_base_math_state(243)
    for _part_fn in (
        _test_math_misc_coverage_part_243,
        _test_math_misc_coverage_part_244,
        _test_math_misc_coverage_part_245,
        _test_math_misc_coverage_part_246,
        _test_math_misc_coverage_part_247,
        _test_math_misc_coverage_part_248,
        _test_math_misc_coverage_part_249,
        _test_math_misc_coverage_part_250,
        _test_math_misc_coverage_part_251,
        _test_math_misc_coverage_part_252,
        _test_math_misc_coverage_part_253,
        _test_math_misc_coverage_part_254,
        _test_math_misc_coverage_part_255,
        _test_math_misc_coverage_part_256,
        _test_math_misc_coverage_part_257,
        _test_math_misc_coverage_part_258,
        _test_math_misc_coverage_part_259,
        _test_math_misc_coverage_part_260,
        _test_math_misc_coverage_part_261,
        _test_math_misc_coverage_part_262,
        _test_math_misc_coverage_part_263,
        _test_math_misc_coverage_part_264,
        _test_math_misc_coverage_part_265,
        _test_math_misc_coverage_part_266,
        _test_math_misc_coverage_part_267,
        _test_math_misc_coverage_part_268,
        _test_math_misc_coverage_part_269,
        _test_math_misc_coverage_part_270,
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([0.0, 0.0]))
