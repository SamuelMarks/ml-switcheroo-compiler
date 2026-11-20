"""Tests for test_core_math_log_exp."""

from __future__ import annotations

import types
import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_log_exp import (
    _expm1,
    _float_power,
    _frexp,
    _ldexp,
    _np_xlog1py,
    _np_xlogy,
    _slogdet,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import MockTestStub, coerce_operand_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class _TestMathLogExpCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_math_log_exp_coverage` split helpers."""

    FloatPowerBackend: type
    PowerBackend: type
    eb: types.ModuleType


def _test_math_log_exp_coverage_part_1(_state: _TestMathLogExpCoverageState) -> None:
    """Execute part 1 of test_math_log_exp_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class Expm1Backend(MockTestStub):
        """Backend with expm1."""

        def expm1(self, x: BackendArg) -> str:
            """Mock expm1.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `expm1`.
            """
            return "expm1_val"

    assert _expm1(Expm1Backend(), 1.0) == "expm1_val"

    # Case B: backend lacks expm1 -> fallback to backend.exp(x) - 1.0
    class ExpBackend(MockTestStub):
        """Backend with exp only."""

        def exp(self, x: BackendArg) -> float:
            """Mock exp.

            Args:
                x: Test parameter `x`.

            Returns:
                float: Computed output value from `exp`.
            """
            return 2.0

    assert _expm1(ExpBackend(), 1.0) == 1.0

    # 2. _float_power
    class FloatPowerBackend:
        """Backend with float_power."""

        def float_power(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock float_power.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `float_power`.
            """
            return "float_power_val"

    class PowerBackend:
        """Backend with power."""

        def power(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock power.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `power`.
            """
            return "power_val"

    _state.update({"FloatPowerBackend": FloatPowerBackend, "PowerBackend": PowerBackend})


def _test_math_log_exp_coverage_part_2(_state: _TestMathLogExpCoverageState) -> None:
    """Execute part 2 of test_math_log_exp_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    FloatPowerBackend, PowerBackend = _state["FloatPowerBackend"], _state["PowerBackend"]

    class EmptyBackend(MockTestStub):
        """Empty backend."""

        pass

    eb = EmptyBackend()
    assert _float_power(FloatPowerBackend(), 2.0, 3.0) == "float_power_val"
    assert _float_power(PowerBackend(), 2.0, 3.0) == "power_val"
    assert _float_power(eb, 2.0, 3.0) is None

    # 3. _frexp and _ldexp
    class FrexpBackend(MockTestStub):
        """Backend with frexp."""

        def frexp(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock frexp.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `frexp`.
            """
            return "frexp_val"

        def ldexp(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock ldexp.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `ldexp`.
            """
            return "ldexp_val"

    fb = FrexpBackend()
    assert _frexp(fb, 3.0) == "frexp_val"
    assert _ldexp(fb, 0.75, 2) == "ldexp_val"
    assert _frexp(eb, 8.0) == (0.5, 4)
    assert _ldexp(eb, 0.5, 4) == 8.0
    _state.update({"eb": eb})
    _state.update({"FloatPowerBackend": FloatPowerBackend, "PowerBackend": PowerBackend})


def _test_math_log_exp_coverage_part_3(_state: _TestMathLogExpCoverageState) -> None:
    """Execute part 3 of test_math_log_exp_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class LinalgSlogdet:
        """Container with slogdet."""

        def slogdet(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock slogdet.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `slogdet`.
            """
            return "linalg_slogdet"

    class BackendLinalg(MockTestStub):
        """Backend with linalg container."""

        linalg = LinalgSlogdet()

    assert _slogdet(BackendLinalg(), [[1, 2], [3, 4]]) == "linalg_slogdet"

    # Case B: backend has slogdet on root
    class BackendRootSlogdet(MockTestStub):
        """Backend with root slogdet."""

        def slogdet(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock slogdet.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `slogdet`.
            """
            return "root_slogdet"

    assert _slogdet(BackendRootSlogdet(), [[1, 2], [3, 4]]) == "root_slogdet"


def _test_math_log_exp_coverage_part_4(_state: _TestMathLogExpCoverageState) -> None:
    """Execute part 4 of test_math_log_exp_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class LinalgProxy:
        """Proxy that only has slogdet on second check."""

        def __init__(self) -> None:
            """Initialize proxy."""
            self.calls = 0

        @property
        def slogdet(self) -> BackendArg:
            """Dynamic slogdet property raising on first check.

            Returns:
                BackendArg: Computed output value from `slogdet`.

            Raises:
                AttributeError: When raised during `slogdet` (Not yet).
            """
            self.calls += 1
            if self.calls == 1:
                raise AttributeError("Not yet")
            return lambda arr: "fallback_slogdet_arr"

    class BackendProxyLinalg(MockTestStub):
        """Backend with proxy linalg."""

        def __init__(self) -> None:
            """Initialize proxy."""
            self.linalg = LinalgProxy()

        def asarray(self, x: BackendArg) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
            """Mock asarray.

            Args:
                x: Test parameter `x`.

            Returns:
                np.ndarray[tuple[int, ...], np.dtype[np.generic]]: Computed output value from `asarray`.
            """
            return np.asarray(x)

    assert _slogdet(BackendProxyLinalg(), [[1, 0], [0, 1]]) == "fallback_slogdet_arr"


def _test_math_log_exp_coverage_part_5(_state: _TestMathLogExpCoverageState) -> None:
    """Execute part 5 of test_math_log_exp_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    eb = _state["eb"]

    class Xlog1pyBackend(MockTestStub):
        """Backend with xlog1py."""

        def xlog1py(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock xlog1py.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `xlog1py`.
            """
            return "xlog1py_val"

    assert _np_xlog1py(Xlog1pyBackend(), 1.0, 2.0) == "xlog1py_val"

    # Case B: fallback to numpy
    res_xlog1py = _np_xlog1py(eb, np.array([0.0, 2.0]), np.array([1.0, 1.0]))
    assert ensure_ndarray(res_xlog1py)[0] == 0.0
    assert np.isclose(ensure_ndarray(coerce_operand_list(res_xlog1py)[1]), 2.0 * np.log1p(1.0))

    # 6. _np_xlogy
    # Case A: backend has xlogy
    class XlogyBackend(MockTestStub):
        """Backend with xlogy."""

        def xlogy(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock xlogy.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `xlogy`.
            """
            return "xlogy_val"

    assert _np_xlogy(XlogyBackend(), 1.0, 2.0) == "xlogy_val"

    # Case B: fallback to numpy
    res_xlogy = _np_xlogy(eb, np.array([0.0, 2.0]), np.array([1.0, 2.0]))
    assert ensure_ndarray(res_xlogy)[0] == 0.0
    assert np.isclose(ensure_ndarray(coerce_operand_list(res_xlogy)[1]), 2.0 * np.log(2.0))

    # Registry verification
    for name in ["Expm1", "FloatPower", "Frexp", "Ldexp", "Slogdet", "Xlog1py", "Xlogy"]:
        assert global_eager_registry.get(name) is not None


def test_math_log_exp_coverage() -> None:
    """Test all branches and lines of math_log_exp.

    Returns:
        None
    """
    _state: _TestMathLogExpCoverageState = {}
    _test_math_log_exp_coverage_part_1(_state)
    _test_math_log_exp_coverage_part_2(_state)
    _test_math_log_exp_coverage_part_3(_state)
    _test_math_log_exp_coverage_part_4(_state)
    _test_math_log_exp_coverage_part_5(_state)
    assert len(_state) > 0
