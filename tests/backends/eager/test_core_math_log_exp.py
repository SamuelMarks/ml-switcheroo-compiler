"""Tests for test_core_math_log_exp."""

from __future__ import annotations

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
from tests.testing_utils import as_callable, as_mod, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def _test_math_log_exp_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_log_exp_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class Expm1Backend:
        """Backend with expm1."""

        def expm1(self, x: EagerValue) -> str:
            """Mock expm1.

            Args:
                x: Test parameter `x`.
            """
            return "expm1_val"

    assert _expm1(as_mod(Expm1Backend()), 1.0) == "expm1_val"

    # Case B: backend lacks expm1 -> fallback to backend.exp(x) - 1.0
    class ExpBackend:
        """Backend with exp only."""

        def exp(self, x: EagerValue) -> float:
            """Mock exp.

            Args:
                x: Test parameter `x`.
            """
            return 2.0

    assert _expm1(as_mod(ExpBackend()), 1.0) == 1.0

    # 2. _float_power
    class FloatPowerBackend:
        """Backend with float_power."""

        def float_power(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock float_power.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "float_power_val"

    class PowerBackend:
        """Backend with power."""

        def power(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock power.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "power_val"

    getattr(_state, "update")({"ExpBackend": ExpBackend, "Expm1Backend": Expm1Backend, "FloatPowerBackend": FloatPowerBackend, "PowerBackend": PowerBackend})
    _state.update(locals())


def _test_math_log_exp_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_log_exp_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    FloatPowerBackend, PowerBackend = _state["FloatPowerBackend"], _state["PowerBackend"]

    class EmptyBackend:
        """Empty backend."""

        pass

    eb = EmptyBackend()
    assert _float_power(FloatPowerBackend(), 2.0, 3.0) == "float_power_val"
    assert _float_power(PowerBackend(), 2.0, 3.0) == "power_val"
    assert _float_power(as_mod(eb), 2.0, 3.0) is None

    # 3. _frexp and _ldexp
    class FrexpBackend:
        """Backend with frexp."""

        def frexp(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock frexp.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "frexp_val"

        def ldexp(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock ldexp.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "ldexp_val"

    fb = FrexpBackend()
    assert _frexp(as_mod(fb), 3.0) == "frexp_val"
    assert _ldexp(as_mod(fb), 0.75, 2) == "ldexp_val"
    assert _frexp(as_mod(eb), 8.0) == (0.5, 4)
    assert _ldexp(as_mod(eb), 0.5, 4) == 8.0
    getattr(_state, "update")({"EmptyBackend": EmptyBackend, "FrexpBackend": FrexpBackend, "eb": eb, "fb": fb})
    _state.update(locals())


def _test_math_log_exp_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_log_exp_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class LinalgSlogdet:
        """Container with slogdet."""

        def slogdet(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock slogdet.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "linalg_slogdet"

    class BackendLinalg:
        """Backend with linalg container."""

        linalg = LinalgSlogdet()

    assert _slogdet(as_mod(BackendLinalg()), [[1, 2], [3, 4]]) == "linalg_slogdet"

    # Case B: backend has slogdet on root
    class BackendRootSlogdet:
        """Backend with root slogdet."""

        def slogdet(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock slogdet.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "root_slogdet"

    assert _slogdet(as_mod(BackendRootSlogdet()), [[1, 2], [3, 4]]) == "root_slogdet"
    getattr(_state, "update")({"BackendLinalg": BackendLinalg, "BackendRootSlogdet": BackendRootSlogdet, "LinalgSlogdet": LinalgSlogdet})
    _state.update(locals())


def _test_math_log_exp_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_math_log_exp_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class LinalgProxy:
        """Proxy that only has slogdet on second check."""

        def __init__(self) -> None:
            """Initialize proxy."""
            self.calls = 0

        def __getattr__(self, name: str) -> EagerValue:
            """Dynamic attribute lookup.

            Args:
                name: Test parameter `name`.
            """
            if name == "slogdet":
                self.calls += 1
                if self.calls == 1:
                    raise AttributeError("Not yet")
                return lambda arr: "fallback_slogdet_arr"
            raise AttributeError(name)

    class BackendProxyLinalg:
        """Backend with proxy linalg."""

        def __init__(self) -> None:
            """Initialize proxy."""
            self.linalg = LinalgProxy()

        def asarray(self, x: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
            """Mock asarray.

            Args:
                x: Test parameter `x`.
            """
            return np.asarray(x)

    assert _slogdet(as_mod(BackendProxyLinalg()), [[1, 0], [0, 1]]) == "fallback_slogdet_arr"
    getattr(_state, "update")({"BackendProxyLinalg": BackendProxyLinalg, "LinalgProxy": LinalgProxy})
    _state.update(locals())


def _test_math_log_exp_coverage_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_math_log_exp_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    eb = _state["eb"]

    class Xlog1pyBackend:
        """Backend with xlog1py."""

        def xlog1py(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock xlog1py.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "xlog1py_val"

    assert _np_xlog1py(as_mod(Xlog1pyBackend()), 1.0, 2.0) == "xlog1py_val"

    # Case B: fallback to numpy
    res_xlog1py = _np_xlog1py(eb, np.array([0.0, 2.0]), np.array([1.0, 1.0]))
    assert as_tensor(getattr(res_xlog1py, "__getitem__")(0)) == 0.0
    assert as_callable(np.isclose)(as_tensor(getattr(res_xlog1py, "__getitem__")(1)), 2.0 * as_callable(np.log1p)(1.0))

    # 6. _np_xlogy
    # Case A: backend has xlogy
    class XlogyBackend:
        """Backend with xlogy."""

        def xlogy(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock xlogy.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "xlogy_val"

    assert _np_xlogy(as_mod(XlogyBackend()), 1.0, 2.0) == "xlogy_val"

    # Case B: fallback to numpy
    res_xlogy = _np_xlogy(eb, np.array([0.0, 2.0]), np.array([1.0, 2.0]))
    assert as_tensor(getattr(res_xlogy, "__getitem__")(0)) == 0.0
    assert as_callable(np.isclose)(as_tensor(getattr(res_xlogy, "__getitem__")(1)), 2.0 * as_callable(np.log)(2.0))

    # Registry verification
    for name in ["Expm1", "FloatPower", "Frexp", "Ldexp", "Slogdet", "Xlog1py", "Xlogy"]:
        assert global_eager_registry.get(name) is not None
    getattr(_state, "update")({"Xlog1pyBackend": Xlog1pyBackend, "XlogyBackend": XlogyBackend, "res_xlog1py": res_xlog1py, "res_xlogy": res_xlogy})


def test_math_log_exp_coverage() -> None:
    """Test all branches and lines of math_log_exp.

    Returns:
        None
    """
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_log_exp_coverage_part_1(_state)
    _test_math_log_exp_coverage_part_2(_state)
    _test_math_log_exp_coverage_part_3(_state)
    _test_math_log_exp_coverage_part_4(_state)
    _test_math_log_exp_coverage_part_5(_state)
    assert len(_state) > 0
