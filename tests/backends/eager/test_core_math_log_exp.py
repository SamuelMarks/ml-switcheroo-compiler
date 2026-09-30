"""Tests for test_core_math_log_exp."""

from __future__ import annotations

from typing import Any

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


def test_math_log_exp_coverage() -> None:
    """Test all branches and lines of math_log_exp.

    Returns:
        None
    """

    # 1. _expm1
    # Case A: backend has expm1
    class Expm1Backend:
        """Backend with expm1."""

        def expm1(self, x: Any) -> str:
            """Mock expm1."""
            return "expm1_val"

    assert _expm1(Expm1Backend(), 1.0) == "expm1_val"

    # Case B: backend lacks expm1 -> fallback to backend.exp(x) - 1.0
    class ExpBackend:
        """Backend with exp only."""

        def exp(self, x: Any) -> float:
            """Mock exp."""
            return 2.0

    assert _expm1(ExpBackend(), 1.0) == 1.0

    # 2. _float_power
    class FloatPowerBackend:
        """Backend with float_power."""

        def float_power(self, *args: Any, **kwargs: Any) -> str:
            """Mock float_power."""
            return "float_power_val"

    class PowerBackend:
        """Backend with power."""

        def power(self, *args: Any, **kwargs: Any) -> str:
            """Mock power."""
            return "power_val"

    class EmptyBackend:
        """Empty backend."""

        pass

    eb = EmptyBackend()
    assert _float_power(FloatPowerBackend(), 2.0, 3.0) == "float_power_val"
    assert _float_power(PowerBackend(), 2.0, 3.0) == "power_val"
    assert _float_power(eb, 2.0, 3.0) is None

    # 3. _frexp and _ldexp
    class FrexpBackend:
        """Backend with frexp."""

        def frexp(self, *args: Any, **kwargs: Any) -> str:
            """Mock frexp."""
            return "frexp_val"

        def ldexp(self, *args: Any, **kwargs: Any) -> str:
            """Mock ldexp."""
            return "ldexp_val"

    fb = FrexpBackend()
    assert _frexp(fb, 3.0) == "frexp_val"
    assert _ldexp(fb, 0.75, 2) == "ldexp_val"
    assert _frexp(eb, 8.0) == (0.5, 4)
    assert _ldexp(eb, 0.5, 4) == 8.0

    # 4. _slogdet
    # Case A: backend.linalg has slogdet
    class LinalgSlogdet:
        """Container with slogdet."""

        def slogdet(self, *args: Any, **kwargs: Any) -> str:
            """Mock slogdet."""
            return "linalg_slogdet"

    class BackendLinalg:
        """Backend with linalg container."""

        linalg = LinalgSlogdet()

    assert _slogdet(BackendLinalg(), [[1, 2], [3, 4]]) == "linalg_slogdet"

    # Case B: backend has slogdet on root
    class BackendRootSlogdet:
        """Backend with root slogdet."""

        def slogdet(self, *args: Any, **kwargs: Any) -> str:
            """Mock slogdet."""
            return "root_slogdet"

    assert _slogdet(BackendRootSlogdet(), [[1, 2], [3, 4]]) == "root_slogdet"

    # Case C: backend without root slogdet and linalg has no slogdet initially
    class LinalgProxy:
        """Proxy that only has slogdet on second check."""

        def __init__(self) -> None:
            """Initialize proxy."""
            self.calls = 0

        def __getattr__(self, name: str) -> Any:
            """Dynamic attribute lookup."""
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

        def asarray(self, x: Any) -> np.ndarray:
            """Mock asarray."""
            return np.asarray(x)

    assert _slogdet(BackendProxyLinalg(), [[1, 0], [0, 1]]) == "fallback_slogdet_arr"

    # 5. _np_xlog1py
    # Case A: backend has xlog1py
    class Xlog1pyBackend:
        """Backend with xlog1py."""

        def xlog1py(self, *args: Any, **kwargs: Any) -> str:
            """Mock xlog1py."""
            return "xlog1py_val"

    assert _np_xlog1py(Xlog1pyBackend(), 1.0, 2.0) == "xlog1py_val"

    # Case B: fallback to numpy
    res_xlog1py = _np_xlog1py(eb, np.array([0.0, 2.0]), np.array([1.0, 1.0]))
    assert res_xlog1py[0] == 0.0
    assert np.isclose(res_xlog1py[1], 2.0 * np.log1p(1.0))

    # 6. _np_xlogy
    # Case A: backend has xlogy
    class XlogyBackend:
        """Backend with xlogy."""

        def xlogy(self, *args: Any, **kwargs: Any) -> str:
            """Mock xlogy."""
            return "xlogy_val"

    assert _np_xlogy(XlogyBackend(), 1.0, 2.0) == "xlogy_val"

    # Case B: fallback to numpy
    res_xlogy = _np_xlogy(eb, np.array([0.0, 2.0]), np.array([1.0, 2.0]))
    assert res_xlogy[0] == 0.0
    assert np.isclose(res_xlogy[1], 2.0 * np.log(2.0))

    # Registry verification
    for name in ["Expm1", "FloatPower", "Frexp", "Ldexp", "Slogdet", "Xlog1py", "Xlogy"]:
        assert global_eager_registry.get(name) is not None
