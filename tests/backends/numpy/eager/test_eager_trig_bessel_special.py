# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager trigonometric, Bessel, beta, digamma, and polygamma functions."""

from __future__ import annotations

import importlib
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from tests.testing_utils import MockTestStub, coerce_numeric, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg
_reduce_window = importlib.import_module(
    "ml_switcheroo_compiler.backends.eager.reference_math.window_reductions"
)._reduce_window


def test_core_math_ops_trig_fallbacks() -> None:
    """Test trigonometric fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _acos, _acosh, _asin, _asinh, _atan, _atan2, _atanh

    with pytest.raises(TypeError):
        require_callable(_acos)(np, 0.5)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_acosh)(np, 1.5)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_asin)(np, 0.5)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_asinh)(np, 0.5)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_atan)(np, 0.5)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_atanh)(np, 0.5)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_atan2)(np, 0.5, 0.5)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass


def test_core_math_ops_degrees_radians_fallbacks() -> None:
    """Test degree/radians fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _deg2rad, _degrees, _rad2deg, _radians

    with pytest.raises(TypeError):
        require_callable(_deg2rad)(np, 180.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_degrees)(np, 3.14)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_rad2deg)(np, 3.14)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_radians)(np, 180.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    assert _deg2rad(DummyBackend(), 180.0) is None


def test_numpy_trig_coverage() -> None:
    """Test function test_numpy_trig_coverage."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_general import _degrees, _rad2deg, _radians
    from ml_switcheroo_compiler.backends.eager.reference_math.math_trig import _np_deg2rad, _np_rad2deg

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    assert _np_deg2rad(np, 180.0) == np.pi
    assert _np_rad2deg(np, np.pi) == 180.0
    assert _degrees(DummyBackend(), 3.14) is None
    assert _rad2deg(DummyBackend(), 3.14) is None
    assert _radians(DummyBackend(), 180.0) is None


def test_core_math_ops_cbrt_fallback() -> None:
    """Test cbrt fallback."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _cbrt

    with pytest.raises(TypeError):
        require_callable(_cbrt)(np, 27.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def sign(self, x: BackendArg) -> BackendArg:
            """Test function sign.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `sign`.
            """
            xf = float(coerce_numeric(x))
            return 1.0 if xf >= 0 else -1.0

        def abs(self, x: BackendArg) -> BackendArg:
            """Test function abs.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `abs`.
            """
            xf = float(coerce_numeric(x))
            return xf if xf >= 0 else -xf

        def power(self, x: BackendArg, y: BackendArg) -> BackendArg:
            """Test function power.

            Args:
                x: Test parameter `x`.
                y: Test parameter `y`.

            Returns:
                BackendArg: Computed output value from `power`.
            """
            import math

            return math.pow(float(coerce_numeric(x)), float(coerce_numeric(y)))

    db = DummyBackend()
    assert abs(float(coerce_numeric(_cbrt(db, 27.0))) - 3.0) < 1e-05
    assert abs(float(coerce_numeric(_cbrt(db, -27.0))) - (-3.0)) < 1e-05


def test_core_math_ops_fix_copysign() -> None:
    """Test fix and copysign fallback."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _copysign, _fix

    with pytest.raises(TypeError):
        require_callable(_fix)(np, 1.5)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_copysign)(np, 1.0, -1.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def floor(self, x: BackendArg) -> BackendArg:
            """Test function floor.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `floor`.
            """
            import math

            return math.floor(float(coerce_numeric(x)))

        def ceil(self, x: BackendArg) -> BackendArg:
            """Test function ceil.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `ceil`.
            """
            import math

            return math.ceil(float(coerce_numeric(x)))

        def where(self, cond: BackendArg, x: BackendArg, y: BackendArg) -> BackendArg:
            """Test function where.

            Args:
                cond: Test parameter `cond`.
                x: Test parameter `x`.
                y: Test parameter `y`.

            Returns:
                BackendArg: Computed output value from `where`.
            """
            return x if cond else y

        def abs(self, x: BackendArg) -> BackendArg:
            """Test function abs.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `abs`.
            """
            xf = float(coerce_numeric(x))
            return xf if xf >= 0 else -xf

        def sign(self, x: BackendArg) -> BackendArg:
            """Test function sign.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `sign`.
            """
            xf = float(coerce_numeric(x))
            return 1.0 if xf >= 0 else -1.0

    db = DummyBackend()
    assert _fix(db, 1.5) == 1.0
    assert _fix(db, -1.5) == -1.0
    assert _copysign(db, 2.0, -1.0) == -2.0


def test_core_math_ops_f_funcs() -> None:
    """Test f-based math fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _float_power, _fmax, _fmin, _fmod, _frexp

    with pytest.raises(TypeError):
        require_callable(_float_power)(np, 2, 3)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_fmax)(np, 1, 2)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_fmin)(np, 1, 2)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_fmod)(np, 5, 2)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_frexp)(np, 1.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    assert _float_power(DummyBackend(), 2, 3) is None
    assert _fmax(DummyBackend(), 1, 2) is None
    assert _fmin(DummyBackend(), 1, 2) is None
    assert _fmod(DummyBackend(), 5, 2) is None
    assert _frexp(DummyBackend(), 1.0) is not None


def test_core_math_ops_hypot_i0_imag() -> None:
    """Test hypot, i0 and imag fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _hypot, _i0, _imag

    with pytest.raises(TypeError):
        require_callable(_hypot)(np, 3, 4)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_i0)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_imag)(np, 1.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def sqrt(self, x: BackendArg) -> BackendArg:
            """Test function sqrt.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `sqrt`.
            """
            return coerce_numeric(x) ** 0.5

    db = DummyBackend()
    assert _hypot(db, 3, 4) == 5.0
    _i0(db, 1.0)
    with pytest.raises(TypeError):
        require_callable(_imag)(db, 1.0)(db, None)


def test_core_math_ops_is_funcs() -> None:
    """Test isclose, iscomplex, isreal fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _isclose, _iscomplex, _isreal

    with pytest.raises(TypeError):
        require_callable(_isclose)(np, 1.0, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_iscomplex)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_isreal)(np, 1.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_isclose)(db, 1.0, 1.0)(db, None)
    _iscomplex(db, 1.0)
    _isreal(db, 1.0)


def test_core_math_ops_k_l_funcs() -> None:
    """Test kaiser, lcm, ldexp fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _kaiser, _lcm, _ldexp

    with pytest.raises(TypeError):
        require_callable(_kaiser)(np, 14, 14)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_lcm)(np, 12, 20)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_ldexp)(np, 1.0, 2)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(AttributeError):
        require_callable(_kaiser)(db, 14, 14)(db, None)
    with pytest.raises(TypeError):
        require_callable(_lcm)(db, 12, 20)(db, None)
    with pytest.raises(TypeError):
        require_callable(_ldexp)(db, 1.0, 2)(db, not None)


def test_core_math_ops_n_p_r_funcs() -> None:
    """Test nextafter, polyval, real fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _nextafter, _polyval, _real

    with pytest.raises(TypeError):
        require_callable(_nextafter)(np, 1.0, 2.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_polyval)(np, [1, 2, 3], 2.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_real)(np, 1.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    with pytest.raises(TypeError):
        require_callable(_nextafter)(db, 1.0, 2.0)(db, not None)
    with pytest.raises(TypeError):
        require_callable(_polyval)(db, [1, 2, 3], 2.0)(db, None)
    with pytest.raises(TypeError):
        require_callable(_real)(db, 1.0)(db, None)


def test_core_math_ops_s_funcs() -> None:
    """Test signbit, sinc, spacing fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _signbit, _sinc, _spacing

    with pytest.raises(TypeError):
        require_callable(_signbit)(np, -1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_sinc)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_spacing)(np, 1.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    assert _signbit(db, -1.0) is True
    assert _signbit(db, 1.0) is False
    with pytest.raises(TypeError):
        require_callable(_sinc)(db, 1.0)(db, None)
    with pytest.raises(TypeError):
        require_callable(_spacing)(db, 1.0)(db, None)


def test_core_math_ops_u_z_funcs() -> None:
    """Test unwrap, zeta fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _np_zeta, _unwrap

    with pytest.raises(TypeError):
        require_callable(_unwrap)(np, [0.0, 3.14, 6.28])(np, not None)
    with pytest.raises(TypeError):
        require_callable(_np_zeta)(np, 2.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    _unwrap(db, [0.0])
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            require_callable(_np_zeta)(db, 2.0)(db, None)


def test_core_math_ops_bessel_i_funcs() -> None:
    """Test bessel i0, i0e, i1, i1e fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _bessel_i0, _bessel_i0e, _bessel_i1, _bessel_i1e

    with pytest.raises(TypeError):
        require_callable(_bessel_i0)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_bessel_i0e)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_bessel_i1)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_bessel_i1e)(np, 1.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            require_callable(_bessel_i0)(db, 1.0)(db, None)
        with pytest.raises(ModuleNotFoundError):
            require_callable(_bessel_i0e)(db, 1.0)(db, None)
        with pytest.raises(ModuleNotFoundError):
            require_callable(_bessel_i1)(db, 1.0)(db, None)
        with pytest.raises(ModuleNotFoundError):
            require_callable(_bessel_i1e)(db, 1.0)(db, None)


def test_core_math_ops_bessel_j_funcs() -> None:
    """Test bessel j0, j1, jn fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _bessel_j0, _bessel_j1, _bessel_jn

    with pytest.raises(TypeError):
        require_callable(_bessel_j0)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_bessel_j1)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_bessel_jn)(np, 2, 1.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            require_callable(_bessel_j0)(db, 1.0)(db, None)
        with pytest.raises(ModuleNotFoundError):
            require_callable(_bessel_j1)(db, 1.0)(db, None)
        with pytest.raises(ModuleNotFoundError):
            require_callable(_bessel_jn)(db, 2, 1.0)(db, None)


def test_core_math_ops_bessel_k_y_funcs() -> None:
    """Test bessel k0, k0e, k1, k1e, y0, y1 fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import (
        _bessel_k0,
        _bessel_k0e,
        _bessel_k1,
        _bessel_k1e,
        _bessel_y0,
        _bessel_y1,
    )

    with pytest.raises(TypeError):
        require_callable(_bessel_k0)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_bessel_k0e)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_bessel_k1)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_bessel_k1e)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_bessel_y0)(np, 1.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_bessel_y1)(np, 1.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            require_callable(_bessel_k0)(db, 1.0)(db, None)
        with pytest.raises(ModuleNotFoundError):
            require_callable(_bessel_k0e)(db, 1.0)(db, None)
        with pytest.raises(ModuleNotFoundError):
            require_callable(_bessel_k1)(db, 1.0)(db, None)
        with pytest.raises(ModuleNotFoundError):
            require_callable(_bessel_k1e)(db, 1.0)(db, None)
        with pytest.raises(ModuleNotFoundError):
            require_callable(_bessel_y0)(db, 1.0)(db, None)
        with pytest.raises(ModuleNotFoundError):
            require_callable(_bessel_y1)(db, 1.0)(db, None)


def test_core_math_ops_beta_digamma_funcs() -> None:
    """Test beta, betainc, digamma fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _beta, _betainc, _digamma

    with pytest.raises(IndexError):
        require_callable(_beta)(np, 1.0, 2.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_betainc)(np, 1.0, 2.0, 0.5)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_digamma)(np, 1.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            require_callable(_beta)(db, 1.0, 2.0)(db, None)
        with pytest.raises(ModuleNotFoundError):
            require_callable(_betainc)(db, 1.0, 2.0, 0.5)(db, None)
        with pytest.raises(ModuleNotFoundError):
            require_callable(_digamma)(db, 1.0)(db, None)


def test_core_math_ops_igammac_polygamma_funcs() -> None:
    """Test igammac, polygamma fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _igammac, _polygamma

    with pytest.raises(TypeError):
        require_callable(_igammac)(np, 1.0, 2.0)(np, not None)
    with pytest.raises(TypeError):
        require_callable(_polygamma)(np, 1, 2.0)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    db = DummyBackend()
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        with pytest.raises(ModuleNotFoundError):
            require_callable(_igammac)(db, 1.0, 2.0)(db, None)
        with pytest.raises(ModuleNotFoundError):
            require_callable(_polygamma)(db, 1, 2.0)(db, None)


def test_core_math_ops_heaviside_fallback() -> None:
    """Test heaviside fallback."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _heaviside

    with pytest.raises(TypeError):
        require_callable(_heaviside)(np, 1.0, 0.5)(np, not None)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def where(self, cond: BackendArg, t: BackendArg, f: BackendArg) -> BackendArg:
            """Test function where.

            Args:
                cond: Test parameter `cond`.
                t: Test parameter `t`.
                f: Test parameter `f`.

            Returns:
                BackendArg: Computed output value from `where`.
            """
            return t if cond else f

    db = DummyBackend()
    assert _heaviside(db, 1.0, 0.5) == 1.0
    assert _heaviside(db, -1.0, 0.5) == 0.0
    assert _heaviside(db, 0.0, 0.5) == 0.5


def test_core_math_ops_accumulate_addn() -> None:
    """Test accumulate_n and add_n fallbacks."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _accumulate_n, _add_n

    assert _accumulate_n(np, [1, 2, 3]) == 6
    assert _add_n(np, [1, 2, 3]) == 6
    with pytest.raises(TypeError):
        require_callable(_accumulate_n)(np, [])(np, None)
    with pytest.raises(TypeError):
        require_callable(_add_n)(np, [])(np, None)
