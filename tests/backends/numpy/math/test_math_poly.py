"""Tests for test_math_poly."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_poly as poly_mod


def test_math_poly_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test all polynomial functions and recurrence branches in math_poly.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for monkeypatching.

    Returns:
        None
    """
    # Test recurrence edge cases
    # max_n < 0
    neg_res = poly_mod._poly_recurrence(-1, 0.5, 1.0, lambda x: x, lambda i, x, t1, t2: 2.0 * x * t1 - t2)
    assert np.all(neg_res == 0.0)

    # max_n == 0
    zero_res = poly_mod._poly_recurrence(0, 0.5, 1.0, lambda x: x, lambda i, x, t1, t2: 2.0 * x * t1 - t2)
    assert zero_res == 1.0

    # max_n == 1
    one_res = poly_mod._poly_recurrence(1, 0.5, 1.0, lambda x: x, lambda i, x, t1, t2: 2.0 * x * t1 - t2)
    assert one_res == 0.5

    # max_n >= 2
    two_res = poly_mod._poly_recurrence(2, 0.5, 1.0, lambda x: x, lambda i, x, t1, t2: 2.0 * x * t1 - t2)
    assert np.isclose(two_res, 2.0 * (0.5**2) - 1.0)

    # chebyshev_polynomial_t
    with pytest.raises(ValueError, match="Expected 2 arguments x and n."):
        poly_mod._np_chebyshev_polynomial_t(np, 0.5)
    t_val = poly_mod._np_chebyshev_polynomial_t(np, 0.5, 2)
    assert np.isclose(t_val, -0.5)
    assert np.isclose(numpy_eager_registry.get("chebyshev_polynomial_t")(np, 0.5, 2), -0.5)

    # chebyshev_polynomial_u
    with pytest.raises(ValueError, match="Expected 2 arguments x and n."):
        poly_mod._np_chebyshev_polynomial_u(np, 0.5)
    u_val = poly_mod._np_chebyshev_polynomial_u(np, 0.5, 2)
    assert np.isclose(numpy_eager_registry.get("chebyshev_polynomial_u")(np, 0.5, 2), u_val)

    # hermite_polynomial_h
    with pytest.raises(ValueError, match="Expected 2 arguments x and n."):
        poly_mod._np_hermite_polynomial_h(np)
    h_val = poly_mod._np_hermite_polynomial_h(np, 0.5, 2)
    assert np.isclose(numpy_eager_registry.get("hermite_polynomial_h")(np, 0.5, 2), h_val)

    # hermite_polynomial_he
    with pytest.raises(ValueError, match="Expected 2 arguments x and n."):
        poly_mod._np_hermite_polynomial_he(np)
    he_val = poly_mod._np_hermite_polynomial_he(np, 0.5, 2)
    assert np.isclose(numpy_eager_registry.get("hermite_polynomial_he")(np, 0.5, 2), he_val)

    # laguerre_polynomial_l
    with pytest.raises(ValueError, match="Expected 2 arguments x and n."):
        poly_mod._np_laguerre_polynomial_l(np)
    l_val = poly_mod._np_laguerre_polynomial_l(np, 0.5, 2)
    assert np.isclose(numpy_eager_registry.get("laguerre_polynomial_l")(np, 0.5, 2), l_val)

    # legendre_polynomial_p
    with pytest.raises(ValueError, match="Expected 2 arguments x and n."):
        poly_mod._np_legendre_polynomial_p(np)
    p_val = poly_mod._np_legendre_polynomial_p(np, 0.5, 2)
    assert np.isclose(numpy_eager_registry.get("legendre_polynomial_p")(np, 0.5, 2), p_val)

    # Shifted chebyshev polynomials
    # Missing args
    assert poly_mod._np_shifted_chebyshev_polynomial_t(np) is None
    assert poly_mod._np_shifted_chebyshev_polynomial_u(np) is None
    assert poly_mod._np_shifted_chebyshev_polynomial_v(np) is None
    assert poly_mod._np_shifted_chebyshev_polynomial_w(np) is None

    # Valid with scipy
    assert poly_mod._np_shifted_chebyshev_polynomial_t(np, 2, 0.5) is not None
    assert numpy_eager_registry.get("shifted_chebyshev_polynomial_t")(np, 2, 0.5) is not None

    assert poly_mod._np_shifted_chebyshev_polynomial_u(np, 2, 0.5) is not None
    assert numpy_eager_registry.get("shifted_chebyshev_polynomial_u")(np, 2, 0.5) is not None

    assert poly_mod._np_shifted_chebyshev_polynomial_v(np, 2, 0.5) is not None
    assert numpy_eager_registry.get("shifted_chebyshev_polynomial_v")(np, 2, 0.5) is not None

    assert poly_mod._np_shifted_chebyshev_polynomial_w(np, 2, 0.5) is not None
    assert numpy_eager_registry.get("shifted_chebyshev_polynomial_w")(np, 2, 0.5) is not None

    # None scipy branch
    monkeypatch.setattr(poly_mod, "_get_sc", lambda: None)
    assert poly_mod._np_shifted_chebyshev_polynomial_t(np, 2, 0.5) is None
    assert poly_mod._np_shifted_chebyshev_polynomial_u(np, 2, 0.5) is None
    assert poly_mod._np_shifted_chebyshev_polynomial_v(np, 2, 0.5) is None
    assert poly_mod._np_shifted_chebyshev_polynomial_w(np, 2, 0.5) is None
