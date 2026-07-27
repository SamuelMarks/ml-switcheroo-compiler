"""Tests for extra."""

from ml_switcheroo_compiler.backends.numpy.eager.math_unary import _np_erf, _np_erfc


class DummyModule:
    """Module without asarray."""

    def vectorize(self, f):
        """Vectorize."""

        def _f(x):
            """F."""
            return [f(i) for i in x]

        return _f


def test_erf_erfc_no_asarray() -> None:
    """Test coverage for branches."""
    mod = DummyModule()

    # x is a scalar
    res_erf = _np_erf(mod, 0.0)
    assert res_erf == 0.0

    res_erfc = _np_erfc(mod, 0.0)
    assert res_erfc == 1.0
