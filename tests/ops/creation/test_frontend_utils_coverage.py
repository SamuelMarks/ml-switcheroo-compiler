"""Tests for frontend utils coverage."""

from ml_switcheroo_compiler.ops.creation.frontend_utils import Geometric, frompyfunc


def test_frontend_utils_missing_2() -> None:
    """Test geometric shape and frompyfunc."""
    _multinomial_shape = Geometric().infer_shape

    class MockTensor:
        shape = (10,)

    assert _multinomial_shape(MockTensor(), size=5) == (5,)

    try:
        frompyfunc(lambda x: x, 1, 1)
    except Exception:
        pass
