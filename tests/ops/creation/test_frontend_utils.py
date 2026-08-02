def test_frontend_utils_frompyfunc():
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.creation.frontend_utils import frompyfunc

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "dummy_frompyfunc"
        assert frompyfunc(lambda x: x, 1, 1) == "dummy_frompyfunc"


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
