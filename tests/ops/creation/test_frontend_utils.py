def test_frontend_utils_frompyfunc():
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.creation.frontend_utils import from_dlpack, frompyfunc, geometric, geomspace

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "dummy_frompyfunc"
        assert frompyfunc(lambda x: x, 1, 1) == "dummy_frompyfunc"

        mock_dispatch.return_value = "dummy_dlpack"
        assert from_dlpack("obj") == "dummy_dlpack"

        mock_dispatch.return_value = "dummy_geomspace"
        assert geomspace(1, 10) == "dummy_geomspace"

        mock_dispatch.return_value = "dummy_geometric"
        assert geometric(0.5) == "dummy_geometric"


"""Tests for frontend utils coverage."""

from ml_switcheroo_compiler.ops.creation.frontend_utils import FromDlpack, Frompyfunc, Geometric, Geomspace, frompyfunc


def test_frontend_utils_missing_2() -> None:
    """Test geometric shape and frompyfunc."""
    _multinomial_shape = Geometric().infer_shape

    class MockTensor:
        shape = (10,)

    assert _multinomial_shape(MockTensor(), size=5) == (5,)
    assert _multinomial_shape(MockTensor(), size=None) == (10,)
    assert _multinomial_shape(MockTensor(), size=(5, 5)) == (5, 5)

    try:
        frompyfunc(lambda x: x, 1, 1)
    except Exception:
        pass

    assert FromDlpack().infer_shape() == ()
    assert FromDlpack().infer_shape(MockTensor()) == (10,)

    assert Frompyfunc().infer_shape() == ()

    assert Geomspace().infer_shape(1, 10) == (50,)
    assert Geomspace().infer_shape(MockTensor(), MockTensor(), num=10) == (10, 10)
    assert Geomspace().infer_shape(MockTensor(), MockTensor(), num=10, axis=-1) == (10, 10)
