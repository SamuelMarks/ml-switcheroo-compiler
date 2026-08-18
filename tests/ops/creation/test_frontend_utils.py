"""Module test_frontend_utils.py."""


def test_frontend_utils_frompyfunc():
    """test_frontend_utils_frompyfunc."""
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
        """MockTensor."""

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


def test_from_dlpack_func():
    """test_from_dlpack_func."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.creation.frontend_utils import from_dlpack

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "mock_result"
        assert from_dlpack("capsule") == "mock_result"
        mock_dispatch.assert_called_once_with("FromDlpack", "capsule")


def test_geomspace_func():
    """test_geomspace_func."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.creation.frontend_utils import geomspace

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "mock_result"
        assert geomspace(1, 10, num=50, endpoint=True, dtype="float32", axis=0) == "mock_result"
        mock_dispatch.assert_called_once_with("Geomspace", 1, 10, num=50, endpoint=True, dtype="float32", axis=0)


def test_geometric_func():
    """test_geometric_func."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.creation.frontend_utils import geometric

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "mock_result"
        assert geometric(0.5, size=(2, 2)) == "mock_result"
        mock_dispatch.assert_called_once_with("Geometric", 0.5, size=(2, 2))


def test_frontend_utils_infer_shapes():
    """test_frontend_utils_infer_shapes."""
    from ml_switcheroo_compiler.ops.creation.frontend_utils import FromDlpack, Frompyfunc, Geometric, Geomspace

    # FromDlpack
    assert FromDlpack().infer_shape() == ()

    class Dummy:
        """Dummy."""

        shape = (2, 2)

    assert FromDlpack().infer_shape(Dummy()) == (2, 2)

    # Frompyfunc
    assert Frompyfunc().infer_shape() == ()

    # Geomspace
    assert Geomspace().infer_shape() == (50,)
    assert Geomspace().infer_shape(Dummy(), Dummy(), 10, axis=0) == (10, 2, 2)
    assert Geomspace().infer_shape(Dummy(), Dummy(), 10, axis=-1) == (2, 2, 10)

    # Geometric
    assert Geometric().infer_shape(Dummy()) == (2, 2)
    assert Geometric().infer_shape(Dummy(), 10) == (10,)
    assert Geometric().infer_shape(Dummy(), size=(3, 3)) == (3, 3)
