"""Module test_frontend_utils.py."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.ops.creation.frontend_utils import FromDlpack, Frompyfunc, Geometric, Geomspace, frompyfunc
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


def test_frontend_utils_frompyfunc() -> None:
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


def test_frontend_utils_missing_2() -> None:
    """Test geometric shape and frompyfunc."""
    _multinomial_shape = Geometric().infer_shape

    class MockTensor:
        """MockTensor."""

        shape = (10,)

    assert as_callable(_multinomial_shape)(as_callable(MockTensor)(), size=5) == (5,)
    assert as_callable(_multinomial_shape)(as_callable(MockTensor)(), size=None) == (10,)
    assert as_callable(_multinomial_shape)(as_callable(MockTensor)(), size=(5, 5)) == (5, 5)
    with pytest.raises(RuntimeError):
        frompyfunc(lambda x: x, 1, 1)
    assert FromDlpack().infer_shape() == ()
    assert as_callable(FromDlpack)().infer_shape(as_callable(MockTensor)()) == (10,)
    assert Frompyfunc().infer_shape() == ()
    assert as_callable(Geomspace)().infer_shape(1, 10) == (50,)
    assert as_callable(Geomspace)().infer_shape(as_callable(MockTensor)(), as_callable(MockTensor)(), num=10) == (10, 10)
    assert as_callable(Geomspace)().infer_shape(as_callable(MockTensor)(), as_callable(MockTensor)(), num=10, axis=-1) == (10, 10)


def test_from_dlpack_func() -> None:
    """test_from_dlpack_func."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.creation.frontend_utils import from_dlpack

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "mock_result"
        assert from_dlpack("capsule") == "mock_result"
        mock_dispatch.assert_called_once_with("FromDlpack", "capsule")


def test_geomspace_func() -> None:
    """test_geomspace_func."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.creation.frontend_utils import geomspace

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "mock_result"
        assert geomspace(1, 10, num=50, endpoint=True, dtype="float32", axis=0) == "mock_result"
        mock_dispatch.assert_called_once_with("Geomspace", 1, 10, num=50, endpoint=True, dtype="float32", axis=0)


def test_geometric_func() -> None:
    """test_geometric_func."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.creation.frontend_utils import geometric

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "mock_result"
        assert geometric(0.5, size=(2, 2)) == "mock_result"
        mock_dispatch.assert_called_once_with("Geometric", 0.5, size=(2, 2))


def test_frontend_utils_infer_shapes() -> None:
    """test_frontend_utils_infer_shapes."""
    from ml_switcheroo_compiler.ops.creation.frontend_utils import FromDlpack, Frompyfunc, Geometric, Geomspace

    assert FromDlpack().infer_shape() == ()

    class Dummy:
        """Dummy."""

        shape = (2, 2)

    assert as_callable(FromDlpack)().infer_shape(as_callable(Dummy)()) == (2, 2)
    assert Frompyfunc().infer_shape() == ()
    assert Geomspace().infer_shape() == (50,)
    assert as_callable(Geomspace)().infer_shape(as_callable(Dummy)(), as_callable(Dummy)(), 10, axis=0) == (10, 2, 2)
    assert as_callable(Geomspace)().infer_shape(as_callable(Dummy)(), as_callable(Dummy)(), 10, axis=-1) == (2, 2, 10)
    assert as_callable(Geometric)().infer_shape(as_callable(Dummy)()) == (2, 2)
    assert as_callable(Geometric)().infer_shape(as_callable(Dummy)(), 10) == (10,)
    assert as_callable(Geometric)().infer_shape(as_callable(Dummy)(), size=(3, 3)) == (3, 3)
