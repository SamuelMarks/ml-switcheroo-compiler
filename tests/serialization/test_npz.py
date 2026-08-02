"""Coverage tests for npz."""

from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.serialization.formats.npz import NpzWeightFormat


def test_npz_load_save():
    format = NpzWeightFormat()

    # Mock backend with load_npz and save_npz
    mock_backend = MagicMock()
    mock_backend.load_npz.return_value = {"a": 1}

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend):
        res = format.load("dummy.npz")
        assert res == {"a": 1}
        mock_backend.load_npz.assert_called_with("dummy.npz")

        format.save({"b": 2}, "dummy.npz")
        mock_backend.save_npz.assert_called_with({"b": 2}, "dummy.npz")

    # Mock backend that raises exception
    mock_backend_err = MagicMock()
    mock_backend_err.load_npz.side_effect = Exception("err")
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend_err):
        with patch("ml_switcheroo_compiler.serialization.formats.npz.parse_npz", return_value={"c": 3}):
            with pytest.warns(UserWarning, match="Backend load_npz failed"):
                res = format.load("dummy.npz")
                assert res == {"c": 3}

    # Mock backend without save_npz
    mock_backend_no_save = MagicMock()
    del mock_backend_no_save.save_npz

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend_no_save):
        with patch("numpy.savez") as mock_savez:
            format.save({"b": 2}, "dummy.npz")
            mock_savez.assert_called_with("dummy.npz", b=2)
