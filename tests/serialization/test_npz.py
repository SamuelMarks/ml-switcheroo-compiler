"""Coverage tests for npz."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.serialization.formats.npz import NpzWeightFormat
from tests.testing_utils import ensure_eager_tensor


def test_npz_load_save(tmp_path: os.PathLike[str]) -> None:
    """Verify `NpzWeightFormat` backend hooks, warning fallback, and pure-Python `.npz` round-trip.

    Args:
        tmp_path (os.PathLike[str]): Temporary filesystem directory fixture.
    """
    import os
    import zipfile

    import numpy as np

    format = NpzWeightFormat()

    # Mock backend with load_npz and save_npz
    mock_backend = MagicMock()
    mock_backend.load_npz.return_value = {"a": 1}

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend):
        res = format.load("dummy.npz")
        assert ensure_eager_tensor(res) == ensure_eager_tensor({"a": 1})
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
                assert ensure_eager_tensor(res) == ensure_eager_tensor({"c": 3})

    # Plain backend without save_npz or load_npz: pure-Python .npz round-trip
    class _PlainNpzBackend:
        """Backend stub without save_npz, load_npz, or get_module hooks."""

    npz_path = os.path.join(str(tmp_path), "weights.npz")
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=_PlainNpzBackend()):
        format.save({"b": np.array([2.0, 3.0], dtype=np.float32), "c.npy": np.array([4.0], dtype=np.float32)}, npz_path)
        assert zipfile.is_zipfile(npz_path)
        loaded = format.load(npz_path)
        assert loaded["b"] == [2.0, 3.0]
        assert loaded["c"] == [4.0]
