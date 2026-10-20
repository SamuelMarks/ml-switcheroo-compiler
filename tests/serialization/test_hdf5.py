"""Test HDF5 serialization format."""

from __future__ import annotations

import os
import tempfile
import typing

import numpy as np

from ml_switcheroo_compiler.serialization.formats.hdf5 import HDF5WeightLoader, HDF5WeightSaver

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_hdf5_save_and_load() -> None:
    """Test saving and loading weights using HDF5 format."""
    weights = {
        "layer1/weight": np.random.rand(10, 10),
        "layer1/bias": np.random.rand(10),
        "layer2/weight": np.random.rand(5, 10),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.h5")

        saver = HDF5WeightSaver()
        typing.cast("EagerCallableProtocol", saver.save)(weights, filepath)

        assert os.path.exists(filepath)

        loader = HDF5WeightLoader()
        loaded_weights = loader.load(filepath)

        assert len(loaded_weights) == 3
        for k, v in weights.items():
            assert k in loaded_weights
            typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerTensorProtocol", getattr(loaded_weights, "__getitem__")(k)), v)


def test_hdf5_schema_validation() -> None:
    """Test schema validation for HDF5."""
    from ml_switcheroo_compiler.serialization.formats.hdf5 import WeightSchema

    # Should construct successfully
    schema = typing.cast("EagerCallableProtocol", WeightSchema)(data={"a": typing.cast("EagerCallableProtocol", np.array)([1, 2, 3])})
    assert "a" in schema.data


def test_hdf5_missing_h5py(monkeypatch: EagerTensorProtocol) -> None:
    """Test that HDF5WeightLoader and HDF5WeightSaver raise ImportError when h5py is None.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import pytest

    import ml_switcheroo_compiler.serialization.formats.hdf5 as hdf5_mod

    monkeypatch.setattr(hdf5_mod, "h5py", None)
    loader = hdf5_mod.HDF5WeightLoader()
    with pytest.raises(ImportError, match="h5py is required for HDF5 weight loading"):
        loader.load("dummy.h5")

    saver = hdf5_mod.HDF5WeightSaver()
    with pytest.raises(ImportError, match="h5py is required for HDF5 weight saving"):
        typing.cast("EagerCallableProtocol", saver.save)({"a": typing.cast("EagerCallableProtocol", np.array)([1])}, "dummy.h5")


def test_h5_and_hdf5_import_error_coverage(tmp_path: EagerTensorProtocol) -> None:
    """Test fallback import branch when h5py is missing.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    import importlib
    import sys
    from unittest import mock

    fake_h5_file = tmp_path / "test.h5"
    fake_h5_file.write_bytes(b"placeholder")

    with mock.patch.dict(sys.modules, {"h5py": None}):
        import ml_switcheroo_compiler.serialization.formats.h5 as h5_mod
        import ml_switcheroo_compiler.serialization.formats.hdf5 as hdf5_mod

        importlib.reload(h5_mod)
        importlib.reload(hdf5_mod)

        assert h5_mod.h5py is None
        assert hdf5_mod.h5py is None

    import ml_switcheroo_compiler.serialization.formats.h5 as h5_mod
    import ml_switcheroo_compiler.serialization.formats.hdf5 as hdf5_mod

    importlib.reload(h5_mod)
    importlib.reload(hdf5_mod)
