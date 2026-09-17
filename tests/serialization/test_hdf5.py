"""Test HDF5 serialization format."""

import os
import tempfile

import numpy as np

from ml_switcheroo_compiler.serialization.formats.hdf5 import HDF5WeightLoader, HDF5WeightSaver


def test_hdf5_save_and_load():
    """Test saving and loading weights using HDF5 format."""
    weights = {
        "layer1/weight": np.random.rand(10, 10),
        "layer1/bias": np.random.rand(10),
        "layer2/weight": np.random.rand(5, 10),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.h5")

        saver = HDF5WeightSaver()
        saver.save(weights, filepath)

        assert os.path.exists(filepath)

        loader = HDF5WeightLoader()
        loaded_weights = loader.load(filepath)

        assert len(loaded_weights) == 3
        for k, v in weights.items():
            assert k in loaded_weights
            np.testing.assert_allclose(loaded_weights[k], v)


def test_hdf5_schema_validation():
    """Test schema validation for HDF5."""
    from ml_switcheroo_compiler.serialization.formats.hdf5 import WeightSchema

    # Should construct successfully
    schema = WeightSchema(data={"a": np.array([1, 2, 3])})
    assert "a" in schema.data


def test_hdf5_missing_h5py(monkeypatch):
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
        saver.save({"a": np.array([1])}, "dummy.h5")


def test_h5_and_hdf5_import_error_coverage(tmp_path) -> None:
    """Test fallback import branch when h5py is missing."""
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
