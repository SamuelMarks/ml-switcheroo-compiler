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
