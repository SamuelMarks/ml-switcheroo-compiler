# ruff: noqa: E501
import tempfile

import numpy as np

from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat


def test_h5_load_save(mocker):
    fmt = H5WeightFormat()
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    del mock_backend.load_h5
    del mock_backend.save_h5
    with tempfile.NamedTemporaryFile(suffix=".h5") as f:

        class MockNumpy:
            def numpy(self):
                return np.array([1, 2])

        class MockData:
            def __init__(self):
                self.data = MockNumpy()

        class MockList:
            def tolist(self):
                return [3, 4]

        weights = {"a": np.array([1, 2]), "b": MockNumpy(), "c": MockData(), "d": MockList()}
        fmt.save(weights, f.name)
        loaded = fmt.load(f.name)
        assert np.array_equal(loaded["a"], [1, 2])
        assert np.array_equal(loaded["b"], [1, 2])
        assert np.array_equal(loaded["c"], [1, 2])
        assert np.array_equal(loaded["d"], [3, 4])


def test_h5_load_save_backend(mocker):
    fmt = H5WeightFormat()
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    mock_backend.load_h5.return_value = {"a": 1}
    assert fmt.load("test.h5") == {"a": 1}
    fmt.save({"a": 1}, "test.h5")
    mock_backend.save_h5.assert_called_with({"a": 1}, "test.h5")
