import json
import os
import struct
import tempfile

import numpy as np

from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat


def test_safetensors_format():
    fmt = SafetensorsWeightFormat()
    data = {
        "a": np.array([1.0, 2.0], dtype=np.float32),
        "b": np.array([3, 4], dtype=np.int32),
        "c": "not_a_tensor",  # Should be ignored because no dtype
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "weights.safetensors")
        fmt.save(data, filepath)

        loaded = fmt.load(filepath)
        assert "a" in loaded
        assert "b" in loaded
        assert "c" not in loaded

        # In our implementation, since we might not have a backend, it returns dicts
        # with 'buffer', 'dtype', 'shape' or directly backends.
        if isinstance(loaded["a"], dict):
            assert loaded["a"]["dtype"] == "F32"
            assert loaded["b"]["dtype"] == "I32"


def test_safetensors_invalid_header():
    fmt = SafetensorsWeightFormat()
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "invalid.safetensors")
        with open(filepath, "wb") as f:
            f.write(b"short")  # Less than 8 bytes
        loaded = fmt.load(filepath)
        assert loaded == {}


def test_safetensors_metadata():
    fmt = SafetensorsWeightFormat()
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "metadata.safetensors")
        # Create a safetensors file manually with __metadata__
        header = {"__metadata__": {"foo": "bar"}, "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
        header_bytes = json.dumps(header).encode("utf-8")
        padding = (8 - (len(header_bytes) % 8)) % 8
        header_bytes += b" " * padding
        with open(filepath, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(np.array([1.0, 2.0], dtype=np.float32).tobytes())

        loaded = fmt.load(filepath)
        assert "a" in loaded
        assert "__metadata__" not in loaded


def test_safetensors_with_backend_mock(mocker):
    fmt = SafetensorsWeightFormat()

    class MockBackend:
        def from_buffer(self, buffer, dtype, shape):
            return "mocked_tensor"

    mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockBackend())

    data = {
        "a": np.array([1.0, 2.0], dtype=np.float32),
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "weights_backend.safetensors")
        fmt.save(data, filepath)

        loaded = fmt.load(filepath)
        assert loaded["a"] == "mocked_tensor"


def test_safetensors_missing_attrs():
    fmt = SafetensorsWeightFormat()

    class MissingDtype:
        pass

    class MissingShape:
        dtype = "float32"

    class MissingTobytes:
        dtype = "float32"
        shape = (1,)

    data = {
        "a": MissingDtype(),
        "b": MissingShape(),
        "c": MissingTobytes(),
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "weights_missing.safetensors")
        fmt.save(data, filepath)
