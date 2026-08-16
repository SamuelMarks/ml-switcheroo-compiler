import os
import tempfile

from ml_switcheroo_compiler.serialization.formats.pickle_format import PickleWeightFormat


def test_pickle_format():
    fmt = PickleWeightFormat()
    data = {"a": 1, "b": "test"}
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "weights.pkl")
        fmt.save(data, filepath)
        loaded = fmt.load(filepath)
        assert loaded == data
