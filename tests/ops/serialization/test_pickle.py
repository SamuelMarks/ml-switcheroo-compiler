# ruff: noqa: E501
import tempfile

from ml_switcheroo_compiler.serialization.formats.pickle_format import PickleWeightFormat


def test_pickle_load_save():
    fmt = PickleWeightFormat()
    with tempfile.NamedTemporaryFile() as f:
        fmt.save({"a": 1}, f.name)
        assert fmt.load(f.name) == {"a": 1}
