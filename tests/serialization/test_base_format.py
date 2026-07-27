"""Test base serialization format."""

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


def test_base_format():
    """Test base classes."""

    class DummyLoader(WeightLoader):
        def load(self, filepath):
            return super().load(filepath)

    class DummySaver(WeightSaver):
        def save(self, weights_np, filepath):
            return super().save(weights_np, filepath)

    assert DummyLoader().load("dummy") is None
    assert DummySaver().save({}, "dummy") is None
