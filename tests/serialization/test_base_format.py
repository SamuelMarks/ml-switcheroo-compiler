"""Test base serialization format."""

import pytest

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


def test_base_format():
    """Test base classes."""

    class DummyLoader(WeightLoader):
        def load(self, filepath):
            return super().load(filepath)

    class DummySaver(WeightSaver):
        def save(self, weights_np, filepath):
            return super().save(weights_np, filepath)

    with pytest.raises(NotImplementedError):
        DummyLoader().load("dummy")
    with pytest.raises(NotImplementedError):
        DummySaver().save({}, "dummy")


def test_base_format_coverage():
    """Test base format coverage."""

    class DummyLoader(WeightLoader):
        def load(self, filepath):
            return super().load(filepath)

    class DummySaver(WeightSaver):
        def save(self, weights_np, filepath):
            return super().save(weights_np, filepath)

    with pytest.raises(NotImplementedError):
        DummySaver().save({}, "test.txt")
    with pytest.raises(NotImplementedError):
        DummyLoader().load("test.txt")
