"""Test base serialization format."""

from __future__ import annotations

from collections.abc import Mapping

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


def test_base_format() -> None:
    """Test base classes."""

    class DummyLoader(WeightLoader):
        """Test class DummyLoader."""

        def load(self, filepath: str) -> Mapping[str, OpValue]:
            """Test function load.

            Args:
                filepath: Test parameter `filepath`.

            Returns:
                Mapping[str, OpValue]: Computed output value from `load`.
            """
            return WeightLoader.load(self, filepath)

    class DummySaver(WeightSaver):
        """Test class DummySaver."""

        def save(self, weights_np: Mapping[str, OpValue], filepath: str) -> None:
            """Test function save.

            Args:
                weights_np: Test parameter `weights_np`.
                filepath: Test parameter `filepath`.
            """
            WeightSaver.save(self, weights_np, filepath)

    assert DummyLoader().load("dummy") is None
    DummySaver().save({}, "dummy")


def test_base_format_coverage() -> None:
    """Test base format coverage."""

    class DummyLoader(WeightLoader):
        """Test class DummyLoader."""

        def load(self, filepath: str) -> Mapping[str, OpValue]:
            """Test function load.

            Args:
                filepath: Test parameter `filepath`.

            Returns:
                Mapping[str, OpValue]: Computed output value from `load`.
            """
            return WeightLoader.load(self, filepath)

    class DummySaver(WeightSaver):
        """Test class DummySaver."""

        def save(self, weights_np: Mapping[str, OpValue], filepath: str) -> None:
            """Test function save.

            Args:
                weights_np: Test parameter `weights_np`.
                filepath: Test parameter `filepath`.
            """
            WeightSaver.save(self, weights_np, filepath)

    DummySaver().save({}, "test.txt")
    assert DummyLoader().load("test.txt") is None
