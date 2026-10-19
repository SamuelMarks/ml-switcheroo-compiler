"""Test base serialization format."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue
    from ml_switcheroo_compiler.interpreter.evaluator import TensorLike


def test_base_format() -> None:
    """Test base classes."""

    class DummyLoader(WeightLoader):
        """Test class DummyLoader."""

        def load(self, filepath: EagerValue) -> dict[str, TensorLike]:
            """Test function load.

            Args:
                filepath: Test parameter `filepath`.
            """
            return cast("dict[str, TensorLike]", cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(super(), "load")))(filepath))

    class DummySaver(WeightSaver):
        """Test class DummySaver."""

        def save(self, weights_np: EagerValue, filepath: EagerValue) -> None:
            """Test function save.

            Args:
                weights_np: Test parameter `weights_np`.
                filepath: Test parameter `filepath`.
            """
            return cast("None", cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(super(), "save")))(weights_np, filepath))

    assert DummyLoader().load("dummy") is None
    assert cast("EagerCallableProtocol", DummySaver)().save({}, "dummy") is None


def test_base_format_coverage() -> None:
    """Test base format coverage."""

    class DummyLoader(WeightLoader):
        """Test class DummyLoader."""

        def load(self, filepath: EagerValue) -> dict[str, TensorLike]:
            """Test function load.

            Args:
                filepath: Test parameter `filepath`.
            """
            return cast("dict[str, TensorLike]", cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(super(), "load")))(filepath))

    class DummySaver(WeightSaver):
        """Test class DummySaver."""

        def save(self, weights_np: EagerValue, filepath: EagerValue) -> None:
            """Test function save.

            Args:
                weights_np: Test parameter `weights_np`.
                filepath: Test parameter `filepath`.
            """
            return cast("None", cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(super(), "save")))(weights_np, filepath))

    assert cast("EagerCallableProtocol", DummySaver)().save({}, "test.txt") is None
    assert DummyLoader().load("test.txt") is None
