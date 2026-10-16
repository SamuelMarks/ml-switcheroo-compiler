"""Test base serialization format."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
    from ml_switcheroo_compiler.interpreter.evaluator import TensorLike


def test_base_format() -> None:
    """Test base classes."""

    class DummyLoader(WeightLoader):
        """Test class DummyLoader."""

        def load(self, filepath: EagerValue) -> dict[str, TensorLike]:
            """Test function load."""
            return cast("dict[str, TensorLike]", cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(super(), "load")))(filepath))

    class DummySaver(WeightSaver):
        """Test class DummySaver."""

        def save(self, weights_np: EagerValue, filepath: EagerValue) -> None:
            """Test function save."""
            return cast("None", cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(super(), "save")))(weights_np, filepath))

    assert DummyLoader().load("dummy") is None
    assert cast("Callable[..., EagerTensorProtocol]", DummySaver)().save({}, "dummy") is None


def test_base_format_coverage() -> None:
    """Test base format coverage."""

    class DummyLoader(WeightLoader):
        """Test class DummyLoader."""

        def load(self, filepath: EagerValue) -> dict[str, TensorLike]:
            """Test function load."""
            return cast("dict[str, TensorLike]", cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(super(), "load")))(filepath))

    class DummySaver(WeightSaver):
        """Test class DummySaver."""

        def save(self, weights_np: EagerValue, filepath: EagerValue) -> None:
            """Test function save."""
            return cast("None", cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(super(), "save")))(weights_np, filepath))

    assert cast("Callable[..., EagerTensorProtocol]", DummySaver)().save({}, "test.txt") is None
    assert DummyLoader().load("test.txt") is None
