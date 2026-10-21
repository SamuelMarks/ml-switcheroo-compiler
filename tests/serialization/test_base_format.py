"""Test base serialization format."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver
from tests.testing_utils import as_callable, as_tensor, as_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue
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
            return as_tensor_dict(as_callable(as_tensor(getattr(super(), "load")))(filepath))

    class DummySaver(WeightSaver):
        """Test class DummySaver."""

        def save(self, weights_np: EagerValue, filepath: EagerValue) -> None:
            """Test function save.

            Args:
                weights_np: Test parameter `weights_np`.
                filepath: Test parameter `filepath`.
            """
            return typing.cast("None", as_callable(as_tensor(getattr(super(), "save")))(weights_np, filepath))

    assert DummyLoader().load("dummy") is None
    assert as_callable(DummySaver)().save({}, "dummy") is None


def test_base_format_coverage() -> None:
    """Test base format coverage."""

    class DummyLoader(WeightLoader):
        """Test class DummyLoader."""

        def load(self, filepath: EagerValue) -> dict[str, TensorLike]:
            """Test function load.

            Args:
                filepath: Test parameter `filepath`.
            """
            return as_tensor_dict(as_callable(as_tensor(getattr(super(), "load")))(filepath))

    class DummySaver(WeightSaver):
        """Test class DummySaver."""

        def save(self, weights_np: EagerValue, filepath: EagerValue) -> None:
            """Test function save.

            Args:
                weights_np: Test parameter `weights_np`.
                filepath: Test parameter `filepath`.
            """
            return typing.cast("None", as_callable(as_tensor(getattr(super(), "save")))(weights_np, filepath))

    assert as_callable(DummySaver)().save({}, "test.txt") is None
    assert DummyLoader().load("test.txt") is None
