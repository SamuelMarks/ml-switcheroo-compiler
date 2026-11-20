"""Tests for test_inference."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from pytest_mock import MockerFixture

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape_inference import (
    _SHAPE_INFERENCE_REGISTRY,
    ShapeType,
    infer_shape,
    register_shape_inference,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_register_shape_inference() -> None:
    """Test function test_register_shape_inference."""

    @register_shape_inference("TestShapeOp")
    def my_infer_shape(
        *args: ShapeType | Sequence[int] | OpValue | None,
        **kwargs: int | float | str | bool | ShapeType | Sequence[int] | OpValue | None,
    ) -> ShapeType:
        """Test function my_infer_shape.

        Args:
            *args (ShapeType | Sequence[int] | OpValue | None): Test parameter `args`.
            **kwargs (int | float | str | bool | ShapeType | Sequence[int] | OpValue | None): Test parameter `kwargs`.

        Returns:
            ShapeType: Computed output value from `my_infer_shape`.
        """
        del args, kwargs
        return (2, 3)

    assert _SHAPE_INFERENCE_REGISTRY["TestShapeOp"] == my_infer_shape
    assert infer_shape("TestShapeOp") == (2, 3)


def test_infer_shape_fallback() -> None:
    """Test function test_infer_shape_fallback."""

    @register_op("TestShapeOp2")
    class TestShapeOp2(OpDef):
        """Test class TestShapeOp2."""

        def infer_shape(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function infer_shape.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `infer_shape`.
            """
            return (4, 5)

    assert infer_shape("TestShapeOp2") == (4, 5)


def test_infer_shape_empty(mocker: MockerFixture) -> None:
    """Test function test_infer_shape_empty.

    Args:
        mocker: Test parameter `mocker`.
    """

    class NoShapeOp:
        """Test class NoShapeOp."""

        pass

    mocker.patch("ml_switcheroo_compiler.ops.registry.get_op", return_value=NoShapeOp)
    assert infer_shape("TestShapeOp3") == ()


def test_infer_shape_args_fallback() -> None:
    """Test fallback to instance.infer_shape(*args) when *shapes fails."""

    @register_op("TestShapeOpArgsFallback")
    class TestShapeOpArgsFallback(OpDef):
        """Test class TestShapeOpArgsFallback."""

        def infer_shape(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function infer_shape.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `infer_shape`.

            Raises:
                ValueError: When raised during `infer_shape` (Expected raw objects).
            """
            if any(isinstance(a, tuple) for a in args):
                raise ValueError("Expected raw objects")
            return (10, 20)

    class MockWithShape(list[int]):
        """List-backed shape holder for fallback testing."""

        shape: tuple[int, int] = (1, 2)

    assert infer_shape("TestShapeOpArgsFallback", MockWithShape([1, 2])) == (10, 20)


def test_infer_shape_exception_handling(mocker: MockerFixture) -> None:
    """Test outer exception handler when op instantiation or inference fails.

    Args:
        mocker: Test parameter `mocker`.
    """

    class FailingOp:
        """Test class FailingOp."""

        def __init__(self) -> None:
            """Test function __init__.

            Raises:
                RuntimeError: When raised during `__init__` (Init failed).
            """
            raise RuntimeError("Init failed")

        def infer_shape(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function infer_shape.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `infer_shape`.
            """
            return ()

    mocker.patch("ml_switcheroo_compiler.ops.registry.get_op", return_value=FailingOp)
    mocker.patch("ml_switcheroo_compiler.ops.shape_inference.infer_op_shape_declarative", return_value=(99,))
    assert infer_shape("TestFailingOp") == (99,)


def test_infer_shape_with_yaml_data(mocker: MockerFixture) -> None:
    """Test op with _yaml_data bypassing instance.infer_shape.

    Args:
        mocker: Test parameter `mocker`.
    """

    class OpWithYaml:
        """Test class OpWithYaml."""

        _yaml_data = {"some": "data"}

        def infer_shape(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function infer_shape.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `infer_shape`.
            """
            return (1, 1)

    mocker.patch("ml_switcheroo_compiler.ops.registry.get_op", return_value=OpWithYaml)
    mocker.patch("ml_switcheroo_compiler.ops.shape_inference.infer_op_shape_declarative", return_value=(7, 7))
    assert infer_shape("OpWithYaml") == (7, 7)


def test_normalize_inputs_list_and_non_shape(mocker: MockerFixture) -> None:
    """Test _normalize_inputs_to_shapes branches for list of shapes and non-shape items.

    Args:
        mocker (MockerFixture): Test parameter `mocker`.
    """

    class _ShapedInt(int):
        """Integer subclass carrying a `shape` attribute."""

        shape: tuple[int, int] = (1, 2)

    # args[0] is a list of items with shape attributes
    assert infer_shape("Add", [_ShapedInt(1), _ShapedInt(2)]) == (1, 2)

    # inputs contain items without shape attributes or list/tuple
    mocker.patch("ml_switcheroo_compiler.ops.shape_inference.infer_op_shape_declarative", return_value=())
    assert infer_shape("Add", inputs=[0, 123]) == ()
