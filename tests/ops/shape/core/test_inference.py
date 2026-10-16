"""Tests for test_inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape_inference import _SHAPE_INFERENCE_REGISTRY, infer_shape, register_shape_inference

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_register_shape_inference() -> None:
    """Test function test_register_shape_inference."""

    @cast("Callable[..., EagerTensorProtocol]", register_shape_inference)("TestShapeOp")
    def my_infer_shape(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function my_infer_shape."""
        return (2, 3)

    assert _SHAPE_INFERENCE_REGISTRY["TestShapeOp"] == my_infer_shape
    assert infer_shape("TestShapeOp") == (2, 3)


def test_infer_shape_fallback() -> None:
    """Test function test_infer_shape_fallback."""

    @register_op("TestShapeOp2")
    class TestShapeOp2(OpDef):
        """Test class TestShapeOp2."""

        def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function infer_shape."""
            return (4, 5)

    assert infer_shape("TestShapeOp2") == (4, 5)


def test_infer_shape_empty(mocker: EagerTensorProtocol) -> None:
    """Test function test_infer_shape_empty."""

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

        def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function infer_shape."""
            if any(isinstance(a, tuple) for a in args):
                raise ValueError("Expected raw objects")
            return (10, 20)

    class MockWithShape:
        """Test class MockWithShape."""

        shape = (1, 2)

    assert cast("Callable[..., EagerTensorProtocol]", infer_shape)("TestShapeOpArgsFallback", cast("Callable[..., EagerTensorProtocol]", MockWithShape)()) == (10, 20)


def test_infer_shape_exception_handling(mocker: EagerTensorProtocol) -> None:
    """Test outer exception handler when op instantiation or inference fails."""

    class FailingOp:
        """Test class FailingOp."""

        def __init__(self) -> None:
            """Test function __init__."""
            raise RuntimeError("Init failed")

        def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function infer_shape."""
            return ()

    mocker.patch("ml_switcheroo_compiler.ops.registry.get_op", return_value=FailingOp)
    mocker.patch("ml_switcheroo_compiler.ops.shape_inference.infer_op_shape_declarative", return_value=(99,))
    assert infer_shape("TestFailingOp") == (99,)


def test_infer_shape_with_yaml_data(mocker: EagerTensorProtocol) -> None:
    """Test op with _yaml_data bypassing instance.infer_shape."""

    class OpWithYaml:
        """Test class OpWithYaml."""

        _yaml_data = {"some": "data"}

        def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function infer_shape."""
            return (1, 1)

    mocker.patch("ml_switcheroo_compiler.ops.registry.get_op", return_value=OpWithYaml)
    mocker.patch("ml_switcheroo_compiler.ops.shape_inference.infer_op_shape_declarative", return_value=(7, 7))
    assert infer_shape("OpWithYaml") == (7, 7)


def test_normalize_inputs_list_and_non_shape(mocker: EagerTensorProtocol) -> None:
    """Test _normalize_inputs_to_shapes branches for list of shapes and non-shape items."""
    # Line 919: args[0] is a list/tuple of items with shape or tuples
    assert cast("Callable[..., EagerTensorProtocol]", infer_shape)("Add", [(1, 2), (1, 2)]) == (1, 2)

    # Branch 932->927: inputs contain items without shape attributes or list/tuple
    mocker.patch("ml_switcheroo_compiler.ops.shape_inference.infer_op_shape_declarative", return_value=())
    assert cast("Callable[..., EagerTensorProtocol]", infer_shape)("Add", inputs=[None, 123]) == ()
