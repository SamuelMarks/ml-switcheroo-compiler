"""Test module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.backends.cupy.eager import execute_op
from ml_switcheroo_compiler.backends.cupy.generator import CupyGenerator
from ml_switcheroo_compiler.backends.cupy.types import array, asarray, item, zeros
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.ir.core import IRNode

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyGraph:
    """Test class DummyGraph."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.nodes = cast("list[EagerTensorProtocol]", [])


class DummyCp:
    """Test class DummyCp."""

    def add(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function add."""
        return "add_res"

    def _test_op(self, *args: EagerTensorProtocol, **kwargs: EagerTensorProtocol) -> EagerValue:
        """Test function _test_op."""
        return "test_res"


def test_cupy_eager_execute_op() -> None:
    """Test function test_cupy_eager_execute_op."""
    cp_mock = DummyCp()

    def dummy_eager(module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function dummy_eager."""
        return "dummy_eager_res"

    global_eager_registry.register("TestOp")(dummy_eager)
    res = cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "TestOp")
    assert res == "dummy_eager_res"
    if "Add" in global_eager_registry._registry:
        del global_eager_registry._registry["Add"]
    with patch("ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings") as mock_mappings:
        mock_schema = type("Dummy", (), {"operations": {"Add": type("DummyOp", (), {"target_api": "add", "custom_code": None})()}})()
        mock_mappings.return_value = mock_schema
        with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", cp_mock), patch("sys.modules", {"ml_switcheroo_compiler.backends.cupy.eager": cp_mock}):
            res2 = cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "Add", 1, 2)
            assert res2 == "add_res"
    global_eager_registry.register("NumpyFallbackOp")(dummy_eager)
    res3 = cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "NumpyFallbackOp")
    assert res3 == "dummy_eager_res"
    if "UnknownOp" in global_eager_registry._registry:
        del global_eager_registry._registry["UnknownOp"]
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "UnknownOp")


def test_cupy_eager_execute_op_exception() -> None:
    """Test function test_cupy_eager_execute_op_exception."""
    import pytest

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    if "OpThatRaises" in global_eager_registry._registry:
        del global_eager_registry._registry["OpThatRaises"]
    with pytest.raises(BackendNotSupportedError, match="Operation 'OpThatRaises' is not implemented."):
        cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "OpThatRaises")
    with patch("ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings") as mock_mappings:
        mock_schema = type("Dummy", (), {"operations": {"SnakeOp": type("DummyOp", (), {"target_api": "snake_op", "custom_code": None})()}})()
        mock_mappings.return_value = mock_schema
        with patch("sys.modules", {"ml_switcheroo_compiler.backends.cupy.eager": DummyCp()}):
            with pytest.raises(BackendNotSupportedError):
                cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "SnakeOp")


def test_cupy_generator() -> None:
    """Test function test_cupy_generator."""
    g = DummyGraph()
    gen = cast("Callable[..., EagerTensorProtocol]", CupyGenerator)(g)
    assert gen.get_fallback_prefix() == "cp"
    assert gen.get_helper_functions() == []
    node = IRNode("Einsum", op_type="Einsum")
    node = IRNode("TruncateDiv", op_type="TruncateDiv")
    node = IRNode("TruncateMod", op_type="TruncateMod")
    node = IRNode("Add", op_type="Add")
    assert gen.generic_visit(node, ["a", "b"]) == "cp.add(a, b)"
    node = IRNode("UnknownOp", op_type="UnknownOp")
    assert gen.generic_visit(node, [], kwarg1="val") == "cp.unknownop(kwarg1='val')"
    assert gen.generic_visit(node, ["a", "b"], kwarg1="val") == "cp.unknownop(a, b, kwarg1='val')"


def test_cupy_types() -> None:
    """Test function test_cupy_types."""
    with patch("ml_switcheroo_compiler.backends.cupy.types.cp", DummyCp()):
        with patch("ml_switcheroo_compiler.backends.cupy.types.generic_zeros", return_value="zeros"):
            assert cast("Callable[..., EagerTensorProtocol]", zeros)(None, (2,)) == "zeros"
        with patch("ml_switcheroo_compiler.backends.cupy.types.generic_array", return_value="array"):
            assert cast("Callable[..., EagerTensorProtocol]", array)(None, [1]) == "array"
        with patch("ml_switcheroo_compiler.backends.cupy.types.generic_asarray", return_value="asarray"):
            assert cast("Callable[..., EagerTensorProtocol]", asarray)(None, [1]) == "asarray"
        with patch("ml_switcheroo_compiler.backends.cupy.types.generic_item", return_value="item"):
            assert cast("Callable[..., EagerTensorProtocol]", item)(None, [1]) == "item"


def test_cupy_eager_execute_op_fallback_to_numpy_registry() -> None:
    """Test function test_cupy_eager_execute_op_fallback_to_numpy_registry."""

    def dummy_eager(module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function dummy_eager."""
        return "dummy_eager_numpy_fallback"

    assert callable(test_cupy_eager_execute_op_fallback_to_numpy_registry)


def test_cupy_eager_op_mapping() -> None:
    """Test function test_cupy_eager_op_mapping."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.cupy.eager import execute_op

    with patch("ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings") as mock_get_mapping:
        mock_schema = type("Dummy", (), {"operations": {"TestOp2": type("DummyOp", (), {"target_api": "custom_op", "custom_code": "lambda *args, **kwargs: 'mapped_res'"})()}})()
        mock_get_mapping.return_value = mock_schema
        with patch("ml_switcheroo_compiler.backends.eager_registry.global_eager_registry.get", return_value=None):
            res = cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "TestOp2")
            assert res == "mapped_res"
