"""Tests for test_export_dot."""

from __future__ import annotations

import io
import typing
from unittest.mock import patch

import pytest

import ml_switcheroo_compiler.ops.export as export_mod
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import global_tracing_state

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


class DummyWithShape:
    """Mock operand providing shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 123


def test_export_to_dot_full_coverage(tmp_path: pytest.TempPathFactory) -> None:
    """Test full coverage for graph export to DOT format.

    Args:
        tmp_path (pytest.TempPathFactory): Temporary directory fixture.
    """
    with pytest.raises(RuntimeError, match="No active graph to export"):
        with patch.object(global_tracing_state, "active_graph", None):
            export_mod.export_to_dot("dummy.dot")

    class MockNodeInput:
        """Mock input reference with an id attribute."""

        def __init__(self, node_id: str) -> None:
            """Initialize input with id.

            Args:
                node_id (str): Identifier.
            """
            self.id: str = node_id

    class MockNode:
        """Mock IR node representation for graph export."""

        def __init__(self, op_type: str, inputs: list[MockNodeInput | str]) -> None:
            """Initialize node.

            Args:
                op_type (str): Operation type name.
                inputs (list[MockNodeInput | str]): Input dependencies.
            """
            self.op_type: str = op_type
            self.inputs: list[MockNodeInput | str] = inputs

    class MockGraph:
        """Mock IR graph container."""

        def __init__(self) -> None:
            """Initialize node dictionary."""
            self.nodes: dict[str, MockNode] = {}

    graph = MockGraph()
    n1 = MockNode("Add", [])
    n2 = MockNode("Mul", [MockNodeInput("node_1"), "raw_inp_str"])
    graph.nodes["node_1"] = n1
    graph.nodes["node_2"] = n2

    class MockProxy:
        """Mock proxy tensor with id attribute."""

        def __init__(self, node_id: str) -> None:
            """Initialize proxy.

            Args:
                node_id (str): Identifier.
            """
            self.id: str = node_id

    t_arr = typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockProxy)("node_2"), typing.cast("EagerCallableProtocol", TensorConfig)((1,), "float32", "cpu"))
    t_untracked = typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", MockProxy)("missing_node"), typing.cast("EagerCallableProtocol", TensorConfig)((1,), "float32", "cpu"))
    t_no_id = typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", DummyWithoutShape)(), typing.cast("EagerCallableProtocol", TensorConfig)((1,), "float32", "cpu"))

    with patch.object(global_tracing_state, "active_graph", graph):
        str_io = io.StringIO()
        typing.cast("EagerCallableProtocol", export_mod.export_to_dot)(str_io, t_arr, t_untracked, t_no_id)
        dot_str = str_io.getvalue()
        assert "digraph G {" in dot_str
        assert '"node_2"' in dot_str
        assert '"node_1"' in dot_str

        file_path = typing.cast("EagerCallableProtocol", str)(typing.cast("EagerTensorProtocol", tmp_path) / typing.cast("EagerTensorProtocol", "out.dot"))
        typing.cast("EagerCallableProtocol", export_mod.export_to_dot)(file_path, t_arr)
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
            assert "digraph G {" in content
