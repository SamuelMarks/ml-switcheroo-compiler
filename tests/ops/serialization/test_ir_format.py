"""Tests for test_ir_format."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.serialization.ir_format import graph_to_flatbuffers, graph_to_json, graph_to_protobuf, json_to_graph

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_ir_format() -> None:
    """Test function test_ir_format."""
    graph = IRGraph()
    getattr(graph.nodes, "__setitem__")("test_node", cast("Callable[..., EagerTensorProtocol]", IRNode)("test_node", "TestOp", ["input1"]))
    json_str = graph_to_json(graph)
    assert "TestOp" in json_str
    loaded_graph = json_to_graph(json_str)
    assert "test_node" in loaded_graph.nodes
    assert loaded_graph.nodes["test_node"].op_type == "TestOp"
    assert graph_to_protobuf(graph) == b""
    assert graph_to_flatbuffers(graph) == b""
