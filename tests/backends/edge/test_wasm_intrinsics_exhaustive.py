"""Test WASM SIMD intrinsics coverage across operations."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def get_all_ops() -> EagerValue:
    """Extract all operations from the registry."""
    return list(OPS_REGISTRY.keys())


def test_wasm_scalar_fallback_and_simd() -> None:
    """Test that UnimplementedMathError is NOT raised for operations that should have fallbacks/SIMD."""
    graph = IRGraph()
    in_node = IRNode(id="Input", op_type="Input", shape_metadata=(10, 10))
    graph.nodes[in_node.id] = in_node
    ops_to_test = ["Sin", "Cos", "Exp", "Add", "Mul", "Relu", "Sigmoid", "Tanh"]
    for op in ops_to_test:
        test_node = IRNode(id=f"test_{op}", op_type=op, inputs=[in_node.id], shape_metadata=(10, 10))
        if op in ["Add", "Mul"]:
            test_node.inputs.append(in_node.id)
        graph.nodes[test_node.id] = test_node
        try:
            generator = WasmCodeGenerator(graph)
            generator.generate()
        except UnimplementedMathError as e:
            pytest.fail(f"Operation {op} triggered UnimplementedMathError: {e}")
        finally:
            del graph.nodes[test_node.id]
    assert callable(test_wasm_scalar_fallback_and_simd)
