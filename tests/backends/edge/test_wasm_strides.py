"""Tests for test_wasm_strides."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


@pytest.fixture(autouse=True)
def global_wasm_mock() -> Iterator[None]:
    """Test function global_wasm_mock."""
    import copy

    from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY

    saved_registry = copy.deepcopy(OPS_REGISTRY)

    ops_to_mock = ["UnknownOp", "Conv2D", "MaxPool2D", "BatchNorm", "LayerNorm", "AvgPool2D", "Add", "Constant", "DotGeneral", "Transpose", "MatMul", "ReduceSum", "ReduceMax", "Tanh", "BroadcastTo", "DummyOp", "Dummy", "Exp", "Input"]
    for op in ops_to_mock:
        if op not in OPS_REGISTRY:
            OPS_REGISTRY[op] = {"variants": {}}
        if "variants" not in OPS_REGISTRY[op]:
            OPS_REGISTRY[op]["variants"] = {}
        getattr(OPS_REGISTRY[op]["variants"], "__setitem__")("edge_wasm_simd", {"template": "mock_template_" + op})

    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template") as mock_get_wasm_template:
        mock_bodies = {
            "mock_template_Add": "wasm_f32x4_add(a, b); wasm_v128_load",
            "mock_template_Constant": "wasm_f32x4_splat(a);",
            "mock_template_Exp": "std::exp(in0_val);",
            "mock_template_Tanh": "std::tanh(in0_val);",
            "mock_template_Conv2D": "Dummy Pool/Conv",
            "mock_template_MaxPool2D": "Dummy Pool/Conv",
            "mock_template_AvgPool2D": "Dummy Pool/Conv",
            "mock_template_UnknownOp": "Unimplemented UnknownOp",
        }

        def mock_template_resolver(template_name: EagerValue) -> EagerValue:
            """Test function mock_template_resolver."""
            return {"body": cast("Callable[..., EagerTensorProtocol]", mock_bodies.get)(template_name, cast("EagerTensorProtocol", "// ") + cast("EagerTensorProtocol", template_name))}

        mock_get_wasm_template.side_effect = mock_template_resolver

        try:
            yield
        finally:
            OPS_REGISTRY.clear()
            OPS_REGISTRY.update(saved_registry)


def test_wasm_simd_broadcast_generation() -> None:
    """Test function test_wasm_simd_broadcast_generation."""
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = (32,)
    n2 = LogicalNode(id="in2", op_type="Input")
    n2.shape_metadata = (32, 64)
    n3 = LogicalNode(id="out", op_type="Add", inputs=["in1", "in2"])
    n3.shape_metadata = (32, 64)
    graph.nodes = {"in1": n1, "in2": n2, "out": n3}

    gen = WasmCodeGenerator(graph)
    code = gen.generate()

    assert "wasm_f32x4_add" in code


def test_wasm_simd_tanh() -> None:
    """Test function test_wasm_simd_tanh."""
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = (32,)
    n2 = LogicalNode(id="out", op_type="Tanh", inputs=["in1"])
    graph.nodes = {"in1": n1, "out": n2}

    gen = WasmCodeGenerator(graph)
    code = gen.generate()

    assert "std::tanh" in code or "wasm_f32x4" in code
