from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def global_wasm_mock():
    import copy

    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

    saved_registry = copy.deepcopy(OPS_REGISTRY)

    ops_to_mock = ["UnknownOp", "Conv2D", "MaxPool2D", "BatchNorm", "LayerNorm", "AvgPool2D", "Add", "Constant", "DotGeneral", "Transpose", "MatMul", "ReduceSum", "ReduceMax", "Tanh", "BroadcastTo", "DummyOp", "Dummy", "Exp", "Input"]
    for op in ops_to_mock:
        if op not in OPS_REGISTRY:
            OPS_REGISTRY[op] = {"variants": {}}
        if "variants" not in OPS_REGISTRY[op]:
            OPS_REGISTRY[op]["variants"] = {}
        OPS_REGISTRY[op]["variants"]["edge_wasm_simd"] = {"template": "mock_template_" + op}

    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template") as mock_get_wasm_template:

        def mock_template_resolver(template_name):
            body = "// " + template_name
            if template_name == "mock_template_Add":
                body = "wasm_f32x4_add(a, b); wasm_v128_load"
            elif template_name == "mock_template_Constant":
                body = "wasm_f32x4_splat(a);"
            elif template_name == "mock_template_Exp":
                body = "std::exp(in0_val);"
            elif template_name == "mock_template_Tanh":
                body = "std::tanh(in0_val);"
            elif template_name == "mock_template_Conv2D":
                body = "Dummy Pool/Conv"
            elif template_name == "mock_template_MaxPool2D":
                body = "Dummy Pool/Conv"
            elif template_name == "mock_template_AvgPool2D":
                body = "Dummy Pool/Conv"
            elif template_name == "mock_template_UnknownOp":
                body = "Unimplemented UnknownOp"
            return {"body": body}

        mock_get_wasm_template.side_effect = mock_template_resolver

        yield

    OPS_REGISTRY.clear()
    OPS_REGISTRY.update(saved_registry)


from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_wasm_simd_broadcast_generation():
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


def test_wasm_simd_tanh():
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = (32,)
    n2 = LogicalNode(id="out", op_type="Tanh", inputs=["in1"])
    graph.nodes = {"in1": n1, "out": n2}

    gen = WasmCodeGenerator(graph)
    code = gen.generate()

    assert "std::tanh" in code or "wasm_f32x4" in code
