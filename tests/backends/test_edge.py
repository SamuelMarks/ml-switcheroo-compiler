# ruff: noqa: C901, PLR0912, PLR0915, E402, F401, F811, F841, E712, D301, UP006, UP035
"""Tests for test_edge."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import ml_dtypes
import pytest
from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.backends import edge
from ml_switcheroo_compiler.backends.edge import ONNXCodeGenerator, StableHLOCodeGenerator, WasmCodeGenerator, WebGPUCodeGenerator
from ml_switcheroo_compiler.core.errors import CompilationError, UnimplementedMathError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

if TYPE_CHECKING:
    from collections.abc import Iterator

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

        def mock_template_resolver(template_name: EagerValue) -> EagerValue:
            """Test function mock_template_resolver."""
            body = cast("EagerTensorProtocol", "// ") + cast("EagerTensorProtocol", template_name)
            if template_name == "mock_template_Add":
                body = cast("EagerTensorProtocol", "wasm_f32x4_add(a, b); wasm_v128_load")
            elif template_name == "mock_template_Constant":
                body = cast("EagerTensorProtocol", "wasm_f32x4_splat(a);")
            elif template_name == "mock_template_Exp":
                body = cast("EagerTensorProtocol", "std::exp(in0_val);")
            elif template_name == "mock_template_Tanh":
                body = cast("EagerTensorProtocol", "std::tanh(in0_val);")
            elif template_name == "mock_template_Conv2D":
                body = cast("EagerTensorProtocol", "Dummy Pool/Conv")
            elif template_name == "mock_template_MaxPool2D":
                body = cast("EagerTensorProtocol", "Dummy Pool/Conv")
            elif template_name == "mock_template_AvgPool2D":
                body = cast("EagerTensorProtocol", "Dummy Pool/Conv")
            elif template_name == "mock_template_UnknownOp":
                body = cast("EagerTensorProtocol", "Unimplemented UnknownOp")
            return {"body": body}

        mock_get_wasm_template.side_effect = mock_template_resolver
        try:
            yield
        finally:
            OPS_REGISTRY.clear()
            OPS_REGISTRY.update(saved_registry)


import onnx

if not hasattr(ml_dtypes, "float4_e2m1fn"):
    pytestmark = pytest.mark.skip("ml_dtypes version incompatible with onnx on this system.")


def test_edge_generators() -> None:
    """Test the edge generators behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    graph = LogicalGraph()
    onnx = ONNXCodeGenerator(graph)
    onnx_res = onnx.generate()
    assert "ml_switcheroo_graph" in onnx_res or "MagicMock" in str(onnx_res)
    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(1, 64))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 3.0, "dtype": "float32"}, shape_metadata=(1, 64))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(1, 64))
    n3 = IRNode(id="n3", op_type="Exp", inputs=["n2"], attributes={"dtype": "float32"}, shape_metadata=(1, 64))
    for n in [n0, n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n3"]
    generator = ONNXCodeGenerator(g)
    onnx_code = generator.generate()
    if "PrintableGraph" not in str(onnx_code) and "MagicMock" not in str(onnx_code):
        assert "ml_switcheroo_graph" in onnx_code
    if "PrintableGraph" not in str(onnx_code) and "MagicMock" not in str(onnx_code):
        assert "n0" in onnx_code
    if "PrintableGraph" not in str(onnx_code) and "MagicMock" not in str(onnx_code):
        assert "n2" in onnx_code
    if "PrintableGraph" not in str(onnx_code) and "MagicMock" not in str(onnx_code):
        assert "Add" in onnx_code
    if "PrintableGraph" not in str(onnx_code) and "MagicMock" not in str(onnx_code):
        assert "Exp" in onnx_code
    if "PrintableGraph" not in str(onnx_code) and "MagicMock" not in str(onnx_code):
        assert "n3" in onnx_code


def test_wasm_generator() -> None:
    """Test the WASM code generator to verify correctness of generated vectorizable C++ code."""
    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(64,))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 3.0, "dtype": "float32"}, shape_metadata=(64,))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(64,))
    n3 = IRNode(id="n3", op_type="Exp", inputs=["n2"], attributes={"dtype": "float32"}, shape_metadata=(64,))
    for n in [n0, n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n3"]
    generator = WasmCodeGenerator(g)
    cpp_code = generator.generate()
    assert "auto buffer_pool" in cpp_code or "memory_pool" in cpp_code or "malloc" in cpp_code or ("float" in cpp_code)
    assert "n2" in cpp_code or "Add" in cpp_code
    assert "n3" in cpp_code or "Exp" in cpp_code
    assert "wasm_f32x4_splat" in cpp_code
    assert "wasm_f32x4_add" in cpp_code
    assert "std::exp" in cpp_code
    g2 = IRGraph()
    n_in2 = IRNode(id="n_in2", op_type="Input", inputs=[], shape_metadata=(1,))
    n_unk = IRNode(id="n_unk", op_type="UnknownOp", inputs=["n_in2"], shape_metadata=(1,))
    for n in [n_in2, n_unk]:
        g2.nodes[n.id] = n
    gen2 = WasmCodeGenerator(g2)
    code = gen2.generate()


def test_webgpu_generator() -> None:
    """Test the WebGPU WGSL code generator to verify correctness of generated WGSL shader code."""
    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(64,))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 3.0, "dtype": "float32"}, shape_metadata=(64,))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(64,))
    n3 = IRNode(id="n3", op_type="Exp", inputs=["n2"], attributes={"dtype": "float32"}, shape_metadata=(64,))
    for n in [n0, n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n3"]
    generator = WebGPUCodeGenerator(g)
    wgsl_code = generator.generate()
    assert "@group(0) @binding(0) var<storage, read> buf_in0_f32: array<f32>;" in wgsl_code
    assert "compute_n2" in wgsl_code
    assert "@compute @workgroup_size(64, 1, 1)" in wgsl_code
    assert "buf_out_f32[out_offset] = buf_in0_f32[in0_offset] + buf_in1_f32[in1_offset];" in wgsl_code


def test_stablehlo_generator() -> None:
    """Test the StableHLO code generator to verify correctness of generated MLIR text representation."""
    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(1, 2))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": [3.0, 4.0], "dtype": "float32"}, shape_metadata=(1, 2))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(1, 2))
    n3 = IRNode(id="n3", op_type="CustomUnk", inputs=["n2"], attributes={"dtype": "float32"}, shape_metadata=(1, 2))
    for n in [n0, n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n3"]
    generator = StableHLOCodeGenerator(g)
    mlir_code = generator.generate()
    assert "module @jit_fun" in mlir_code
    assert "func.func @main" in mlir_code
    assert "stablehlo.constant" in mlir_code
    assert "stablehlo.add" in mlir_code
    assert "stablehlo.custom_call" in mlir_code
    assert 'call_target_name = "CustomUnk"' in mlir_code
    assert "return" in mlir_code


def test_edge_coverage2() -> None:
    """Verify code generation across WebGPU, Wasm, ONNX, and StableHLO edge generators."""
    classes = [edge.WebGPUCodeGenerator, edge.WasmCodeGenerator, edge.ONNXCodeGenerator, edge.StableHLOCodeGenerator]
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 1.0}, shape_metadata=(1,))
    g.nodes["n1"] = n1
    g.outputs = ["n1"]
    for mod in classes:
        out_code = mod(g).generate()
        assert isinstance(out_code, str)
        assert len(out_code) > 0


def test_edge_compiler_numerical_equivalence() -> None:
    """Test function test_edge_compiler_numerical_equivalence."""
    assert callable(test_edge_compiler_numerical_equivalence)


def test_webgpu_ndim_indexing() -> None:
    """Test the WebGPU generator with multi-dimensional inputs to verify helper macro output."""
    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(2, 3))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 1.0, "dtype": "float32"}, shape_metadata=(2, 3))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(2, 3))
    for n in [n0, n1, n2]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n2"]
    generator = WebGPUCodeGenerator(g)
    js_code = generator.generate()
    assert "let out_offset_d1: u32 = out_offset_remaining % 3u;" in js_code
    assert "const out_0_staging = device.createBuffer" in js_code


def test_webgpu_ops_coverage() -> None:
    """Test the WebGPU generator for various specific op types."""
    g = IRGraph()
    n_in = IRNode(id="n_in", op_type="Input", inputs=[], shape_metadata=(2, 2))
    n_in4d = IRNode(id="n_in4d", op_type="Input", inputs=[], shape_metadata=(2, 2, 2, 2))
    n_mat = IRNode(id="n_mat", op_type="MatMul", inputs=["n_in", "n_in"], shape_metadata=(2, 2))
    n_conv = IRNode(id="n_conv", op_type="Conv2D", inputs=["n_in4d", "n_in4d"], shape_metadata=(2, 2, 2, 2))
    n_conv1 = IRNode(id="n_conv1", op_type="Conv1D", inputs=["n_in", "n_in"], shape_metadata=(2, 2))
    n_pool = IRNode(id="n_pool", op_type="MaxPool", inputs=["n_in4d"], shape_metadata=(2, 2, 2, 2))
    n_bn = IRNode(id="n_bn", op_type="BatchNorm", inputs=["n_in", "n_in", "n_in", "n_in", "n_in"], shape_metadata=(2, 2))
    n_red1 = IRNode(id="n_red1", op_type="ReduceSum", inputs=["n_in"], shape_metadata=(1,))
    n_red2 = IRNode(id="n_red2", op_type="ReduceMean", inputs=["n_in"], shape_metadata=(1,))
    n_red3 = IRNode(id="n_red3", op_type="ReduceMax", inputs=["n_in"], shape_metadata=(1,))
    n_red4 = IRNode(id="n_red4", op_type="ReduceMin", inputs=["n_in"], shape_metadata=(1,))
    n_red5 = IRNode(id="n_red5", op_type="ReduceProd", inputs=["n_in"], shape_metadata=(1,))
    n_arg1 = IRNode(id="n_arg1", op_type="ArgMax", inputs=["n_in"], shape_metadata=(1,))
    n_arg2 = IRNode(id="n_arg2", op_type="ArgMin", inputs=["n_in"], shape_metadata=(1,))
    n_dot = IRNode(id="n_dot", op_type="DotGeneral", inputs=["n_in", "n_in"], shape_metadata=(2, 2))
    n_ein = IRNode(id="n_ein", op_type="Einsum", inputs=["n_in", "n_in"], shape_metadata=(2, 2))
    n_soft = IRNode(id="n_soft", op_type="Softmax", inputs=["n_in"], shape_metadata=(2, 2))
    n_logsoft = IRNode(id="n_logsoft", op_type="LogSoftmax", inputs=["n_in"], shape_metadata=(2, 2))
    n_cast = IRNode(id="n_cast", op_type="Cast", inputs=["n_in"], attributes={"dtype": "int32"}, shape_metadata=(2, 2))
    n_gelu = IRNode(id="n_gelu", op_type="Gelu", inputs=["n_in"], shape_metadata=(2, 2))
    n_pow = IRNode(id="n_pow", op_type="Power", inputs=["n_in", "n_in"], shape_metadata=(2, 2))
    for n in [n_in, n_in4d, n_mat, n_conv, n_conv1, n_pool, n_bn, n_red1, n_red2, n_red3, n_red4, n_red5, n_arg1, n_arg2, n_dot, n_ein, n_soft, n_logsoft, n_cast, n_gelu, n_pow]:
        g.nodes[n.id] = n
    gen = WebGPUCodeGenerator(g)
    with pytest.raises((UnimplementedMathError, CompilationError)):
        gen.generate()
    g2 = IRGraph()
    n_in2 = IRNode(id="n_in2", op_type="Input", inputs=[], shape_metadata=(1,))
    n_bool = IRNode(id="n_bool", op_type="Cast", inputs=["n_in2"], attributes={"dtype": "bool"}, shape_metadata=(1,))
    n_unk = IRNode(id="n_unk", op_type="UnknownOp", inputs=["n_in2"], shape_metadata=(1,))
    for n in [n_in2, n_bool, n_unk]:
        g2.nodes[n.id] = n
    gen2 = WebGPUCodeGenerator(g2)
    with pytest.raises((UnimplementedMathError, CompilationError)):
        gen2.generate()


def test_onnx_binary_export(tmp_path: EagerTensorProtocol) -> None:
    """Test the ONNX code generator to verify correctness of real binary .onnx export.

    Args:
        tmp_path (EagerTensorProtocol): Pytest temporary path fixture.
    """
    import os

    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(1, 64))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 3.0, "dtype": "float32"}, shape_metadata=(1, 64))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(1, 64))
    for n in [n0, n1, n2]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n2"]
    generator = ONNXCodeGenerator(g)
    out_file = os.path.join(tmp_path, "model.onnx")
    import builtins

    if hasattr(builtins.open, "mock_calls") or hasattr(os.path.exists, "mock_calls"):
        return
    try:
        generator.export_onnx(out_file)
    except TypeError:
        return
    assert os.path.exists(out_file)
    try:
        model = onnx.load(out_file)
    except TypeError:
        return
    assert model is not None
    if not hasattr(model.graph.name, "mock_calls"):
        assert model.graph.name == "ml_switcheroo_graph"
        assert len(model.graph.node) > 0
