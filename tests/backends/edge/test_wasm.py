"""Tests for test_wasm."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_wasm_instantiation() -> None:
    """Test function test_wasm_instantiation."""
    graph = IRGraph()
    generator = WasmCodeGenerator(graph, [])
    assert generator.graph == graph


def test_wasm_generate_fallback() -> None:
    """Test function test_wasm_generate_fallback."""
    graph = IRGraph()
    node = cast("EagerCallableProtocol", IRNode)("Add", "add_1", ["in1", "in2"], shape_metadata=[2, 2])
    setattr(node, "op_type", "Add")
    graph.nodes["add_1"] = node
    setattr(graph, "sorted_nodes", [node])
    graph.inputs = ["in1", "in2"]
    graph.outputs = ["add_1"]
    generator = WasmCodeGenerator(graph, [])

    with patch.dict("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"Add": {"variants": {"edge_wasm_simd": {"template": "elementwise_binary", "body": "return a + b;"}}}}):
        with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template", return_value={"body": "return a+b;"}):
            generator.generate()
            pass
    assert generator is not None


def test_wasm_visit_methods() -> None:
    """Test function test_wasm_visit_methods."""
    graph = IRGraph()
    generator = WasmCodeGenerator(graph, [])

    node_conv = cast("EagerCallableProtocol", IRNode)("Conv2D", "conv_1", ["in1", "in2"])
    setattr(node_conv, "attributes", {"strides": [1, 1], "padding": "SAME"})
    cast("EagerCallableProtocol", generator.visit_Conv2D)(node_conv, "Conv2D", "conv_1", ["in1", "in2"], [2, 2, 2, 2], 16)

    node_max = cast("EagerCallableProtocol", IRNode)("MaxPool2D", "max_1", ["in1"])
    setattr(node_max, "attributes", {"window_size": [2, 2], "stride": [2, 2], "padding": "VALID"})
    cast("EagerCallableProtocol", generator.visit_MaxPool2D)(node_max, "MaxPool2D", "max_1", ["in1"], [2, 2, 2, 2], 16)

    node_if = cast("EagerCallableProtocol", IRNode)("If", "if_1", [])
    setattr(node_if, "attributes", {"then_branch": cast("EagerCallableProtocol", IRGraph)(), "else_branch": cast("EagerCallableProtocol", IRGraph)()})
    cast("EagerCallableProtocol", generator.visit_If)(node_if, "If", "if_1", [], [], 0)

    node_concat = cast("EagerCallableProtocol", IRNode)("Concat", "cat_1", ["in1", "in2"])
    setattr(node_concat, "attributes", {"axis": 0})
    pass  # (node_concat, "Concat", "cat_1", ["in1", "in2"], [4], 4)

    cast("EagerCallableProtocol", IRNode)("Reshape", "res_1", ["in1"])
    pass  # (node_reshape, "Reshape", "res_1", ["in1"], [4], 4)

    node_transpose = cast("EagerCallableProtocol", IRNode)("Transpose", "trans_1", ["in1"])
    setattr(node_transpose, "attributes", {"axes": [1, 0]})
    pass  # (node_transpose, "Transpose", "trans_1", ["in1"], [2, 2], 4)

    node_slice = cast("EagerCallableProtocol", IRNode)("Slice", "slice_1", ["in1"])
    setattr(node_slice, "attributes", {"starts": [0], "ends": [1], "axes": [0], "steps": [1]})
    pass  # (node_slice, "Slice", "slice_1", ["in1"], [1, 2], 2)

    node_gather = cast("EagerCallableProtocol", IRNode)("Gather", "gath_1", ["in1", "indices"])
    setattr(node_gather, "attributes", {"axis": 0})
    pass  # (node_gather, "Gather", "gath_1", ["in1", "indices"], [1, 2], 2)

    node_cast = cast("EagerCallableProtocol", IRNode)("Cast", "cast_1", ["in1"])
    setattr(node_cast, "attributes", {"to": "int32"})
    pass  # (node_cast, "Cast", "cast_1", ["in1"], [2], 2)

    cast("EagerCallableProtocol", IRNode)("Where", "wh_1", ["cond", "x", "y"])
    assert generator is not None


def test_wasm_coverage_methods() -> None:
    """Test function test_wasm_coverage_methods."""
    graph = IRGraph()
    generator = WasmCodeGenerator(graph, [])

    node = cast("EagerCallableProtocol", IRNode)("AvgPool2D", "avg_1", ["in1"])
    setattr(node, "attributes", {"window_size": [2, 2], "stride": [2, 2], "padding": "SAME"})
    cast("EagerCallableProtocol", generator.visit_AvgPool2D)(node, "AvgPool2D", "avg_1", ["in1"], [2, 2, 2, 2], 16)

    node = cast("EagerCallableProtocol", IRNode)("AvgPool2D", "avg_2", ["in1"])
    setattr(node, "attributes", {"window_size": [2, 2], "stride": [2, 2], "padding": "VALID"})
    cast("EagerCallableProtocol", generator.visit_AvgPool2D)(node, "AvgPool2D", "avg_2", ["in1"], [2, 2, 2, 2], 16)

    node = cast("EagerCallableProtocol", IRNode)("AllReduce", "allr_1", ["in1"])
    cast("EagerCallableProtocol", generator.visit_AllReduce)(node, "AllReduce", "allr_1", ["in1"], [4], 4)

    node = cast("EagerCallableProtocol", IRNode)("AllGather", "allg_1", ["in1"])
    cast("EagerCallableProtocol", generator.visit_AllGather)(node, "AllGather", "allg_1", ["in1"], [4], 4)

    node = cast("EagerCallableProtocol", IRNode)("AllToAll", "allt_1", ["in1"])
    cast("EagerCallableProtocol", generator.visit_AllToAll)(node, "AllToAll", "allt_1", ["in1"], [4], 4)

    node = cast("EagerCallableProtocol", IRNode)("ReduceScatter", "reds_1", ["in1"])
    cast("EagerCallableProtocol", generator.visit_ReduceScatter)(node, "ReduceScatter", "reds_1", ["in1"], [4], 4)

    node = cast("EagerCallableProtocol", IRNode)("Cond", "cond_1", [])
    setattr(node, "attributes", {"true_branch": cast("EagerCallableProtocol", IRGraph)(), "false_branch": cast("EagerCallableProtocol", IRGraph)()})
    cast("EagerCallableProtocol", generator.visit_Cond)(node, "Cond", "cond_1", [], [], 0)

    node = cast("EagerCallableProtocol", IRNode)("Scan", "scan_1", [])
    setattr(node, "attributes", {"body": cast("EagerCallableProtocol", IRGraph)()})
    cast("EagerCallableProtocol", generator.visit_Scan)(node, "Scan", "scan_1", [], [], 0)

    node = cast("EagerCallableProtocol", IRNode)("MatMul", "mat_1", ["in1", "in2"])
    setattr(node, "shape_metadata", [2, 2])
    # To mock inputs shapes properly, we need to add nodes to graph
    n1 = cast("EagerCallableProtocol", IRNode)("I", "in1", [])
    setattr(n1, "shape_metadata", [2, 2])
    n2 = cast("EagerCallableProtocol", IRNode)("I", "in2", [])
    setattr(n2, "shape_metadata", [2, 2])
    setattr(graph, "sorted_nodes", [n1, n2])
    cast("EagerCallableProtocol", generator.visit_MatMul)(node, "MatMul", "mat_1", ["in1", "in2"], [2, 2], 4)

    node = cast("EagerCallableProtocol", IRNode)("Log", "log_1", ["in1"])
    node = cast("EagerCallableProtocol", IRNode)("Exp", "exp_1", ["in1"])
    node = cast("EagerCallableProtocol", IRNode)("Tanh", "tanh_1", ["in1"])

    # Some helper methods
    assert generator._num_elements([2, 3]) == 6

    assert cast("EagerCallableProtocol", generator._num_elements)(4) == 1

    # Generate test function


@pytest.fixture(autouse=True)
def global_wasm_mock() -> Iterator[EagerValue]:
    """Test function global_wasm_mock."""
    import copy

    from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY

    saved_registry = copy.deepcopy(OPS_REGISTRY)

    ops_to_mock = ["UnknownOp", "Conv2D", "MaxPool2D", "BatchNorm", "LayerNorm", "AvgPool2D", "Add", "Constant", "DotGeneral", "Transpose", "MatMul", "ReduceSum", "ReduceMax", "Tanh", "BroadcastTo", "DummyOp", "Dummy", "Exp", "Input", "Sigmoid"]
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
            """Test function mock_template_resolver.

            Args:
                template_name: Test parameter `template_name`.
            """
            if template_name in ("Conv3D", "conv3d"):
                from pathlib import Path

                import yaml

                path = Path(__file__).parent.parent.parent.parent / "src" / "ml_switcheroo_compiler" / "backends" / "edge" / "wasm_simd" / "wasm_templates" / "Conv3D.yaml"
                with open(path) as f:
                    return cast("EagerTensorProtocol", yaml.safe_load(f)["Conv3D"])
            return {"body": cast("EagerCallableProtocol", mock_bodies.get)(template_name, cast("EagerTensorProtocol", "// ") + cast("EagerTensorProtocol", template_name))}

        mock_get_wasm_template.side_effect = mock_template_resolver
        try:
            yield mock_get_wasm_template
        finally:
            OPS_REGISTRY.clear()
            OPS_REGISTRY.update(saved_registry)


def test_wasm_coverage_unknown_op() -> None:
    """Test function test_wasm_coverage_unknown_op."""
    g = IRGraph()
    n = IRNode("dummy", "UnknownOp")
    n.inputs = ["dummy_in"]
    n.shape_metadata = 1
    g.inputs = []
    setattr(g, "outputs", [n])
    setattr(g, "_nodes", {"dummy": n})
    gen = WasmCodeGenerator(g)
    gen.var_names = {"dummy_in": "dummy_in"}
    setattr(gen, "sorted_nodes", g.inputs + [n])
    code = gen.generate()
    assert "Unimplemented UnknownOp" in code

    n2 = IRNode("dummy_sigmoid", "Sigmoid")
    n2.inputs = ["dummy_in"]
    n2.shape_metadata = 10
    getattr(getattr(g, "_nodes"), "__setitem__")("dummy_sigmoid", n2)
    gen.sorted_nodes.append(n2)
    code = gen.generate()
    assert "std::exp" in code

    n3 = IRNode("dummy_exp", "Exp")
    n3.inputs = ["dummy_in"]
    n3.shape_metadata = 10
    getattr(getattr(g, "_nodes"), "__setitem__")("dummy_exp", n3)
    gen.sorted_nodes.append(n3)
    code = gen.generate()
    assert "std::exp" in code


def test_wasm_conv2d_pool2d_norm_coverage() -> None:
    """Test function test_wasm_conv2d_pool2d_norm_coverage."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_conv = IRNode("conv", "Conv2D", inputs=["in1", "in2"], attributes={"window_strides": (2, 2), "padding": ((1, 1), (1, 1))})
    n_conv.shape_metadata = (1, 16, 16, 16)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = (1, 3, 32, 32)
    in2 = IRNode("in2", "Input")
    in2.shape_metadata = (16, 3, 3, 3)

    n_pool = IRNode("pool", "MaxPool2D", inputs=["conv"], attributes={"window_dimensions": (2, 2), "window_strides": (2, 2), "padding": ((0, 0), (0, 0))})
    n_pool.shape_metadata = (1, 16, 8, 8)

    n_norm = IRNode("norm", "BatchNorm", inputs=["pool", "w", "b", "rm", "rv"], attributes={"epsilon": 1e-5})
    n_norm.shape_metadata = (1, 16, 8, 8)

    w = IRNode("w", "Input")
    w.shape_metadata = (16,)
    b = IRNode("b", "Input")
    b.shape_metadata = (16,)
    rm = IRNode("rm", "Input")
    rm.shape_metadata = (16,)
    rv = IRNode("rv", "Input")
    rv.shape_metadata = (16,)

    n_ln = IRNode("ln", "LayerNorm", inputs=["pool", "w", "b"], attributes={"epsilon": 1e-5})
    n_ln.shape_metadata = (1, 16, 8, 8)

    g.nodes = {"in1": in1, "in2": in2, "conv": n_conv, "pool": n_pool, "norm": n_norm, "w": w, "b": b, "rm": rm, "rv": rv, "ln": n_ln}
    g.inputs = ["in1", "in2", "w", "b", "rm", "rv"]
    g.outputs = ["norm", "ln"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, w, b, rm, rv, n_conv, n_pool, n_norm, n_ln]

    code = gen.generate()
    assert len(code) > 0
    assert "\x00asm" in code or "module" in code or "// max_pool2d" in code or "// MaxPool2D" in code
    assert "// max_pool2d" in code or "// MaxPool2D" in code
    assert "// max_pool2d" in code or "// MaxPool2D" in code


def test_wasm_avgpool2d_coverage() -> None:
    """Test function test_wasm_avgpool2d_coverage."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_pool = IRNode("pool", "AvgPool2D", inputs=["in1"], attributes={"window_dimensions": (2, 2), "window_strides": (2, 2), "padding": (0, 0)})
    n_pool.shape_metadata = (1, 16, 8, 8)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = (1, 16, 16, 16)

    g.nodes = {"in1": in1, "pool": n_pool}
    g.inputs = ["in1"]
    g.outputs = ["pool"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, n_pool]

    code = gen.generate()
    assert len(code) > 0
    assert len(code) > 0


def test_wasm_conv2d_fallback_coverage() -> None:
    """Test function test_wasm_conv2d_fallback_coverage."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_conv = IRNode("conv", "Conv2D", inputs=["in1", "in2"], attributes={})
    n_conv.shape_metadata = (1, 16, 16, 16)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = (1, 3, 32)  # not 4D
    in2 = IRNode("in2", "Input")
    in2.shape_metadata = (16, 3, 3)

    g.nodes = {"in1": in1, "in2": in2, "conv": n_conv}
    g.inputs = ["in1", "in2"]
    g.outputs = ["conv"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, n_conv]

    code = gen.generate()
    assert len(code) > 0
    assert len(code) > 0


def test_wasm_scalar_shapes3() -> None:
    """Test function test_wasm_scalar_shapes3."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_conv = IRNode("conv", "Conv2D", inputs=["in1", "in2"], attributes={"padding": (1, 1)})
    n_conv.shape_metadata = (1, 16, 16, 16)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = 5.0
    in2 = IRNode("in2", "Input")
    in2.shape_metadata = 5.0

    g.nodes = {"in1": in1, "in2": in2, "conv": n_conv}
    g.inputs = ["in1", "in2"]
    g.outputs = ["conv"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, n_conv]
    gen.generate()
    assert gen is not None


def test_wasm_scalar_pool_norm() -> None:
    """Test function test_wasm_scalar_pool_norm."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_pool = IRNode("pool", "MaxPool2D", inputs=["in1"], attributes={"padding": (1, 1)})
    n_pool.shape_metadata = (1, 16, 8, 8)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = 5.0

    g.nodes = {"in1": in1, "pool": n_pool}
    g.inputs = ["in1"]
    g.outputs = ["pool"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, n_pool]
    gen.generate()

    # LayerNorm with bad inputs
    g2 = IRGraph()
    in_ln = IRNode("in_ln", "Input")
    in_ln.shape_metadata = 5.0
    n_ln = IRNode("ln", "LayerNorm", inputs=["in_ln"], attributes={"padding": (1, 1)})
    n_ln.shape_metadata = (1, 16, 8, 8)
    g2.nodes = {"in_ln": in_ln, "ln": n_ln}
    g2.inputs = ["in_ln"]
    g2.outputs = ["ln"]
    gen2 = WasmCodeGenerator(g2)
    gen2.sorted_nodes = [in_ln, n_ln]
    gen2.generate()
    assert gen is not None


def test_wasm_conv2d_scalar_padding() -> None:
    """Test function test_wasm_conv2d_scalar_padding."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_conv = IRNode("conv", "Conv2D", inputs=["in1", "in2"], attributes={"padding": (1, 1)})
    n_conv.shape_metadata = (1, 16, 16, 16)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = (1, 3, 32, 32)
    in2 = IRNode("in2", "Input")
    in2.shape_metadata = (16, 3, 3, 3)

    n_pool = IRNode("pool", "MaxPool2D", inputs=["conv"], attributes={"padding": (1, 1)})
    n_pool.shape_metadata = (1, 16, 8, 8)

    g.nodes = {"in1": in1, "in2": in2, "conv": n_conv, "pool": n_pool}
    g.inputs = ["in1", "in2"]
    g.outputs = ["pool"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, n_conv, n_pool]
    gen.generate()
    assert gen is not None


def test_wasm_norm_no_shape() -> None:
    """Test function test_wasm_norm_no_shape."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    in1 = IRNode("in1", "Input")
    in1.shape_metadata = None

    n_ln = IRNode("ln", "LayerNorm", inputs=["in1"], attributes={})
    n_ln.shape_metadata = (1, 16, 8, 8)

    g.nodes = {"in1": in1, "ln": n_ln}
    g.inputs = ["in1"]
    g.outputs = ["ln"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, n_ln]
    gen.generate()
    assert gen is not None


def test_wasm_missing_attributes_and_strides() -> None:
    """Test function test_wasm_missing_attributes_and_strides."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_conv = IRNode("conv", "Conv2D", inputs=["in1", "in2"])
    n_conv.attributes = {}  # force hasattr(node, "attributes") to be False
    n_conv.shape_metadata = (1, 16, 16, 16)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = (1, 3, 32, 32)
    in2 = IRNode("in2", "Input")
    in2.shape_metadata = (16, 3, 3, 3)

    n_conv2 = IRNode("conv2", "Conv2D", inputs=["in1", "in2"], attributes={"window_dimensions": [1], "window_strides": [1], "padding": 1})
    n_conv2.shape_metadata = (1, 16, 16, 16)

    n_pool = IRNode("pool", "MaxPool2D", inputs=["conv"])
    n_pool.attributes = {}
    n_pool.shape_metadata = (1, 16, 8, 8)

    n_pool2 = IRNode("pool2", "MaxPool2D", inputs=["conv"], attributes={"window_dimensions": [1], "window_strides": [1], "padding": 1})
    n_pool2.shape_metadata = (1, 16, 8, 8)

    n_norm = IRNode("norm", "BatchNorm", inputs=["pool", "w", "b", "rm", "rv"])
    n_norm.attributes = {}
    n_norm.shape_metadata = (1, 16, 8, 8)

    w = IRNode("w", "Input")
    w.shape_metadata = (16,)
    b = IRNode("b", "Input")
    b.shape_metadata = (16,)
    rm = IRNode("rm", "Input")
    rm.shape_metadata = (16,)
    rv = IRNode("rv", "Input")
    rv.shape_metadata = (16,)

    n_ln = IRNode("ln", "LayerNorm", inputs=["pool", "w", "b"])
    n_ln.attributes = {}
    n_ln.shape_metadata = (1, 16, 8, 8)

    g.nodes = {"in1": in1, "in2": in2, "conv": n_conv, "conv2": n_conv2, "pool": n_pool, "pool2": n_pool2, "norm": n_norm, "w": w, "b": b, "rm": rm, "rv": rv, "ln": n_ln}
    g.inputs = ["in1", "in2", "w", "b", "rm", "rv"]
    g.outputs = ["norm", "ln"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, w, b, rm, rv, n_conv, n_conv2, n_pool, n_pool2, n_norm, n_ln]

    code = gen.generate()
    assert len(code) > 0
    assert len(code) > 0


def test_wasm_matmul_and_conv_tiling_and_webrtc() -> None:
    """Test function test_wasm_matmul_and_conv_tiling_and_webrtc."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    # MatMul with tiling
    n_matmul_tiled = IRNode("matmul_tiled", "MatMul", inputs=["in1", "in2"], attributes={"tiling": True})
    n_matmul_tiled.shape_metadata = (16, 16)

    # Conv2D with tiling
    n_conv_tiled = IRNode("conv_tiled", "Conv2D", inputs=["in_img", "in_w"], attributes={"tiling": True, "stride": (1, 1)})
    n_conv_tiled.shape_metadata = (1, 16, 16, 16)

    # WebRTC ops
    n_allreduce = IRNode("allreduce", "AllReduce", inputs=["in1"])
    n_allgather = IRNode("allgather", "AllGather", inputs=["in1"])
    n_alltoall = IRNode("alltoall", "AllToAll", inputs=["in1"])
    n_reducescatter = IRNode("reducescatter", "ReduceScatter", inputs=["in1"])

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = (16, 16)
    in2 = IRNode("in2", "Input")
    in2.shape_metadata = (16, 16)
    in_img = IRNode("in_img", "Input")
    in_img.shape_metadata = (1, 3, 16, 16)
    in_w = IRNode("in_w", "Input")
    in_w.shape_metadata = (16, 3, 3, 3)

    g.nodes = {"in1": in1, "in2": in2, "in_img": in_img, "in_w": in_w, "matmul_tiled": n_matmul_tiled, "conv_tiled": n_conv_tiled, "allreduce": n_allreduce, "allgather": n_allgather, "alltoall": n_alltoall, "reducescatter": n_reducescatter}
    g.inputs = ["in1", "in2", "in_img", "in_w"]
    g.outputs = ["matmul_tiled", "conv_tiled", "allreduce", "allgather", "alltoall"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, in_img, in_w, n_matmul_tiled, n_conv_tiled, n_allreduce, n_allgather, n_alltoall, n_reducescatter]

    code = gen.generate()
    assert code is not None


def test_wasm_folded_batch_norm_coverage() -> None:
    """Test function test_wasm_folded_batch_norm_coverage."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_conv = IRNode("conv", "Conv2D", inputs=["in1", "in2"], attributes={"folded_batch_norm": True, "bn_inputs": ["w", "b", "rm", "rv"]})
    n_conv.shape_metadata = (1, 16, 16, 16)

    n_conv_bad_bn = IRNode("conv_bad_bn", "Conv2D", inputs=["in1", "in2"], attributes={"folded_batch_norm": True, "bn_inputs": ["w"]})
    n_conv_bad_bn.shape_metadata = (1, 16, 16, 16)

    in1 = IRNode("in1", "Input")
    in2 = IRNode("in2", "Input")
    w = IRNode("w", "Input")
    b = IRNode("b", "Input")
    rm = IRNode("rm", "Input")
    rv = IRNode("rv", "Input")

    g.nodes = {"in1": in1, "in2": in2, "conv": n_conv, "conv_bad_bn": n_conv_bad_bn, "w": w, "b": b, "rm": rm, "rv": rv}
    g.inputs = ["in1", "in2", "w", "b", "rm", "rv"]
    g.outputs = ["conv", "conv_bad_bn"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, w, b, rm, rv, n_conv, n_conv_bad_bn]
    code = gen.generate()
    assert "Folded BatchNorm for conv" in code


def test_wasm_while_loop_coverage() -> None:
    """Test function test_wasm_while_loop_coverage."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    body_graph = IRGraph()
    b_in = IRNode("b_in", "Input")
    b_out = IRNode("b_out", "Add", inputs=["b_in", "b_in"])
    body_graph.nodes = {"b_in": b_in, "b_out": b_out}
    body_graph.inputs = ["b_in"]
    body_graph.outputs = ["b_out"]

    g = IRGraph()
    n_while = IRNode("while", "WhileLoop", inputs=["in1"], attributes={"body": body_graph})
    n_while.shape_metadata = (1,)

    in1 = IRNode("in1", "Input")

    g.nodes = {"in1": in1, "while": n_while}
    g.inputs = ["in1"]
    g.outputs = ["while"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, n_while]
    code = gen.generate()
    assert "wasm_f32x4_add" in code


def test_wasm_cond_coverage() -> None:
    """Test function test_wasm_cond_coverage."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    then_graph = IRGraph()
    t_in = IRNode("t_in", "Input")
    t_out = IRNode("t_out", "Exp", inputs=["t_in"])
    then_graph.nodes = {"t_in": t_in, "t_out": t_out}
    then_graph.inputs = ["t_in"]
    then_graph.outputs = ["t_out"]

    else_graph = IRGraph()
    e_in = IRNode("e_in", "Input")
    e_out = IRNode("e_out", "Tanh", inputs=["e_in"])
    else_graph.nodes = {"e_in": e_in, "e_out": e_out}
    else_graph.inputs = ["e_in"]
    else_graph.outputs = ["e_out"]

    g = IRGraph()
    n_cond = IRNode("cond", "Cond", inputs=["cond_in", "t_in", "e_in"], attributes={"then_branch": then_graph, "else_branch": else_graph})
    n_cond.shape_metadata = (1,)

    cond_in = IRNode("cond_in", "Input")
    in2 = IRNode("t_in", "Input")
    in3 = IRNode("e_in", "Input")

    g.nodes = {"cond_in": cond_in, "t_in": in2, "e_in": in3, "cond": n_cond}
    g.inputs = ["cond_in", "t_in", "e_in"]
    g.outputs = ["cond"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [cond_in, in2, in3, n_cond]
    code = gen.generate()
    assert "std::exp" in code
    assert "std::tanh" in code


class DummyGraph:
    """Dummy graph."""

    nodes = cast("list[EagerTensorProtocol]", [])


def test_wasm_aligned_alloc() -> None:
    """Test aligned alloc."""
    gen = cast("EagerCallableProtocol", WasmCodeGenerator)(cast("EagerCallableProtocol", DummyGraph)())
    assert gen._allocate_aligned_memory(128, 16) == "std::aligned_alloc(16, 128);"


def test_wasm_generate_striding_logic() -> None:
    """Test striding logic."""
    gen = cast("EagerCallableProtocol", WasmCodeGenerator)(cast("EagerCallableProtocol", DummyGraph)())
    strides, c_code = gen._generate_striding_logic([])
    assert strides == []
    assert c_code == "0"

    strides, c_code = gen._generate_striding_logic([2, 3])
    assert strides == [3, 1]
    assert c_code == "((idx / 3) % 2) * 3 + (idx % 3) * 1"


def test_wasm_compile_clang_fallback(mocker: EagerTensorProtocol) -> None:
    """Test compile clang fallback.

    Args:
        mocker: Mocker fixture.
    """
    gen = cast("EagerCallableProtocol", WasmCodeGenerator)(cast("EagerCallableProtocol", DummyGraph)())
    mocker.patch("shutil.which", side_effect=lambda x: "clang" if x == "clang" else None)
    mocker.patch("subprocess.run")
    js, wasm = gen.compile_wasm("/tmp")
    assert wasm == "/tmp/kernel.wasm"


def test_wasm_compile_no_compiler(mocker: EagerTensorProtocol) -> None:
    """Test compile no compiler.

    Args:
        mocker: Mocker fixture.
    """
    gen = cast("EagerCallableProtocol", WasmCodeGenerator)(cast("EagerCallableProtocol", DummyGraph)())
    mocker.patch("shutil.which", return_value=None)
    with pytest.raises(CompilationError, match="Neither emcc nor clang found"):
        gen.compile_wasm("/tmp")


def test_wasm_compile_subprocess_error(mocker: EagerTensorProtocol) -> None:
    """Test compile subprocess error.

    Args:
        mocker: Mocker fixture.
    """
    import subprocess

    gen = cast("EagerCallableProtocol", WasmCodeGenerator)(cast("EagerCallableProtocol", DummyGraph)())
    mocker.patch("shutil.which", return_value="emcc")
    cast("EagerCallableProtocol", mocker.patch)("subprocess.run", side_effect=cast("EagerCallableProtocol", subprocess.CalledProcessError)(1, "emcc", b"out", b"err"))
    with pytest.raises(CompilationError, match="WASM compilation failed: err"):
        gen.compile_wasm("/tmp")


def test_wasm_compile_generic_error(mocker: EagerTensorProtocol) -> None:
    """Test compile generic error.

    Args:
        mocker: Mocker fixture.
    """
    gen = cast("EagerCallableProtocol", WasmCodeGenerator)(cast("EagerCallableProtocol", DummyGraph)())
    mocker.patch("shutil.which", return_value="emcc")
    cast("EagerCallableProtocol", mocker.patch)("subprocess.run", side_effect=cast("EagerCallableProtocol", Exception)("foo"))
    with pytest.raises(CompilationError, match="WASM compilation failed with unknown error: foo"):
        gen.compile_wasm("/tmp")


def test_wasm_generic_visit() -> None:
    """Test generic visit."""
    gen = cast("EagerCallableProtocol", WasmCodeGenerator)(cast("EagerCallableProtocol", DummyGraph)())

    class DummyNode:
        """Dummy node."""

        id = "dummy_id"

    assert cast("EagerCallableProtocol", gen.generic_visit)(cast("EagerCallableProtocol", DummyNode)(), []) == "dummy_id"


def test_wasm_visit_conv2d_shapes(mocker: EagerTensorProtocol) -> None:
    """Test visit conv2d shapes.

    Args:
        mocker: Mocker fixture.
    """
    gen = cast("EagerCallableProtocol", WasmCodeGenerator)(cast("EagerCallableProtocol", DummyGraph)())

    class DummyNode:
        """Dummy node."""

        id = "conv_id"
        attributes = {"stride": 1, "padding": "valid"}

    # Mock get_wasm_template and get_cpp_helpers
    mocker.patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template", return_value={"body": "body"})

    cast("EagerCallableProtocol", gen.visit_Conv2D)(cast("EagerCallableProtocol", DummyNode)(), "Conv2D", "conv_id", ["in0", "in1"], 5, 5)  # shape is int
    cast("EagerCallableProtocol", gen.visit_Conv2D)(cast("EagerCallableProtocol", DummyNode)(), "Conv2D", "conv_id", ["in0", "in1"], [1, 5], 5)  # shape length < 4
    assert gen is not None


def test_wasm_visit_pool2d_shapes(mocker: EagerTensorProtocol) -> None:
    """Test visit pool2d shapes.

    Args:
        mocker: Mocker fixture.
    """
    gen = cast("EagerCallableProtocol", WasmCodeGenerator)(cast("EagerCallableProtocol", DummyGraph)())

    class DummyNode:
        """Dummy node."""

        id = "pool_id"
        attributes = {"kernel_size": 2, "stride": 2}

    mocker.patch("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"MaxPool2D": {"variants": {"edge_wasm_simd": {"template": "max_pool2d"}}}})
    mocker.patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template", return_value={"body": "body"})

    # shape and in0_shape are int/float and length < 4
    cast("EagerCallableProtocol", gen._generate_pooling2d)(cast("EagerCallableProtocol", DummyNode)(), "pool_id", ["in0"], 5, "max_pool2d")
    cast("EagerCallableProtocol", gen._generate_pooling2d)(cast("EagerCallableProtocol", DummyNode)(), "pool_id", ["in0"], [1, 5], "max_pool2d")
    assert gen is not None


def test_wasm_generate_op_missing_template(mocker: EagerTensorProtocol) -> None:
    """Test generate op missing template.

    Args:
        mocker: Mocker fixture.
    """
    gen = cast("EagerCallableProtocol", WasmCodeGenerator)(cast("EagerCallableProtocol", DummyGraph)())

    class DummyNode:
        """Dummy node."""

        id = "op_id"

    mocker.patch("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"FakeOp": {"variants": {"edge_wasm_simd": {}}}})
    UnimplementedMathError = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.backends.edge.wasm"), "UnimplementedMathError"))

    with cast("EagerCallableProtocol", pytest.raises)(UnimplementedMathError, match="Missing WASM SIMD template"):
        cast("EagerCallableProtocol", gen._generate_op)(cast("EagerCallableProtocol", DummyNode)(), "FakeOp", "op_id", ["in0"], 5, 5)


def test_wasm_generate_op_missing_body(mocker: EagerTensorProtocol) -> None:
    """Test generate op missing body.

    Args:
        mocker: Mocker fixture.
    """
    gen = cast("EagerCallableProtocol", WasmCodeGenerator)(cast("EagerCallableProtocol", DummyGraph)())

    class DummyNode:
        """Dummy node."""

        id = "op_id"

    mocker.patch("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"FakeOp": {"variants": {"edge_wasm_simd": {"template": "fake_template"}}}})
    mocker.patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template", return_value={"not_body": "x"})

    UnimplementedMathError = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.backends.edge.wasm"), "UnimplementedMathError"))

    with cast("EagerCallableProtocol", pytest.raises)(UnimplementedMathError, match="MISSING BODY FOR: FakeOp"):
        cast("EagerCallableProtocol", gen._generate_op)(cast("EagerCallableProtocol", DummyNode)(), "FakeOp", "op_id", ["in0"], 5, 5)


def test_wasm_generate_op_shapes(mocker: EagerTensorProtocol) -> None:
    """Test generate op shapes.

    Args:
        mocker: Mocker fixture.
    """
    gen = cast("EagerCallableProtocol", WasmCodeGenerator)(cast("EagerCallableProtocol", DummyGraph)())

    class DummyNode:
        """Dummy node."""

        id = "op_id"

    mocker.patch("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"FakeOp": {"variants": {"edge_wasm_simd": {"template": "fake_template"}}}})
    mocker.patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template", return_value={"body": "body_code"})

    cast("EagerCallableProtocol", gen._generate_op)(cast("EagerCallableProtocol", DummyNode)(), "FakeOp", "op_id", ["in0"], 5, 5)
    cast("EagerCallableProtocol", gen._generate_op)(cast("EagerCallableProtocol", DummyNode)(), "FakeOp", "op_id", ["in0"], [1, 5], 5)
    assert gen is not None


def test_wasm_visit_cond_mapped_inputs(mocker: EagerTensorProtocol) -> None:
    """Test visit cond mapped inputs.

    Args:
        mocker: Mocker fixture.
    """
    gen = cast("EagerCallableProtocol", WasmCodeGenerator)(cast("EagerCallableProtocol", DummyGraph)())

    class DummySubNode:
        """Dummy sub node."""

        op_type = "Add"
        id = "sub_id"
        inputs = ["not_in_branch_inputs", "in_branch_inputs"]
        shape_metadata = [1]

    class DummyBranchGraph:
        """Dummy branch graph."""

        inputs = ["in_branch_inputs"]

        def get_sorted_nodes(self) -> EagerValue:
            """Get sorted nodes.

            Returns:
                list: Dummy subnode.
            """
            return [cast("EagerCallableProtocol", DummySubNode)()]

    class DummyNode:
        """Dummy node."""

        id = "cond_id"
        attributes = {"true_graph": DummyBranchGraph(), "false_graph": DummyBranchGraph()}

    mocker.patch("ml_switcheroo_compiler.backends.edge.wasm.WasmCodeGenerator._generate_op")
    cast("EagerCallableProtocol", gen.visit_Cond)(cast("EagerCallableProtocol", DummyNode)(), "Cond", "cond_id", ["pred", "in_branch_inputs"], [1], 1)
    assert gen is not None


def test_wasm_missing_coverage() -> None:
    """Test function test_wasm_missing_coverage."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    n1 = IRNode("if_node", "If", inputs=["in1"], attributes={"branch_graphs": [IRGraph()]})
    graph.nodes["if_node"] = n1
    graph.inputs = ["in1"]
    graph.outputs = ["if_node"]

    gen = WasmCodeGenerator(graph)
    gen.visit_If(n1, "If", "if_node", ["in1"], [10], 10)

    n2 = IRNode("cond_node", "Cond", inputs=["in1"], attributes={"branch_graphs": [IRGraph(), IRGraph()]})
    gen.visit_Cond(n2, "Cond", "cond_node", ["in1"], [10], 10)
    assert gen is not None


def test_webgpu_missing_shader() -> None:
    """Test function test_webgpu_missing_shader."""
    from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    n0 = IRNode("in1", "Input", inputs=[], shape_metadata=[1])
    n1 = IRNode("bad", "BadOp", inputs=["in1"], shape_metadata=[1])
    graph.nodes["in1"] = n0
    graph.nodes["bad"] = n1
    graph.inputs = ["in1"]
    graph.outputs = ["bad"]

    # Use a custom generator to force exception in _get_wgsl_for_op
    class MyGen(WebGPUCodeGenerator):
        """Test class MyGen."""

        def _get_wgsl_for_op(self, node: EagerValue, shape: EagerValue, nelem: EagerValue, clean_id: EagerValue) -> tuple[list[str], str, str, str]:
            """Test function _get_wgsl_for_op.

            Args:
                node: Test parameter `node`.
                shape: Test parameter `shape`.
                nelem: Test parameter `nelem`.
                clean_id: Test parameter `clean_id`.
            """
            raise ValueError("Forced error")

    gen = MyGen(graph)
    import pytest

    with pytest.raises(ValueError, match="Forced error"):
        gen.generate()


def test_webgl_missing_shader() -> None:
    """Test function test_webgl_missing_shader."""
    from ml_switcheroo_compiler.backends.edge.webgl import WebGLCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    n1 = IRNode("bad", "BadOp", inputs=["in1"])
    graph.nodes["bad"] = n1
    graph.inputs = ["in1"]
    graph.outputs = ["bad"]
    gen = WebGLCodeGenerator(graph)
    import pytest

    with pytest.raises(ValueError, match="Missing WebGL shader template for operation: BadOp"):
        gen.generate()


def test_wasm_missing_coverage_empty_branches() -> None:
    """Test function test_wasm_missing_coverage_empty_branches."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    n1 = IRNode("if_node", "If", inputs=["in1"], attributes={"branch_graphs": []})
    gen = WasmCodeGenerator(graph)
    gen.visit_If(n1, "If", "if_node", ["in1"], [10], 10)
    assert gen is not None


def test_wasm_compile_aot_impl() -> None:
    """Test function test_wasm_compile_aot_impl."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph1 = IRGraph()
    n1 = IRNode("x", "Input", inputs=[])
    graph1.nodes["x"] = n1
    graph1.inputs = ["x"]
    graph1.outputs = ["x"]

    gen = WasmCodeGenerator(graph1)

    graph2 = IRGraph()
    n2 = IRNode("y", "Input", inputs=[])
    graph2.nodes["y"] = n2
    graph2.inputs = ["y"]
    graph2.outputs = ["y"]

    with patch.object(gen, "compile_wasm", return_value=("a.js", "b.wasm")):
        res = gen._compile_aot_impl(graph2, output_dir="/tmp/wasm_out")
        assert res == ("a.js", "b.wasm")
        assert gen.graph == graph2

    with patch.object(gen, "compile_wasm", return_value=None):
        res2 = gen._compile_aot_impl(graph2)
        assert res2 == ("", "")


def test_wasm_shape_telemetry() -> None:
    """Test function test_wasm_shape_telemetry."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    n_in = IRNode("in0", "Input", inputs=[], shape_metadata=[2, 4])
    n_mul = IRNode("mul0", "Mul", inputs=["in0", "in0"], shape_metadata=[2, 4])
    graph.nodes["in0"] = n_in
    graph.nodes["mul0"] = n_mul
    graph.inputs = ["in0"]
    graph.outputs = ["mul0"]

    gen = WasmCodeGenerator(graph)
    payload = gen.export_graph_payload()

    assert payload["inputs"] == ["in0"]
    assert payload["outputs"] == ["mul0"]
    assert len(payload["nodes"]) == 2
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(payload, "__getitem__")("nodes")), "__getitem__")(0)), "__getitem__")("id")) == "in0"
    assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(payload, "__getitem__")("nodes")), "__getitem__")(0)), "__getitem__")("shape_metadata")) == [2, 4]

    # Test applying telemetry back
    gen.apply_shape_telemetry({"mul0": [4, 8]})
    assert n_mul.shape_metadata == (4, 8)


def test_wasm_conv3d_generation() -> None:
    """Test WASM SIMD Conv3D code generation with full parameter coverage."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    n_in = IRNode("in0", "Input", inputs=[], shape_metadata=[1, 2, 4, 8, 8])
    n_w = IRNode("w0", "Input", inputs=[], shape_metadata=[4, 2, 3, 3, 3])
    n_conv = IRNode(
        "conv3d_0",
        "Conv3D",
        inputs=["in0", "w0"],
        shape_metadata=[1, 4, 4, 8, 8],
        attributes={"strides": (1, 1, 1), "padding": "SAME", "dilations": (1, 1, 1)},
    )
    graph.nodes["in0"] = n_in
    graph.nodes["w0"] = n_w
    graph.nodes["conv3d_0"] = n_conv
    graph.inputs = ["in0", "w0"]
    graph.outputs = ["conv3d_0"]

    gen = WasmCodeGenerator(graph)
    code = gen.generate()
    assert "SIMD Conv3D (Standard)" in code
    assert "int filter_d = 3;" in code
    assert "int filter_h = 3;" in code
    assert "int filter_w = 3;" in code
    assert "buf_conv3d_0[out_idx] = acc;" in code


def test_wasm_conv3d_branches_coverage() -> None:
    """Test WASM Conv3D with scalar and 2-tuple padding, strides, dilations, and non-5D shapes."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    n_in_scalar = IRNode("in_scalar", "Input", inputs=[], shape_metadata=4)
    n_in_empty = IRNode("in_empty", "Input", inputs=[], shape_metadata=[])
    n_in_3d = IRNode("in_3d", "Input", inputs=[], shape_metadata=[2, 3, 4])
    n_w_empty = IRNode("w_empty", "Input", inputs=[], shape_metadata=[])
    n_w_scalar = IRNode("w_scalar", "Input", inputs=[], shape_metadata=3)
    n_w_3d = IRNode("w_3d", "Input", inputs=[], shape_metadata=[2, 3, 4])

    graph.nodes["in_scalar"] = n_in_scalar
    graph.nodes["in_empty"] = n_in_empty
    graph.nodes["in_3d"] = n_in_3d
    graph.nodes["w_empty"] = n_w_empty
    graph.nodes["w_scalar"] = n_w_scalar
    graph.nodes["w_3d"] = n_w_3d

    gen = WasmCodeGenerator(graph)

    # Test Conv3D with 3D in0_shape (< 5), 3D w_shape (< 5), 3D shape (< 5), and padding="VALID"
    node_c0 = IRNode(
        "conv3d_3d",
        "Conv3D",
        inputs=["in_3d", "w_3d"],
        shape_metadata=[2, 3, 4],
        attributes={"stride": 1, "padding": "VALID", "dilation": 2},
    )
    gen.visit_Conv3D(node_c0, "Conv3D", "conv3d_3d", ["in_3d", "w_3d"], [2, 3, 4], 24)

    # Test Conv3D with empty in0_shape and empty w_shape and empty shape
    node_c1 = IRNode(
        "conv3d_empty",
        "Conv3D",
        inputs=["in_empty", "w_empty"],
        shape_metadata=[],
        attributes={"stride": 1, "padding": "SAME", "dilation": 1},
    )
    gen.visit_Conv3D(node_c1, "Conv3D", "conv3d_empty", ["in_empty", "w_empty"], [], 1)

    # Test Conv3D with scalar in0_shape, scalar w_shape, and scalar shape
    node_c2 = IRNode(
        "conv3d_scalar",
        "Conv3D",
        inputs=["in_scalar", "w_scalar"],
        shape_metadata=8,
        attributes={"stride": (2, 2, 2), "padding": (1, 1, 1), "dilation": (1, 1, 1)},
    )
    cast("EagerCallableProtocol", gen.visit_Conv3D)(node_c2, "Conv3D", "conv3d_scalar", ["in_scalar", "w_scalar"], 8, 8)

    # Test Conv3D with 2-element stride and padding
    node_c3 = IRNode(
        "conv3d_2elem",
        "Conv3D",
        inputs=["in_scalar", "w_scalar"],
        shape_metadata=[1, 1, 4, 4, 4],
        attributes={"stride": (2, 2), "padding": (1, 1), "dilation": 1},
    )
    gen.visit_Conv3D(node_c3, "Conv3D", "conv3d_2elem", ["in_scalar", "w_scalar"], [1, 1, 4, 4, 4], 64)

    # Test Conv3D with 1-element list stride and padding
    node_c4 = IRNode(
        "conv3d_1elem",
        "Conv3D",
        inputs=["in_scalar", "w_scalar"],
        shape_metadata=[1, 1, 4, 4, 4],
        attributes={"stride": [1], "padding": [0], "dilation": [1]},
    )
    gen.visit_Conv3D(node_c4, "Conv3D", "conv3d_1elem", ["in_scalar", "w_scalar"], [1, 1, 4, 4, 4], 64)
    assert gen is not None


def test_wasm_scan_body_graph_and_blank_lines() -> None:
    """Verify WASM scan with body_graph and template emission with blank lines."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    sub_graph = IRGraph(name="sub")
    sub_node = IRNode("sub_in", "Input", inputs=[], shape_metadata=[2, 2])
    sub_graph.nodes["sub_in"] = sub_node
    sub_graph.inputs = ["sub_in"]
    sub_graph.outputs = ["sub_in"]

    main_graph = IRGraph(name="main")
    in_node = IRNode("in0", "Input", inputs=[], shape_metadata=[2, 2])
    main_graph.nodes["in0"] = in_node
    main_graph.inputs = ["in0"]

    gen = WasmCodeGenerator(main_graph)

    # visit_Scan with body_graph
    scan_node = IRNode("scan_0", "Scan", inputs=["in0"], attributes={"body_graph": sub_graph})
    gen.visit_Scan(scan_node, "Scan", "scan_0", ["in0"], [2, 2], 4)

    # _generate_op with template having blank lines to cover line.strip() branch
    with (
        patch.dict("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"CustomOp": {"variants": {"edge_wasm_simd": {"template": "custom_tpl"}}}}),
        patch(
            "ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template",
            return_value={"body": "line_one;\n\nline_two;\n  \nline_three;"},
        ),
    ):
        dummy_node = IRNode("custom_op_0", "CustomOp", inputs=["in0"])
        gen._generate_op(dummy_node, "CustomOp", "custom_op_0", ["in0"], [2, 2], 4)
    assert gen is not None
