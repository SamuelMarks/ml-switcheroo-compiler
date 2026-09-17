"""Unit tests for the declarative symbolic autodiff expression compiler."""

from __future__ import annotations

import pytest
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.grad.symbolic_compiler import SymbolicExpressionCompiler
from ml_switcheroo_compiler.ir.core import IRGraph


def test_symbolic_compiler_init_and_load(tmp_path) -> None:
    """Test compiler initialization, manifest loading, caching, and missing file error handling."""
    compiler = SymbolicExpressionCompiler()
    manifest = compiler.load_manifest()
    assert manifest is not None
    assert compiler.load_manifest() is manifest

    missing_compiler = SymbolicExpressionCompiler(manifest_path=str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError):
        missing_compiler.load_manifest()


def test_split_nested_args() -> None:
    """Test splitting nested argument strings with complex nested parenthesis and brackets."""
    compiler = SymbolicExpressionCompiler()
    args_str = "Mul($tangent[0], $input[1]), MatMul($input[0], Add($tangent[1], Constant(1.0))), axis=-1"
    parts = compiler.split_nested_args(args_str)
    assert len(parts) == 3
    assert parts[0] == "Mul($tangent[0], $input[1])"
    assert parts[1] == "MatMul($input[0], Add($tangent[1], Constant(1.0)))"
    assert parts[2] == "axis=-1"

    trailing = compiler.split_nested_args("a, b,")
    assert len(trailing) == 2
    assert trailing == ["a", "b"]


def test_compile_primitive_expressions() -> None:
    """Test compiling basic symbolic primitives such as constants, zeros, inputs, and cotangents."""
    graph = IRGraph()
    compiler = SymbolicExpressionCompiler()
    node = LogicalNode(id="n0", op_type="Cos", inputs=["x_in"], shape_metadata=[2, 3])
    graph.nodes["n0"] = node

    assert compiler.compile_expression(graph, "3.14", node) == "3.14"

    cot_id = compiler.compile_expression(graph, "$cotangent", node, cotangent="cot_0")
    assert cot_id == "cot_0"

    out_id = compiler.compile_expression(graph, "$output", node)
    assert out_id == "n0"

    in0_id = compiler.compile_expression(graph, "$input[0]", node)
    assert in0_id == "x_in"

    in_oob = compiler.compile_expression(graph, "$input[99]", node)
    assert in_oob == ""

    tan0_id = compiler.compile_expression(graph, "$tangent[0]", node, tangents=["tan_0"])
    assert tan0_id == "tan_0"

    tan_oob = compiler.compile_expression(graph, "$tangent[99]", node, tangents=["tan_0"])
    assert tan_oob == ""

    zero_id = compiler.compile_expression(graph, "$zero", node)
    assert zero_id in graph.nodes
    assert graph.nodes[zero_id].op_type == "ZerosLike"

    neg_id = compiler.compile_expression(graph, "- $cotangent", node, cotangent="cot_0")
    assert neg_id in graph.nodes
    assert graph.nodes[neg_id].op_type == "Negative"

    raw_str = compiler.compile_expression(graph, "unrecognized_raw_symbol", node)
    assert raw_str == "unrecognized_raw_symbol"


def test_build_call_node_variants() -> None:
    """Test build_call_node handling of special ops like Constant, BroadcastAlign, and attributes."""
    graph = IRGraph()
    compiler = SymbolicExpressionCompiler()
    dummy_node = LogicalNode(id="dummy", op_type="CustomOp", inputs=["in_a", "in_b"], attributes={"flag": True})
    graph.nodes["dummy"] = dummy_node
    target_node = LogicalNode(id="tgt", op_type="Input", inputs=[], shape_metadata=[4, 4])
    graph.nodes["tgt"] = target_node

    cst_id = compiler.build_call_node(graph, "Constant", ["2.5"], dummy_node)
    assert cst_id in graph.nodes
    assert graph.nodes[cst_id].op_type == "Constant"
    assert graph.nodes[cst_id].attributes["value"] == 2.5

    bc_reduce_id = compiler.build_call_node(graph, "BroadcastReduce", ["in_a", "tgt"], dummy_node)
    assert bc_reduce_id in graph.nodes
    assert graph.nodes[bc_reduce_id].op_type == "BroadcastReduce"

    bc_like_id = compiler.build_call_node(graph, "BroadcastLike", ["in_a", "tgt"], dummy_node)
    assert bc_like_id in graph.nodes
    assert graph.nodes[bc_like_id].op_type == "BroadcastLike"

    call_id = compiler.build_call_node(graph, "CustomOp", ["in_a", "key=123", "bad_literal=invalid("], dummy_node)
    assert call_id in graph.nodes
    assert graph.nodes[call_id].attributes["key"] == 123
    assert graph.nodes[call_id].attributes["bad_literal"] == "invalid("


def test_compile_vjp_and_jvp() -> None:
    """Test compiling high-level VJP and JVP expressions directly from the manifest."""
    graph = IRGraph()
    compiler = SymbolicExpressionCompiler()
    add_node = LogicalNode(id="add_0", op_type="Add", inputs=["x1", "x2"], shape_metadata=[2, 2])
    graph.nodes["add_0"] = add_node

    vjp_adjs = compiler.compile_vjp(graph, "Add", add_node, cotangent="cot_add")
    assert vjp_adjs is not None
    assert len(vjp_adjs) == 2
    assert vjp_adjs[0] == "cot_add"
    assert vjp_adjs[1] == "cot_add"

    jvp_tan = compiler.compile_jvp(graph, "Add", add_node, tangents=["t1", "t2"])
    assert jvp_tan is not None
    assert jvp_tan in graph.nodes
    assert graph.nodes[jvp_tan].op_type == "Add"

    assert compiler.compile_vjp(graph, "NonExistentOp", add_node, "cot_add") is None
    assert compiler.compile_jvp(graph, "NonExistentOp", add_node, ["t1", "t2"]) is None

    # Test single-string vjp spec
    manifest = compiler.load_manifest()
    from ml_switcheroo_compiler.grad.config_models import AutodiffRuleModel

    manifest.vjp_rules["StringVjpOp"] = AutodiffRuleModel(opcode="StringVjpOp", vjp="Transpose($cotangent)")
    vjp_single = compiler.compile_vjp(graph, "StringVjpOp", add_node, cotangent="cot_tr")
    assert vjp_single is not None
    assert len(vjp_single) == 1

    # Test None vjp spec
    manifest.vjp_rules["NoneVjpOp"] = AutodiffRuleModel(opcode="NoneVjpOp", vjp=None)
    assert compiler.compile_vjp(graph, "NoneVjpOp", add_node, "cot_add") is None


def test_composite_activation_gradients() -> None:
    """Test composite activation gradient generation for GELU, SiLU, Softmax, and LogSoftmax."""
    graph = IRGraph()
    compiler = SymbolicExpressionCompiler()

    gelu_grad_id = compiler.compile_gelu_grad(graph, cotangent="cot_g", x="x_g")
    assert gelu_grad_id in graph.nodes

    silu_grad_id = compiler.compile_silu_grad(graph, cotangent="cot_s", x="x_s", out="out_s")
    assert silu_grad_id in graph.nodes

    silu_grad_no_out = compiler.compile_silu_grad(graph, cotangent="cot_s", x="x_s", out=None)
    assert silu_grad_no_out in graph.nodes

    softmax_grad_id = compiler.compile_softmax_grad(graph, cotangent="cot_sm", out="out_sm", axis=-1)
    assert softmax_grad_id in graph.nodes

    log_softmax_grad_id = compiler.compile_log_softmax_grad(graph, cotangent="cot_lsm", out="out_lsm", axis=-1)
    assert log_softmax_grad_id in graph.nodes


def test_composite_tensor_and_normalization_gradients() -> None:
    """Test composite gradients for matrix operations, transpose, reshape, and normalization layers."""
    graph = IRGraph()
    compiler = SymbolicExpressionCompiler()

    mm_a = compiler.compile_matmul_grad_a(graph, cotangent="cot_mm", b="b_node")
    assert mm_a in graph.nodes

    mm_b = compiler.compile_matmul_grad_b(graph, cotangent="cot_mm", a="a_node")
    assert mm_b in graph.nodes

    bmm_a, bmm_b = compiler.compile_batch_matmul_grad(graph, cotangent="cot_bmm", a="a_bmm", b="b_bmm")
    assert bmm_a in graph.nodes
    assert bmm_b in graph.nodes

    trans_adj = compiler.compile_transpose_grad(graph, cotangent="cot_tr")
    assert trans_adj in graph.nodes

    reshape_adj = compiler.compile_reshape_grad(graph, cotangent="cot_rs")
    assert reshape_adj in graph.nodes

    ln_adj = compiler.compile_layer_norm_grad(graph, cotangent="cot_ln", x="x_ln", weight="w_ln")
    assert ln_adj in graph.nodes

    bn_adj = compiler.compile_batch_norm_grad(graph, cotangent="cot_bn", x="x_bn", scale="s_bn")
    assert bn_adj in graph.nodes

    rmsn_adj = compiler.compile_rms_norm_grad(graph, cotangent="cot_rmsn", x="x_rmsn", weight="w_rmsn")
    assert rmsn_adj in graph.nodes
