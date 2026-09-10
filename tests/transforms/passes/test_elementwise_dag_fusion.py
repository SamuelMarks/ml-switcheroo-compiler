"""Exhaustive tests for elementwise DAG cluster fusion in operator_fusion pass."""

from ml_switcheroo_compiler.backends.cuda.cuda import CudaCodeGenerator
from ml_switcheroo_compiler.backends.metal.metal import MetalCodeGenerator
from ml_switcheroo_compiler.backends.rocm.rocm import RocmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
    _get_scalar_expression_snippet,
    fuse_elementwise_clusters,
    operator_fusion_pass,
)


def test_elementwise_cluster_fusion_chain() -> None:
    """Verify that a chain of elementwise operations is fused into a single FusedElementwise node."""
    graph = IRGraph(name="test_fusion")
    graph.nodes["in_a"] = IRNode(id="in_a", op_type="Input", shape_metadata=(4, 8))
    graph.nodes["in_b"] = IRNode(id="in_b", op_type="Input", shape_metadata=(4, 8))
    graph.nodes["add1"] = IRNode(id="add1", op_type="Add", inputs=["in_a", "in_b"], shape_metadata=(4, 8))
    graph.nodes["mul1"] = IRNode(id="mul1", op_type="Mul", inputs=["add1", "in_a"], shape_metadata=(4, 8))
    graph.nodes["relu1"] = IRNode(id="relu1", op_type="Relu", inputs=["mul1"], shape_metadata=(4, 8))
    graph.inputs = ["in_a", "in_b"]
    graph.outputs = ["relu1"]

    modified = fuse_elementwise_clusters(graph)
    assert modified is True
    assert "add1" not in graph.nodes
    assert "mul1" not in graph.nodes
    assert "relu1" in graph.nodes

    fused = graph.nodes["relu1"]
    assert fused.op_type == "FusedElementwise"
    assert fused.inputs == ["in_a", "in_b"]
    assert "Add" in fused.attributes["fused_ops"]
    assert "Mul" in fused.attributes["fused_ops"]
    assert "Relu" in fused.attributes["fused_ops"]
    assert "fmaxf" in fused.attributes["scalar_expr"]

    # Verify CUDA generation includes fused kernel
    cuda_code = CudaCodeGenerator(graph).generate()
    assert "fused_elementwise_kernel" in cuda_code
    assert "fmaxf" in cuda_code

    # Verify ROCm generation includes fused kernel
    rocm_code = RocmCodeGenerator(graph).generate()
    assert "fused_elementwise_kernel" in rocm_code

    # Verify Metal generation includes fused compute shader
    metal_code = MetalCodeGenerator(graph).generate()
    assert "fused_elementwise" in metal_code


def test_elementwise_fusion_multi_consumer_not_fused() -> None:
    """Verify that intermediate nodes with multiple consumers are not fused."""
    graph = IRGraph(name="test_multi_consumer")
    graph.nodes["in_a"] = IRNode(id="in_a", op_type="Input", shape_metadata=(4,))
    graph.nodes["add1"] = IRNode(id="add1", op_type="Add", inputs=["in_a", "in_a"], shape_metadata=(4,))
    graph.nodes["relu1"] = IRNode(id="relu1", op_type="Relu", inputs=["add1"], shape_metadata=(4,))
    graph.nodes["sig1"] = IRNode(id="sig1", op_type="Sigmoid", inputs=["add1"], shape_metadata=(4,))
    graph.inputs = ["in_a"]
    graph.outputs = ["relu1", "sig1"]

    modified = fuse_elementwise_clusters(graph)
    assert modified is False
    assert "add1" in graph.nodes
    assert graph.nodes["add1"].op_type == "Add"


def test_elementwise_fusion_output_node_not_inlined() -> None:
    """Verify that nodes listed in graph outputs are preserved as consumers and not inlined away."""
    graph = IRGraph(name="test_output_preserved")
    graph.nodes["in_a"] = IRNode(id="in_a", op_type="Input", shape_metadata=(4,))
    graph.nodes["add1"] = IRNode(id="add1", op_type="Add", inputs=["in_a", "in_a"], shape_metadata=(4,))
    graph.nodes["relu1"] = IRNode(id="relu1", op_type="Relu", inputs=["add1"], shape_metadata=(4,))
    graph.inputs = ["in_a"]
    graph.outputs = ["add1", "relu1"]

    modified = fuse_elementwise_clusters(graph)
    assert modified is False
    assert "add1" in graph.nodes


def test_elementwise_fusion_incompatible_shapes_not_fused() -> None:
    """Verify that elementwise nodes with different non-matching shapes are not fused."""
    graph = IRGraph(name="test_shape_mismatch")
    graph.nodes["in_a"] = IRNode(id="in_a", op_type="Input", shape_metadata=(2, 3))
    graph.nodes["add1"] = IRNode(id="add1", op_type="Add", inputs=["in_a", "in_a"], shape_metadata=(2, 3))
    graph.nodes["relu1"] = IRNode(id="relu1", op_type="Relu", inputs=["add1"], shape_metadata=(4, 6))
    graph.inputs = ["in_a"]
    graph.outputs = ["relu1"]

    modified = fuse_elementwise_clusters(graph)
    assert modified is False
    assert "add1" in graph.nodes


def test_elementwise_fusion_empty_graph() -> None:
    """Verify fuse_elementwise_clusters handles empty graph safely."""
    graph = IRGraph()
    assert fuse_elementwise_clusters(graph) is False


def test_scalar_expression_snippets_coverage() -> None:
    """Test scalar expression snippet generation for all supported elementwise operations."""
    assert _get_scalar_expression_snippet("sub", ["a", "b"]) == "(a - b)"
    assert _get_scalar_expression_snippet("div", ["a", "b"]) == "(a / b)"
    assert _get_scalar_expression_snippet("truedivide", ["a", "b"]) == "(a / b)"
    assert _get_scalar_expression_snippet("sigmoid", ["a"]) == "(1.0f / (1.0f + expf(-a)))"
    assert _get_scalar_expression_snippet("neg", ["a"]) == "(-a)"
    assert _get_scalar_expression_snippet("abs", ["a"]) == "fabsf(a)"
    assert _get_scalar_expression_snippet("exp", ["a"]) == "expf(a)"
    assert _get_scalar_expression_snippet("log", ["a"]) == "logf(a)"
    assert _get_scalar_expression_snippet("sqrt", ["a"]) == "sqrtf(a)"
    assert _get_scalar_expression_snippet("tanh", ["a"]) == "tanhf(a)"
    assert "0.5f" in _get_scalar_expression_snippet("gelu", ["a"])
    assert "expf" in _get_scalar_expression_snippet("silu", ["a"])
    assert _get_scalar_expression_snippet("maximum", ["a", "b"]) == "fmaxf(a, b)"
    assert _get_scalar_expression_snippet("minimum", ["a", "b"]) == "fminf(a, b)"
    assert _get_scalar_expression_snippet("unknown_op", ["a"]) == "a"
    assert _get_scalar_expression_snippet("add", []) == "(0.0f + 0.0f)"


def test_operator_fusion_pass_with_elementwise_cluster() -> None:
    """Verify operator_fusion_pass successfully triggers elementwise cluster fusion and DCE."""
    graph = IRGraph(name="test_pass")
    graph.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(10,))
    graph.nodes["n_sub"] = IRNode(id="n_sub", op_type="Sub", inputs=["x", "x"], shape_metadata=(10,))
    graph.nodes["n_abs"] = IRNode(id="n_abs", op_type="Abs", inputs=["n_sub"], shape_metadata=(10,))
    graph.inputs = ["x"]
    graph.outputs = ["n_abs"]

    result = operator_fusion_pass(graph)
    assert result is True
    assert "n_sub" not in graph.nodes
    assert "n_abs" in graph.nodes
    assert graph.nodes["n_abs"].op_type == "FusedElementwise"


def test_operator_fusion_fused_node_as_consumer_and_producer() -> None:
    """Verify fusion where both producer and consumer can be FusedElementwise nodes."""
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
        OperatorFusionPass,
        apply_operator_fusion,
    )

    graph = IRGraph(name="test_multi_fusion")
    graph.nodes["a"] = IRNode(id="a", op_type="Input")
    graph.nodes["b"] = IRNode(id="b", op_type="Input")
    graph.nodes["n1"] = IRNode(id="n1", op_type="Add", inputs=["a", "b"])
    graph.nodes["n2"] = IRNode(id="n2", op_type="Mul", inputs=["n1", "a"])
    graph.nodes["n3"] = IRNode(id="n3", op_type="Sub", inputs=["n2", "b"])
    graph.nodes["n4"] = IRNode(id="n4", op_type="Relu", inputs=["n3"])
    graph.inputs = ["a", "b"]
    graph.outputs = ["n4"]

    # Pre-fuse n3 and n4 manually to make consumer FusedElementwise
    fused_n4 = IRNode(
        id="n4",
        op_type="FusedElementwise",
        inputs=["n2", "b"],
        attributes={"fused_ops": ["Sub", "Relu"], "scalar_expr": "fmaxf(0.0f, (in0 - in1))"},
    )
    graph.nodes["n4"] = fused_n4
    del graph.nodes["n3"]

    modified = fuse_elementwise_clusters(graph)
    assert modified is True
    assert graph.nodes["n4"].op_type == "FusedElementwise"

    # OperatorFusionPass class run
    pass_runner = OperatorFusionPass()
    assert pass_runner.run(graph) is False

    # apply_operator_fusion
    g2 = apply_operator_fusion(graph)
    assert g2 is graph


def test_elementwise_fusion_non_elementwise_and_missing_inputs() -> None:
    """Verify elementwise fusion handles missing producer and non-elementwise producers."""
    graph = IRGraph(name="test_non_elem")
    graph.nodes["conv"] = IRNode(id="conv", op_type="Conv2D", inputs=["missing_input"])
    graph.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=["conv", "completely_missing_producer"])
    graph.outputs = ["relu", "nonexistent_output"]

    # conv is not elementwise, and completely_missing_producer is None for elementwise relu
    modified = fuse_elementwise_clusters(graph)
    assert modified is False
