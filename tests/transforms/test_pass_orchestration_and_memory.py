"""Tests for middle-end pass orchestration, operator fusion, and memory arena planning."""

import os

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import PassManager
from ml_switcheroo_compiler.transforms.passes.buffer_allocation import (
    BufferAllocationPass,
)
from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
    NodePattern,
)


def test_memory_layouts_yaml_alignments() -> None:
    """Verify declarative memory alignment schemas in memory_layouts.yaml."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "transforms", "passes", "memory_layouts.yaml")
    assert os.path.exists(yaml_path)
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "memory_alignments" in data
    alignments = data["memory_alignments"]
    assert "avx512" in alignments and alignments["avx512"]["alignment_bytes"] == 64
    assert "avx2" in alignments and alignments["avx2"]["alignment_bytes"] == 32
    assert "gpu_global" in alignments and alignments["gpu_global"]["alignment_bytes"] == 128
    assert "webgpu" in alignments and alignments["webgpu"]["alignment_bytes"] == 256
    assert "wasm_simd" in alignments and alignments["wasm_simd"]["alignment_bytes"] == 16


def test_fusion_patterns_yaml_definitions() -> None:
    """Verify expanded operator fusion patterns in fusion_patterns.yaml."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "transforms", "fusion_patterns.yaml")
    assert os.path.exists(yaml_path)
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    pattern_names = {p["name"] for p in data.get("patterns", [])}
    assert "ScaledDotProductAttention" in pattern_names
    assert "ConvBNActivation" in pattern_names
    assert "LayerNorm" in pattern_names
    assert "MatMulBiasGelu" in pattern_names


def test_subgraph_pattern_matcher_wildcard_and_commutativity() -> None:
    """Verify NodePattern support for wildcards and commuting inputs."""
    # Commutative add pattern: Add(A, B) matching in any order
    p_in0 = NodePattern(op_type="Constant", capture="c0")
    p_in1 = NodePattern(op_type="Input", capture="c1")
    p_add = NodePattern(op_type="Add", inputs=[p_in0, p_in1], commutative=True, capture="root")

    graph = IRGraph()
    n_in = IRNode(id="x", op_type="Input")
    n_const = IRNode(id="c", op_type="Constant")
    # Order reversed: inputs are [Input, Constant]
    n_add = IRNode(id="add_node", op_type="Add", inputs=["x", "c"])
    graph.nodes = {"x": n_in, "c": n_const, "add_node": n_add}

    from ml_switcheroo_compiler.transforms.passes.operator_fusion import match_pattern

    cap_map: dict[str, str | IRNode] = {}
    assert match_pattern(graph, "add_node", p_add, cap_map)
    assert cap_map["c0"].id == "c"
    assert cap_map["c1"].id == "x"

    # Wildcard pattern
    p_wild = NodePattern(op_type="*", capture="any_op")
    cap_wild: dict[str, str | IRNode] = {}
    assert match_pattern(graph, "add_node", p_wild, cap_wild)
    assert cap_wild["any_op"].id == "add_node"


def test_dynamic_buffer_allocation_symbolic_batch() -> None:
    """Verify dynamic buffer allocation handles symbolic batch dimension B using stride multipliers."""
    graph = IRGraph()
    n_in = IRNode(id="in_0", op_type="Input", shape_metadata=("B", 64, 64), attributes={"shape": ("B", 64, 64), "dtype": "float32"})
    n_out = IRNode(id="out_0", op_type="Relu", inputs=["in_0"], shape_metadata=("B", 64, 64), attributes={"shape": ("B", 64, 64), "dtype": "float32"})
    graph.nodes = {"in_0": n_in, "out_0": n_out}
    graph.outputs = ["out_0"]

    alloc_pass = BufferAllocationPass(alignment=64)
    alloc_pass.calculate_arena_offsets(graph)

    assert n_in.attributes.get("dynamic_batch") is True
    assert n_in.attributes.get("stride_multiplier") == 64 * 64 * 4

    modified = alloc_pass.run(graph)
    assert modified is True
    assert "buffer_offset_symbolic" in n_in.attributes
    assert "dynamic_memory_schema" in graph.attributes


def test_pass_manager_optimization_levels_and_fixpoint() -> None:
    """Verify PassManager loads O0, O1, O2, O3 optimization levels and runs to convergence."""
    pm_o0 = PassManager()
    pm_o0.load_from_config(opt_level="O0")
    assert len(pm_o0.passes) > 0
    assert "type_promotion_explicitizer" in pm_o0.pass_names

    pm_o2 = PassManager()
    pm_o2.load_from_config(opt_level="O2")
    assert "operator_fusion" in pm_o2.pass_names

    # Run fixpoint convergence test on a small graph
    graph = IRGraph()
    n_in = IRNode(id="x", op_type="Input", shape_metadata=(4, 4), attributes={"shape": (4, 4), "dtype": "float32"})
    n_relu = IRNode(id="r", op_type="Relu", inputs=["x"], shape_metadata=(4, 4), attributes={"shape": (4, 4), "dtype": "float32"})
    graph.nodes = {"x": n_in, "r": n_relu}
    graph.outputs = ["r"]

    pm = PassManager()
    pm.add_pass(lambda g: False, name="identity_pass")
    converged_graph = pm.run_until_converged(graph, max_iterations=5)
    assert converged_graph is not None
    assert "r" in converged_graph.nodes
