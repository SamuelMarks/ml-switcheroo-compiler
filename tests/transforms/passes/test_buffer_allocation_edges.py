# ruff: noqa: E501
"""Edge case and exhaustive unit tests for buffer allocation and graph coloring allocator."""

from unittest.mock import patch

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, LogicalNode
from ml_switcheroo_compiler.transforms.passes.buffer_allocation import (
    BufferAllocationPass,
    GreedyOffsetAllocator,
    InterferenceGraph,
    InterferenceGraphColoringAllocator,
    _get_node_byte_size,
)


def test_get_node_byte_size_string() -> None:
    """Test _get_node_byte_size with symbolic dimensions."""
    n = LogicalNode(id="n1", op_type="Input", shape_metadata=["B", 2, 2])
    res = _get_node_byte_size(n)
    assert res == "B * 2 * 2 * 4"


def test_allocator_dynamic() -> None:
    """Test dynamic allocation block creation in GreedyOffsetAllocator."""
    alloc = GreedyOffsetAllocator()
    res = alloc.allocate_dynamic("B * 10", 0, 5, "var1")
    assert res == "offset_var1"
    assert len(alloc.dynamic_blocks) == 1


def test_interference_graph_readd_node() -> None:
    """Test adding the same node twice to cover existing adj branch."""
    ig = InterferenceGraph()
    ig.add_node("n1", size=128, birth=0, death=2)
    ig.add_node("n1", size=128, birth=0, death=2)
    assert "n1" in ig.nodes
    assert len(ig.adj["n1"]) == 0


def test_coloring_allocator_exhaustive() -> None:
    """Test InterferenceGraphColoringAllocator methods and peak memory reduction."""
    g = IRGraph(name="test_coloring")
    n1 = IRNode(id="n1", op_type="Input", shape_metadata=(16, 16))
    n1.attributes["dtype"] = "float32"
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"], shape_metadata=(16, 16))
    n2.attributes["dtype"] = "float32"
    n3 = IRNode(id="n3", op_type="Relu", inputs=["n2"], shape_metadata=(16, 16))
    n3.attributes["dtype"] = "float32"
    g.nodes = {"n1": n1, "n2": n2, "n3": n3}
    g.outputs = ["n3"]

    allocator = InterferenceGraphColoringAllocator(alignment=256)
    allocs = allocator.allocate_colored_buffers(g)
    assert "n1" in allocs
    assert "n2" in allocs
    assert "n3" in allocs
    assert "color" in allocs["n1"]
    assert "offset" in allocs["n1"]
    assert "size" in allocs["n1"]

    # Test with pre-sorted nodes
    sorted_nodes = [n1, n2, n3]
    allocs_sorted = allocator.allocate_colored_buffers(g, sorted_nodes=sorted_nodes)
    assert len(allocs_sorted) == 3

    # Peak reduction test
    reduction = allocator.compute_peak_memory_reduction(g)
    assert "naive_peak_bytes" in reduction
    assert "colored_peak_bytes" in reduction
    assert "reduction_pct" in reduction

    # Peak reduction with empty graph (naive_peak == 0 branch)
    empty_g = IRGraph(name="empty")
    reduction_empty = allocator.compute_peak_memory_reduction(empty_g)
    assert reduction_empty["reduction_pct"] == 0.0


def test_coloring_allocator_with_symbolic_sizes() -> None:
    """Test InterferenceGraphColoringAllocator with non-integer node sizes."""
    g = IRGraph(name="test_symbolic")
    n1 = IRNode(id="n1", op_type="Input", inputs=["external_input_not_in_nodes"], shape_metadata=["B", 16])
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"], shape_metadata=["B", 16])
    g.nodes = {"n1": n1, "n2": n2}
    g.outputs = ["n2", "non_existent_output_id"]

    allocator = InterferenceGraphColoringAllocator(alignment=256)
    allocs = allocator.allocate_colored_buffers(g)
    assert len(allocs) == 2


def test_buffer_allocation_pass_colored_wrappers() -> None:
    """Test BufferAllocationPass allocate_colored and compute_peak_memory_reduction wrappers."""
    pass_obj = BufferAllocationPass(alignment=256)
    g = IRGraph(name="test_wrappers")
    n1 = IRNode(id="n1", op_type="Input", shape_metadata=(8, 8))
    n1.attributes["dtype"] = "float32"
    g.nodes = {"n1": n1}
    g.outputs = ["n1"]

    allocs = pass_obj.allocate_colored(g, alignment=256)
    assert "n1" in allocs

    reduction = pass_obj.compute_peak_memory_reduction(g, alignment=256)
    assert "naive_peak_bytes" in reduction


def test_buffer_allocation_run_branches() -> None:
    """Test BufferAllocationPass.run branching for skip_coloring, existing attributes, and exceptions."""
    pass_obj = BufferAllocationPass(alignment=256)

    # 1. Branch where skip_coloring is True and graph.attributes is pre-populated
    g = IRGraph(name="test_skip")
    n1 = IRNode(id="n1", op_type="Input", shape_metadata=(8, 8))
    n1.attributes["dtype"] = "float32"
    g.nodes = {"n1": n1}
    g.attributes = {"skip_coloring": True}
    res = pass_obj.run(g)
    assert res is True
    assert "buffer_color" not in n1.attributes

    # 2. Branch where graph.attributes is initially None
    g2 = IRGraph(name="test_none_attrs")
    n2 = IRNode(id="n2", op_type="Input", shape_metadata=(8, 8))
    n2.attributes["dtype"] = "float32"
    g2.nodes = {"n2": n2}
    g2.attributes = None  # type: ignore[assignment]
    res2 = pass_obj.run(g2)
    assert res2 is True
    assert "buffer_color" in n2.attributes

    # 3. Branch where coloring raises an exception (handled by except Exception: pass)
    g3 = IRGraph(name="test_coloring_exception")
    n3 = IRNode(id="n3", op_type="Input", shape_metadata=(8, 8))
    n3.attributes["dtype"] = "float32"
    g3.nodes = {"n3": n3}
    with patch.object(InterferenceGraph, "color_registers", side_effect=RuntimeError("Forced coloring error")):
        res3 = pass_obj.run(g3)
        assert res3 is True
        assert "buffer_color" not in n3.attributes

    # 4. Branch where a node in colors is missing from graph.nodes (nid in graph.nodes is False)
    g4 = IRGraph(name="test_missing_nid")
    n4 = IRNode(id="n4", op_type="Input", shape_metadata=(8, 8))
    n4.attributes["dtype"] = "float32"
    phantom = IRNode(id="phantom", op_type="Input", shape_metadata=(8, 8))
    phantom.attributes["dtype"] = "float32"
    g4.nodes = {"n4": n4}  # phantom is not in g4.nodes
    with patch("ml_switcheroo_compiler.transforms.pass_manager.DAGTopologicalSorter.sort", return_value=[n4, phantom]):
        res4 = pass_obj.run(g4)
        assert res4 is True
        assert "buffer_color" in n4.attributes


def test_allocate_arena_symbolic_1d_shape() -> None:
    """Test allocate_arena with symbolic node size and 1D shape covering branch 412->418."""
    pass_obj = BufferAllocationPass(alignment=256)
    g = IRGraph(name="test_symbolic_1d")
    n1 = IRNode(id="n1", op_type="Input", shape_metadata=["B"])
    n1.attributes["dtype"] = "float32"
    g.nodes = {"n1": n1}
    g.outputs = ["n1"]

    with patch("ml_switcheroo_compiler.transforms.passes.buffer_allocation._get_node_byte_size", return_value="sym_size"):
        offsets = pass_obj.calculate_arena_offsets(g)
        assert offsets == {}
        assert n1.attributes["stride_multiplier"] == 1
        assert n1.attributes["dynamic_batch"] is True
