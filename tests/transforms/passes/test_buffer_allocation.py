"""Behavioral unit tests for buffer allocation."""

from __future__ import annotations

import typing
from unittest.mock import patch

import pytest
from ml_switcheroo_ir import IRGraph, IRNode, LogicalNode, NodeDict

from ml_switcheroo_compiler.transforms.passes.buffer_allocation import (
    BufferAllocationPass,
    GreedyOffsetAllocator,
    InterferenceGraph,
    InterferenceGraphColoringAllocator,
    _get_node_byte_size,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_buffer_allocation_coverage() -> None:
    """Test function test_buffer_allocation_coverage."""
    from unittest.mock import patch

    from ml_switcheroo_ir import IRGraph, IRNode

    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import buffer_allocation_pass

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Add", inputs=[])
    n2 = IRNode(id="n2", op_type="Sub", inputs=["n1"])  # in place safe
    n3 = IRNode(id="n3", op_type="MatMul", inputs=["n1", "n2"])

    g.nodes = NodeDict({"n1": n1, "n2": n2, "n3": n3})
    g.outputs = ["n1"]

    with patch("ml_switcheroo_compiler.transforms.pass_manager.DAGTopologicalSorter.sort", return_value=[n1, n2, n3]):
        with patch("ml_switcheroo_compiler.transforms.passes.buffer_allocation._get_node_byte_size", return_value=128):
            buffer_allocation_pass(g)

            assert "buffer_offset" in n1.attributes


def test_buffer_allocation_in_place() -> None:
    """Test function test_buffer_allocation_in_place."""
    from unittest.mock import patch

    from ml_switcheroo_ir import IRGraph, IRNode

    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import buffer_allocation_pass

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[])
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"])  # n1 used once

    g.nodes = NodeDict({"n1": n1, "n2": n2})

    with patch("ml_switcheroo_compiler.transforms.pass_manager.DAGTopologicalSorter.sort", return_value=[n1, n2]):
        with patch("ml_switcheroo_compiler.transforms.passes.buffer_allocation._get_node_byte_size", return_value=128):
            buffer_allocation_pass(g)
            assert n1.attributes["buffer_offset"] == n2.attributes["buffer_offset"]


def test_buffer_allocation_dynamic() -> None:
    """Test function test_buffer_allocation_dynamic."""
    from unittest.mock import patch

    from ml_switcheroo_ir import IRGraph, IRNode

    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import buffer_allocation_pass

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[])
    g.nodes = NodeDict({"n1": n1})

    with patch("ml_switcheroo_compiler.transforms.pass_manager.DAGTopologicalSorter.sort", return_value=[n1]):
        with patch(
            "ml_switcheroo_compiler.transforms.passes.buffer_allocation._get_node_byte_size", return_value="size_expr"
        ):
            buffer_allocation_pass(g)
            assert n1.attributes["buffer_size_symbolic"] == "size_expr"
    from unittest.mock import patch

    from ml_switcheroo_ir import IRNode

    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import _get_node_byte_size

    node = IRNode("n1", "Add")

    class MockCostModel:
        """Test class MockCostModel."""

        def get_memory_cost(self, n: BackendArg) -> BackendArg:
            """Test function get_memory_cost.

            Args:
                n: Test parameter `n`.

            Returns:
                BackendArg: Computed output value from `get_memory_cost`.
            """
            return 256

    with patch(
        "ml_switcheroo_compiler.transforms.passes.graph_scheduling.DefaultCostModel", return_value=MockCostModel()
    ):
        assert _get_node_byte_size(node) == 256

    class MockCostModelStr:
        """Test class MockCostModelStr."""

        def get_memory_cost(self, n: BackendArg) -> BackendArg:
            """Test function get_memory_cost.

            Args:
                n: Test parameter `n`.

            Returns:
                BackendArg: Computed output value from `get_memory_cost`.
            """
            return "symbolic"

    with patch(
        "ml_switcheroo_compiler.transforms.passes.graph_scheduling.DefaultCostModel", return_value=MockCostModelStr()
    ):
        assert _get_node_byte_size(node) == "symbolic"

    class MockCostModelFloat:
        """Test class MockCostModelFloat."""

        def get_memory_cost(self, n: BackendArg) -> BackendArg:
            """Test function get_memory_cost.

            Args:
                n: Test parameter `n`.

            Returns:
                BackendArg: Computed output value from `get_memory_cost`.
            """
            return 128.5

    with patch(
        "ml_switcheroo_compiler.transforms.passes.graph_scheduling.DefaultCostModel", return_value=MockCostModelFloat()
    ):
        assert _get_node_byte_size(node) == "128.5"


def test_buffer_allocation_extra_coverage() -> None:
    """Test function test_buffer_allocation_extra_coverage."""
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import GreedyOffsetAllocator, buffer_allocation_pass

    # 1. Empty graph
    g = IRGraph()
    res = buffer_allocation_pass(g)
    assert res is False

    # 2. Allocator hole (break condition)
    alloc = GreedyOffsetAllocator()
    alloc.allocate_static(10, 0, 5)  # offset 0, ends at 10
    alloc.allocate_static(10, 0, 15)  # offset 10, ends at 20
    # First block expires at 5, second expires at 15.
    # Current time = 6. First block will be freed.

    # We try to allocate 5 bytes.
    # The gap before 10 is 10 bytes (0 to 10), which is >= 5, so it should break and use offset 0.
    offset = alloc.allocate_static(5, 6, 10)
    assert offset == 0

    # 3. allocate alias
    alloc.allocate(5, 6, 10)


def test_buffer_allocation_branch_coverage() -> None:
    """Test function test_buffer_allocation_branch_coverage."""
    from unittest.mock import patch

    from ml_switcheroo_ir import IRGraph, IRNode

    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import buffer_allocation_pass

    g = IRGraph()
    # node n1 is an output but not in sorted_nodes
    g.outputs = ["n1", "missing_out"]
    n1 = IRNode(id="n1", op_type="Input", inputs=["missing_inp"])
    # give n1 the same offset to skip the attribute setting
    n1.attributes["buffer_offset"] = 0
    n1.attributes["buffer_size"] = 128

    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"])
    # Not enough size for reuse
    n2.attributes["buffer_offset"] = 128

    n3 = IRNode(id="n3", op_type="Add", inputs=["n2"])

    # Missing node in graph for reuse
    n3.inputs.append("missing_node")

    g.nodes = NodeDict({"n1": n1, "n2": n2, "n3": n3})

    with patch("ml_switcheroo_compiler.transforms.pass_manager.DAGTopologicalSorter.sort", return_value=[n1, n2, n3]):
        with patch(
            "ml_switcheroo_compiler.transforms.passes.buffer_allocation._get_node_byte_size",
            side_effect=[128, 256, 256],
        ):
            buffer_allocation_pass(g)

    # Dynamic branch skip
    g2 = IRGraph()
    n4 = IRNode(id="n4", op_type="Input", inputs=[])
    n4.attributes["buffer_offset_symbolic"] = "offset_n4"
    g2.nodes = NodeDict({"n4": n4})
    with patch("ml_switcheroo_compiler.transforms.pass_manager.DAGTopologicalSorter.sort", return_value=[n4]):
        with patch(
            "ml_switcheroo_compiler.transforms.passes.buffer_allocation._get_node_byte_size", return_value="size_expr"
        ):
            buffer_allocation_pass(g2)
    assert n4.attributes.get("buffer_offset_symbolic") == "offset_n4"


def test_buffer_allocation_pass_class() -> None:
    """Test function test_buffer_allocation_pass_class."""
    from ml_switcheroo_ir import IRGraph, IRNode

    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import BufferAllocationPass

    alloc_pass = BufferAllocationPass(alignment=16)

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", shape_metadata=(16, 16))
    n1.attributes["dtype"] = "float32"
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"], shape_metadata=(16, 16))
    n2.attributes["dtype"] = "float32"
    n3 = IRNode(id="n3", op_type="Relu", inputs=["n2"], shape_metadata=(16, 16))
    n3.attributes["dtype"] = "float32"

    g.nodes = NodeDict({"n1": n1, "n2": n2, "n3": n3})
    g.outputs = ["n3"]

    intervals = alloc_pass.compute_liveness_intervals(g)
    assert intervals["n1"][0] == 0
    assert intervals["n1"][1] == 1  # used by n2 at step 1
    assert intervals["n2"][0] == 1
    assert intervals["n2"][1] == 2  # used by n3 at step 2
    assert intervals["n3"][1] == 3  # graph output, alive at end

    offsets = alloc_pass.calculate_arena_offsets(g, alignment=16)
    assert "n1" in offsets
    assert "n2" in offsets
    assert "n3" in offsets
    # Offsets must be multiples of 16
    for off in offsets.values():
        assert off % 16 == 0

    modified = alloc_pass.run(g)
    assert modified is True
    assert "buffer_offset" in n1.attributes

    # Coverage for lines 194->193, 199->198, and 227
    from unittest.mock import patch

    g_extra = IRGraph()
    n_extra = IRNode(id="n_extra", op_type="Input", inputs=["external_non_node"], shape_metadata=(16, 16))
    g_extra.nodes = NodeDict({"n_extra": n_extra})
    g_extra.outputs = ["n_extra", "missing_out_id"]
    intervals_extra = alloc_pass.compute_liveness_intervals(g_extra)
    assert "n_extra" in intervals_extra

    with patch(
        "ml_switcheroo_compiler.transforms.passes.buffer_allocation._get_node_byte_size", return_value="symbolic_dim"
    ):
        offsets_symbolic = alloc_pass.calculate_arena_offsets(g_extra)
        assert len(offsets_symbolic) == 0


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
    g.nodes = NodeDict({"n1": n1, "n2": n2, "n3": n3})
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
    g.nodes = NodeDict({"n1": n1, "n2": n2})
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
    g.nodes = NodeDict({"n1": n1})
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
    g.nodes = NodeDict({"n1": n1})
    g.attributes = {"skip_coloring": True}
    res = pass_obj.run(g)
    assert res is True
    assert "buffer_color" not in n1.attributes

    from tests.testing_utils import MockGraphStub

    # 2. Branch where graph.attributes is initially None
    g2 = MockGraphStub(name="test_none_attrs")
    n2 = IRNode(id="n2", op_type="Input", shape_metadata=(8, 8))
    n2.attributes["dtype"] = "float32"
    g2.nodes = NodeDict({"n2": n2})
    pytest.MonkeyPatch().setattr(g2, "attributes", None, raising=False)
    res2 = pass_obj.run(g2)
    assert res2 is True
    assert "buffer_color" in n2.attributes

    # 3. Branch where coloring raises an exception (handled by except Exception: pass)
    g3 = IRGraph(name="test_coloring_exception")
    n3 = IRNode(id="n3", op_type="Input", shape_metadata=(8, 8))
    n3.attributes["dtype"] = "float32"
    g3.nodes = NodeDict({"n3": n3})
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
    g4.nodes = NodeDict({"n4": n4})  # phantom is not in g4.nodes
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
    g.nodes = NodeDict({"n1": n1})
    g.outputs = ["n1"]

    with patch(
        "ml_switcheroo_compiler.transforms.passes.buffer_allocation._get_node_byte_size", return_value="sym_size"
    ):
        offsets = pass_obj.calculate_arena_offsets(g)
        assert offsets == {}
        assert n1.attributes["stride_multiplier"] == 1
        assert n1.attributes["dynamic_batch"] is True
