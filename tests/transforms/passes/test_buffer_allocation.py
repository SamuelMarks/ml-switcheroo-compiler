"""Test buffer allocation."""

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.buffer_allocation import GreedyOffsetAllocator, buffer_allocation_pass


def test_buffer_allocation_edge() -> None:
    graph = IRGraph()
    graph.nodes["in0"] = IRNode(id="in0", op_type="Input", shape_metadata=[10])
    graph.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["in0", "in0"], shape_metadata=[10])
    graph.outputs = ["add"]

    modified = buffer_allocation_pass(graph)
    assert modified

    assert graph.nodes["add"].attributes.get("buffer_offset") is not None
    assert graph.nodes["add"].attributes.get("buffer_id") == 0

    gen_webgpu = WebGPUCodeGenerator(graph)
    res_wgsl = gen_webgpu.generate()
    assert "buf_arena_0" in res_wgsl

    gen_wasm = WasmCodeGenerator(graph)
    res_wasm = gen_wasm.generate()
    assert "buf_arena_0" in res_wasm


def test_buffer_reuse() -> None:
    graph = IRGraph()
    graph.nodes["in0"] = IRNode(id="in0", op_type="Input", shape_metadata=[10])
    graph.nodes["add1"] = IRNode(id="add1", op_type="Add", inputs=["in0", "in0"], shape_metadata=[10])
    graph.nodes["add2"] = IRNode(id="add2", op_type="Add", inputs=["add1", "in0"], shape_metadata=[10])
    graph.nodes["add3"] = IRNode(id="add3", op_type="Add", inputs=["add2", "in0"], shape_metadata=[10])
    graph.outputs = ["add3"]

    modified = buffer_allocation_pass(graph)
    assert modified

    # Check if offsets are reused
    offsets = [graph.nodes[n].attributes.get("buffer_offset") for n in ["add1", "add2", "add3"]]
    assert len(set(offsets)) < 3  # at least one buffer should be reused since depth is linear!


def test_buffer_allocation_empty() -> None:
    assert buffer_allocation_pass(IRGraph()) is False


def test_buffer_allocation_dynamic_or_none() -> None:
    graph = IRGraph()
    graph.nodes["n1"] = IRNode(id="n1", op_type="Input", shape_metadata=None)

    class DynamicNode(IRNode):
        @property
        def is_dynamic_shape(self):
            return True

    n2 = DynamicNode(id="n2", op_type="Input", shape_metadata=[10])
    graph.nodes["n2"] = n2
    graph.outputs = ["n1", "n2"]
    buffer_allocation_pass(graph)


def test_buffer_allocation_branch_coverage() -> None:
    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", shape_metadata=[10])
    # out_id not in last_use
    n1.attributes["buffer_offset"] = 0  # To hit node.attributes.get("buffer_offset") != assigned_offset being False
    graph.nodes["n1"] = n1
    graph.outputs = ["missing_out"]
    buffer_allocation_pass(graph)

    # Try reuse when size is too small
    graph2 = IRGraph()
    graph2.nodes["in0"] = IRNode(id="in0", op_type="Input", shape_metadata=[5])
    graph2.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["in0"], shape_metadata=[10])
    graph2.outputs = ["add"]
    buffer_allocation_pass(graph2)


def test_buffer_allocation_not_safe() -> None:
    graph = IRGraph()
    graph.nodes["in0"] = IRNode(id="in0", op_type="Input", shape_metadata=[10])
    # Conv is not in IN_PLACE_SAFE_OPS
    graph.nodes["conv"] = IRNode(id="conv", op_type="Conv", inputs=["in0"], shape_metadata=[10])
    graph.outputs = ["conv"]
    buffer_allocation_pass(graph)


def test_buffer_allocation_input_not_in_graph() -> None:
    graph = IRGraph()
    # Missing input node 'in0'
    graph.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["in0"], shape_metadata=[10])
    graph.outputs = ["add"]
    buffer_allocation_pass(graph)


def test_greedy_allocator() -> None:
    allocator = GreedyOffsetAllocator()
    allocator.allocate(10, 0, 5)
    allocator.allocate(10, 0, 5)  # Should go to offset 10
    allocator.allocate(10, 2, 8)  # Should go to offset 20
    # Now advance current_time to 6, so the first two expire
    off = allocator.allocate(10, 6, 10)
    assert off == 0  # Reuse first block
    # Ensure active_allocations has >1 elements to sort
    allocator.allocate_at(50, 10, 20)
    allocator.allocate_at(30, 10, 20)
    allocator.allocate(10, 6, 10)  # will sort the allocations
