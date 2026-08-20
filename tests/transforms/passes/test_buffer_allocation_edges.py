from ml_switcheroo_compiler.ir.core import LogicalNode
from ml_switcheroo_compiler.transforms.passes.buffer_allocation import GreedyOffsetAllocator, _get_node_byte_size


def test_get_node_byte_size_string():
    n = LogicalNode(id="n1", op_type="Input", shape_metadata=["B", 2, 2])
    res = _get_node_byte_size(n)
    assert res == "B * 2 * 2 * 4"


def test_allocator_dynamic():
    alloc = GreedyOffsetAllocator()
    res = alloc.allocate_dynamic("B * 10", 0, 5, "var1")
    assert res == "offset_var1"
    assert len(alloc.dynamic_blocks) == 1
