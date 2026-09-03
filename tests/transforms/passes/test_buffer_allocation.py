def test_buffer_allocation_coverage():
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import buffer_allocation_pass

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Add", inputs=[])
    n2 = IRNode(id="n2", op_type="Sub", inputs=["n1"])  # in place safe
    n3 = IRNode(id="n3", op_type="MatMul", inputs=["n1", "n2"])

    g.nodes = {"n1": n1, "n2": n2, "n3": n3}
    g.outputs = ["n1"]

    with patch("ml_switcheroo_compiler.transforms.pass_manager.DAGTopologicalSorter.sort", return_value=[n1, n2, n3]):
        with patch("ml_switcheroo_compiler.transforms.passes.buffer_allocation._get_node_byte_size", return_value=128):
            g_opt = buffer_allocation_pass(g)

            assert "buffer_offset" in n1.attributes


def test_buffer_allocation_in_place():
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import buffer_allocation_pass

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[])
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"])  # n1 used once

    g.nodes = {"n1": n1, "n2": n2}

    with patch("ml_switcheroo_compiler.transforms.pass_manager.DAGTopologicalSorter.sort", return_value=[n1, n2]):
        with patch("ml_switcheroo_compiler.transforms.passes.buffer_allocation._get_node_byte_size", return_value=128):
            buffer_allocation_pass(g)
            assert n1.attributes["buffer_offset"] == n2.attributes["buffer_offset"]


def test_buffer_allocation_dynamic():
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import buffer_allocation_pass

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[])
    g.nodes = {"n1": n1}

    with patch("ml_switcheroo_compiler.transforms.pass_manager.DAGTopologicalSorter.sort", return_value=[n1]):
        with patch("ml_switcheroo_compiler.transforms.passes.buffer_allocation._get_node_byte_size", return_value="size_expr"):
            buffer_allocation_pass(g)
            assert n1.attributes["buffer_size_symbolic"] == "size_expr"
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import _get_node_byte_size

    node = IRNode("n1", "Add")

    class MockCostModel:
        def get_memory_cost(self, n):
            return 256

    with patch("ml_switcheroo_compiler.transforms.passes.graph_scheduling.DefaultCostModel", return_value=MockCostModel()):
        assert _get_node_byte_size(node) == 256

    class MockCostModelStr:
        def get_memory_cost(self, n):
            return "symbolic"

    with patch("ml_switcheroo_compiler.transforms.passes.graph_scheduling.DefaultCostModel", return_value=MockCostModelStr()):
        assert _get_node_byte_size(node) == "symbolic"

    class MockCostModelFloat:
        def get_memory_cost(self, n):
            return 128.5

    with patch("ml_switcheroo_compiler.transforms.passes.graph_scheduling.DefaultCostModel", return_value=MockCostModelFloat()):
        assert _get_node_byte_size(node) == "128.5"


def test_buffer_allocation_extra_coverage():
    from ml_switcheroo_compiler.ir.core import IRGraph
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


def test_buffer_allocation_branch_coverage():
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
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

    g.nodes = {"n1": n1, "n2": n2, "n3": n3}

    with patch("ml_switcheroo_compiler.transforms.pass_manager.DAGTopologicalSorter.sort", return_value=[n1, n2, n3]):
        with patch("ml_switcheroo_compiler.transforms.passes.buffer_allocation._get_node_byte_size", side_effect=[128, 256, 256]):
            buffer_allocation_pass(g)

    # Dynamic branch skip
    g2 = IRGraph()
    n4 = IRNode(id="n4", op_type="Input", inputs=[])
    n4.attributes["buffer_offset_symbolic"] = "offset_n4"
    g2.nodes = {"n4": n4}
    with patch("ml_switcheroo_compiler.transforms.pass_manager.DAGTopologicalSorter.sort", return_value=[n4]):
        with patch("ml_switcheroo_compiler.transforms.passes.buffer_allocation._get_node_byte_size", return_value="size_expr"):
            buffer_allocation_pass(g2)
