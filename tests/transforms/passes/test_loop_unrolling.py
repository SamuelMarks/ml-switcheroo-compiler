"""Test loop unrolling pass."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.loop_unrolling import loop_unrolling_pass


def test_loop_unrolling_pass():
    """Test unrolling loops with statically analyzable bound."""
    # Create body graph
    body_graph = IRGraph()
    body_graph.inputs = ["in_val"]

    n1 = IRNode(id="n1", op_type="Add", inputs=["in_val", "const_1"])
    body_graph.nodes["n1"] = n1
    body_graph.outputs = ["n1"]

    # Create main graph
    graph = IRGraph()
    loop_node = IRNode(id="loop", op_type="WhileLoop", inputs=["init_val"], attributes={"max_iterations": 2, "body": body_graph})
    graph.nodes["loop"] = loop_node

    # Run unrolling pass
    optimized = loop_unrolling_pass(graph)

    # The loop should be unrolled twice (n1 and n1_iter0, etc.)
    # We expect 3 nodes: Identity, loop_iter0_n1, loop_iter1_n1
    assert "loop" in optimized.nodes
    assert optimized.nodes["loop"].op_type == "Identity"
    assert optimized.nodes["loop"].inputs == ["loop_iter1_n1"]

    assert "loop_iter0_n1" in optimized.nodes
    assert optimized.nodes["loop_iter0_n1"].inputs == ["init_val", "const_1"]

    assert "loop_iter1_n1" in optimized.nodes
    assert optimized.nodes["loop_iter1_n1"].inputs == ["loop_iter0_n1", "const_1"]


def test_loop_unrolling_no_unroll():
    """Test when loop should not be unrolled."""
    graph = IRGraph()
    loop_node = IRNode(
        id="loop",
        op_type="WhileLoop",
        inputs=["init_val"],
        attributes={"max_iterations": 100},  # Exceeds default limit (10)
    )
    graph.nodes["loop"] = loop_node

    optimized = loop_unrolling_pass(graph)
    assert "loop" in optimized.nodes
    assert optimized.nodes["loop"].op_type == "WhileLoop"


def test_detect_static_bound():
    """Test heuristics boundary detection."""
    from ml_switcheroo_compiler.transforms.passes.loop_unrolling import detect_static_bound

    node = IRNode(id="n", op_type="WhileLoop")
    # Empty
    assert detect_static_bound(node, []) is None

    # Heuristics
    heuristics = [{"op_type": "WhileLoop", "max_iterations": 5}]
    assert detect_static_bound(node, heuristics) == 5


def test_get_initial_constants():
    """Test stub for coverage."""
    from ml_switcheroo_compiler.transforms.passes.loop_unrolling import _get_initial_constants

    assert _get_initial_constants() == []


def test_loop_unrolling_extra_coverage():
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, LogicalNode
    from ml_switcheroo_compiler.transforms.passes.loop_unrolling import _load_config, detect_static_bound, loop_unrolling_pass

    # test yaml missing
    with patch("os.path.exists", return_value=False):
        assert _load_config() == {}

    # detect_static_bound branches
    node = IRNode(id="n1", op_type="Add")
    assert detect_static_bound(node, []) is None

    node = IRNode(id="n1", op_type="WhileLoop")
    heuristics = [{"op_type": "Add"}, {"op_type": "WhileLoop", "other": 1}, {"op_type": "WhileLoop", "max_iterations": "string"}, {"op_type": "WhileLoop", "max_iterations": 5}]
    assert detect_static_bound(node, heuristics) == 5

    # graph loops unrolling branches
    g = IRGraph()
    ln = LogicalNode(id="ln1", op_type="Input")
    g.nodes["ln1"] = ln

    n2 = IRNode(id="n2", op_type="WhileLoop")
    n2.attributes["max_iterations"] = 3
    # No body_graph, should skip
    g.nodes["n2"] = n2

    # Body graph with logical nodes
    n3 = IRNode(id="n3", op_type="WhileLoop", inputs=["ln1"])
    n3.attributes["max_iterations"] = 1

    body = IRGraph()
    body.inputs = ["body_in"]
    body.outputs = ["body_out"]
    bln = LogicalNode(id="body_node", op_type="Add", inputs=["body_in"])
    body.nodes["body_node"] = bln
    n3.attributes["body"] = body

    g.nodes["n3"] = n3

    # Add stream and device missing on IRNode
    n4 = IRNode(id="n4", op_type="WhileLoop", inputs=["ln1"])
    n4.attributes["max_iterations"] = 1

    body2 = IRGraph()
    body2.inputs = ["body_in"]
    body2.outputs = ["body_out"]
    n4_inner = IRNode(id="n4_inner", op_type="Add", inputs=["body_in"])
    # Delete stream and device if they exist
    if hasattr(n4_inner, "stream"):
        del n4_inner.stream
    if hasattr(n4_inner, "device"):
        del n4_inner.device
    body2.nodes["n4_inner"] = n4_inner
    n4.attributes["body"] = body2
    g.nodes["n4"] = n4

    loop_unrolling_pass(g)
