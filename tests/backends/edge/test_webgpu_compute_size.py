"""Extra tests for webgpu."""

from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_webgpu_compute_total_size():
    """Test _compute_total_size."""
    graph = IRGraph()
    gen = WebGPUCodeGenerator(graph)

    # Empty output ids
    assert gen._compute_total_size([]) == 1

    # Missing out node
    assert gen._compute_total_size(["missing_id"]) == 1

    # Found node
    n = LogicalNode(id="found_id", op_type="Add")
    n.shape_metadata = [2, 3]
    gen.sorted_nodes = [n]
    assert gen._compute_total_size(["found_id"]) == 6
