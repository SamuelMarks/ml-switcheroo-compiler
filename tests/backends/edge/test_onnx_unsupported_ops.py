import pytest

from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_onnx_backend_not_supported_error():
    graph = IRGraph()
    # Mock node with an op_type definitely not in ONNX schema
    node = IRNode("unsupported_op_1", "SomeNonExistentOp", ["in1"], shape_metadata=[2, 2])
    graph.nodes["unsupported_op_1"] = node
    graph.sorted_nodes = [node]
    graph.inputs = ["in1"]
    graph.outputs = ["unsupported_op_1"]

    generator = ONNXCodeGenerator(graph, [])
    # Set schema to empty to trigger error
    generator.schema = {}

    with pytest.raises(BackendNotSupportedError) as exc_info:
        generator.generate()

    assert "Operation 'SomeNonExistentOp' not supported in ONNX schema" in str(exc_info.value)
