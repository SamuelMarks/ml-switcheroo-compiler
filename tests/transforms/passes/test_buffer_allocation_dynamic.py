from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.buffer_allocation import buffer_allocation_pass


class DynamicNode(IRNode):
    @property
    def is_dynamic_shape(self):
        return True


def test_dynamic_allocation_pass():
    graph = IRGraph()
    # Symbolic string shape "B"
    n1 = DynamicNode(id="n1", op_type="Input", shape_metadata=["B", 64, 64])
    graph.nodes["n1"] = n1
    graph.outputs = ["n1"]

    assert buffer_allocation_pass(graph)

    # Check attributes
    schema = graph.attributes.get("dynamic_memory_schema")
    assert schema is not None
    assert len(schema["dynamic_offsets"]) == 1
    offset_info = schema["dynamic_offsets"][0]
    assert offset_info["symbolic_math"] == "B * 64 * 64 * 4"
    assert offset_info["node_id"] == "n1"

    assert n1.attributes.get("buffer_size_symbolic") == "B * 64 * 64 * 4"
