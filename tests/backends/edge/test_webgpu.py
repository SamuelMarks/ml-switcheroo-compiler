"""Tests for webgpu backend coverage."""

from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_webgpu_coverage() -> None:
    """Test webgpu generic visit."""
    gen = WebGPUCodeGenerator(IRGraph())
    nodes = [
        LogicalNode(id="n_add", op_type="Add", inputs=["in1", "in2"]),
        LogicalNode(id="n_exp", op_type="Exp", inputs=["in1"]),
        LogicalNode(id="n_min", op_type="Min", inputs=["in1", "in2"]),
        LogicalNode(id="n_neg", op_type="Negative", inputs=["in1"]),
        LogicalNode(id="n_other", op_type="OtherOp", inputs=["in1"]),
        None,
    ]
    for n in nodes:
        gen.generic_visit(n, [])
    gen._generate_js_orchestrator("", [], [], 0, 1)


def test_webgpu_more() -> None:
    """Test webgpu more operations."""
    gen = WebGPUCodeGenerator(IRGraph())
    for op in ["Log", "Sqrt", "Abs", "Max", "Subtract", "Multiply", "TrueDivide", "Div", "SomeOtherOp", "Neg"]:
        n = LogicalNode(id="n_" + op, op_type=op, inputs=["in1", "in2"])
        gen.generic_visit(n, [])


def test_webgpu_map_type() -> None:
    """Test map_type."""
    gen = WebGPUCodeGenerator(IRGraph())
    assert gen._map_type("float32") == "f32"
    assert gen._map_type("float64") == "f32"
    assert gen._map_type("int32") == "i32"
    assert gen._map_type("bool") == "bool"
    assert gen._map_type("unknown") == "f32"


def test_webgpu_shape_and_strides() -> None:
    """Test shape and strides."""
    gen = WebGPUCodeGenerator(IRGraph())

    node = LogicalNode(id="n1", op_type="Add")
    assert gen._get_shape_and_strides(node) == ([], [])

    node.shape_metadata = []
    assert gen._get_shape_and_strides(node) == ([], [])

    node.shape_metadata = 5
    assert gen._get_shape_and_strides(node) == ([5], [1])

    node.shape_metadata = [2, 3, 4]
    assert gen._get_shape_and_strides(node) == ([2, 3, 4], [12, 4, 1])


def test_webgpu_generate() -> None:
    """Test generate method end-to-end."""
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = [2, 3]
    n1.dtype = "float32"

    n2 = LogicalNode(id="in2", op_type="Input")
    n2.shape_metadata = [6]
    n2.dtype = "int32"

    n3 = LogicalNode(id="c1", op_type="Constant", attributes={"value": 42.0})
    n3.shape_metadata = [1]

    n4 = LogicalNode(id="add", op_type="Add", inputs=["in1", "c1"])
    n4.shape_metadata = [2, 3]

    n_no_id = LogicalNode(id="", op_type="Add")

    graph.nodes = {"in1": n1, "in2": n2, "c1": n3, "add": n4, "": n_no_id}
    graph.outputs = ["add", "in2"]

    gen = WebGPUCodeGenerator(graph)

    code = gen.generate()
    assert "fn get_offset_in1" in code
    assert "f32" in code
    assert "i32" in code
    assert "function run" in code


def test_webgpu_generate_empty() -> None:
    """Test generate with empty graph."""
    graph = IRGraph()
    graph.outputs = None
    gen = WebGPUCodeGenerator(graph)
    code = gen.generate()
    assert "function run" in code


def test_webgpu_visit_input() -> None:
    """Test generic_visit on Input node."""
    gen = WebGPUCodeGenerator(IRGraph())

    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = [2, 3]
    assert gen.generic_visit(n1, []) == "in_0[get_offset_in1(idx)]"

    n2 = LogicalNode(id="in2", op_type="Input")
    n2.shape_metadata = [6]
    assert gen.generic_visit(n2, []) == "in_1[idx]"
