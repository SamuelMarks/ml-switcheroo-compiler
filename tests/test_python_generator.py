"""Tests for the native Python target code generation."""

from ml_switcheroo.backends.python_generator import PythonCodeGenerator
from ml_switcheroo_ir import LogicalGraph, LogicalNode


def test_python_generator_basic() -> None:
    """Test basic Python code generation."""
    graph = LogicalGraph(name="test_python", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["const1"] = LogicalNode(
        id="const1", op_type="Constant", attributes={"value": 42.0}
    )
    graph.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in1", "const1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["add"])

    generator = PythonCodeGenerator(graph)
    code = generator.generate()

    assert "import ml_switcheroo.ops as ops" in code
    assert "def apply_model(*args, **kwargs):" in code
    assert "input_0 = args[0]" in code
    assert "input_1 = args[1]" in code
    assert "const_2 = 42.0" in code
    assert "tensor_3 = ops.add(input_0, const_2)" in code
    assert "return tensor_3" in code


def test_python_generator_expand_shape() -> None:
    """Test Python code generation with Expand op."""
    graph = LogicalGraph(name="test_python", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["expand"] = LogicalNode(
        id="expand", op_type="Expand", inputs=["in1"], shape_metadata=(1, 2, 3)
    )
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["expand"])

    generator = PythonCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = ops.expand(input_0, shape=(1, 2, 3))" in code


def test_python_generator_unknown_op() -> None:
    """Test Python code generation falls back to lower-case op mapping."""
    graph = LogicalGraph(name="test_python", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])

    generator = PythonCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = ops.fooop(input_0)" in code


def test_python_generator_no_output() -> None:
    """Test Python code generation without explicit output node."""
    graph = LogicalGraph(name="test_python")
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")

    generator = PythonCodeGenerator(graph)
    code = generator.generate()

    assert "return None" in code


def test_python_generator_assign_var_name() -> None:
    """Test deterministic variable naming."""
    graph = LogicalGraph(name="test_python")
    generator = PythonCodeGenerator(graph)
    v1 = generator.assign_var_name("node1", "tensor")
    v2 = generator.assign_var_name("node2", "tensor")
    v3 = generator.assign_var_name("node1", "tensor")  # Already assigned
    assert v1 == "tensor_0"
    assert v2 == "tensor_1"
    assert v3 == "tensor_0"


def test_python_generator_get_indent() -> None:
    """Test indentation formatting."""
    graph = LogicalGraph(name="test_python")
    generator = PythonCodeGenerator(graph)
    assert generator.get_indent() == ""
    generator.indent_level = 2
    assert generator.get_indent() == "        "
