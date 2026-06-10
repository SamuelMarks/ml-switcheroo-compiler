"""Tests for MLX target code generation."""

from ml_switcheroo.backends.mlx import MLXCodeGenerator
from ml_switcheroo_ir import LogicalGraph, LogicalNode


def test_mlx_generator_basic() -> None:
    """Test basic MLX code generation."""
    graph = LogicalGraph(name="test_mlx", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["const1"] = LogicalNode(
        id="const1", op_type="Constant", attributes={"value": 42.0}
    )
    graph.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in1", "const1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["add"])

    generator = MLXCodeGenerator(graph)
    code = generator.generate()

    assert "import mlx.core as mx" in code
    assert "class CompiledModel(nn.Module):" in code
    assert "def __init__(self):" in code
    assert "def __call__(self, *args, **kwargs):" in code
    assert "input_0 = args[0]" in code
    assert "input_1 = args[1]" in code
    assert "const_2 = mx.array(42.0)" in code
    assert "tensor_3 = mx.add(input_0, const_2)" in code
    assert "return tensor_3" in code


def test_mlx_generator_expand_shape() -> None:
    """Test MLX code generation with Expand op."""
    graph = LogicalGraph(name="test_mlx", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["expand"] = LogicalNode(
        id="expand", op_type="Expand", inputs=["in1"], shape_metadata=(1, 2, 3)
    )
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["expand"])

    generator = MLXCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = mx.broadcast_to(input_0, (1, 2, 3))" in code


def test_mlx_generator_unknown_op() -> None:
    """Test MLX code generation falls back to lower-case op mapping."""
    graph = LogicalGraph(name="test_mlx", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])

    generator = MLXCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = mx.fooop(input_0)" in code


def test_mlx_generator_no_output() -> None:
    """Test MLX code generation without explicit output node."""
    graph = LogicalGraph(name="test_mlx")
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")

    generator = MLXCodeGenerator(graph)
    code = generator.generate()

    assert "return None" in code
