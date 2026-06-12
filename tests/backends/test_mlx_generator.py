"""Unit tests for the MLX code generator backend.

This module contains tests that verify the compilation of logical graphs into MLX-
compatible Python code, covering basic operations, shape broadcasting, fallback behavior
for unknown operations, and edge cases like empty outputs.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo.backends.mlx import MLXCodeGenerator


def test_mlx_generator_basic() -> None:
    """Verifies basic MLX code generation from a simple logical graph.

    This test constructs a logical graph containing input nodes, a constant node,
    an addition node, and an output node. It asserts that the generated MLX code
    correctly imports MLX, defines a model class, initializes constants,
    performs the addition, and returns the expected output

    Returns:
    None
    """
    graph = LogicalGraph(name="test_mlx", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["const1"] = LogicalNode(
        id="const1",
        op_type="Constant",
        attributes={"value": 42.0},
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
    """Verifies MLX code generation for the BroadcastTo (Expand) operation.

    This test constructs a logical graph with an input node and a BroadcastTo
    node specifying a target shape. It asserts that the generated MLX code
    correctly uses `mx.broadcast_to` with the specified shape metadata

    Returns:
    None
    """
    graph = LogicalGraph(name="test_mlx", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["expand"] = LogicalNode(
        id="expand",
        op_type="BroadcastTo",
        inputs=["in1"],
        shape_metadata=(1, 2, 3),
    )
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["expand"])

    generator = MLXCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = mx.broadcast_to(input_0, (1, 2, 3))" in code


def test_mlx_generator_unknown_op() -> None:
    """Verifies that the MLX generator falls back to a lowercased operation name for.

    unknown ops

    This test constructs a logical graph containing an unrecognized operation type
    ("FooOp"). It asserts that the generator falls back to generating a call to
    `mx.fooop` by lowercasing the operation type

    Returns:
    None
    """
    graph = LogicalGraph(name="test_mlx", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])

    generator = MLXCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = mx.fooop(input_0)" in code


def test_mlx_generator_no_output() -> None:
    """Verifies MLX code generation when the logical graph has no explicit output nodes.

    This test constructs a logical graph with only an input node and no defined
    outputs. It asserts that the generated MLX code returns `None` in its call
    method

    Returns:
    None
    """
    graph = LogicalGraph(name="test_mlx")
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")

    generator = MLXCodeGenerator(graph)
    code = generator.generate()

    assert "return None" in code
