"""Unit tests for the PyTorch code generator backend.

This module contains tests that verify the correctness of the PyTorch code generation
from a logical graph representation, including handling of inputs, constants, broadcast
operations, unknown operations, and empty outputs.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo.backends.pytorch import PyTorchCodeGenerator


def test_pytorch_generator_basic() -> None:
    """Verifies basic PyTorch code generation with inputs, constants, and operations.

    This test constructs a logical graph with two inputs, a constant value, an
    addition
    operation, and an output. It then asserts that the generated PyTorch code
    correctly
    defines a `nn.Module` subclass, registers the constant as a parameter, and
    implements
    the forward pass with the expected PyTorch operations

    Returns:
    None
    """
    graph = LogicalGraph(name="test_pt", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["const1"] = LogicalNode(
        id="const1",
        op_type="Constant",
        attributes={"value": 42.0},
    )
    graph.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in1", "const1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["add"])

    generator = PyTorchCodeGenerator(graph)
    code = generator.generate()

    assert "import torch" in code
    assert "class CompiledModel(nn.Module):" in code
    assert "def __init__(self):" in code
    assert (
        "self.register_parameter('const_0', nn.Parameter(torch.tensor(42.0)))" in code
    )
    assert "def forward(self, *args, **kwargs):" in code
    assert "input_1 = args[0]" in code
    assert "input_2 = args[1]" in code
    assert "const_0 = self.const_0" in code
    assert "tensor_3 = torch.add(input_1, const_0)" in code
    assert "return tensor_3" in code


def test_pytorch_generator_no_params() -> None:
    """Verifies PyTorch code generation when the graph contains no constant parameters.

    This test constructs a logical graph with only inputs and an addition operation,
    ensuring that the generated `__init__` method of the PyTorch module is empty
    (i.e., only calls `super().__init__()` and contains `pass`)

    Returns:
    None
    """
    graph = LogicalGraph(name="test_pt", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in1", "in2"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["add"])

    generator = PyTorchCodeGenerator(graph)
    code = generator.generate()

    assert "def __init__(self):\n        super().__init__()\n        pass" in code


def test_pytorch_generator_expand_shape() -> None:
    """Verifies PyTorch code generation for the BroadcastTo (expand) operation.

    This test constructs a logical graph containing a `BroadcastTo` node with shape
    metadata
    It asserts that the generated PyTorch code correctly translates this operation
    to the
    `.expand()` method call on the input tensor with the specified shape

    Returns:
    None
    """
    graph = LogicalGraph(name="test_pt", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["expand"] = LogicalNode(
        id="expand",
        op_type="BroadcastTo",
        inputs=["in1"],
        shape_metadata=(1, 2, 3),
    )
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["expand"])

    generator = PyTorchCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = input_0.expand((1, 2, 3))" in code


def test_pytorch_generator_unknown_op() -> None:
    """Verifies that the PyTorch generator falls back to a lowercase mapping for unknown.

    operations

    This test constructs a logical graph with an unrecognized operation type
    (`FooOp`)
    It asserts that the generator falls back to generating a call to `torch.fooop`
    by converting the operation type to lowercase

    Returns:
    None
    """
    graph = LogicalGraph(name="test_pt", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])

    generator = PyTorchCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = torch.fooop(input_0)" in code


def test_pytorch_generator_no_output() -> None:
    """Verifies PyTorch code generation when the logical graph has no explicit output.

    nodes

    This test constructs a logical graph with only an input node and no output
    nodes
    It asserts that the generated PyTorch code's forward method returns `None`

    Returns:
    None
    """
    graph = LogicalGraph(name="test_pt")
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")

    generator = PyTorchCodeGenerator(graph)
    code = generator.generate()

    assert "return None" in code
