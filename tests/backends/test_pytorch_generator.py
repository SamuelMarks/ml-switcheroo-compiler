"""Tests for PyTorch target code generation."""

from ml_switcheroo.backends.pytorch import PyTorchCodeGenerator
from ml_switcheroo_ir import LogicalGraph, LogicalNode


def test_pytorch_generator_basic() -> None:
    """Test basic PyTorch code generation."""
    graph = LogicalGraph(name="test_pt", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["const1"] = LogicalNode(
        id="const1", op_type="Constant", attributes={"value": 42.0}
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
    """Test PyTorch code generation without constants."""
    graph = LogicalGraph(name="test_pt", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in1", "in2"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["add"])

    generator = PyTorchCodeGenerator(graph)
    code = generator.generate()

    assert "def __init__(self):\n        super().__init__()\n        pass" in code


def test_pytorch_generator_expand_shape() -> None:
    """Test PyTorch code generation with Expand op."""
    graph = LogicalGraph(name="test_pt", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["expand"] = LogicalNode(
        id="expand", op_type="BroadcastTo", inputs=["in1"], shape_metadata=(1, 2, 3)
    )
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["expand"])

    generator = PyTorchCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = input_0.expand((1, 2, 3))" in code


def test_pytorch_generator_unknown_op() -> None:
    """Test PyTorch code generation falls back to lower-case op mapping."""
    graph = LogicalGraph(name="test_pt", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])

    generator = PyTorchCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = unknown_op_fooop(input_0)" in code


def test_pytorch_generator_no_output() -> None:
    """Test PyTorch code generation without explicit output node."""
    graph = LogicalGraph(name="test_pt")
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")

    generator = PyTorchCodeGenerator(graph)
    code = generator.generate()

    assert "return None" in code
