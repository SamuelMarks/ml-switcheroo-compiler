"""Module test pytorch."""

from typing import Any, Optional

from ml_switcheroo_compiler.backends.pytorch import PyTorchCodeGenerator
from ml_switcheroo_compiler.ir.core import LogicalGraph


class MockNode:
    """Mock Node."""

    def __init__(
        self,
        id: str,
        op_type: str,
        inputs: list[str],
        attributes: dict[str, Any],
        shape_metadata: Optional[tuple[int, ...]],
    ) -> None:
        """Init."""
        self.id = id
        self.op_type = op_type
        self.inputs = inputs
        self.attributes = attributes
        self.shape_metadata = shape_metadata


def test_pytorch_generator_coverage() -> None:
    """Test pytorch generator coverage."""
    gen = PyTorchCodeGenerator(LogicalGraph("foo"))

    class DummyNode:
        op_type = "Sum"

    res = gen.visit(DummyNode(), ["a"], unrelated="hi")
    assert res == "torch.sum(a, keepdim=False)"

    # Test kwargs placeholder replacement (line 48)
    class ReshapeNode:
        op_type = "Reshape"

    res2 = gen.visit(ReshapeNode(), ["a"], shape="(2, 2)")
    assert res2 == "torch.reshape(a, (2, 2))"

    # Test generic fallback with axis and keepdims (lines 62, 64)
    class ReluNode:
        op_type = "Relu"

    res3 = gen.visit(ReluNode(), ["a"], axis=1, keepdims=True)
    assert res3 == "torch.relu(a, dim=1, keepdim=True)"


def test_pytorch_generator_basic() -> None:
    """Test basic functionality."""
    gen = PyTorchCodeGenerator(LogicalGraph("foo"))
    assert gen is not None


def test_pytorch_generator_generate() -> None:
    """Test full code generation."""
    graph = LogicalGraph("test")
    # Empty graph
    gen1 = PyTorchCodeGenerator(graph)
    code1 = gen1.generate()
    assert "class CompiledModel(nn.Module):" in code1
    assert "pass" in code1

    # Graph with constants
    graph2 = LogicalGraph("test2")
    n1 = MockNode("n1", "Constant", [], {"value": 42.0}, None)
    n2 = MockNode("n2", "Relu", ["n1"], {}, None)
    graph2.nodes = {"n1": n1, "n2": n2}

    gen2 = PyTorchCodeGenerator(graph2)
    # mock emit_constant
    gen2.emit_constant = lambda node: "42.0"
    code2 = gen2.generate()
    assert "self.register_parameter" in code2
    assert "pass" not in code2
