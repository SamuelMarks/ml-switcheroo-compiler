"""Coverage tests for base generator."""

import pytest
from ml_switcheroo.backends.base_generator import BaseGenerator
from ml_switcheroo_ir import LogicalGraph, LogicalNode


class DummyGenerator(BaseGenerator):
    """Docstring."""

    def _dispatch_op_template(
        self, op_instance: object, *args: object, **kwargs: object
    ) -> str:
        """Docstring."""
        return "dummy_op"

    def generate(self) -> str:
        """Docstring."""
        self._generate_body()
        return "\n".join(self.code)


def test_cycle_detection() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["n2"])
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Add", inputs=["n1"])
    with pytest.raises(ValueError, match="Cycle detected"):
        DummyGenerator(g)


def test_missing_node_in_graph() -> None:
    """Docstring."""
    # To hit line 37->exit / 41->40 we might need a node with inputs that don't exist
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["missing"])
    DummyGenerator(g)
    # The missing node won't be processed but shouldn't crash


def test_assign_var_name_existing() -> None:
    """Docstring."""
    g = LogicalGraph()
    gen = DummyGenerator(g)
    gen.assign_var_name("n1")
    assert gen.assign_var_name("n1") == "tensor_0"
    assert gen.assign_var_name("n2") == "tensor_1"


def test_output_assignment() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Output", inputs=["n2"])
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Input")
    gen = DummyGenerator(g)
    code = gen.generate()
    assert "return input_0" in code
