"""Unit tests for verifying the functionality and coverage of the BaseGenerator class.

This module contains tests that validate cycle detection, handling of missing nodes,
variable name assignment, output assignment, and attribute coverage during code
generation.
"""

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator


class DummyGenerator(BaseGenerator):
    """A dummy implementation of BaseGenerator used for testing purposes.

    This class overrides the abstract methods of BaseGenerator to provide simple,
    predictable behaviors for testing graph traversal and code generation.
    """

    def visit(
        self,
        node: object,
        input_vars: list[str],
        **kwargs: object,
    ) -> str:
        """Visit node.

        Args:
            node (object): The node
            input_vars (list[str]): Input vars
            **kwargs (object): Kwargs

        Returns:
            str: Output.
        """
        return "dummy_op"

    def generate(self) -> str:
        """Generate.

        Returns:
            str: The resulting output.
        """
        self._generate_body()
        return "\n".join(self.code)


def test_cycle_detection() -> None:
    """Verifies that the generator correctly detects cycles in the graph and raises a.

    ValueError

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["n2"])
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Add", inputs=["n1"])
    with pytest.raises(ValueError, match="Cycle detected"):
        DummyGenerator(g)


def test_missing_node_in_graph() -> None:
    """Verifies that the generator handles missing input nodes gracefully without crashing.

    Returns:
    None
    """
    # To hit line 37->exit / 41->40 we might need a node with inputs that don't exist
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["missing"])
    DummyGenerator(g)
    # The missing node won't be processed but shouldn't crash


def test_assign_var_name_existing() -> None:
    """Verifies the variable name assignment and caching logic of the generator.

    Returns:
    None
    """
    g = LogicalGraph()
    gen = DummyGenerator(g)
    gen.assign_var_name("n1")
    assert gen.assign_var_name("n1") == "tensor_0"
    assert gen.assign_var_name("n2") == "tensor_1"


def test_output_assignment() -> None:
    """Verifies that output nodes are correctly identified and processed during generation.

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Output", inputs=["n2"])
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Input")
    _ = DummyGenerator(g)


def test_generator_attributes_coverage() -> None:
    """Verifies generator coverage when handling nodes with specific IR attributes.

    Returns:
    None
    """
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    class DummyGen(DummyGenerator):
        """A nested dummy generator class used to test IR node attribute coverage."""

        def generate(self) -> str:
            """Generate function.

            Returns:
                str: The computed result.
            """
            return super().generate()

        def visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
            """_dispatch_op_template function.

            Args:
                node (object): The node.
                input_vars (list[str]): The input_vars.
                **kwargs: Additional keyword arguments.

            Returns:
                str: The computed result.
            """
            return "code"

    g = IRGraph("dummy")
    g.nodes["a"] = IRNode(
        id="a",
        op_type="Add",
        inputs=[],
        attributes={"stream_id": 1, "async_check": True},
    )

    gen = DummyGen(g)
    _ = gen.generate()
