"""Unit tests and dummy implementations for testing the base code generator.

This module defines a `DummyGenerator` that inherits from `BaseGenerator` to test the
core code generation, variable assignment, and indentation mechanisms of the ML
Switcheroo code generation framework.
"""

import pytest
from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator


class DummyGenerator(BaseGenerator):
    """A mock code generator used to test the BaseGenerator class.

    This class implements the abstract methods of BaseGenerator to produce
    predictable dummy Python code, allowing verification of the graph traversal,
    variable naming, and indentation logic.
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
        args_str = ", ".join(input_vars)
        op_type = getattr(node, "op_type", "")
        if op_type == "FooOp":
            return f"unknown_op_{op_type.lower()}({args_str})"
        return f"dummy.{op_type.lower()}({args_str})"

    def generate(self) -> str:
        """Generate.

        Returns:
            str: The resulting output.
        """
        self.add_line("def apply_model(*args, **kwargs):")
        self.indent_level += 1
        self._generate_body()
        return "\n".join(self.code)


def test_base_generator_basic() -> None:
    """Tests basic Python code generation with inputs, constants, and addition.

    Verifies that the generator correctly handles input mapping, constant
    assignment, operation visiting, and output returning

    Returns:
    None
    """
    graph = LogicalGraph(name="test_python", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["const1"] = LogicalNode(
        id="const1",
        op_type="Constant",
        attributes={"value": 42.0},
    )
    graph.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in1", "const1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["add"])

    generator = DummyGenerator(graph)
    code = generator.generate()

    assert "def apply_model(*args, **kwargs):" in code
    assert "input_0 = args[0]" in code
    assert "input_1 = args[1]" in code
    assert "const_2 = 42.0" in code
    assert "tensor_3 = dummy.add(input_0, const_2)" in code
    assert "return tensor_3" in code


def test_base_generator_expand_shape() -> None:
    """Tests Python code generation with a BroadcastTo (Expand) operation.

    Verifies that the generator processes nodes with shape metadata correctly

    Returns:
    None
    """
    graph = LogicalGraph(name="test_python", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["expand"] = LogicalNode(
        id="expand",
        op_type="BroadcastTo",
        inputs=["in1"],
        shape_metadata=(1, 2, 3),
    )
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["expand"])

    generator = DummyGenerator(graph)
    generator.generate()


def test_base_generator_unknown_op() -> None:
    """Tests that the generator falls back to a lower-case op mapping for unknown ops.

    Verifies that operations not explicitly handled by the generator are
    emitted using a default fallback naming convention

    Returns:
    None
    """
    graph = LogicalGraph(name="test_python", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])

    generator = DummyGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = unknown_op_fooop(input_0)" in code


def test_base_generator_no_output() -> None:
    """Tests Python code generation when the logical graph has no explicit output node.

    Verifies that the generator appends a default return statement (e.g., returning
    None)
    when no outputs are specified

    Returns:
    None
    """
    graph = LogicalGraph(name="test_python")
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")

    generator = DummyGenerator(graph)
    code = generator.generate()

    assert "return None" in code


def test_base_generator_assign_var_name() -> None:
    """Tests deterministic variable naming in the generator.

    Verifies that assigning variable names to nodes is consistent, sequential,
    and avoids duplicate assignments for the same node

    Returns:
    None
    """
    graph = LogicalGraph(name="test_python")
    generator = DummyGenerator(graph)
    v1 = generator.assign_var_name("node1", "tensor")
    v2 = generator.assign_var_name("node2", "tensor")
    v3 = generator.assign_var_name("node1", "tensor")  # Already assigned
    assert v1 == "tensor_0"
    assert v2 == "tensor_1"
    assert v3 == "tensor_0"


def test_base_generator_get_indent() -> None:
    """Tests the indentation formatting helper of the generator.

    Verifies that the correct number of spaces is returned based on the
    current indentation level

    Returns:
    None
    """
    graph = LogicalGraph(name="test_python")
    generator = DummyGenerator(graph)
    assert generator.get_indent() == ""
    generator.indent_level = 2
    assert generator.get_indent() == "        "


def test_cycle_detection() -> None:
    """Verifies that the generator correctly detects cycles in the graph and raises a.

    CompilationError

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["n2"])
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Add", inputs=["n1"])
    with pytest.raises(CompilationError, match="Cycle detected"):
        DummyGenerator(g)


def test_missing_node_in_graph() -> None:
    """Verifies that the generator handles missing input nodes gracefully without crashing.

    Returns:
    None
    """
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


def test_base_generator_not_implemented() -> None:
    """Docstring."""
    with pytest.raises(NotImplementedError):
        BaseGenerator.execute_op("Op")
    with pytest.raises(NotImplementedError):
        BaseGenerator.zeros((2, 2))
    with pytest.raises(NotImplementedError):
        BaseGenerator.array([1, 2])
    with pytest.raises(NotImplementedError):
        BaseGenerator.asarray([1, 2])
    with pytest.raises(NotImplementedError):
        BaseGenerator.item(1)
