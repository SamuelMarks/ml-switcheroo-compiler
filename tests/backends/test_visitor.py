"""Test module."""

from __future__ import annotations

import pytest
from ml_switcheroo_ir import AttributeValue, IRNode

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.generator_mixins import GeneratorLifecycleMixin
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from tests.testing_utils import MockNodeStub


class DummyGenerator(GeneratorLifecycleMixin):
    """Test class DummyGenerator."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.sorted_nodes: list[IRNode] = []
        self.var_names: dict[str, str] = {"in1": "var_in1"}
        self.input_idx: int = 0
        self._output_returns: list[str] | None = []
        self.lines: list[str] = []

    def _emit_body_return(self, returns: list[str]) -> None:
        """Test function _emit_body_return.

        Args:
            returns (list[str]): Test parameter `returns`.
        """
        self.lines.append(f"return {returns}")

    def emit_constant(self, node: IRNode) -> str:
        """Test function emit_constant.

        Args:
            node (IRNode): Test parameter `node`.

        Returns:
            str: Computed output value from `emit_constant`.
        """
        del node
        return "42.0"

    def assign_var_name(self, node_id: str, prefix: str = "tensor") -> str:
        """Test function assign_var_name.

        Args:
            node_id (str): Test parameter `node_id`.
            prefix (str): Test parameter `prefix`.

        Returns:
            str: Computed output value from `assign_var_name`.
        """
        del node_id
        return f"{prefix}_1"

    def _emit_constant_assignment(self, var: str, val: str) -> None:
        """Test function _emit_constant_assignment.

        Args:
            var (str): Test parameter `var`.
            val (str): Test parameter `val`.
        """
        self.lines.append(f"{var} = {val}")

    def _emit_input_assignment(self, var: str, node: IRNode, pref: str, idx: int) -> None:
        """Test function _emit_input_assignment.

        Args:
            var (str): Test parameter `var`.
            node (IRNode): Test parameter `node`.
            pref (str): Test parameter `pref`.
            idx (int): Test parameter `idx`.
        """
        del node
        self.lines.append(f"{var} = {pref}[{idx}]")

    def _emit_output_assignment(self, node: IRNode, invars: list[str], returns: str) -> None:
        """Test function _emit_output_assignment.

        Args:
            node (IRNode): Test parameter `node`.
            invars (list[str]): Test parameter `invars`.
            returns (str): Test parameter `returns`.
        """
        del node, invars
        assert self._output_returns is not None
        self._output_returns.append(returns)

    def visit(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Test function visit.

        Args:
            node (IRNode): Test parameter `node`.
            input_vars (list[str]): Test parameter `input_vars`.
            **kwargs (BackendArg): Test parameter `kwargs`.

        Returns:
            str: Computed output value from `visit`.
        """
        del node, input_vars, kwargs
        return "visited"

    def add_line(self, line: str) -> None:
        """Test function add_line.

        Args:
            line (str): Test parameter `line`.
        """
        self.lines.append(line)


class DummyNode(MockNodeStub):
    """Test class DummyNode."""

    def __init__(self, op_type: str, id: str, inputs: list[str] | None = None) -> None:
        """Test function __init__.

        Args:
            op_type (str): Test parameter `op_type`.
            id (str): Test parameter `id`.
            inputs (list[str] | None): Test parameter `inputs`.
        """
        self.op_type = op_type
        self.id = id
        self.inputs = inputs or []
        self.attributes: dict[str, AttributeValue] = {}
        self.shape_metadata = ()


def test_visitor() -> None:
    """Test function test_visitor."""
    gen = DummyGenerator()
    vis = CodeGeneratorVisitor(gen)

    n_const = DummyNode("Constant", "c1")
    n_in = DummyNode("Input", "i1")
    n_out = DummyNode("Output", "o1", inputs=["in1"])
    n_op = DummyNode("Op", "op1", inputs=["in1"])
    pytest.MonkeyPatch().setattr(n_op, "attributes", {"stream_id": 1, "async_check": True}, raising=False)
    pytest.MonkeyPatch().setattr(n_op, "shape_metadata", (1,), raising=False)

    pytest.MonkeyPatch().setattr(gen, "sorted_nodes", [n_const, n_in, n_op, n_out], raising=False)

    vis.generate_body("args")

    assert "const_1 = 42.0" in gen.lines
    assert "input_1 = args[0]" in gen.lines
    assert "tensor_1 = visited" in gen.lines
    assert "return ['var_in1']" in gen.lines


def test_visitor_compute_node_branches() -> None:
    """Test function test_visitor_compute_node_branches."""
    gen = DummyGenerator()
    vis = CodeGeneratorVisitor(gen)

    n_op = DummyNode("Op", "op2")
    n_op.attributes = {}
    n_op.shape_metadata = ()
    vis.handle_compute_node(n_op)
    assert "tensor_1 = visited" in gen.lines
