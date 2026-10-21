"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from tests.testing_utils import as_callable, as_tensor_dict, as_tensor_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class DummyGenerator:
    """Test class DummyGenerator."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.sorted_nodes = as_tensor_list([])
        self.var_names = {"in1": "var_in1"}
        self.input_idx = 0
        self._output_returns = as_tensor_list([])
        self.lines = as_tensor_list([])

    def _emit_body_return(self, returns: EagerValue) -> None:
        """Test function _emit_body_return.

        Args:
            returns: Test parameter `returns`.
        """
        as_callable(self.lines.append)(f"return {returns}")

    def emit_constant(self, node: EagerValue) -> EagerValue:
        """Test function emit_constant.

        Args:
            node: Test parameter `node`.
        """
        return "42.0"

    def assign_var_name(self, id: EagerValue, prefix: EagerValue = "tensor") -> EagerValue:
        """Test function assign_var_name.

        Args:
            id: Test parameter `id`.
            prefix: Test parameter `prefix`.
        """
        return f"{prefix}_1"

    def _emit_constant_assignment(self, var: EagerValue, val: EagerValue) -> None:
        """Test function _emit_constant_assignment.

        Args:
            var: Test parameter `var`.
            val: Test parameter `val`.
        """
        as_callable(self.lines.append)(f"{var} = {val}")

    def _emit_input_assignment(self, var: EagerValue, node: EagerValue, pref: EagerValue, idx: EagerValue) -> None:
        """Test function _emit_input_assignment.

        Args:
            var: Test parameter `var`.
            node: Test parameter `node`.
            pref: Test parameter `pref`.
            idx: Test parameter `idx`.
        """
        as_callable(self.lines.append)(f"{var} = {pref}[{idx}]")

    def _emit_output_assignment(self, node: EagerValue, invars: EagerValue, returns: EagerValue) -> None:
        """Test function _emit_output_assignment.

        Args:
            node: Test parameter `node`.
            invars: Test parameter `invars`.
            returns: Test parameter `returns`.
        """
        as_callable(self._output_returns.append)(returns)

    def visit(self, node: EagerValue, invars: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function visit.

        Args:
            node: Test parameter `node`.
            invars: Test parameter `invars`.
            kwargs: Test parameter `kwargs`.
        """
        return "visited"

    def add_line(self, line: EagerValue) -> None:
        """Test function add_line.

        Args:
            line: Test parameter `line`.
        """
        as_callable(self.lines.append)(line)


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, op_type: EagerValue, id: EagerValue, inputs: EagerValue = None) -> None:
        """Test function __init__.

        Args:
            op_type: Test parameter `op_type`.
            id: Test parameter `id`.
            inputs: Test parameter `inputs`.
        """
        self.op_type = op_type
        self.id = id
        self.inputs = inputs or []
        self.attributes = as_tensor_dict({})
        self.shape_metadata = ()


def test_visitor() -> None:
    """Test function test_visitor."""
    gen = DummyGenerator()
    vis = as_callable(CodeGeneratorVisitor)(gen)

    n_const = DummyNode("Constant", "c1")
    n_in = DummyNode("Input", "i1")
    n_out = DummyNode("Output", "o1", inputs=["in1"])
    n_op = DummyNode("Op", "op1", inputs=["in1"])
    setattr(n_op, "attributes", {"stream_id": 1, "async_check": True})
    setattr(n_op, "shape_metadata", (1,))

    setattr(gen, "sorted_nodes", [n_const, n_in, n_op, n_out])

    vis.generate_body("args")

    assert "const_1 = 42.0" in gen.lines
    assert "input_1 = args[0]" in gen.lines
    assert "tensor_1 = visited" in gen.lines
    assert "return ['var_in1']" in gen.lines


def test_visitor_compute_node_branches() -> None:
    """Test function test_visitor_compute_node_branches."""
    gen = DummyGenerator()
    vis = as_callable(CodeGeneratorVisitor)(gen)

    n_op = DummyNode("Op", "op2")
    n_op.attributes = {}
    n_op.shape_metadata = ()
    as_callable(vis.handle_compute_node)(n_op)
    assert "tensor_1 = visited" in gen.lines
