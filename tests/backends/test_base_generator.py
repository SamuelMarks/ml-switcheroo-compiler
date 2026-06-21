"""Test base generator."""

import pytest

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_base_generator_not_implemented() -> None:
    """Docstring."""
    with pytest.raises(NotImplementedError):
        BaseGenerator.execute_op("Add", 1, 2)
    with pytest.raises(NotImplementedError):
        BaseGenerator.zeros((2, 2))
    with pytest.raises(NotImplementedError):
        BaseGenerator.array([1, 2])
    with pytest.raises(NotImplementedError):
        BaseGenerator.asarray([1, 2])
    with pytest.raises(NotImplementedError):
        BaseGenerator.item([1.5])


class MockGenerator(BaseGenerator):
    """Docstring."""

    def generate(self) -> str:
        """Docstring."""
        return ""

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Docstring."""
        return "mock"


def test_base_generator_compute_node_coverage() -> None:
    """Docstring."""
    graph = IRGraph()
    node = IRNode(
        "id1",
        "MockOp",
        attributes={"stream_id": 1, "async_check": True},
        shape_metadata=(1, 2),
    )
    graph.nodes["id1"] = node

    from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor

    gen = MockGenerator(graph)
    visitor = CodeGeneratorVisitor(gen)
    visitor.handle_compute_node(node)
    assert gen.code[0] == "tensor_0 = mock"
