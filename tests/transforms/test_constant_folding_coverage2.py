"""Module docstring."""

from unittest.mock import MagicMock

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.constant_folding import constant_folding_pass


def test_constant_folding_numel_branch(monkeypatch: object) -> None:
    """Docstring."""
    graph = IRGraph()
    n0 = IRNode(id="n0", op_type="Constant", inputs=[], attributes={"value": 1})
    n1 = IRNode(id="n1", op_type="Add", inputs=["n0"], attributes={})
    graph.nodes["n0"] = n0
    graph.nodes["n1"] = n1

    # Mock evaluate_graph to return something with numel
    class MockVal:
        """Docstring."""

        def numel(self) -> int:
            """Docstring."""
            return 1

    def mock_eval(g: object, inputs: dict) -> dict:
        """Docstring."""
        return {"n1": MockVal()}

    monkeypatch.setattr("ml_switcheroo_compiler.interpreter.evaluate_graph", mock_eval)

    mock_backend = MagicMock()
    mock_backend.item.return_value = 42
    monkeypatch.setattr(
        "ml_switcheroo_compiler.backends.registry.get_active_backend",
        lambda: mock_backend,
    )

    constant_folding_pass(graph)
    assert graph.nodes["n1"].op_type == "Constant"
    assert graph.nodes["n1"].attributes["value"] == 42
