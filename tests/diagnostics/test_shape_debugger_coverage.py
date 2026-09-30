"""Tests for test_shape_debugger_coverage."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode

import ml_switcheroo_compiler.diagnostics.shape_debugger as shape_debug_mod
from ml_switcheroo_compiler.core.tensor import Tensor


class _DummyBadShape:
    """Shape object whose iterator raises TypeError."""

    def __iter__(self) -> _DummyBadShape:
        """Return self as iterator.

        Returns:
            _DummyBadShape: Self.
        """
        return self

    def __next__(self) -> int:
        """Raise TypeError on iteration.

        Raises:
            TypeError: Simulated invalid dimension.
        """
        raise TypeError("Not iterable dimension")


def test_shape_debugger_exhaustive() -> None:
    """Verify shape debugging tables, Graphviz conversion, and HTML exports."""
    from unittest.mock import patch

    # 0. Test when formatters.yaml does not exist
    shape_debug_mod._FORMATTERS = {}
    with patch("os.path.exists", return_value=False):
        shape_debug_mod._load_formatters()
        assert shape_debug_mod._FORMATTERS == {}

    # Reset _FORMATTERS to exercise loading from formatters.yaml
    shape_debug_mod._FORMATTERS = {}
    shape_debug_mod._load_formatters()
    # Call _load_formatters again to exercise cache branch
    shape_debug_mod._load_formatters()

    # 1. debug_shapes with standard output having shape
    def model_with_shape(x: Tensor) -> Tensor:
        """Identity model returning input tensor."""
        return x

    md_out = shape_debug_mod.debug_shapes(model_with_shape, (2, 4))
    assert "| Node | Shape | DType |" in md_out
    assert "| input | (2, 4) | float64 |" in md_out
    assert "| output | (2, 4) | float64 |" in md_out

    # 2. debug_shapes with output lacking shape attribute
    def model_scalar_out(x: Tensor) -> None:
        """Model returning None without shape."""
        return None

    md_scalar = shape_debug_mod.debug_shapes(model_scalar_out, (3,))
    assert "| output | unknown | float64 |" in md_scalar

    # 3. debug_shapes when callable raises RuntimeError
    def model_raising(x: Tensor) -> Tensor:
        """Model raising RuntimeError for test."""
        raise RuntimeError("Model execution failed")

    md_err = shape_debug_mod.debug_shapes(model_raising, (1,))
    assert "| Node | Shape | DType |" in md_err

    # 4. to_graphviz
    graph = LogicalGraph(name="test_viz", outputs=["n1"])
    graph.nodes["n0"] = LogicalNode("n0", "Input")
    graph.nodes["n1"] = LogicalNode("n1", "Relu", inputs=["n0"])

    dot_str = shape_debug_mod.to_graphviz(graph)
    assert "digraph G {" in dot_str
    assert '"n0" [label="Input"];' in dot_str
    assert '"n1" [label="Relu"];' in dot_str
    assert '"n0" -> "n1";' in dot_str

    # 5. to_html
    html_str = shape_debug_mod.to_html(graph)
    assert "<html><body><h1>IR Graph</h1></body></html>" in html_str
