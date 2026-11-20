"""Behavioral unit tests for shape debugger."""

from __future__ import annotations

import typing

from ml_switcheroo_ir import LogicalGraph, LogicalNode, NodeDict

import ml_switcheroo_compiler.diagnostics.shape_debugger as shape_debug_mod
from ml_switcheroo_compiler.core.tensor import Tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_shape_debugger() -> None:
    """Test function test_shape_debugger."""
    from unittest.mock import mock_open, patch

    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    import ml_switcheroo_compiler.diagnostics.shape_debugger as sd
    from ml_switcheroo_compiler.diagnostics.shape_debugger import _load_formatters, debug_shapes, to_graphviz, to_html

    sd._FORMATTERS = {}

    # Test _load_formatters without file
    with patch("os.path.exists", return_value=False):
        _load_formatters()
        assert sd._FORMATTERS == {}

    sd._FORMATTERS = {}

    # Test _load_formatters with empty file
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="")):
            _load_formatters()
            assert sd._FORMATTERS == {}

    sd._FORMATTERS = {}

    # Test _load_formatters with valid file
    with patch("os.path.exists", return_value=True):
        with patch(
            "builtins.open",
            mock_open(read_data="markdown_table:\n  header: '| Node | Shape |\\n'\n  row: '| {name} | {shape} |\\n'"),
        ):
            _load_formatters()
            assert "markdown_table" in sd._FORMATTERS

    # Test debug_shapes success
    def dummy_model(x: BackendArg) -> BackendArg:
        """Test function dummy_model.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `dummy_model`.
        """
        return x

    res = debug_shapes(dummy_model, (2, 2))
    assert "| Node | Shape |" in res
    assert "| input | (2, 2) |" in res
    assert "| output | (2, 2) |" in res

    # Test debug_shapes error
    def error_model(x: BackendArg) -> None:
        """Test function error_model.

        Args:
            x: Test parameter `x`.

        Raises:
            RuntimeError: When raised during `error_model` (boom).
        """
        raise RuntimeError("boom")

    res = debug_shapes(error_model, (2, 2))
    assert "| Node | Shape |" in res
    assert "| input | (2, 2) |" not in res  # because error happens and it resets

    # Test debug_shapes unknown shape
    def unknown_model(x: BackendArg) -> BackendArg:
        """Test function unknown_model.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `unknown_model`.
        """

        class Dummy:
            """Test class Dummy."""

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
                """No-op callable satisfying CallableObjectProtocol without `.shape`.

                Args:
                    *args: Ignored positional args.
                    **kwargs: Ignored keyword args.
                """
                del args, kwargs

        return Dummy()

    res = debug_shapes(unknown_model, (2, 2))
    assert "| output | unknown |" in res

    # Test to_graphviz
    g = LogicalGraph()
    n1 = LogicalNode(id="n1", op_type="Input")
    n2 = LogicalNode(id="n2", op_type="Add", inputs=["n1"])
    g.nodes = NodeDict({"n1": n1, "n2": n2})

    dot = to_graphviz(g)
    assert "digraph G {" in dot
    assert '"n1" [label="Input"];' in dot
    assert '"n1" -> "n2";' in dot

    # Test to_html
    html = to_html(g)
    assert "<html>" in html


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

        Returns:
            int: Computed output value from `__next__`.
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
        """Identity model returning input tensor.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor: Computed output value from `model_with_shape`.
        """
        return x

    md_out = shape_debug_mod.debug_shapes(model_with_shape, (2, 4))
    assert "| Node | Shape | DType |" in md_out
    assert "| input | (2, 4) | float64 |" in md_out
    assert "| output | (2, 4) | float64 |" in md_out

    # 2. debug_shapes with output lacking shape attribute
    def model_scalar_out(x: Tensor) -> None:
        """Model returning None without shape.

        Args:
            x: Test parameter `x`.
        """
        return None

    md_scalar = shape_debug_mod.debug_shapes(model_scalar_out, (3,))
    assert "| output | unknown | float64 |" in md_scalar

    # 3. debug_shapes when callable raises RuntimeError
    def model_raising(x: Tensor) -> Tensor:
        """Model raising RuntimeError for test.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor: Computed output value from `model_raising`.

        Raises:
            RuntimeError: When raised during `model_raising` (Model execution failed).
        """
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
