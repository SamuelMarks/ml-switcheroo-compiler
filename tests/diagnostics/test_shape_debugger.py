def test_shape_debugger():
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
        with patch("builtins.open", mock_open(read_data="markdown_table:\n  header: '| Node | Shape |\\n'\n  row: '| {name} | {shape} |\\n'")):
            _load_formatters()
            assert "markdown_table" in sd._FORMATTERS

    # Test debug_shapes success
    def dummy_model(x):
        return x

    res = debug_shapes(dummy_model, (2, 2))
    assert "| Node | Shape |" in res
    assert "| input | (2, 2) |" in res
    assert "| output | (2, 2) |" in res

    # Test debug_shapes error
    def error_model(x):
        raise RuntimeError("boom")

    res = debug_shapes(error_model, (2, 2))
    assert "| Node | Shape |" in res
    assert "| input | (2, 2) |" not in res  # because error happens and it resets

    # Test debug_shapes unknown shape
    def unknown_model(x):
        class Dummy:
            pass

        return Dummy()

    res = debug_shapes(unknown_model, (2, 2))
    assert "| output | unknown |" in res

    # Test to_graphviz
    g = LogicalGraph()
    n1 = LogicalNode(id="n1", op_type="Input")
    n2 = LogicalNode(id="n2", op_type="Add", inputs=["n1"])
    g.nodes = {"n1": n1, "n2": n2}

    dot = to_graphviz(g)
    assert "digraph G {" in dot
    assert '"n1" [label="Input"];' in dot
    assert '"n1" -> "n2";' in dot

    # Test to_html
    html = to_html(g)
    assert "<html>" in html
