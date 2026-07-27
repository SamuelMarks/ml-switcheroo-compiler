"""Test module."""

from ml_switcheroo_compiler.backends.formatters import CodeFormatter, FallbackHandler, FormatterContext, OpFormatter


def test_formatter_context():
    ctx = FormatterContext("np", "Add", ["a"], {"axis": 0})
    assert ctx.prefix == "np"


def test_op_formatter():
    assert OpFormatter.format_backend_string("{0} + {val}", ["a"], {"val": 1}) == "a + 1"

    ctx1 = FormatterContext("np", "Sum", ["a"], {"axis": 0, "keepdims": True})
    assert OpFormatter.format_generic_fallback(ctx1) == "np.sum(a, axis=0, keepdims=True)"


def test_fallback_handler():
    class DummyNode:
        op_type = "Mean"

    assert FallbackHandler.generate_fallback_code(DummyNode(), ["a"], "np", axis=0) == "np.mean(a, axis=0)"


def test_code_formatter():
    f = CodeFormatter()
    assert f.get_indent() == ""
    f.indent_level = 1
    assert f.get_indent() == "    "
    f.add_line("test")
    assert f.code == ["    test"]
    assert f.assign_var_name("1") == "tensor_0"
    assert f.assign_var_name("1") == "tensor_0"


def test_op_formatter_edge_cases():
    assert OpFormatter.format_backend_string("{0} + x", ["a"], {"val": 1}) == "a + x"

    ctx2 = FormatterContext("np", "Sum", ["a"], {"axis": None, "keepdims": False})
    assert OpFormatter.format_generic_fallback(ctx2) == "np.sum(a)"


def test_op_formatter_edge_cases_extended():
    ctx = FormatterContext("np", "Cast", ["a"], {"dtype": "float32"})
    assert OpFormatter.format_generic_fallback(ctx) == "np.cast(a, dtype='float32')"


def test_op_formatter_edge_cases_continue():
    ctx = FormatterContext("np", "Cast", ["a"], {"node_id": "n1", "shape_metadata": ()})
    assert OpFormatter.format_generic_fallback(ctx) == "np.cast(a)"
