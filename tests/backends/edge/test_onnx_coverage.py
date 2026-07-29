"""Test ONNX backend edge cases coverage."""

import sys
from unittest.mock import MagicMock

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator


def test_onnx_evaluate_node_none():
    """Test generic_visit with None."""
    gen = ONNXCodeGenerator(LogicalGraph())
    assert gen.generic_visit(None, []) == "onnx_op"


def test_onnx_generic_visit_input():
    """Explicitly test generic_visit on Input nodes to cover lines 43-45."""
    gen = ONNXCodeGenerator(LogicalGraph())
    n = LogicalNode(id="n1", op_type="Input")
    assert gen.generic_visit(n, []) == "n1"
    assert gen.var_map["n1"] == "n1"


def test_onnx_get_proto_type():
    """Test _get_proto_type."""
    gen = ONNXCodeGenerator(LogicalGraph())

    class MockTensorProto:
        DOUBLE = 11
        INT32 = 6
        BOOL = 9
        FLOAT = 1

    assert gen._get_proto_type("float64", MockTensorProto) == 11
    assert gen._get_proto_type("int32", MockTensorProto) == 6
    assert gen._get_proto_type("bool", MockTensorProto) == 9
    assert gen._get_proto_type("unknown", MockTensorProto) == 1


def test_onnx_generate_text_fallback():
    """Test _generate_text_fallback and import error coverage."""
    g = LogicalGraph(outputs=["n2"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input", shape_metadata=(1, 2))
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Add", inputs=["n1"])
    gen = ONNXCodeGenerator(g)

    # Mock ImportError
    original_onnx = sys.modules.get("onnx")
    sys.modules["onnx"] = None
    try:
        res = gen.generate()
        assert "ir_version: 7" in res
        assert 'input: "n1"' in res
        assert '"n2" = Add("n1")' in res
        assert 'output: "n2"' in res
    finally:
        if original_onnx:
            sys.modules["onnx"] = original_onnx
        else:
            del sys.modules["onnx"]


def test_onnx_generate_success(monkeypatch):
    """Test generate with mocked ONNX to cover success path."""
    g = LogicalGraph(outputs=["n1", "n3"])
    n_inp = LogicalNode(id="inp", op_type="Input", shape_metadata=(1, 2))
    n_inp.dtype = "float32"
    g.nodes["inp"] = n_inp
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Constant", attributes={"value": 5.0}, shape_metadata=(2,))
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Constant", attributes={"value": 1.0})
    g.nodes["n3"] = LogicalNode(id="n3", op_type="Add", inputs=["inp", "n1"])

    gen = ONNXCodeGenerator(g)

    mock_onnx = MagicMock()
    original_onnx = sys.modules.get("onnx")
    sys.modules["onnx"] = mock_onnx

    try:
        res = gen.generate()
        assert res is not None
    finally:
        if original_onnx:
            sys.modules["onnx"] = original_onnx
        else:
            del sys.modules["onnx"]


def test_onnx_export_constant_no_shape(monkeypatch):
    """Test exporting constant node without shape."""
    g = LogicalGraph(outputs=["n1", "n2", "n3"])

    n_inp = LogicalNode(id="inp", op_type="Input", shape_metadata=(1, 2))
    n_inp.dtype = "float32"
    g.nodes["inp"] = n_inp

    g.nodes["n1"] = LogicalNode(id="n1", op_type="Constant", attributes={"value": 5.0})
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Add", inputs=["inp", "n1"])
    g.nodes["n3"] = LogicalNode(id="n3", op_type="Subtract", inputs=["n2", "n1"])

    gen = ONNXCodeGenerator(g)
    mock_file = MagicMock()
    mock_open = MagicMock(return_value=mock_file)
    mock_file.__enter__.return_value = mock_file

    monkeypatch.setattr("builtins.open", mock_open)

    mock_onnx = MagicMock()
    original_onnx = sys.modules.get("onnx")
    sys.modules["onnx"] = mock_onnx

    try:
        gen.export_onnx("dummy.onnx")
    finally:
        if original_onnx:
            sys.modules["onnx"] = original_onnx
        else:
            del sys.modules["onnx"]

    mock_open.assert_called_once_with("dummy.onnx", "wb")
