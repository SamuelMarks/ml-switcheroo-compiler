"""Tests for ONNX backend coverage."""

import sys
from unittest.mock import MagicMock

mock_onnx = MagicMock()
mock_onnx.TensorProto.DOUBLE = 11
mock_onnx.TensorProto.INT32 = 6
mock_onnx.TensorProto.BOOL = 9
mock_onnx.TensorProto.FLOAT = 1
mock_onnx.helper.make_tensor_value_info.return_value = "ValueInfo"
mock_onnx.helper.make_tensor.return_value = "Tensor"
mock_onnx.helper.make_node.return_value = "Node"
mock_onnx.helper.make_graph.return_value = MagicMock()
mock_onnx.helper.printable_graph.return_value = "PrintableGraph"
sys.modules["onnx"] = mock_onnx

import unittest
from unittest.mock import patch

from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


class TestONNXCodeGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.graph = IRGraph()
        self.graph.outputs = ["n_Add"]

        input_node = LogicalNode(id="in1", op_type="Input")
        input_node.shape_metadata = (2, 3)
        input_node.dtype = "float32"
        self.graph.nodes = {"in1": input_node}

        self.gen = ONNXCodeGenerator(self.graph)
        self.gen.sorted_nodes = [input_node]

        for op in ["Constant", "Add", "Subtract", "Multiply", "TrueDivide", "Div", "Exp", "Log", "Negative", "Neg", "Other"]:
            n = LogicalNode(id="n_" + op, op_type=op, inputs=["in1", "in2"])
            n.shape_metadata = (2, 3)
            n.dtype = "float32"
            if op == "Constant":
                n.attributes = {"value": 1.0}
            self.gen.sorted_nodes.append(n)

        n_none = LogicalNode(id="n_NoneShape", op_type="Add", inputs=["in1"])
        n_none.shape_metadata = None
        n_none.dtype = "float32"
        self.gen.sorted_nodes.append(n_none)

        n_const_none = LogicalNode(id="n_ConstNoneShape", op_type="Constant", inputs=[])
        n_const_none.shape_metadata = None
        n_const_none.attributes = {"value": 1.0}
        self.gen.sorted_nodes.append(n_const_none)

    def test_generic_visit(self):
        """Test generic visit."""
        self.assertEqual(self.gen.generic_visit(None, []), "onnx_op")

        node = LogicalNode(id="test_node", op_type="Add")
        self.assertEqual(self.gen.generic_visit(node, []), "test_node")

        # Node without id
        class NodeNoId:
            pass

        node_noid = NodeNoId()
        name = self.gen.generic_visit(node_noid, [])
        self.assertTrue(isinstance(name, str))
        self.assertGreater(len(name), 0)

    def test_get_proto_type(self):
        """Test get proto type."""

        class TensorProto:
            DOUBLE = 11
            INT32 = 6
            BOOL = 9
            FLOAT = 1

        self.assertEqual(self.gen._get_proto_type("float64", TensorProto), TensorProto.DOUBLE)
        self.assertEqual(self.gen._get_proto_type("int32", TensorProto), TensorProto.INT32)
        self.assertEqual(self.gen._get_proto_type("bool", TensorProto), TensorProto.BOOL)
        self.assertEqual(self.gen._get_proto_type("float32", TensorProto), TensorProto.FLOAT)

    def test_generate_text_fallback(self):
        """Test generate text fallback."""
        # Add output to graph
        self.graph.outputs = ["n_Add"]
        fallback = self.gen._generate_text_fallback()
        self.assertIn("ir_version: 7", fallback)
        self.assertIn('input: "in1"', fallback)
        self.assertIn('"n_Add" = Add("in1", "in2")', fallback)
        self.assertIn('output: "n_Add"', fallback)

    def test_generate_with_onnx(self):
        """Test generate with onnx."""
        mock_onnx.helper.printable_graph.return_value = "PrintableGraph"

        # Test with dynamic axes
        dynamic_axes = {"in1": {0: "batch_size"}}
        res = self.gen.generate(dynamic_axes=dynamic_axes)
        self.assertEqual(res, "PrintableGraph")

        # Test export
        mock_model = MagicMock()
        mock_model.SerializeToString.return_value = b"test"
        mock_onnx.helper.make_model.return_value = mock_model

        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.close()
            self.gen.export_onnx(tf.name, dynamic_axes=dynamic_axes)
            mock_model.SerializeToString.assert_called_once()
            os.unlink(tf.name)

    def test_generate_without_onnx(self):
        """Test generate without onnx."""
        # Test Import Error behavior by forcing ImportError in generate
        with patch("builtins.__import__", side_effect=ImportError("mocked import error")):
            res = self.gen.generate()
            self.assertIn("ir_version: 7", res)

    def test_get_node_and_name_none_node(self):
        """Test get node and name none node."""
        node, name = self.gen._get_node_and_name("missing_output", is_output=True)
        self.assertIsNone(node)
        self.assertEqual(name, "missing_output")

    def test_build_single_value_info_missing_node(self):
        """Test build single value info missing node."""
        mock_onnx.helper.make_tensor_value_info.reset_mock()
        self.gen._build_single_value_info("missing_output", None, mock_onnx.TensorProto, is_output=True)
        mock_onnx.helper.make_tensor_value_info.assert_called_with("missing_output", 1, [])

    def test_fallback_no_outputs_or_inputs(self):
        """Test fallback no outputs or inputs."""
        self.gen.graph.outputs = None
        self.gen.sorted_nodes = []
        res = self.gen._generate_text_fallback()
        self.assertIn("ir_version: 7", res)

    def test_onnx_proto_types(self) -> None:
        """Test exhaustive proto types mapping."""

        class DummyTensorProto:
            DOUBLE = 11
            FLOAT = 1
            FLOAT16 = 10
            BFLOAT16 = 16
            INT64 = 7
            INT32 = 6
            INT16 = 5
            INT8 = 3
            UINT64 = 8
            UINT32 = 13
            UINT16 = 4
            UINT8 = 2
            BOOL = 9

        gen = ONNXCodeGenerator(IRGraph())
        assert gen._get_proto_type("float64", DummyTensorProto) == DummyTensorProto.DOUBLE
        assert gen._get_proto_type("float16", DummyTensorProto) == DummyTensorProto.FLOAT16
        assert gen._get_proto_type("bfloat16", DummyTensorProto) == DummyTensorProto.BFLOAT16
        assert gen._get_proto_type("int64", DummyTensorProto) == DummyTensorProto.INT64
        assert gen._get_proto_type("int8", DummyTensorProto) == DummyTensorProto.INT8
        assert gen._get_proto_type("uint32", DummyTensorProto) == DummyTensorProto.UINT32
        assert gen._get_proto_type("uint8", DummyTensorProto) == DummyTensorProto.UINT8
        assert gen._get_proto_type("unknown", DummyTensorProto) == DummyTensorProto.FLOAT
        assert gen._get_proto_type("int16", DummyTensorProto) == DummyTensorProto.INT16
        assert gen._get_proto_type("uint64", DummyTensorProto) == DummyTensorProto.UINT64
        assert gen._get_proto_type("uint16", DummyTensorProto) == DummyTensorProto.UINT16

    def test_onnx_extended_ops(self) -> None:
        """Test generation of extended neural network and tensor operations."""
        graph = IRGraph()
        graph.outputs = ["out"]
        gen = ONNXCodeGenerator(graph)

        ops = ["MatMul", "Conv2D", "MaxPool", "AvgPool2D", "BatchNorm", "Reshape", "Transpose", "Squeeze", "Concat", "Slice", "Gather", "ScatterND", "ReduceSum", "ReduceMean"]

        for op in ops:
            n = LogicalNode(id="n_" + op, op_type=op, inputs=["in1", "in2"])
            gen.sorted_nodes.append(n)

        try:
            import onnx  # noqa: F401

            has_onnx = True
        except ImportError:
            has_onnx = False

        if has_onnx:
            from onnx import TensorProto

            nodes = gen._build_onnx_nodes(TensorProto)
            # 14 ops were added
            assert len(nodes) == 14
            # ops generated successfully
            # Onnx ops don't expose op_type like our nodes directly here if it's a NodeProto,
            # we should check string representation or just accept it generated 14 nodes.

    def test_printer_to_text(self) -> None:
        """Test the printer.to_text execution path if available."""
        with patch("ml_switcheroo_compiler.backends.edge.onnx.ONNXCodeGenerator._build_onnx_graph") as mock_build:
            mock_build.return_value = "GraphDef"
            res = self.gen.generate()
            self.assertEqual(res, "PrintableGraph")

    def test_printer_to_text_string(self) -> None:
        """Test the printer.to_text execution path returning a string."""
        with patch("ml_switcheroo_compiler.backends.edge.onnx.ONNXCodeGenerator._build_onnx_graph") as mock_build:
            mock_build.return_value = "GraphDef"

            class MockPrinter:
                def to_text(self, graph):
                    return "RealPrintableGraph"

            with patch("onnx.printer", new=MockPrinter()):
                res = self.gen.generate()
                self.assertEqual(res, "RealPrintableGraph")

    def test_printer_to_text_np_fallback(self) -> None:
        """Test the printer.to_text execution path falling back to mock logic."""
        with patch("ml_switcheroo_compiler.backends.edge.onnx.ONNXCodeGenerator._build_onnx_graph") as mock_build:
            mock_build.return_value = "GraphDef"

            class MockPrinter:
                def to_text(self, graph):
                    return MagicMock()

            with patch("onnx.printer", new=MockPrinter()):
                res = self.gen.generate()
                self.assertEqual(res, "PrintableGraph")

    def test_printer_to_text_import_error(self) -> None:
        """Test the printer.to_text execution path throwing ImportError."""
        with patch("ml_switcheroo_compiler.backends.edge.onnx.ONNXCodeGenerator._build_onnx_graph") as mock_build:
            mock_build.return_value = "GraphDef"

            orig_import = __import__
            with patch("builtins.__import__") as mock_import:

                def side_effect(name, *args, **kwargs):
                    if name == "onnx":
                        raise ImportError()
                    return orig_import(name, *args, **kwargs)

                mock_import.side_effect = side_effect

                with patch("ml_switcheroo_compiler.backends.edge.onnx.ONNXCodeGenerator._generate_text_fallback") as mock_fallback:
                    mock_fallback.return_value = "Fallback"
                    res = self.gen.generate()
                    self.assertEqual(res, "Fallback")

    def test_printer_to_text_inner_import_error(self) -> None:
        """Test the printer.to_text execution path throwing an inner ImportError on onnx.printer."""
        with patch("ml_switcheroo_compiler.backends.edge.onnx.ONNXCodeGenerator._build_onnx_graph") as mock_build:
            mock_build.return_value = "GraphDef"

            class MockPrinter:
                def to_text(self, g):
                    raise ImportError("Mocked")

            with patch("onnx.printer", new=MockPrinter()):
                with patch("onnx.helper.printable_graph") as mock_printable_graph:
                    mock_printable_graph.return_value = "MockedPrintableGraph"
                    res = self.gen.generate()
                    self.assertEqual(res, "MockedPrintableGraph")

    def test_onnx_subgraphs_if_loop(self):
        """Test subgraph generation for If and Loop ops."""
        g = IRGraph()
        n_if = LogicalNode(id="n_If", op_type="If", inputs=["cond"])

        then_graph = IRGraph()
        then_node = LogicalNode(id="then_add", op_type="Add", inputs=["a", "b"])
        then_graph.nodes = {"then_add": then_node}
        then_graph.outputs = ["then_add"]
        n_if.attributes = {"then_branch": then_graph}

        else_graph = IRGraph()
        else_node = LogicalNode(id="else_sub", op_type="Sub", inputs=["a", "b"])
        else_graph.nodes = {"else_sub": else_node}
        else_graph.outputs = ["else_sub"]
        n_if.attributes["else_branch"] = else_graph

        n_loop = LogicalNode(id="n_Loop", op_type="Loop", inputs=["len"])
        body_graph = IRGraph()
        body_node = LogicalNode(id="body_mul", op_type="Multiply", inputs=["a", "b"])
        body_graph.nodes = {"body_mul": body_node}
        body_graph.outputs = ["body_mul"]
        n_loop.attributes = {"body": body_graph}

        g.nodes = {"n_If": n_if, "n_Loop": n_loop}
        gen = ONNXCodeGenerator(g)
        gen.sorted_nodes = [n_if, n_loop]

        nodes = gen._build_onnx_nodes(mock_onnx.TensorProto)
        # Should have called make_node with kwargs then_branch, else_branch, body
        self.assertEqual(len(nodes), 2)

        # Check that mock_onnx.helper.make_node was called with the right kwargs
        # Since it's a MagicMock, we can inspect its call args
        calls = mock_onnx.helper.make_node.call_args_list
        if_call = next(c for c in calls if c[0][0] == "If")
        loop_call = next(c for c in calls if c[0][0] == "Loop")

        self.assertIn("then_branch", if_call[1])
        self.assertIn("else_branch", if_call[1])
        self.assertIn("body", loop_call[1])

    def test_onnx_subgraphs_missing_branches(self):
        """Test subgraph generation for If and Loop ops missing branches."""
        g = IRGraph()
        n_if_no_then = LogicalNode(id="n_If_nothen", op_type="If", inputs=["cond"])
        n_if_no_then.attributes = {"else_branch": IRGraph()}

        n_if_no_else = LogicalNode(id="n_If_noelse", op_type="If", inputs=["cond"])
        n_if_no_else.attributes = {"then_branch": IRGraph()}

        n_loop_no_body = LogicalNode(id="n_Loop_nobody", op_type="Loop", inputs=["len"])
        n_loop_no_body.attributes = {}

        g.nodes = {"n_If_nothen": n_if_no_then, "n_If_noelse": n_if_no_else, "n_Loop_nobody": n_loop_no_body}
        gen = ONNXCodeGenerator(g)
        gen.sorted_nodes = [n_if_no_then, n_if_no_else, n_loop_no_body]

        nodes = gen._build_onnx_nodes(mock_onnx.TensorProto)
        self.assertEqual(len(nodes), 3)
