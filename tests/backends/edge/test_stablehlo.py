"""Tests for StableHLO backend coverage."""

import unittest

from ml_switcheroo_compiler.backends.edge.stablehlo import StableHLOCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


class TestStableHLOCodeGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.graph = IRGraph()
        self.graph.outputs = ["n_Add"]

        self.input_node = LogicalNode(id="in1", op_type="Input")
        self.input_node.shape_metadata = (2, 3)
        self.input_node.dtype = "float32"

        self.graph.nodes = {"in1": self.input_node}

        self.gen = StableHLOCodeGenerator(self.graph)
        self.gen.sorted_nodes = [self.input_node]

        for op in ["Constant", "Add", "Subtract", "Multiply", "TrueDivide", "Div", "Exp", "Log", "Negative", "Neg", "Other"]:
            n = LogicalNode(id="n_" + op, op_type=op, inputs=["in1", "missing_node"])
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

    def test_empty_edge_stablehlo_variant(self):
        """Test fallback when edge_stablehlo exists but lacks opcode/generator."""
        from unittest.mock import patch

        mock_registry = {"EmptyVariantOp": {"variants": {"edge_stablehlo": {}}}}

        n = LogicalNode(id="n_empty", op_type="EmptyVariantOp", inputs=["in1"])
        self.gen.sorted_nodes.append(n)

        with patch("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", mock_registry):
            code = self.gen.generate()
            self.assertIn("stablehlo.custom_call", code)

            import os
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.close()
                self.gen.export_mlirbc(tf.name)
                os.unlink(tf.name)

    def test_map_type(self):
        """Test map type."""
        self.assertEqual(self.gen._map_type((2, 3), "float32"), "tensor<2x3xf32>")
        self.assertEqual(self.gen._map_type((), "float32"), "tensor<f32>")
        self.assertEqual(self.gen._map_type((2,), "float64"), "tensor<2xf64>")
        self.assertEqual(self.gen._map_type((2, 3, 4), "int32"), "tensor<2x3x4xi32>")
        self.assertEqual(self.gen._map_type((2,), "bool"), "tensor<2xi1>")
        self.assertEqual(self.gen._map_type((2,), "unknown"), "tensor<2xf32>")

    def test_resolve_input_types(self):
        """Test resolve input types."""
        n = LogicalNode(id="test_node", op_type="Add", inputs=["in1", "missing_node"])
        types = self.gen._resolve_input_types(n, "tensor<f32>")
        self.assertEqual(types, ["tensor<2x3xf32>", "tensor<f32>"])

    def test_generic_visit_input(self):
        """Test generic visit input."""
        name = self.gen.generic_visit(self.input_node, [])
        self.assertTrue(name.startswith("%arg"))

    def test_export_mlirbc(self):
        """Test export to MLIR bytecode."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.close()
            self.gen.export_mlirbc(tf.name)

            with open(tf.name, "rb") as f:
                content = f.read()
            self.assertTrue(content.startswith(b"ML\xefR\x01"))
            os.unlink(tf.name)

    def test_generate(self):
        """Test generate."""
        code = self.gen.generate()
        self.assertIn("module @jit_fun", code)
        self.assertIn("func.func @main", code)
        self.assertIn("stablehlo.constant", code)
        self.assertIn("stablehlo.add", code)
        self.assertIn("stablehlo.subtract", code)
        self.assertIn("stablehlo.multiply", code)
        self.assertIn("stablehlo.divide", code)
        self.assertIn("stablehlo.exponential", code)
        self.assertIn("stablehlo.log", code)
        self.assertIn("stablehlo.negate", code)
        self.assertIn("stablehlo.custom_call", code)
        self.assertIn("return %v_n_Add", code)

    def test_build_out_types(self):
        """Test build out types."""
        # Test with missing node
        types = self.gen._build_out_types(["n_Add", "missing_node"])
        self.assertEqual(types, ["tensor<2x3xf32>", "tensor<f32>"])

    def test_get_returns_str(self):
        """Test get returns str."""
        self.assertEqual(self.gen._get_returns_str([]), "tensor<f32>")
        self.assertEqual(self.gen._get_returns_str(["tensor<f32>"]), "tensor<f32>")
        self.assertEqual(self.gen._get_returns_str(["tensor<f32>", "tensor<i32>"]), "tensor<f32>, tensor<i32>")

    def test_get_ret_vars_str(self):
        """Test get ret vars str."""
        self.assertEqual(self.gen._get_ret_vars_str([]), "")
        self.assertEqual(self.gen._get_ret_vars_str(["%v_1"]), "%v_1")
        self.assertEqual(self.gen._get_ret_vars_str(["%v_1", "%v_2"]), "%v_1, %v_2")

    def test_generate_no_outputs(self):
        """Test generate no outputs."""
        self.gen.graph.outputs = None
        code = self.gen.generate()
        self.assertIn("return", code)

    def test_stablehlo_types_and_ops(self) -> None:
        """Test generation of extended neural network and tensor types for StableHLO."""
        gen = StableHLOCodeGenerator(IRGraph())

        # Test types
        assert "f16" in gen._map_type((), "float16")
        assert "bf16" in gen._map_type((), "bfloat16")
        assert "i64" in gen._map_type((), "int64")
        assert "i8" in gen._map_type((), "int8")
        assert "ui32" in gen._map_type((), "uint32")
        assert "ui8" in gen._map_type((), "uint8")

        ops = ["MatMul", "Conv2D", "Reshape", "Transpose", "Broadcast", "Concat", "Slice", "Gather", "ReduceSum", "ReduceMean", "Relu", "Sigmoid"]

        for op in ops:
            n = LogicalNode(id="n_" + op, op_type=op, inputs=["in1", "in2"])
            gen.sorted_nodes.append(n)

        gen.generate()

        output = "\n".join(gen.code)
        assert "stablehlo.dot_general" in output
        assert "stablehlo.convolution" in output
        assert "stablehlo.reduce" in output
        assert "stablehlo.maximum" in output


import pytest


def test_stablehlo_no_schema():
    with pytest.MonkeyPatch.context() as m:
        m.setattr("os.path.exists", lambda x: False)
        graph = IRGraph()
        gen = StableHLOCodeGenerator(graph)
        assert gen.schema == {}
