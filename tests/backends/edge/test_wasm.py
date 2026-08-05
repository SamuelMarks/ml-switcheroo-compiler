"""Tests for WASM backend coverage."""

import os
import unittest
from unittest.mock import patch

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


class TestWasmCodeGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.graph = IRGraph()
        self.graph.outputs = ["n_Add"]

        self.input_node = LogicalNode(id="in1", op_type="Input")
        self.input_node.dtype = "float32"
        self.graph.nodes = {"in1": self.input_node}

        self.gen = WasmCodeGenerator(self.graph)
        self.gen.sorted_nodes = [self.input_node]

        for op in ["Constant", "Add", "Subtract", "Multiply", "TrueDivide", "Div", "Min", "Max", "Sqrt", "Abs", "Negative", "Neg", "Exp", "Log", "Other"]:
            n = LogicalNode(id="n_" + op, op_type=op, inputs=["in1", "in1"])
            n.dtype = "float32"
            if op == "Constant":
                n.attributes = {"value": 1.0}
            self.gen.sorted_nodes.append(n)

        n_none = LogicalNode(id="n_NoneShape", op_type="Add", inputs=["in1", "in1"])
        n_none.dtype = "float64"
        self.gen.sorted_nodes.append(n_none)

    def test_map_type(self):
        """Test map type."""
        self.assertEqual(self.gen._map_type("float32"), "float")
        self.assertEqual(self.gen._map_type("float64"), "double")
        self.assertEqual(self.gen._map_type("int32"), "int")
        self.assertEqual(self.gen._map_type("bool"), "bool")
        self.assertEqual(self.gen._map_type("unknown"), "float")

    def test_generic_visit_none(self):
        """Test generic visit none."""
        self.assertEqual(self.gen.generic_visit(None, []), "")

    def test_generate(self):
        """Test generate."""
        code = self.gen.generate()
        self.assertIn("#include <wasm_simd128.h>", code)
        self.assertIn("void main_kernel", code)

        # Check SIMD ops
        self.assertIn("wasm_f32x4_splat", code)
        self.assertIn("wasm_f32x4_add", code)
        self.assertIn("wasm_f32x4_sub", code)
        self.assertIn("wasm_f32x4_mul", code)
        self.assertIn("wasm_f32x4_div", code)
        self.assertIn("wasm_f32x4_pmin", code)
        self.assertIn("wasm_f32x4_pmax", code)
        self.assertIn("wasm_f32x4_sqrt", code)
        self.assertIn("wasm_f32x4_abs", code)
        self.assertIn("wasm_f32x4_neg", code)
        self.assertIn("std::exp(wasm_f32x4_extract_lane", code)
        self.assertIn("std::log(wasm_f32x4_extract_lane", code)
        self.assertIn("other(wasm_f32x4_extract_lane", code)

        # Check scalar ops
        self.assertIn("float v_n_Add_scalar = ", code)
        self.assertIn("std::exp(in_0[idx])", code)
        self.assertIn("std::min(in_0[idx], in_0[idx])", code)
        self.assertIn("-in_0[idx]", code)

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_compile_wasm_emcc(self, mock_subprocess, mock_which):
        """Test compile wasm emcc."""

        def which_side_effect(cmd):
            if cmd == "emcc":
                return "/usr/bin/emcc"
            return None

        mock_which.side_effect = which_side_effect

        res = self.gen.compile_wasm()
        self.assertIsNotNone(res)
        self.assertEqual(len(res), 2)
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        self.assertEqual(args[0], "/usr/bin/emcc")

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_compile_wasm_clang(self, mock_subprocess, mock_which):
        """Test compile wasm clang."""

        def which_side_effect(cmd):
            if cmd == "clang":
                return "/usr/bin/clang"
            return None

        mock_which.side_effect = which_side_effect

        res = self.gen.compile_wasm()
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "")
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        self.assertEqual(args[0], "/usr/bin/clang")

    @patch("shutil.which")
    def test_compile_wasm_none(self, mock_which):
        """Test compile wasm none."""
        mock_which.return_value = None
        res = self.gen.compile_wasm()
        self.assertIsNone(res)

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_compile_wasm_exception(self, mock_subprocess, mock_which):
        """Test compile wasm exception."""
        mock_which.return_value = "/usr/bin/emcc"
        mock_subprocess.side_effect = Exception("compile error")
        res = self.gen.compile_wasm()
        self.assertIsNone(res)

    def test_output_node(self):
        """Test output node."""
        n = LogicalNode(id="n_Output", op_type="Output", inputs=["in1", "in1"])
        self.gen.sorted_nodes.append(n)
        self.gen.is_simd = True
        self.gen.generic_visit(n, [])
        self.gen.is_simd = False
        self.gen.generic_visit(n, [])

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_compile_wasm_deleted_tempfile(self, mock_subprocess, mock_which):
        """Test compile wasm deleted tempfile."""

        def mock_run_side_effect(cmd, **kwargs):
            # Find the temp file and delete it before compile_wasm does
            for arg in cmd:
                if arg.endswith(".cpp"):
                    os.remove(arg)

        mock_which.return_value = "/usr/bin/emcc"
        mock_subprocess.side_effect = mock_run_side_effect
        self.gen.compile_wasm()

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_compile_wasm_exception_and_deleted(self, mock_subprocess, mock_which):
        """Test compile wasm exception and deleted."""

        def mock_run_side_effect(cmd, **kwargs):
            for arg in cmd:
                if arg.endswith(".cpp"):
                    os.remove(arg)
            raise Exception("compile error")

        mock_which.return_value = "/usr/bin/emcc"
        mock_subprocess.side_effect = mock_run_side_effect
        res = self.gen.compile_wasm()
        self.assertIsNone(res)
