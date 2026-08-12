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

        for op in ["Constant", "Add", "Subtract", "Multiply", "TrueDivide", "Div", "Minimum", "Maximum", "Sqrt", "Abs", "Negative", "Neg", "Exp", "Log", "Relu"]:
            inputs = ["in1", "in1"]
            n = LogicalNode(id="n_" + op, op_type=op, inputs=inputs)
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
        self.assertIn("wasm_f32x4_min", code)  # Minimum, Minimum mapped to min
        self.assertIn("wasm_f32x4_max", code)  # Maximum, Maximum mapped to max
        self.assertIn("wasm_f32x4_sqrt", code)
        self.assertIn("wasm_f32x4_abs", code)
        self.assertIn("wasm_f32x4_neg", code)
        self.assertIn("std::exp(in0_val", code)
        self.assertIn("std::log(in0_val", code)

        # Check scalar fallback inline
        self.assertIn("std::min(in0_val, in1_val)", code)
        self.assertIn("-in0_val", code)

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

    def test_wasm_neural_networks_and_reductions(self) -> None:
        """Test wasm neural networks and reductions operations generation."""
        ops = ["Relu", "Gelu", "Silu", "Tanh", "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ArgMax", "ArgMin", "MatMul", "Conv2D", "ConvTranspose2D", "MaxPool2D", "AvgPool2D", "Reshape", "Transpose", "Concat", "Slice", "Gather", "Scatter", "Softmax", "Sin", "Cos"]
        for op in ops:
            n = LogicalNode(id="n_" + op, op_type=op, inputs=["in1", "in1"])
            self.gen.sorted_nodes.append(n)

        # Test full code generation with all these ops
        self.gen.generate()

    def test_wasm_striding_and_alloc(self) -> None:
        """Test wasm memory alloc and striding."""
        assert "aligned_alloc" in self.gen._allocate_aligned_memory(1024)

        strides, code = self.gen._generate_striding_logic([2, 3, 4])
        assert strides == [12, 4, 1]
        assert "idx % 4" in code

        strides_empty, code_empty = self.gen._generate_striding_logic([])
        assert strides_empty == []
        assert code_empty == "0"

    def test_wasm_scalar_shapes(self) -> None:
        """Test wasm with scalar shapes."""
        graph = IRGraph()
        n1 = LogicalNode(id="n1", op_type="Input")
        n1.shape_metadata = 5.0
        n2 = LogicalNode(id="n2", op_type="Add", inputs=["n1", "n1"])
        n2.shape_metadata = 5
        n3 = LogicalNode(id="n3", op_type="Add", inputs=["n1", "n1"])
        n3.shape_metadata = None
        graph.nodes = {"n1": n1, "n2": n2, "n3": n3}
        gen = WasmCodeGenerator(graph)
        gen.generate()

    def test_wasm_scalar_shapes2(self) -> None:
        """Test wasm with scalar shapes for matmul and reducesum."""
        graph = IRGraph()
        n1 = LogicalNode(id="n1", op_type="Input")
        n1.shape_metadata = 5.0
        n2 = LogicalNode(id="n2", op_type="MatMul", inputs=["n1", "n1"])
        n3 = LogicalNode(id="n3", op_type="ReduceSum", inputs=["n1"])
        graph.nodes = {"n1": n1, "n2": n2, "n3": n3}
        gen = WasmCodeGenerator(graph)
        gen.generate()


def test_wasm_branch_coverage():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    g = IRGraph()
    n = LogicalNode(id="dummy", op_type="MatMul", inputs=["in1", "in2"])
    n.shape_metadata = 1

    in1 = LogicalNode(id="in1", op_type="Input")
    in1.shape_metadata = 1
    in2 = LogicalNode(id="in2", op_type="Input")
    in2.shape_metadata = 1

    g.nodes = {"in1": in1, "in2": in2, "dummy": n}
    g.inputs = ["in1", "in2"]
    g.outputs = ["dummy"]

    gen = WasmCodeGenerator(g)
    code = gen.generate()
    assert "MatMul / DotGeneral Fallback" in code
