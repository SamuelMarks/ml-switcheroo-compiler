"""Tests for WASM backend coverage."""

import os
import unittest
from unittest.mock import patch

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


class TestWasmCodeGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        from unittest.mock import patch

        self.patcher = patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template")
        self.mock_get_wasm_template = self.patcher.start()

        def mock_template_resolver(template_name):
            if template_name == "kernel_headers":
                return {"body": "#include <wasm_simd128.h>\n#include <cmath>\n#include <cstdlib>\n#include <algorithm>\n"}
            if template_name == "kernel_main_start":
                return {"body": 'extern "C" {{\nvoid main_kernel({params_str}) {{\n'}
            if template_name == "mock_template_Add":
                return {"body": "wasm_f32x4_add(a, b);"}
            if template_name == "mock_template_Subtract" or template_name == "mock_template_Sub":
                return {"body": "wasm_f32x4_sub(a, b);"}
            if template_name == "mock_template_Mul" or template_name == "mock_template_Multiply":
                return {"body": "wasm_f32x4_mul(a, b);"}
            if template_name == "mock_template_Div" or template_name == "mock_template_TrueDivide":
                return {"body": "wasm_f32x4_div(a, b);"}
            if template_name == "mock_template_Minimum":
                return {"body": "wasm_f32x4_min(a, b);"}
            if template_name == "mock_template_Maximum":
                return {"body": "wasm_f32x4_max(a, b);"}
            if template_name == "mock_template_Sqrt":
                return {"body": "wasm_f32x4_sqrt(a);"}
            if template_name == "mock_template_Abs":
                return {"body": "wasm_f32x4_abs(a);"}
            if template_name == "mock_template_Negative" or template_name == "mock_template_Neg":
                return {"body": "wasm_f32x4_neg(a);"}
            if template_name == "mock_template_Relu":
                return {"body": "wasm_f32x4_max(a, zero);"}
            if template_name == "mock_template_Exp":
                return {"body": "std::exp(in0_val);"}
            if template_name == "mock_template_Log":
                return {"body": "std::log(in0_val);"}
            if template_name == "mock_template_Constant":
                return {"body": "wasm_f32x4_splat(a);"}
            if template_name == "mock_template_MatMul":
                return {"body": "// MatMul / DotGeneral Fallback"}
            return {"body": "// mock generated template for " + template_name}

        self.mock_get_wasm_template.side_effect = mock_template_resolver

        import copy

        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        self.saved_registry = copy.deepcopy(OPS_REGISTRY)

        ops_to_mock = [
            "Constant",
            "Add",
            "Subtract",
            "Multiply",
            "TrueDivide",
            "Div",
            "Minimum",
            "Maximum",
            "Sqrt",
            "Abs",
            "Negative",
            "Neg",
            "Exp",
            "Log",
            "Relu",
            "Gelu",
            "Silu",
            "Tanh",
            "ReduceSum",
            "ReduceMean",
            "ReduceMax",
            "ReduceMin",
            "ArgMax",
            "ArgMin",
            "MatMul",
            "Conv2D",
            "ConvTranspose2D",
            "MaxPool2D",
            "AvgPool2D",
            "Reshape",
            "Transpose",
            "Concat",
            "Slice",
            "Gather",
            "Scatter",
            "Softmax",
            "Sin",
            "Cos",
            "NoneShape",
        ]
        for op in ops_to_mock:
            if op not in OPS_REGISTRY:
                OPS_REGISTRY[op] = {"variants": {}}
            if "variants" not in OPS_REGISTRY[op]:
                OPS_REGISTRY[op]["variants"] = {}
            OPS_REGISTRY[op]["variants"]["edge_wasm_simd"] = {"template": "mock_template_" + op}

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

    def tearDown(self):
        self.patcher.stop()
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        OPS_REGISTRY.clear()
        OPS_REGISTRY.update(self.saved_registry)

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
        from ml_switcheroo_compiler.core.errors import CompilationError

        mock_which.return_value = None
        try:
            self.gen.compile_wasm()
        except CompilationError:
            pass
        with self.assertRaises(CompilationError):
            self.gen.compile_wasm()

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_compile_wasm_exception(self, mock_subprocess, mock_which):
        """Test compile wasm exception."""
        from ml_switcheroo_compiler.core.errors import CompilationError

        mock_which.return_value = "/usr/bin/emcc"
        mock_subprocess.side_effect = Exception("compile error")
        with self.assertRaises(CompilationError):
            self.gen.compile_wasm()

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
        from ml_switcheroo_compiler.core.errors import CompilationError

        def mock_run_side_effect(cmd, **kwargs):
            import os

            for arg in cmd:
                if arg.endswith(".cpp"):
                    os.remove(arg)
            raise Exception("compile error")

        mock_which.return_value = "/usr/bin/emcc"
        mock_subprocess.side_effect = mock_run_side_effect
        with self.assertRaises(CompilationError):
            self.gen.compile_wasm()

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
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template") as mock_get:
        mock_get.return_value = {"body": "// MatMul / DotGeneral Fallback"}
        import copy

        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        saved = copy.deepcopy(OPS_REGISTRY)
        try:
            if "MatMul" not in OPS_REGISTRY:
                OPS_REGISTRY["MatMul"] = {"variants": {}}
            if "variants" not in OPS_REGISTRY["MatMul"]:
                OPS_REGISTRY["MatMul"]["variants"] = {}
            OPS_REGISTRY["MatMul"]["variants"]["edge_wasm_simd"] = {"template": "mock_template_MatMul"}

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
        finally:
            OPS_REGISTRY.clear()
            OPS_REGISTRY.update(saved)


def test_wasm_control_flow():
    """Test WhileLoop, Cond, and Scan."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = [1]

    n_while = LogicalNode(id="n_while", op_type="WhileLoop", inputs=["in1"])
    n_while.attributes = {"max_iters": 5}
    n_while.shape_metadata = [1]

    n_cond = LogicalNode(id="n_cond", op_type="Cond", inputs=["in1"])
    n_cond.shape_metadata = [1]

    n_scan = LogicalNode(id="n_scan", op_type="Scan", inputs=["in1"])
    n_scan.shape_metadata = [1]

    graph.nodes = {"in1": n1, "n_while": n_while, "n_cond": n_cond, "n_scan": n_scan}
    graph.inputs = ["in1"]
    graph.outputs = ["n_while", "n_cond", "n_scan"]

    gen = WasmCodeGenerator(graph)
    gen.sorted_nodes = [n1, n_while, n_cond, n_scan]
    code = gen.generate()

    assert "for (int i = 0; i < 5; ++i)" in code
    assert "if (!(buf_in1[0] > 0.0)) break;" in code
    assert "if (buf_in1[0] > 0.0)" in code
    assert "acc_n_scan + buf_in1[i]" in code


def test_wasm_control_flow_no_inputs():
    """Test control flow with no inputs."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    n_while = LogicalNode(id="n_while", op_type="WhileLoop", inputs=[])
    n_while.shape_metadata = [1]
    n_cond = LogicalNode(id="n_cond", op_type="Cond", inputs=[])
    n_cond.shape_metadata = [1]
    n_scan = LogicalNode(id="n_scan", op_type="Scan", inputs=[])
    n_scan.shape_metadata = [1]

    gen = WasmCodeGenerator(graph)
    gen.sorted_nodes = [n_while, n_cond, n_scan]
    code = gen.generate()

    assert "for (int i = 0; i < 10; ++i)" in code
    assert "if (!(1)) break;" in code
    assert "if (1)" in code
    assert "acc = 1.0;" in code


def test_missing_body_error():
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template") as mock_get:
        mock_get.return_value = {}  # Empty template, no body
        import copy

        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        saved = copy.deepcopy(OPS_REGISTRY)
        try:
            OPS_REGISTRY["DummyOpNoBody"] = {"variants": {"edge_wasm_simd": {"template": "mock_template_NoBody"}}}

            n = LogicalNode(id="n_NoBody", op_type="DummyOpNoBody", inputs=["in1"])

            # Mock inputs
            in1 = LogicalNode(id="in1", op_type="Input")
            in1.shape_metadata = 1

            graph = IRGraph()
            gen = WasmCodeGenerator(graph)
            gen.sorted_nodes = [in1, n]
            try:
                gen.generate()
                raise AssertionError("Should raise UnimplementedMathError")
            except UnimplementedMathError:
                pass
        finally:
            OPS_REGISTRY.clear()
            OPS_REGISTRY.update(saved)


def test_unimplemented_math_error():
    """Test unimplemented math error."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    n = LogicalNode(id="n_Unknown", op_type="ReallyUnknownOp", inputs=["in1"])
    graph = IRGraph()
    gen = WasmCodeGenerator(graph)
    gen.sorted_nodes = [n]
    try:
        gen.generate()
        raise AssertionError("Should raise UnimplementedMathError")
    except UnimplementedMathError:
        pass


def test_wasm_compile_finally_missing_file():
    from ml_switcheroo_ir import LogicalGraph

    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator

    g = LogicalGraph()
    gen = WasmCodeGenerator(g)

    from unittest.mock import patch

    from ml_switcheroo_compiler.core.errors import CompilationError

    # We patch tempfile.NamedTemporaryFile to return a mock whose name we immediately delete
    with patch("tempfile.NamedTemporaryFile") as mock_temp, patch("shutil.which") as mock_which:

        class MockFile:
            def __init__(self):
                self.name = "fake_deleted_file.cpp"

            def write(self, data):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        mock_temp.return_value = MockFile()
        mock_which.return_value = None
        try:
            gen.compile_wasm()
        except CompilationError:
            pass

    def test_wasm_conv2d_scalar_shapes(self) -> None:
        """Test wasm conv2d with scalar shapes to hit branches 179-196."""
        graph = IRGraph()
        in1 = LogicalNode(id="in1", op_type="Input")
        in1.shape_metadata = 5
        in2 = LogicalNode(id="in2", op_type="Input")
        in2.shape_metadata = 3

        n_conv = LogicalNode(id="n_conv", op_type="Conv2D", inputs=["in1", "in2"])
        n_conv.shape_metadata = 4
        graph.nodes = {"in1": in1, "in2": in2, "n_conv": n_conv}
        gen = WasmCodeGenerator(graph)
        gen.sorted_nodes = [in1, in2, n_conv]
        gen.generate()

    def test_wasm_conv2d_short_tuple_shapes(self) -> None:
        """Test wasm conv2d with short tuple shapes."""
        graph = IRGraph()
        in1 = LogicalNode(id="in1", op_type="Input")
        in1.shape_metadata = (5, 5)
        in2 = LogicalNode(id="in2", op_type="Input")
        in2.shape_metadata = (3, 3)

        n_conv = LogicalNode(id="n_conv", op_type="Conv2D", inputs=["in1", "in2"])
        n_conv.shape_metadata = (4, 4)
        graph.nodes = {"in1": in1, "in2": in2, "n_conv": n_conv}
        gen = WasmCodeGenerator(graph)
        gen.sorted_nodes = [in1, in2, n_conv]
        gen.generate()

    def test_wasm_conv2d_none_shapes(self) -> None:
        """Test wasm conv2d with None shapes."""
        graph = IRGraph()
        in1 = LogicalNode(id="in1", op_type="Input")
        in1.shape_metadata = None
        in2 = LogicalNode(id="in2", op_type="Input")
        in2.shape_metadata = None

        n_conv = LogicalNode(id="n_conv", op_type="Conv2D", inputs=["in1", "in2"])
        n_conv.shape_metadata = None
        graph.nodes = {"in1": in1, "in2": in2, "n_conv": n_conv}
        gen = WasmCodeGenerator(graph)
        gen.sorted_nodes = [in1, in2, n_conv]
        gen.generate()

    def test_wasm_compile_subprocess_error_has_stderr():
        from ml_switcheroo_ir import LogicalGraph

        from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
        from ml_switcheroo_compiler.core.errors import CompilationError

        g = LogicalGraph()
        gen = WasmCodeGenerator(g)

        import subprocess
        from unittest.mock import patch

        with patch("shutil.which") as mock_which, patch("subprocess.run") as mock_run:
            mock_which.return_value = "/usr/bin/emcc"
            mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="cmd", stderr=b"detailed syntax error")
            try:
                gen.compile_wasm()
                raise AssertionError("Should have raised CompilationError")
            except CompilationError as e:
                assert "detailed syntax error" in str(e)

        with patch("shutil.which") as mock_which, patch("subprocess.run") as mock_run:
            mock_which.return_value = "/usr/bin/emcc"
            mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="cmd", stderr=None)
            try:
                gen.compile_wasm()
                raise AssertionError("Should have raised CompilationError")
            except CompilationError as e:
                assert "WASM compilation failed" in str(e)
