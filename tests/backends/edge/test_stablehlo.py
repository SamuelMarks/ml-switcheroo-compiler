"""Tests for StableHLO backend coverage."""

import unittest

import pytest

from ml_switcheroo_compiler.backends.edge.stablehlo import StableHLOCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


class TestStableHLOCodeGenerator(unittest.TestCase):
    """Test suite for StableHLOCodeGenerator."""

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

    def test_stablehlo_dynamic_shapes(self) -> None:
        """Test dynamic dimension formatting with wildcard ? symbol."""
        gen = StableHLOCodeGenerator(IRGraph())
        self.assertEqual(gen._map_type(("batch", 16), "float32"), "tensor<?x16xf32>")
        self.assertEqual(gen._map_type((None, "?", 3), "int32"), "tensor<?x?x3xi32>")
        self.assertEqual(gen._map_type((-1, 32), "float64"), "tensor<?x32xf64>")

    def test_stablehlo_control_flow_if(self) -> None:
        """Test stablehlo.if emission with then and else branches."""
        g = IRGraph(name="if_graph")
        in_cond = LogicalNode(id="cond", op_type="Input", shape_metadata=(), attributes={"dtype": "bool"})
        in_x = LogicalNode(id="x", op_type="Input", shape_metadata=(2, 2), attributes={"dtype": "float32"})

        then_g = IRGraph()
        inp_t = LogicalNode(id="inp_t", op_type="Input")
        t_node = LogicalNode(id="t_add", op_type="Add", inputs=["x", "x"], shape_metadata=(2, 2))
        then_g.nodes = {"inp_t": inp_t, "t_add": t_node}
        then_g.outputs = ["t_add"]
        then_g.sorted_nodes = [inp_t, t_node]

        else_g = IRGraph()
        inp_e = LogicalNode(id="inp_e", op_type="Input")
        e_node = LogicalNode(id="e_sub", op_type="Subtract", inputs=["x", "x"], shape_metadata=(2, 2))
        else_g.nodes = {"inp_e": inp_e, "e_sub": e_node}
        else_g.outputs = ["e_sub"]
        else_g.sorted_nodes = [inp_e, e_node]

        if_node = LogicalNode(
            id="if_res",
            op_type="If",
            inputs=["cond"],
            shape_metadata=(2, 2),
            attributes={"then_branch": then_g, "else_branch": else_g},
        )

        g.nodes = {"cond": in_cond, "x": in_x, "if_res": if_node}
        g.inputs = ["cond", "x"]
        g.outputs = ["if_res"]

        gen = StableHLOCodeGenerator(g)
        code = gen.generate()

        self.assertIn('"stablehlo.if"', code)
        self.assertIn('"stablehlo.return"', code)

    def test_stablehlo_control_flow_while(self) -> None:
        """Test stablehlo.while emission with condition and body branches."""
        g = IRGraph(name="while_graph")
        in_x = LogicalNode(id="x", op_type="Input", shape_metadata=(1,), attributes={"dtype": "float32"})

        cond_g = IRGraph()
        inp_c = LogicalNode(id="inp_c", op_type="Input")
        c_node = LogicalNode(id="c_lt", op_type="Other", inputs=["x"], shape_metadata=(), attributes={"dtype": "bool"})
        cond_g.nodes = {"inp_c": inp_c, "c_lt": c_node}
        cond_g.outputs = ["c_lt"]
        cond_g.sorted_nodes = [inp_c, c_node]

        body_g = IRGraph()
        inp_b = LogicalNode(id="inp_b", op_type="Input")
        b_node = LogicalNode(id="b_add", op_type="Add", inputs=["x", "x"], shape_metadata=(1,))
        body_g.nodes = {"inp_b": inp_b, "b_add": b_node}
        body_g.outputs = ["b_add"]
        body_g.sorted_nodes = [inp_b, b_node]

        while_node = LogicalNode(
            id="while_res",
            op_type="While",
            inputs=["x"],
            shape_metadata=(1,),
            attributes={"cond": cond_g, "body": body_g},
        )

        g.nodes = {"x": in_x, "while_res": while_node}
        g.inputs = ["x"]
        g.outputs = ["while_res"]

        gen = StableHLOCodeGenerator(g)
        code = gen.generate()

        self.assertIn('"stablehlo.while"', code)
        self.assertIn("^bb0", code)
        self.assertIn('"stablehlo.return"', code)

    def test_stablehlo_control_flow_branches_none(self) -> None:
        """Test If and While emission when branch attributes are None."""
        g = IRGraph(name="empty_branches")
        in_cond = LogicalNode(id="cond", op_type="Input", shape_metadata=(), attributes={"dtype": "bool"})
        in_x = LogicalNode(id="x", op_type="Input", shape_metadata=(2,), attributes={"dtype": "float32"})
        if_node = LogicalNode(id="if_res", op_type="If", inputs=["cond"], shape_metadata=(2,), attributes={})
        while_node = LogicalNode(id="while_res", op_type="While", inputs=["x"], shape_metadata=(2,), attributes={})

        g.nodes = {"cond": in_cond, "x": in_x, "if_res": if_node, "while_res": while_node}
        g.inputs = ["cond", "x"]
        g.outputs = ["if_res", "while_res"]

        gen = StableHLOCodeGenerator(g)
        code = gen.generate()
        self.assertIn('"stablehlo.if"', code)
        self.assertIn('"stablehlo.while"', code)

    def test_stablehlo_custom_call_explicit(self) -> None:
        """Test stablehlo.custom_call emission with call_target_name."""
        g = IRGraph(name="custom_call_graph")
        in_x = LogicalNode(id="x", op_type="Input", shape_metadata=(2, 2))
        cc_node = LogicalNode(
            id="cc_res",
            op_type="CustomCall",
            inputs=["x"],
            shape_metadata=(2, 2),
            attributes={"call_target_name": "MyCustomKernel"},
        )
        g.nodes = {"x": in_x, "cc_res": cc_node}
        g.inputs = ["x"]
        g.outputs = ["cc_res"]

        gen = StableHLOCodeGenerator(g)
        code = gen.generate()

        self.assertIn('"stablehlo.custom_call"', code)
        self.assertIn('call_target_name = "MyCustomKernel"', code)


def test_stablehlo_no_schema():
    """Test StableHLOCodeGenerator behavior when schema file does not exist."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr("os.path.exists", lambda x: False)
        graph = IRGraph()
        gen = StableHLOCodeGenerator(graph)
        assert gen.schema == {}


def test_verify_stablehlo_mlir_with_opt_binary(monkeypatch):
    """Test verify_stablehlo_mlir using mocked mlir-opt binary."""
    import subprocess

    from ml_switcheroo_compiler.backends.edge.stablehlo import verify_stablehlo_mlir

    # Success case
    mock_res_success = subprocess.CompletedProcess(args=["mlir-opt"], returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_res_success)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    assert verify_stablehlo_mlir("some mlir", mlir_opt_path="/usr/bin/mlir-opt") is True

    # Failure case
    mock_res_fail = subprocess.CompletedProcess(args=["mlir-opt"], returncode=1, stdout=b"", stderr=b"parse error")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_res_fail)
    with pytest.raises(ValueError, match="mlir-opt verification failed"):
        verify_stablehlo_mlir("bad mlir", mlir_opt_path="/usr/bin/mlir-opt")


def test_verify_stablehlo_mlir_structural(monkeypatch):
    """Test verify_stablehlo_mlir structural fallback checks without mlir-opt."""
    from ml_switcheroo_compiler.backends.edge.stablehlo import verify_stablehlo_mlir

    monkeypatch.setattr("shutil.which", lambda prog: None)

    # Empty text
    with pytest.raises(ValueError, match="MLIR text is empty"):
        verify_stablehlo_mlir("")
    with pytest.raises(ValueError, match="MLIR text is empty"):
        verify_stablehlo_mlir("   ")

    # Unbalanced braces
    with pytest.raises(ValueError, match="Unbalanced braces"):
        verify_stablehlo_mlir("module { func.func @main() { return }")

    # Missing module or func.func
    with pytest.raises(ValueError, match="MLIR text must declare a module and func.func"):
        verify_stablehlo_mlir("func.func @main() { return }")
    with pytest.raises(ValueError, match="MLIR text must declare a module and func.func"):
        verify_stablehlo_mlir("module { foo() { return } }")

    # Missing return
    with pytest.raises(ValueError, match="MLIR function must contain a return statement"):
        verify_stablehlo_mlir("module { func.func @main() { } }")

    # Valid structural MLIR
    valid_mlir = "module @jit { func.func @main(%arg0: tensor<f32>) -> tensor<f32> { return %arg0 : tensor<f32> } }"
    assert verify_stablehlo_mlir(valid_mlir) is True


def test_stablehlo_generator_verify():
    """Test StableHLOCodeGenerator.verify method."""
    g = IRGraph(name="verify_graph")
    in_x = LogicalNode(id="x", op_type="Input", shape_metadata=(2, 2))
    out_node = LogicalNode(id="out", op_type="Identity", inputs=["x"], shape_metadata=(2, 2))
    g.nodes = {"x": in_x, "out": out_node}
    g.inputs = ["x"]
    g.outputs = ["out"]

    gen = StableHLOCodeGenerator(g)
    # verify when self.code is empty initially
    assert gen.verify() is True
    # verify when self.code is already populated
    assert gen.verify() is True


def test_stablehlo_dot_general_variations():
    """Test dot_general emission under various ranks and explicit attributes."""
    # Explicit dot dimension attributes
    g = IRGraph()
    in_x = LogicalNode(id="x", op_type="Input", shape_metadata=(2, 3, 4))
    in_y = LogicalNode(id="y", op_type="Input", shape_metadata=(2, 4, 5))
    dot_explicit = LogicalNode(
        id="dot_exp",
        op_type="MatMul",
        inputs=["x", "y"],
        shape_metadata=(2, 3, 5),
        attributes={
            "lhs_batching_dimensions": [0],
            "rhs_batching_dimensions": [0],
            "lhs_contracting_dimensions": [2],
            "rhs_contracting_dimensions": [1],
        },
    )
    g.nodes = {"x": in_x, "y": in_y, "dot_exp": dot_explicit}
    g.inputs = ["x", "y"]
    g.outputs = ["dot_exp"]
    gen = StableHLOCodeGenerator(g)
    code = gen.generate()
    assert "lhs_batching_dimensions = [0]" in code
    assert "lhs_contracting_dimensions = [2]" in code

    # Rank >= 3 without explicit attributes
    g3 = IRGraph()
    dot3 = LogicalNode(id="dot3", op_type="MatMul", inputs=["x", "y"], shape_metadata=(2, 3, 5))
    g3.nodes = {"x": in_x, "y": in_y, "dot3": dot3}
    g3.inputs = ["x", "y"]
    g3.outputs = ["dot3"]
    gen3 = StableHLOCodeGenerator(g3)
    code3 = gen3.generate()
    assert "lhs_batching_dimensions = [0]" in code3
    assert "lhs_contracting_dimensions = [2]" in code3

    # Rank 1 without explicit attributes
    g1 = IRGraph()
    in_v1 = LogicalNode(id="v1", op_type="Input", shape_metadata=(4,))
    in_v2 = LogicalNode(id="v2", op_type="Input", shape_metadata=(4,))
    dot1 = LogicalNode(id="dot1", op_type="MatMul", inputs=["v1", "v2"], shape_metadata=())
    g1.nodes = {"v1": in_v1, "v2": in_v2, "dot1": dot1}
    g1.inputs = ["v1", "v2"]
    g1.outputs = ["dot1"]
    gen1 = StableHLOCodeGenerator(g1)
    code1 = gen1.generate()
    assert "lhs_contracting_dimensions = [0]" in code1

    # Empty inputs for dot
    gen_empty = StableHLOCodeGenerator(IRGraph())
    dot_no_inputs = LogicalNode(id="dot_none", op_type="MatMul", inputs=[])
    gen_empty.sorted_nodes = [dot_no_inputs]
    code_no_inputs = gen_empty.generate()
    assert "lhs_contracting_dimensions = [1]" in code_no_inputs


def test_stablehlo_convolution_attribute_variations():
    """Test convolution emission with various stride, dilation, and padding formats."""
    g = IRGraph()
    in_x = LogicalNode(id="x", op_type="Input", shape_metadata=(1, 3, 32, 32))
    in_w = LogicalNode(id="w", op_type="Input", shape_metadata=(16, 3, 3, 3))

    # int stride, int dilation, int padding
    conv1 = LogicalNode(
        id="conv1",
        op_type="Conv2D",
        inputs=["x", "w"],
        shape_metadata=(1, 16, 30, 30),
        attributes={"stride": 2, "dilation": 2, "lhs_dilation": 1, "padding": 1},
    )

    # 2-element padding tuple, list stride & dilation
    conv2 = LogicalNode(
        id="conv2",
        op_type="Conv2D",
        inputs=["x", "w"],
        shape_metadata=(1, 16, 30, 30),
        attributes={"strides": [1, 1], "rhs_dilation": [1, 1], "lhs_dilation": [1, 1], "pads": (2, 3)},
    )

    # 4-element padding tuple
    conv3 = LogicalNode(
        id="conv3",
        op_type="Conv2D",
        inputs=["x", "w"],
        shape_metadata=(1, 16, 30, 30),
        attributes={"padding": (1, 2, 3, 4)},
    )

    # nested list padding with mixed/inner elements
    conv4 = LogicalNode(
        id="conv4",
        op_type="Conv2D",
        inputs=["x", "w"],
        shape_metadata=(1, 16, 30, 30),
        attributes={"padding": [[1, 1], 2]},
    )

    # invalid/string padding
    conv5 = LogicalNode(
        id="conv5",
        op_type="Conv2D",
        inputs=["x", "w"],
        shape_metadata=(1, 16, 30, 30),
        attributes={"padding": "VALID"},
    )

    g.nodes = {"x": in_x, "w": in_w, "conv1": conv1, "conv2": conv2, "conv3": conv3, "conv4": conv4, "conv5": conv5}
    g.inputs = ["x", "w"]
    g.outputs = ["conv1", "conv2", "conv3", "conv4", "conv5"]

    gen = StableHLOCodeGenerator(g)
    code = gen.generate()
    assert '"stablehlo.convolution"' in code


def test_stablehlo_reduce_window_and_pad():
    """Test reduce_window and pad ops emission with all attribute permutations."""
    g = IRGraph()
    in_x = LogicalNode(id="x", op_type="Input", shape_metadata=(1, 28, 28, 1))

    # MaxPool2D with int kernel_size, stride, padding
    max_pool = LogicalNode(
        id="max1",
        op_type="MaxPool2D",
        inputs=["x"],
        shape_metadata=(1, 14, 14, 1),
        attributes={"kernel_size": 2, "stride": 2, "padding": 1},
    )

    # AvgPool2D with 2-element tuple kernel_size, 2-element tuple stride, 2-element padding
    avg_pool = LogicalNode(
        id="avg1",
        op_type="AvgPool2D",
        inputs=["x"],
        shape_metadata=(1, 14, 14, 1),
        attributes={"window_dimensions": (2, 2), "window_strides": (2, 2), "padding": ([1, 1], [1, 1])},
    )

    # ReduceWindow with 4-element lists, custom reducer, explicit init input
    in_init = LogicalNode(id="init_val", op_type="Input", shape_metadata=(), attributes={"dtype": "float32"})
    reduce_win = LogicalNode(
        id="rw1",
        op_type="ReduceWindow",
        inputs=["x", "init_val"],
        shape_metadata=(1, 14, 14, 1),
        attributes={
            "window_dimensions": [1, 2, 2, 1],
            "window_strides": [1, 2, 2, 1],
            "padding": [[0, 0], 1, [2, 2], 0],
            "reducer": "multiply",
            "base_dilations": [1, 1, 1, 1],
            "window_dilations": [1, 1, 1, 1],
        },
    )

    # ReduceWindow with unnested string/other padding, empty inputs
    rw_other = LogicalNode(
        id="rw_other",
        op_type="ReduceWindow",
        inputs=[],
        shape_metadata=(1, 14, 14, 1),
        attributes={"padding": "NONE"},
    )

    # Pad op with explicit pad constant input
    in_pad_val = LogicalNode(id="pad_c", op_type="Input", shape_metadata=(), attributes={"dtype": "float32"})
    pad_op1 = LogicalNode(
        id="pad1",
        op_type="Pad",
        inputs=["x", "pad_c"],
        shape_metadata=(1, 30, 30, 1),
        attributes={
            "edge_padding_low": [0, 1, 1, 0],
            "edge_padding_high": [0, 1, 1, 0],
            "interior_padding": [0, 0, 0, 0],
        },
    )

    # Pad op with constant value attribute, no in_types / no in_vars
    pad_op2 = LogicalNode(
        id="pad2",
        op_type="Pad",
        inputs=[],
        shape_metadata=(1, 32, 32, 1),
        attributes={"value": 0.5},
    )

    # Pad op with scalar shape (rank 0 -> defaults to 4)
    in_scalar = LogicalNode(id="scalar_in", op_type="Input", shape_metadata=())
    pad_op3 = LogicalNode(
        id="pad3",
        op_type="Pad",
        inputs=["scalar_in"],
        shape_metadata=(),
        attributes={"value": -1.0},
    )

    g.nodes = {
        "x": in_x,
        "init_val": in_init,
        "pad_c": in_pad_val,
        "scalar_in": in_scalar,
        "max1": max_pool,
        "avg1": avg_pool,
        "rw1": reduce_win,
        "rw_other": rw_other,
        "pad1": pad_op1,
        "pad2": pad_op2,
        "pad3": pad_op3,
    }
    g.inputs = ["x", "init_val", "pad_c", "scalar_in"]
    g.outputs = ["max1", "avg1", "rw1", "rw_other", "pad1", "pad2", "pad3"]

    gen = StableHLOCodeGenerator(g)
    # Direct tests for elem_t branch variations (tensor<f32> and raw f32)
    node_rw = LogicalNode(id="rw_scalar", op_type="ReduceWindow", inputs=["scalar_in"], shape_metadata=())
    gen._emit_reduce_window(node_rw, "rw_sc", ["%arg0"], ["tensor<f32>"], "tensor<f32>")
    gen._emit_reduce_window(node_rw, "rw_raw", ["%arg0"], ["f32"], "f32")

    node_pad = LogicalNode(id="pad_scalar", op_type="Pad", inputs=["scalar_in"], shape_metadata=())
    gen._emit_pad(node_pad, "pad_sc", ["%arg0"], ["tensor<f32>"], "tensor<f32>")
    gen._emit_pad(node_pad, "pad_raw", ["%arg0"], ["f32"], "f32")

    code = gen.generate()
    assert '"stablehlo.reduce_window"' in code
    assert '"stablehlo.pad"' in code
    assert '"stablehlo.maximum"' in code
    assert '"stablehlo.add"' in code
    assert '"stablehlo.multiply"' in code


def test_stablehlo_while_extra_block_inputs():
    """Test while loop where condition or body has more inputs than in_types, and empty in_types."""
    g = IRGraph(name="while_extra")
    in_x = LogicalNode(id="x", op_type="Input", shape_metadata=(1,), attributes={"dtype": "float32"})

    cond_g = IRGraph()
    inp_c1 = LogicalNode(id="c1", op_type="Input")
    inp_c2 = LogicalNode(id="c2", op_type="Input")
    c_node = LogicalNode(id="c_op", op_type="Other", inputs=["c1"], shape_metadata=(), attributes={"dtype": "bool"})
    cond_g.nodes = {"c1": inp_c1, "c2": inp_c2, "c_op": c_node}
    cond_g.outputs = ["c_op"]
    cond_g.sorted_nodes = [inp_c1, inp_c2, c_node]

    body_g = IRGraph()
    inp_b1 = LogicalNode(id="b1", op_type="Input")
    inp_b2 = LogicalNode(id="b2", op_type="Input")
    b_node = LogicalNode(id="b_op", op_type="Add", inputs=["b1", "b1"], shape_metadata=(1,))
    body_g.nodes = {"b1": inp_b1, "b2": inp_b2, "b_op": b_node}
    body_g.outputs = ["b_op"]
    body_g.sorted_nodes = [inp_b1, inp_b2, b_node]

    while_node = LogicalNode(
        id="while_res",
        op_type="While",
        inputs=["x"],
        shape_metadata=(1,),
        attributes={"cond": cond_g, "body": body_g},
    )

    g.nodes = {"x": in_x, "while_res": while_node}
    g.inputs = ["x"]
    g.outputs = ["while_res"]

    gen = StableHLOCodeGenerator(g)
    gen._emit_while(while_node, "while_test", [], [], "tensor<1xf32>")

    code = gen.generate()
    assert '"stablehlo.while"' in code
