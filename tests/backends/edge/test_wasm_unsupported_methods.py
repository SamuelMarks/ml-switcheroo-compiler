def test_wasm_missing_methods_extra():
    """Test missing NN and math methods in WASM."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_linear = IRNode(id="n_linear", op_type="Linear", inputs=["dummy"], shape_metadata=[2, 2])
    n_attn = IRNode(id="n_attn", op_type="Attention", inputs=["dummy"], shape_metadata=[2, 2])
    n_maxpool = IRNode(id="n_maxpool", op_type="MaxPool", inputs=["dummy"], shape_metadata=[1, 2, 2, 1])
    n_layernorm = IRNode(id="n_layernorm", op_type="LayerNorm", inputs=["dummy"], shape_metadata=[2, 2])
    n_trig = IRNode(id="n_trig", op_type="Trig", inputs=["dummy"], shape_metadata=[2, 2])
    n_reducesum = IRNode(id="n_reducesum", op_type="ReduceSum", inputs=["dummy"], shape_metadata=[2, 2])
    n_exp = IRNode(id="n_exp", op_type="Exp", inputs=["dummy"], shape_metadata=[2, 2])
    n_log = IRNode(id="n_log", op_type="Log", inputs=["dummy"], shape_metadata=[2, 2])
    n_tanh = IRNode(id="n_tanh", op_type="Tanh", inputs=["dummy"], shape_metadata=[2, 2])
    n_sigmoid = IRNode(id="n_sigmoid", op_type="Sigmoid", inputs=["dummy"], shape_metadata=[2, 2])

    for n in [n_linear, n_attn, n_maxpool, n_layernorm, n_trig, n_reducesum, n_exp, n_log, n_tanh, n_sigmoid]:
        g.nodes[n.id] = n

    g.inputs = ["dummy"]
    g.outputs = ["n_linear"]

    gen = WasmCodeGenerator(g)
    code = gen.generate()

    assert "buf_n_linear" in code
    assert "buf_n_attn" in code
    assert "buf_n_maxpool" in code
    assert "buf_n_layernorm" in code
    assert "buf_n_trig" in code
    assert "sum_n_reducesum" in code

    # Check bounds checking
    assert "if (4 > size) return; // Out of bounds" in code


def test_wasm_line_coverage_extra():
    """Test line coverage for shape edge cases in WASM."""
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_conv2d = IRNode(id="n_conv2d", op_type="Conv2D", inputs=["n_dummy", "n_dummy_none"], shape_metadata=[])
    n_dummy = IRNode(id="n_dummy", op_type="Input", inputs=[], shape_metadata=[])
    n_dummy_none = IRNode(id="n_dummy_none", op_type="Input", inputs=[], shape_metadata=None)
    n_avgpool = IRNode(id="n_avgpool", op_type="AvgPool2D", inputs=["n_dummy_none"], shape_metadata=[])

    n_matmul_int = IRNode(id="n_matmul_int", op_type="MatMul", inputs=["n_dummy_int", "n_dummy_int"], shape_metadata=[])
    n_dummy_int = IRNode(id="n_dummy_int", op_type="Input", inputs=[], shape_metadata=1)

    n_cond_unmapped = IRNode(id="n_cond", op_type="Cond", inputs=["dummy"], shape_metadata=[])
    n_cond_unmapped.attributes["then_branch"] = IRGraph()
    n_cond_unmapped.attributes["then_branch"].inputs = ["not_in_parent"]
    n_cond_unmapped.attributes["then_branch"].nodes["sub"] = IRNode("sub", "Add", ["not_in_parent"])

    for n in [n_conv2d, n_dummy, n_dummy_none, n_avgpool, n_matmul_int, n_dummy_int, n_cond_unmapped]:
        g.nodes[n.id] = n

    gen = WasmCodeGenerator(g)

    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template", return_value={"body": ""}):
        gen.generate()

    # Test emcc fallback
    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template", return_value={"body": ""}):
        with patch("shutil.which", side_effect=lambda x: "emcc" if x == "emcc" else None):
            with patch("subprocess.run", return_value=MagicMock()):
                gen.compile_wasm("dummy_out")
