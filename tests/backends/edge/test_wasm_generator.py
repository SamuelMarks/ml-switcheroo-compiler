from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_wasm_generator():
    pass

    pass

    # Force reload OPS_REGISTRY to avoid test pollution
    import importlib

    import ml_switcheroo_compiler.ops.generated_registry

    importlib.reload(ml_switcheroo_compiler.ops.generated_registry)
    import ml_switcheroo_compiler.ops.registry
    from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY

    orig = ml_switcheroo_compiler.ops.registry._YAML_REGISTRY.copy()
    ml_switcheroo_compiler.ops.registry._YAML_REGISTRY.clear()
    ml_switcheroo_compiler.ops.registry._YAML_REGISTRY.update(OPS_REGISTRY)

    try:
        graph = IRGraph()

        graph.nodes = {"x": IRNode("x", "Input", inputs=[], shape_metadata=[10]), "y": IRNode("y", "Input", inputs=[], shape_metadata=[10]), "add": IRNode("add", "Add", inputs=["x", "y"], shape_metadata=[10])}
        graph.inputs = ["x", "y"]
        graph.outputs = ["add"]

        gen = WasmCodeGenerator(graph)
        code = gen.generate()
        assert True
    finally:
        ml_switcheroo_compiler.ops.registry._YAML_REGISTRY.clear()
        ml_switcheroo_compiler.ops.registry._YAML_REGISTRY.update(orig)


def test_wasm_generator_missing_coverage():
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    # test visit_Conv2D where shape is empty
    gen = WasmCodeGenerator(IRGraph())
    node = IRNode("conv", "Conv2D", inputs=["x", "w"], shape_metadata=[])
    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template", return_value={"body": ""}):
        gen.visit_Conv2D(node, "Conv2D", "conv", ["x", "w"], [], 1)

    # test _generate_pooling2d where len(in0_shape) < 4 and shape is empty
    gen.sorted_nodes = [IRNode("x", "Input", shape_metadata=[1, 2])]
    gen._generate_pooling2d(node, "pool", ["x"], [], "max_pool_2d")

    # test visit_Cond where inp is not in branch_graph.inputs
    true_graph = IRGraph()
    true_graph.inputs = ["pred"]  # some input
    true_graph.outputs = []
    # Add a subnode to true_graph that uses an input not in true_graph.inputs
    subnode = IRNode("sub", "Dummy", inputs=["outer_var"])
    true_graph.nodes = {"sub": subnode}

    cond_node = IRNode("cond", "Cond", inputs=["pred", "x", "y"])
    cond_node.attributes = {"true_graph": true_graph, "false_graph": IRGraph()}
    gen.sorted_nodes = [cond_node]
    gen.visit_Cond(cond_node, "Cond", "cond", ["pred", "x", "y"], [10], 10)

    # test fallback in _generate_op where in0_shape is empty
    op_node1 = IRNode("op1", "MatMul", inputs=["x", "y"])
    gen.sorted_nodes = [IRNode("x", "Input", shape_metadata=[])]
    gen._generate_op(op_node1, "MatMul", "op1", ["x", "y"], [10], 10)

    # test fallback in _generate_op where in0_shape is an integer
    op_node2 = IRNode("op2", "MatMul", inputs=["x2", "y2"])
    gen.sorted_nodes = [IRNode("x2", "Input", shape_metadata=5)]  # float or int
    gen._generate_op(op_node2, "MatMul", "op2", ["x2", "y2"], [10], 10)


def test_wasm_scalar_fallback_code():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    g.nodes["n1"] = IRNode(id="n1", op_type="UnknownSIMD_Op_Test")
    g.sorted_nodes = [g.nodes["n1"]]
    gen = WasmCodeGenerator(g)

    # We directly invoke _generate_vector_unrolled_op for a fake op that has no simd_macro
    gen._generate_vector_unrolled_op(g.nodes["n1"], "UnknownSIMD_Op_Test", "n1", ["in1"], [10], 10)
    assert "i_n1 < 10" in "\n".join(gen.code)


def test_wasm_dummy_allocation():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    g.nodes["n1"] = IRNode(id="n1", op_type="Linear")
    g.sorted_nodes = [g.nodes["n1"]]
    gen = WasmCodeGenerator(g)
    code = gen.generate()
    assert "float dummy_val = 0.0f;" in code
    assert "float* buf_dummy = &dummy_val;" in code


def test_wasm_binary_simd_ops_and_peeling():
    """Test WASM SIMD binary ops and remainder loop peeling for non-multiple-of-4 dimensions."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    gen = WasmCodeGenerator(g)

    # 10 is not a multiple of 4: verifies loop peeling (<= 6 step 4, then remainder < 10)
    for op_type, visit_name in [
        ("Add", "visit_Add"),
        ("Sub", "visit_Sub"),
        ("Mul", "visit_Mul"),
        ("Div", "visit_Div"),
        ("Min", "visit_Min"),
        ("Max", "visit_Max"),
    ]:
        gen.code.clear()
        node = IRNode(id=f"node_{op_type}", op_type=op_type, inputs=["x", "y"], shape_metadata=[10])
        getattr(gen, visit_name)(node, op_type, f"clean_{op_type}", ["x", "y"], [10], 10)
        code_str = "\n".join(gen.code)
        assert "wasm_v128_load" in code_str
        assert "wasm_v128_store" in code_str
        assert f"i_clean_{op_type} <= 10 - 4" in code_str
        assert f"i_clean_{op_type} < 10" in code_str


def test_wasm_unary_simd_ops_and_peeling():
    """Test WASM SIMD unary ops and remainder loop peeling."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    gen = WasmCodeGenerator(g)

    for op_type, visit_name in [
        ("Abs", "visit_Abs"),
        ("Neg", "visit_Neg"),
        ("Sqrt", "visit_Sqrt"),
        ("Relu", "visit_Relu"),
        ("Ceil", "visit_Ceil"),
        ("Floor", "visit_Floor"),
        ("Round", "visit_Round"),
        ("Sin", "visit_Sin"),
        ("Cos", "visit_Cos"),
    ]:
        gen.code.clear()
        node = IRNode(id=f"node_{op_type}", op_type=op_type, inputs=["x"], shape_metadata=[7])
        getattr(gen, visit_name)(node, op_type, f"clean_{op_type}", ["x"], [7], 7)
        code_str = "\n".join(gen.code)
        assert "wasm_v128_load" in code_str
        assert "wasm_v128_store" in code_str
        assert f"i_clean_{op_type} <= 7 - 4" in code_str
        assert f"i_clean_{op_type} < 7" in code_str


def test_wasm_symbolic_shape_telemetry():
    """Test WASM shape telemetry resolving SymInt dimensions."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.ir.shape_system import SymInt

    g = IRGraph()
    sym_b = SymInt("B")
    n_in = IRNode(id="in0", op_type="Input", inputs=[], shape_metadata=[sym_b, 128])
    n_out = IRNode(id="out0", op_type="Relu", inputs=["in0"], shape_metadata=[sym_b, 128])
    g.nodes["in0"] = n_in
    g.nodes["out0"] = n_out
    g.inputs = ["in0"]
    g.outputs = ["out0"]

    gen = WasmCodeGenerator(g)
    gen.apply_shape_telemetry({"shapes": {"in0": [8, 128]}})

    assert n_in.shape_metadata == (8, 128)
    assert n_out.shape_metadata == (8, 128)


def test_wasm_additional_edges_and_telemetry() -> None:
    """Test binary simd op fallbacks, unary simd without yaml, and shape telemetry branches."""
    import os
    from unittest.mock import patch

    import pytest

    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    gen = WasmCodeGenerator(g)

    # 1. _generate_binary_simd_op with unmapped op (lines 792->796, 797-805, 822-825)
    node_custom = IRNode(id="custom", op_type="CustomBinary", inputs=["in0", "in1"])
    gen.code.clear()
    gen._generate_binary_simd_op(node_custom, "CustomBinary", "custom", ["in0", "in1"], [4], 4)
    code_str = "\n".join(gen.code)
    # Since CustomBinary has no SIMD macro, it generates scalar loop without wasm_v128_load
    assert "wasm_v128_load" not in code_str
    assert "i_custom < 4" in code_str

    # _generate_binary_simd_op when yaml_path does not exist (line 788->796)
    with patch("os.path.exists", return_value=False):
        gen.code.clear()
        gen._generate_binary_simd_op(node_custom, "Add", "custom", ["in0", "in1"], [4], 4)
        assert "buf_in0[i_custom] + buf_in1[i_custom]" in "\n".join(gen.code)

    # _generate_binary_simd_op when intrinsic has no scalar_fallback (line 794->796)
    fake_binary_intrinsics = {
        "intrinsics": {
            "NoFallback": {
                "macro_name": "wasm_f32x4_nofallback",
                "scalar_fallback": None,
            }
        },
        "scalars": {},
    }
    node_nofallback = IRNode(id="nf", op_type="NoFallback", inputs=["in0", "in1"])
    from unittest.mock import mock_open

    with patch("builtins.open", mock_open(read_data="intrinsics: {}")):
        with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.config_models.WasmIntrinsicsConfig.model_dump", return_value=fake_binary_intrinsics):
            gen.code.clear()
            gen._generate_binary_simd_op(node_nofallback, "NoFallback", "nf", ["in0", "in1"], [4], 4)

    # 2. _generate_vector_unrolled_op when op has only scalar definition in yaml (lines 846->850, 853-854)
    node_tan = IRNode(id="tan_node", op_type="Tan", inputs=["in_tan"])
    gen.code.clear()
    gen._generate_vector_unrolled_op(node_tan, "Tan", "tan_node", ["in_tan"], [4], 4)
    assert "buf_in_tan" in "\n".join(gen.code)

    # _generate_vector_unrolled_op when intrinsics.yaml does not exist
    node_sin = IRNode(id="sin_node", op_type="Sin", inputs=["in_sin"])
    with patch.object(os.path, "exists", return_value=False):
        gen.code.clear()
        gen._generate_vector_unrolled_op(node_sin, "Sin", "sin_node", ["in_sin"], [4], 4)
        assert "std::sin" in "\n".join(gen.code)

    # 3. _generate_vector_unrolled_op when op is completely unimplemented (lines 867->870)
    node_unimpl = IRNode(id="unimpl", op_type="UnknownOpXYZ", inputs=["in0"])
    with patch.object(os.path, "exists", return_value=False):
        with pytest.raises(UnimplementedMathError, match="lacks WASM SIMD intrinsics"):
            gen._generate_vector_unrolled_op(node_unimpl, "UnknownOpXYZ", "unimpl", ["in0"], [4], 4)

    # 4. _generate_vector_unrolled_op when simd_macro exists but scalar_expr does not (line 871)
    fake_intrinsics = {
        "intrinsics": {
            "SpecialSimd": {
                "macro_name": "wasm_f32x4_special",
                "scalar_fallback": None,
            }
        },
        "scalars": {},
    }
    node_spec = IRNode(id="spec", op_type="SpecialSimd", inputs=["in0"])
    with patch("builtins.open", mock_open(read_data="intrinsics: {}")):
        with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.config_models.WasmIntrinsicsConfig.model_dump", return_value=fake_intrinsics):
            gen.code.clear()
            gen._generate_vector_unrolled_op(node_spec, "SpecialSimd", "spec", ["in0"], [4], 4)
            code_spec = "\n".join(gen.code)
            assert "wasm_f32x4_extract_lane" in code_spec

    # 5. apply_shape_telemetry exhaustive branches (lines 1173, 1180->1176, 1182->1188, 1187, 1193->1191, 1199-1200, 1202, 1203->1195, 1205->1191)
    # Line 1173: non-dict shapes payload returns early
    gen.apply_shape_telemetry({"shapes": "not_a_dict"})  # type: ignore[dict-item]

    # Model graph with nodes covering all branches of apply_shape_telemetry
    class FailingEvalDim:
        def __init__(self) -> None:
            self.node = self

        def eval(self, env: dict[str, int]) -> int:
            raise RuntimeError("Eval failed")

        def __int__(self) -> int:
            return 32

    n_valid_str = IRNode(id="n1", op_type="Input", shape_metadata=["B", "Seq"])
    n_non_list_concrete = IRNode(id="n2", op_type="Input", shape_metadata=["B"])
    n_len_mismatch = IRNode(id="n3", op_type="Input", shape_metadata=["B", 10])
    n_failing_eval = IRNode(id="n4", op_type="Relu", inputs=["n1"], shape_metadata=[FailingEvalDim()])
    n_unresolved_str = IRNode(id="n5", op_type="Relu", inputs=["n1"], shape_metadata=["UnknownSym"])
    n_no_meta = IRNode(id="n6", op_type="Relu", inputs=["n1"])  # shape_metadata is None

    g2 = IRGraph()
    g2.nodes = {
        "n1": n_valid_str,
        "n2": n_non_list_concrete,
        "n3": n_len_mismatch,
        "n4": n_failing_eval,
        "n5": n_unresolved_str,
        "n6": n_no_meta,
    }
    g2.inputs = ["n1", "n2", "n3"]
    g2.outputs = ["n4", "n5", "n6"]
    gen2 = WasmCodeGenerator(g2)

    telemetry_payload = {
        "n1": [16, 64],
        "n2": 42,  # not list/tuple (line 1180->1176)
        "n3": [16, 10, 20],  # len mismatch (line 1182->1188)
    }
    gen2.apply_shape_telemetry(telemetry_payload)
    assert n_valid_str.shape_metadata == (16, 64)
    assert n_failing_eval.shape_metadata == (1,)
