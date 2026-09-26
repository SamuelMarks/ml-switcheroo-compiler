"""Unit tests for WASM SIMD 128-bit intrinsics code generation and scalar polyfills."""

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_wasm_simd_activations_codegen() -> None:
    """Verify WASM SIMD code generation for activation operations."""
    ops = [
        ("relu6", "Relu6", "WASM_RELU6_F32X4"),
        ("leaky_relu", "LeakyRelu", "WASM_LEAKY_RELU_F32X4"),
        ("hardswish", "HardSwish", "WASM_HARDSWISH_F32X4"),
        ("hardsigmoid", "HardSigmoid", "WASM_HARDSIGMOID_F32X4"),
        ("gelu", "GELU", "WASM_GELU_F32X4"),
        ("silu", "SiLU", "WASM_SILU_F32X4"),
        ("softplus", "Softplus", "WASM_SOFTPLUS_F32X4"),
        ("softsign", "Softsign", "WASM_SOFTSIGN_F32X4"),
    ]

    for nid, op_type, macro in ops:
        graph = IRGraph(name=f"test_{op_type}")
        in_node = IRNode(id="in_0", op_type="Input", shape_metadata=(16,))
        op_node = IRNode(id=nid, op_type=op_type, inputs=["in_0"], shape_metadata=(16,))
        graph.nodes = {"in_0": in_node, nid: op_node}
        graph.inputs = ["in_0"]
        graph.outputs = [nid]

        gen = WasmCodeGenerator(graph)
        src = gen.generate()

        assert macro in src
        assert "wasm_v128_load" in src
        assert "wasm_v128_store" in src


def test_wasm_simd_unary_math_codegen() -> None:
    """Verify WASM SIMD code generation for unary math operations."""
    ops = [
        ("square", "Square", "WASM_SQUARE_F32X4"),
        ("cube", "Cube", "WASM_CUBE_F32X4"),
        ("reciprocal", "Reciprocal", "WASM_RECIPROCAL_F32X4"),
        ("rsqrt", "Rsqrt", "WASM_RSQRT_F32X4"),
        ("trunc", "Trunc", "WASM_TRUNC_F32X4"),
        ("sign", "Sign", "WASM_SIGN_F32X4"),
    ]

    for nid, op_type, macro in ops:
        graph = IRGraph(name=f"test_{op_type}")
        in_node = IRNode(id="in_0", op_type="Input", shape_metadata=(16,))
        op_node = IRNode(id=nid, op_type=op_type, inputs=["in_0"], shape_metadata=(16,))
        graph.nodes = {"in_0": in_node, nid: op_node}
        graph.inputs = ["in_0"]
        graph.outputs = [nid]

        gen = WasmCodeGenerator(graph)
        src = gen.generate()

        assert macro in src


def test_wasm_simd_comparisons_and_logic() -> None:
    """Verify WASM SIMD code generation for binary comparisons and logic."""
    ops = [
        ("eq", "Equal", "WASM_EQUAL_F32X4"),
        ("ne", "NotEqual", "WASM_NOTEQUAL_F32X4"),
        ("gt", "Greater", "WASM_GREATER_F32X4"),
        ("ge", "GreaterEqual", "WASM_GREATEREQUAL_F32X4"),
        ("lt", "Less", "WASM_LESS_F32X4"),
        ("le", "LessEqual", "WASM_LESSEQUAL_F32X4"),
        ("land", "LogicalAnd", "WASM_LOGICALAND_F32X4"),
        ("lor", "LogicalOr", "WASM_LOGICALOR_F32X4"),
        ("lxor", "LogicalXor", "WASM_LOGICALXOR_F32X4"),
        ("min_op", "Minimum", "WASM_MINIMUM_F32X4"),
        ("max_op", "Maximum", "WASM_MAXIMUM_F32X4"),
    ]

    for nid, op_type, macro in ops:
        graph = IRGraph(name=f"test_{op_type}")
        in_a = IRNode(id="in_a", op_type="Input", shape_metadata=(16,))
        in_b = IRNode(id="in_b", op_type="Input", shape_metadata=(16,))
        op_node = IRNode(id=nid, op_type=op_type, inputs=["in_a", "in_b"], shape_metadata=(16,))
        graph.nodes = {"in_a": in_a, "in_b": in_b, nid: op_node}
        graph.inputs = ["in_a", "in_b"]
        graph.outputs = [nid]

        gen = WasmCodeGenerator(graph)
        src = gen.generate()

        assert macro in src


def test_wasm_scalar_polyfills_for_unmapped_simd() -> None:
    """Verify sound scalar C++ polyfill generation for ops lacking SIMD intrinsics."""
    graph = IRGraph(name="test_polyfills")
    in_node = IRNode(id="in_0", op_type="Input", shape_metadata=(16,))
    bessel_node = IRNode(id="bessel", op_type="bessel_j0", inputs=["in_0"], shape_metadata=(16,))
    graph.nodes = {"in_0": in_node, "bessel": bessel_node}
    graph.inputs = ["in_0"]
    graph.outputs = ["bessel"]

    gen = WasmCodeGenerator(graph)
    src = gen.generate()

    assert "_scalar_bessel_j0" in src


def test_wasm_while_loop_annotations_and_aliases() -> None:
    """Verify WASM code generation for WhileLoop with loop annotations and Loop alias."""
    graph = IRGraph(name="test_while_loop")
    in_node = IRNode(id="in_0", op_type="Input", shape_metadata=(4,))
    loop_node = IRNode(
        id="loop_op",
        op_type="WhileLoop",
        inputs=["in_0"],
        shape_metadata=(4,),
        attributes={
            "parallel_iterations": 4,
            "swap_memory": True,
            "maximum_iterations": 50,
            "shape_invariants": "[4]",
        },
    )
    graph.nodes = {"in_0": in_node, "loop_op": loop_node}
    graph.inputs = ["in_0"]
    graph.outputs = ["loop_op"]

    gen = WasmCodeGenerator(graph)
    src = gen.generate()

    assert "Loop annotations:" in src
    assert "parallel_iterations=4" in src
    assert "swap_memory=True" in src
    assert "maximum_iterations=50" in src
    assert "shape_invariants=[4]" in src

    # Test Loop alias
    alias_node = IRNode(
        id="loop_alias",
        op_type="Loop",
        inputs=["in_0"],
        shape_metadata=(4,),
    )
    graph_alias = IRGraph(name="test_loop_alias")
    graph_alias.nodes = {"in_0": in_node, "loop_alias": alias_node}
    graph_alias.inputs = ["in_0"]
    graph_alias.outputs = ["loop_alias"]
    gen_alias = WasmCodeGenerator(graph_alias)
    src_alias = gen_alias.generate()
    assert "WhileLoop (ND Tensor State)" in src_alias


def test_wasm_unimplemented_and_binary_simd_fallbacks() -> None:
    """Verify UnimplementedMathError on unknown SIMD op and binary SIMD scalar fallbacks."""
    import pytest

    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    # 1. UnimplementedMathError
    graph_unknown = IRGraph(name="test_unknown")
    in_node = IRNode(id="in_0", op_type="Input", shape_metadata=(4,))
    unknown_node = IRNode(id="unknown", op_type="CompletelyUnknownOpXYZ", inputs=["in_0"], shape_metadata=(4,))
    graph_unknown.nodes = {"in_0": in_node, "unknown": unknown_node}
    graph_unknown.inputs = ["in_0"]
    graph_unknown.outputs = ["unknown"]

    gen_unknown = WasmCodeGenerator(graph_unknown)
    with pytest.raises(UnimplementedMathError, match="Missing WASM SIMD template"):
        gen_unknown.generate()

    # 2. Binary SIMD op with scalar fallback without intrinsic (atan2 from intrinsics scalars)
    graph_binary = IRGraph(name="test_binary_simd")
    in_a = IRNode(id="in_a", op_type="Input", shape_metadata=(4,))
    in_b = IRNode(id="in_b", op_type="Input", shape_metadata=(4,))
    node = IRNode(id="atan2_op", op_type="atan2", inputs=["in_a", "in_b"], shape_metadata=(4,))
    graph_binary.nodes = {"in_a": in_a, "in_b": in_b, "atan2_op": node}
    graph_binary.inputs = ["in_a", "in_b"]
    graph_binary.outputs = ["atan2_op"]

    gen_binary = WasmCodeGenerator(graph_binary)
    src_binary = gen_binary.generate()
    assert "_scalar_atan2" in src_binary

    # 3. Direct invocation of _generate_binary_simd_op for fallback binary_scalars mapping
    gen_direct = WasmCodeGenerator(graph_binary)
    gen_direct._generate_binary_simd_op(node, "add", "add_direct", ["in_a", "in_b"], (4,), 4)
    assert any("buf_in_a[i_add_direct] + buf_in_b[i_add_direct]" in line for line in gen_direct.code)


def test_wasm_edge_branches_intrinsics_and_templates() -> None:
    """Verify edge branch coverage: WhileLoop no inputs, missing YAML, template missing body, and scalars dict fallback."""
    from unittest.mock import patch

    import pytest

    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    # 1. WhileLoop with no inputs and annotations
    node_loop_no_inputs = IRNode(
        id="loop_no_inputs",
        op_type="WhileLoop",
        inputs=[],
        shape_metadata=(4,),
        attributes={
            "parallel_iterations": 8,
            "swap_memory": False,
            "maximum_iterations": 20,
            "shape_invariants": "[1, 2]",
        },
    )
    g_loop = IRGraph(name="test_while_no_inputs")
    g_loop.nodes = {"loop_no_inputs": node_loop_no_inputs}
    g_loop.outputs = ["loop_no_inputs"]
    gen_loop = WasmCodeGenerator(g_loop)
    src_loop = gen_loop.generate()
    assert "Loop annotations:" in src_loop
    assert "buf_dummy" in src_loop

    # 1b. WhileLoop with max_iters=None and empty annotations
    node_loop_none = IRNode(
        id="loop_none",
        op_type="WhileLoop",
        inputs=[],
        shape_metadata=(4,),
        attributes={"max_iters": None},
    )
    g_none = IRGraph(name="test_while_none")
    g_none.nodes = {"loop_none": node_loop_none}
    g_none.outputs = ["loop_none"]
    gen_none = WasmCodeGenerator(g_none)
    src_none = gen_none.generate()
    assert "Loop annotations:" not in src_none

    # 2. yaml_path does not exist branch
    with patch("os.path.exists", return_value=False):
        g_missing_yaml = IRGraph(name="t")
        in_0 = IRNode(id="in_0", op_type="Input", shape_metadata=(4,))
        node_unmapped = IRNode(id="op", op_type="NonExistentOpWithoutTemplate", inputs=["in_0"], shape_metadata=(4,))
        g_missing_yaml.nodes = {"in_0": in_0, "op": node_unmapped}
        g_missing_yaml.inputs = ["in_0"]
        g_missing_yaml.outputs = ["op"]
        gen_missing = WasmCodeGenerator(g_missing_yaml)
        with pytest.raises(UnimplementedMathError, match="Missing WASM SIMD template"):
            gen_missing.generate()

    # 3. Template without 'body' key
    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template", return_value={"no_body": "x"}):
        g_no_body = IRGraph(name="t_no_body")
        in_0 = IRNode(id="in_0", op_type="Input", shape_metadata=(4,))
        node_alibi = IRNode(id="op", op_type="ALiBi", inputs=["in_0"], shape_metadata=(4,))
        g_no_body.nodes = {"in_0": in_0, "op": node_alibi}
        g_no_body.inputs = ["in_0"]
        g_no_body.outputs = ["op"]
        gen_no_body = WasmCodeGenerator(g_no_body)
        with pytest.raises(UnimplementedMathError, match="MISSING BODY FOR: ALiBi"):
            gen_no_body.generate()

    # 4. Binary SIMD op with intrinsic without scalar_fallback falling back to scalars dict
    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.config_models.WasmIntrinsicsConfig.model_dump") as mock_dump:
        mock_dump.return_value = {
            "intrinsics": {
                "custom_simd_op": {
                    "macro_name": "WASM_CUSTOM_F32X4",
                }
            },
            "scalars": {
                "custom_simd_op": "custom_func",
            },
        }
        g_custom = IRGraph(name="t_custom")
        in_a = IRNode(id="in_a", op_type="Input", shape_metadata=(4,))
        in_b = IRNode(id="in_b", op_type="Input", shape_metadata=(4,))
        node_custom = IRNode(id="op", op_type="custom_simd_op", inputs=["in_a", "in_b"], shape_metadata=(4,))
        g_custom.nodes = {"in_a": in_a, "in_b": in_b, "op": node_custom}
        g_custom.inputs = ["in_a", "in_b"]
        g_custom.outputs = ["op"]
        gen_custom = WasmCodeGenerator(g_custom)
        src_custom = gen_custom.generate()
        assert "_scalar_custom_simd_op" in src_custom
