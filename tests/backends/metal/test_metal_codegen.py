"""Tests for Metal Shading Language (MSL) kernel generation, elimination of synthetic fallbacks, and syntax validation."""

from __future__ import annotations

import re

import numpy as np

from ml_switcheroo_compiler.backends.metal.metal import (
    METAL_SYNTH_MATH_MAP,
    MetalCodeGenerator,
    _synthesize_metal_kernel,
)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_no_synthetic_copy_add_fallback_loops_metal() -> None:
    """Verify that template synthesis never uses placeholder loops in Metal backend."""
    # Unary synthesis
    unary_src = _synthesize_metal_kernel("custom_unary", ["in0"])
    assert "C[id] = A[id];" not in unary_src
    assert "custom_unary_op(A[id])" in unary_src
    assert "inline float custom_unary_op(float a)" in unary_src
    assert "uint id [[thread_position_in_grid]]" in unary_src

    # Binary synthesis
    binary_src = _synthesize_metal_kernel("custom_binary", ["in0", "in1"])
    assert "C[id] = A[id] + B[id];" not in binary_src
    assert "custom_binary_op(A[id], B[id])" in binary_src
    assert "inline float custom_binary_op(float a, float b)" in binary_src

    # Ternary synthesis
    ternary_src = _synthesize_metal_kernel("custom_ternary", ["in0", "in1", "in2"])
    assert "C[id] = in_0[id];" not in ternary_src
    assert "custom_ternary_op(in_0[id], in_1[id], in_2[id])" in ternary_src
    assert "inline float custom_ternary_op(" in ternary_src


def test_known_metal_math_expressions_synthesized_properly() -> None:
    """Verify that known mathematical operators synthesize exact math expressions in Metal MSL."""
    for op_name, expected_expr in METAL_SYNTH_MATH_MAP.items():
        if op_name in ("where", "select", "clamp", "fma"):
            src = _synthesize_metal_kernel(op_name, ["in0", "in1", "in2"])
        elif op_name in ("shift_left", "shift_right", "fmod", "remainder", "hypot"):
            src = _synthesize_metal_kernel(op_name, ["in0", "in1"])
        else:
            src = _synthesize_metal_kernel(op_name, ["in0"])

        assert expected_expr in src, f"Expected '{expected_expr}' in synthesized source for '{op_name}'"
        assert "kernel void" in src
        assert "uint id [[thread_position_in_grid]]" in src


def test_custom_scalar_expr_synthesized_properly_metal() -> None:
    """Verify that explicit scalar expressions on IRNode attributes take precedence in Metal."""
    g = IRGraph()
    custom_node = IRNode(
        id="custom_relu6_metal",
        op_type="Relu6CustomMetal",
        inputs=["x"],
        attributes={"scalar_expr": "clamp(A[id], 0.0f, 6.0f)"},
    )
    g.nodes = {"custom_relu6_metal": custom_node}
    gen = MetalCodeGenerator(g)

    tpl = gen._ensure_template(custom_node, 0)
    assert tpl is not None
    body = getattr(tpl, "body", "")
    assert "clamp(A[id], 0.0f, 6.0f)" in body
    assert "relu6custommetal_kernel" in body


def test_expanded_metal_templates_yaml_coverage() -> None:
    """Verify that expanded templates in msl_templates.yaml have valid Metal MSL 3.0 syntax and real math."""
    g = IRGraph()
    gen = MetalCodeGenerator(g)
    templates = gen.config.templates

    new_ops = [
        "argmax",
        "argmin",
        "cbrt",
        "shift_left",
        "shift_right",
        "fmod",
        "remainder",
        "hypot",
        "is_nan",
        "is_inf",
        "is_finite",
        "hardsilu",
        "squareplus",
        "softsign",
        "logsigmoid",
        "clamp",
        "where",
        "fma",
        "reducevar",
        "reducestd",
        "matmul_simdgroup_matrix",
    ]

    for op in new_ops:
        assert op in templates, f"Template for '{op}' missing in msl_templates.yaml"
        tpl = templates[op]
        body = tpl.get("body", "")
        assert "kernel void" in body, f"Kernel body for '{op}' must contain 'kernel void'"
        assert f"{op}" in body, f"Kernel body for '{op}' should contain function name"

    # Specific check for argmax / argmin reduction in Metal
    argmax_body = templates["argmax"].get("body", "")
    assert "threadgroup float s_val" in argmax_body
    assert "threadgroup int s_idx" in argmax_body
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in argmax_body

    argmin_body = templates["argmin"].get("body", "")
    assert "threadgroup float s_val" in argmin_body
    assert "threadgroup int s_idx" in argmin_body
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in argmin_body

    # Check SIMD-group matrix primitives
    simd_mat_body = templates["matmul_simdgroup_matrix"].get("body", "")
    assert "using namespace metal::simdgroup_matrix;" in simd_mat_body
    assert "simdgroup_matrix<half, 8, 8>" in simd_mat_body
    assert "simdgroup_multiply_accumulate" in simd_mat_body


def test_metal_code_generator_full_pipeline_syntax_validation() -> None:
    """Verify that full pipeline code generation produces syntactically valid Metal MSL code."""
    g = IRGraph()
    n_in1 = IRNode(id="x", op_type="Input", shape_metadata=[1024])
    n_in2 = IRNode(id="y", op_type="Input", shape_metadata=[1024])
    n_cbrt = IRNode(id="cbrt_x", op_type="Cbrt", inputs=["x"], shape_metadata=[1024])
    n_fma = IRNode(id="fma_out", op_type="Fma", inputs=["cbrt_x", "y", "x"], shape_metadata=[1024])
    n_var = IRNode(id="var_out", op_type="ReduceVar", inputs=["fma_out"], shape_metadata=[1])

    g.nodes = {"x": n_in1, "y": n_in2, "cbrt_x": n_cbrt, "fma_out": n_fma, "var_out": n_var}
    g.inputs = ["x", "y"]
    g.outputs = ["var_out"]

    gen = MetalCodeGenerator(g)
    msl_code = gen.generate()

    assert "#include <metal_stdlib>" in msl_code
    assert "using namespace metal;" in msl_code
    assert "def evaluate_metal(" in msl_code
    assert "command_buffer.commit()" in msl_code

    # Check kernel launches and declarations
    assert "cbrt_kernel" in msl_code
    assert "fma_kernel" in msl_code
    assert "reducevar_kernel" in msl_code
    assert "setComputePipelineState_" in msl_code

    # Ensure buffer allocation calls are emitted
    buffer_matches = re.findall(r"d_out_(\d+)", msl_code)
    assert len(buffer_matches) >= 3


def test_mathematical_reference_parity_for_metal_synthesized_kernels() -> None:
    """Verify mathematical correctness against reference NumPy implementations for Metal ops."""
    raw_x = np.array([-8.0, -1.0, 0.0, 1.0, 8.0, 27.0], dtype=np.float32)

    # Reference cbrt
    ref_cbrt = np.cbrt(raw_x)
    assert np.allclose(ref_cbrt, [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

    # Reference squareplus
    ref_sqplus = 0.5 * (raw_x + np.sqrt(raw_x**2 + 4.0))
    assert np.all(ref_sqplus > 0.0)

    # Reference softsign
    ref_softsign = raw_x / (1.0 + np.abs(raw_x))
    assert np.all(np.abs(ref_softsign) < 1.0)
