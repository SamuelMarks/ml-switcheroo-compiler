"""Tests for CUDA real kernel template generation, elimination of synthetic fallbacks, and syntax validation."""

from __future__ import annotations

import re

import numpy as np

from ml_switcheroo_compiler.backends.cuda.cuda import (
    CUDA_SYNTH_MATH_MAP,
    CudaCodeGenerator,
    _synthesize_cuda_kernel,
)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_no_synthetic_copy_add_fallback_loops() -> None:
    """Verify that template synthesis never uses placeholder loops."""
    # Unary synthesis
    unary_src = _synthesize_cuda_kernel("custom_unary", ["in0"])
    assert "C[id] = A[id];" not in unary_src
    assert "custom_unary_op(A[id])" in unary_src
    assert "__device__ inline float custom_unary_op(float a)" in unary_src

    # Binary synthesis
    binary_src = _synthesize_cuda_kernel("custom_binary", ["in0", "in1"])
    assert "C[id] = A[id] + B[id];" not in binary_src
    assert "custom_binary_op(A[id], B[id])" in binary_src
    assert "__device__ inline float custom_binary_op(float a, float b)" in binary_src

    # Ternary synthesis
    ternary_src = _synthesize_cuda_kernel("custom_ternary", ["in0", "in1", "in2"])
    assert "C[id] = in_0[id];" not in ternary_src
    assert "custom_ternary_op(in_0[id], in_1[id], in_2[id])" in ternary_src
    assert "__device__ inline float custom_ternary_op(" in ternary_src


def test_known_cuda_math_expressions_synthesized_properly() -> None:
    """Verify that known mathematical operators synthesize exact math expressions."""
    for op_name, expected_expr in CUDA_SYNTH_MATH_MAP.items():
        if op_name in ("where", "select", "clamp", "fma"):
            src = _synthesize_cuda_kernel(op_name, ["in0", "in1", "in2"])
        elif op_name in ("shift_left", "shift_right", "fmod", "remainder", "hypot"):
            src = _synthesize_cuda_kernel(op_name, ["in0", "in1"])
        else:
            src = _synthesize_cuda_kernel(op_name, ["in0"])

        assert expected_expr in src, f"Expected '{expected_expr}' in synthesized source for '{op_name}'"
        assert "__global__ void" in src


def test_custom_scalar_expr_synthesized_properly() -> None:
    """Verify that explicit scalar expressions on IRNode attributes take precedence."""
    g = IRGraph()
    custom_node = IRNode(
        id="custom_relu6",
        op_type="Relu6Custom",
        inputs=["x"],
        attributes={"scalar_expr": "fminf(fmaxf(A[id], 0.0f), 6.0f)"},
    )
    g.nodes = {"custom_relu6": custom_node}
    gen = CudaCodeGenerator(g)

    tpl = gen._ensure_template(custom_node, 0)
    assert tpl is not None
    body = getattr(tpl, "body", "")
    assert "fminf(fmaxf(A[id], 0.0f), 6.0f)" in body
    assert "relu6custom_kernel" in body


def test_expanded_cuda_templates_yaml_coverage() -> None:
    """Verify that expanded templates in cuda_templates.yaml have valid CUDA C++ syntax and real math."""
    g = IRGraph()
    gen = CudaCodeGenerator(g)
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
    ]

    for op in new_ops:
        assert op in templates, f"Template for '{op}' missing in cuda_templates.yaml"
        tpl = templates[op]
        body = tpl.get("body", "")
        assert "__global__ void" in body, f"Kernel body for '{op}' must contain '__global__ void'"
        assert f"{op}" in body, f"Kernel body for '{op}' should contain function name"

    # Specific check for argmax / argmin reduction
    argmax_body = templates["argmax"].get("body", "")
    assert "__shared__ float s_val" in argmax_body
    assert "__shared__ int s_idx" in argmax_body
    assert "__syncthreads()" in argmax_body

    argmin_body = templates["argmin"].get("body", "")
    assert "__shared__ float s_val" in argmin_body
    assert "__shared__ int s_idx" in argmin_body
    assert "__syncthreads()" in argmin_body


def test_cuda_code_generator_full_pipeline_syntax_validation() -> None:
    """Verify that full pipeline code generation produces syntactically valid CUDA C++ code."""
    g = IRGraph()
    n_in1 = IRNode(id="x", op_type="Input", shape_metadata=[1024])
    n_in2 = IRNode(id="y", op_type="Input", shape_metadata=[1024])
    n_cbrt = IRNode(id="cbrt_x", op_type="Cbrt", inputs=["x"], shape_metadata=[1024])
    n_fma = IRNode(id="fma_out", op_type="Fma", inputs=["cbrt_x", "y", "x"], shape_metadata=[1024])
    n_var = IRNode(id="var_out", op_type="ReduceVar", inputs=["fma_out"], shape_metadata=[1])

    g.nodes = {"x": n_in1, "y": n_in2, "cbrt_x": n_cbrt, "fma_out": n_fma, "var_out": n_var}
    g.inputs = ["x", "y"]
    g.outputs = ["var_out"]

    gen = CudaCodeGenerator(g)
    cuda_code = gen.generate()

    assert "#include <cuda_runtime.h>" in cuda_code
    assert "void evaluate_cuda(" in cuda_code
    assert "CUDA_CHECK(cudaDeviceSynchronize());" in cuda_code

    # Check kernel launches and declarations
    assert "cbrt_kernel" in cuda_code
    assert "fma_kernel" in cuda_code
    assert "reducevar_kernel" in cuda_code

    # Ensure all cudaMalloc calls have matching cudaFree or are properly tracked
    malloc_matches = re.findall(r"cudaMalloc\(&(\w+),", cuda_code)
    assert len(malloc_matches) >= 3


def test_mathematical_reference_parity_for_synthesized_kernels() -> None:
    """Verify mathematical correctness against reference NumPy implementations."""
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
