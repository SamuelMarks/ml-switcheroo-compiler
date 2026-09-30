"""Tests asserting hardware ISA and compiler IR grounding verifiers against static snapshots."""

from __future__ import annotations

from ml_ecosystem_snapshots.grounding import (
    validate_metal_op,
    validate_mlir_op,
    validate_onnx_op,
    validate_ptx_instruction,
    validate_rdna_instruction,
    validate_sass_instruction,
    validate_stablehlo_op,
    validate_wasm_op,
    validate_wgsl_op,
)


def test_validate_sass_instruction() -> None:
    """Verify NVIDIA SASS instruction grounding across valid and invalid architectures."""
    valid_report = validate_sass_instruction("FFMA", "sm_80", ["R0", "R1", "R2", "R3"])
    assert valid_report.is_grounded is True
    assert not valid_report.diagnostics

    invalid_arch_report = validate_sass_instruction("FFMA", "sm_invalid_arch", ["R0", "R1", "R2", "R3"])
    assert invalid_arch_report.is_grounded is False
    assert len(invalid_arch_report.diagnostics) >= 1

    unknown_mnemonic_report = validate_sass_instruction("NONEXISTENT_MNEMONIC", "sm_80", ["R0"])
    assert unknown_mnemonic_report.is_grounded is False


def test_validate_rdna_instruction() -> None:
    """Verify AMD RDNA instruction grounding across valid and mistyped mnemonics."""
    valid_report = validate_rdna_instruction("v_add_f32", "GFX10", ["v0", "v1", "v2"])
    assert valid_report.is_grounded is True
    assert not valid_report.diagnostics

    unknown_report = validate_rdna_instruction("v_nonexistent_rdna_op", "GFX10", ["v0"])
    assert unknown_report.is_grounded is False
    assert len(unknown_report.diagnostics) >= 1


def test_validate_ptx_instruction() -> None:
    """Verify NVIDIA PTX instruction grounding and argument typing."""
    valid_report = validate_ptx_instruction("add", sm_arch="sm_80", types=["f32"])
    assert valid_report.is_grounded is True
    assert not valid_report.diagnostics

    unknown_report = validate_ptx_instruction("nonexistent_ptx_mnemonic_123")
    assert unknown_report.is_grounded is False


def test_validate_wgsl_op() -> None:
    """Verify WebGPU WGSL builtin operation grounding."""
    valid_report = validate_wgsl_op("workgroupBarrier")
    assert valid_report.is_grounded is True
    assert not valid_report.diagnostics

    unknown_report = validate_wgsl_op("nonexistent_wgsl_barrier_xyz")
    assert unknown_report.is_grounded is False


def test_validate_wasm_op() -> None:
    """Verify WebAssembly SIMD intrinsic opcode grounding."""
    valid_report = validate_wasm_op("f32x4.add")
    assert valid_report.is_grounded is True
    assert not valid_report.diagnostics

    unknown_report = validate_wasm_op("f32x4.nonexistent_operation")
    assert unknown_report.is_grounded is False


def test_validate_metal_op() -> None:
    """Verify Apple Metal Shading Language builtin operation grounding."""
    valid_report = validate_metal_op("simdgroup_barrier")
    assert valid_report.is_grounded is True
    assert not valid_report.diagnostics

    unknown_report = validate_metal_op("nonexistent_metal_simd_op")
    assert unknown_report.is_grounded is False


def test_validate_stablehlo_op() -> None:
    """Verify StableHLO operation grounding against dialect specifications."""
    valid_report = validate_stablehlo_op("add", ["tensor<2x2xf32>", "tensor<2x2xf32>"], {})
    assert valid_report.is_grounded is True
    assert not valid_report.diagnostics

    unknown_report = validate_stablehlo_op("nonexistent_stablehlo_op", [], {})
    assert unknown_report.is_grounded is False


def test_validate_mlir_op() -> None:
    """Verify MLIR dialect and operation grounding."""
    valid_report = validate_mlir_op("arith", "addf", 2, {})
    assert valid_report.is_grounded is True
    assert not valid_report.diagnostics

    unknown_report = validate_mlir_op("arith", "nonexistent_arith_op", 2, {})
    assert unknown_report.is_grounded is False


def test_validate_onnx_op() -> None:
    """Verify ONNX operator grounding against standard operator sets."""
    valid_report = validate_onnx_op("Add", 2)
    assert valid_report.is_grounded is True
    assert not valid_report.diagnostics

    unknown_report = validate_onnx_op("NonexistentONNXOp", 2)
    assert unknown_report.is_grounded is False
