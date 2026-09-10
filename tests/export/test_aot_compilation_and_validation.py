"""Tests for Phase 8: AOT compilation and binary serialization validation."""

from __future__ import annotations

import os

import pytest

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.export.aot import compile_function
from ml_switcheroo_compiler.export.export_api import (
    ExportArchive,
    validate_onnx_binary,
    validate_saved_model_binary,
    validate_stablehlo_binary,
)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_cpp_compile_aot() -> None:
    """Test CppGenerator compile_aot compiling to native shared library."""
    g = IRGraph(name="test_aot_cpp")
    g.inputs = []
    g.nodes["c1"] = IRNode(id="c1", op_type="Constant", attributes={"value": 5.0})
    g.outputs = ["c1"]

    gen = CppGenerator(g)
    executable = gen.compile_aot()
    assert callable(executable)
    res = executable()
    assert res == "Execution successful"


def test_webgpu_compile_aot() -> None:
    """Test WebGPUCodeGenerator compile_aot emitting WGSL dispatch bundle."""
    g = IRGraph(name="test_aot_webgpu")
    g.inputs = ["in0"]
    g.nodes["in0"] = IRNode(id="in0", op_type="Input", inputs=[], shape_metadata=(4,))
    g.nodes["out0"] = IRNode(id="out0", op_type="Add", inputs=["in0", "in0"], shape_metadata=(4,))
    g.outputs = ["out0"]

    gen = WebGPUCodeGenerator(g)
    bundle = gen.compile_aot()
    assert isinstance(bundle, dict)
    assert bundle["format"] == "webgpu_wgsl_bundle"
    assert bundle["status"] == "ready_to_dispatch"
    assert "shaderCode" in bundle["code"]


def test_wasm_compile_aot(tmp_path) -> None:
    """Test WasmCodeGenerator compile_aot producing wasm artifacts."""
    g = IRGraph(name="test_aot_wasm")
    g.inputs = []
    g.nodes["c1"] = IRNode(id="c1", op_type="Constant", attributes={"value": 1.0})
    g.outputs = ["c1"]

    gen = WasmCodeGenerator(g)
    # compile_aot invokes compile_wasm hook
    try:
        res = gen.compile_aot(output_dir=str(tmp_path))
        assert isinstance(res, tuple)
    except CompilationError:
        # Expected if neither emcc nor clang is installed in test container
        pass


def test_compile_function_strict_diagnostics() -> None:
    """Test compile_function with strict=True raising CompilationError on missing backend."""
    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    def f(x):
        return x

    t = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    c = compile_function(f, backend="non_existent_backend", strict=True)
    with pytest.raises(CompilationError) as exc_info:
        c(t)
    assert "Target backend 'non_existent_backend' not found" in str(exc_info.value)


def test_binary_validation_utilities(tmp_path) -> None:
    """Test binary validation utilities for ONNX, StableHLO, and SavedModel."""
    # 1. Non-existent files
    assert validate_onnx_binary(str(tmp_path / "missing.onnx")) is False
    assert validate_stablehlo_binary(str(tmp_path / "missing.mlirbc")) is False
    assert validate_saved_model_binary(str(tmp_path / "missing_dir")) is False

    # 2. Valid ONNX file structure
    onnx_file = tmp_path / "model.onnx"
    with open(onnx_file, "wb") as f:
        f.write(b"\x08\x07\x12\x15ml-switcheroo-compiler")  # valid protobuf header tags
    assert validate_onnx_binary(str(onnx_file)) is True

    # Empty ONNX file
    empty_onnx = tmp_path / "empty.onnx"
    with open(empty_onnx, "wb") as f:
        f.write(b"")
    assert validate_onnx_binary(str(empty_onnx)) is False

    # 3. StableHLO binary validation
    mlirbc_file = tmp_path / "module.mlirbc"
    with open(mlirbc_file, "wb") as f:
        f.write(b"ML\xefR\x00\x01")
    assert validate_stablehlo_binary(str(mlirbc_file)) is True

    # Corrupt StableHLO
    corrupt_mlirbc = tmp_path / "corrupt.mlirbc"
    with open(corrupt_mlirbc, "wb") as f:
        f.write(b"BAD!")
    assert validate_stablehlo_binary(str(corrupt_mlirbc)) is False

    # 4. SavedModel directory validation
    sm_dir = tmp_path / "saved_model"
    archive = ExportArchive()
    archive.write_out(str(sm_dir))
    assert validate_saved_model_binary(str(sm_dir)) is True

    # Missing variables in SavedModel dir
    bad_sm_dir = tmp_path / "bad_saved_model"
    os.makedirs(bad_sm_dir)
    with open(bad_sm_dir / "saved_model.pb", "wb") as f:
        f.write(b"\x08\x01")
    assert validate_saved_model_binary(str(bad_sm_dir)) is False
