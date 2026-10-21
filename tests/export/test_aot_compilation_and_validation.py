"""Tests for Phase 8: AOT compilation and binary serialization validation."""

from __future__ import annotations

import os
import shutil
import typing

import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
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
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


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
    assert isinstance(as_tensor(bundle), dict)
    assert bundle["format"] == "webgpu_wgsl_bundle"
    assert bundle["status"] == "ready_to_dispatch"
    assert "shaderCode" in str(bundle["code"])


def test_wasm_compile_aot(tmp_path: EagerTensorProtocol) -> None:
    """Test WasmCodeGenerator compile_aot producing wasm artifacts and raising CompilationError when toolchain missing.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    from unittest.mock import patch

    g = IRGraph(name="test_aot_wasm")
    g.inputs = []
    g.nodes["c1"] = IRNode(id="c1", op_type="Constant", attributes={"value": 1.0})
    g.outputs = ["c1"]

    gen = WasmCodeGenerator(g)
    with patch.object(shutil, "which", return_value=None):
        with pytest.raises(CompilationError, match="Neither emcc nor clang found"):
            gen.compile_aot(output_dir=str(tmp_path))

    with patch.object(shutil, "which", return_value="/usr/bin/emcc"):
        with patch("subprocess.run"):
            res = gen.compile_aot(output_dir=str(tmp_path))
            assert isinstance(as_tensor(res), tuple)
            assert len(as_tensor(res)) == 2


def test_compile_function_strict_diagnostics() -> None:
    """Test compile_function with strict=True raising CompilationError on missing backend."""
    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    def f(x: EagerValue) -> EagerValue:
        """Identity test function.

        Args:
            x (EagerValue): Input tensor.

        Returns:
            EagerValue: Unmodified input tensor.
        """
        return x

    t = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    c = compile_function(f, backend="non_existent_backend", strict=True)
    with pytest.raises(CompilationError, match="Target backend 'non_existent_backend' not found"):
        as_callable(c)(t)


def test_binary_validation_utilities(tmp_path: EagerTensorProtocol) -> None:
    """Test binary validation utilities for ONNX, StableHLO, and SavedModel.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    assert validate_onnx_binary(str(tmp_path / "missing.onnx")) is False
    assert validate_stablehlo_binary(str(tmp_path / "missing.mlirbc")) is False
    assert validate_saved_model_binary(str(tmp_path / "missing_dir")) is False

    onnx_file = tmp_path / "model.onnx"
    with open(onnx_file, "wb") as f:
        f.write(b"\x08\x07\x12\x15ml-switcheroo-compiler")
    assert validate_onnx_binary(str(onnx_file)) is True

    empty_onnx = tmp_path / "empty.onnx"
    with open(empty_onnx, "wb") as f:
        f.write(b"")
    assert validate_onnx_binary(str(empty_onnx)) is False

    mlirbc_file = tmp_path / "module.mlirbc"
    with open(mlirbc_file, "wb") as f:
        f.write(b"ML\xefR\x00\x01")
    assert validate_stablehlo_binary(str(mlirbc_file)) is True

    corrupt_mlirbc = tmp_path / "corrupt.mlirbc"
    with open(corrupt_mlirbc, "wb") as f:
        f.write(b"BAD!")
    assert validate_stablehlo_binary(str(corrupt_mlirbc)) is False

    sm_dir = tmp_path / "saved_model"
    archive = ExportArchive()
    archive.write_out(str(sm_dir))
    assert validate_saved_model_binary(str(sm_dir)) is True

    bad_sm_dir = tmp_path / "bad_saved_model"
    os.makedirs(bad_sm_dir)
    with open(bad_sm_dir / "saved_model.pb", "wb") as f:
        f.write(b"\x08\x01")
    assert validate_saved_model_binary(str(bad_sm_dir)) is False
