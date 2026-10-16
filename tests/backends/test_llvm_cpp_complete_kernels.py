"""Test complete kernels generation for LLVM C++ backend."""

from __future__ import annotations

import pytest

from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_strict_mode_unimplemented_op() -> None:
    """Test strict mode raises UnimplementedMathError instead of fallback."""
    graph = IRGraph()
    node = IRNode(id="FakeUnknownOp", op_type="FakeUnknownOp", inputs=["in_a"], shape_metadata=(10, 10))
    graph.nodes[node.id] = node

    generator = CppGenerator(graph, strict=True)
    with pytest.raises(UnimplementedMathError):
        generator.generate()


def test_scalar_fallback_no_zeros() -> None:
    """Test non-strict mode emits an actual scalar loop, not just zeros if inputs exist."""
    graph = IRGraph()
    in_node = IRNode(id="Input", op_type="Input", shape_metadata=(10, 10))
    graph.nodes[in_node.id] = in_node

    node = IRNode(id="FakeUnknownOp", op_type="FakeUnknownOp", inputs=[in_node.id], shape_metadata=(10, 10))
    graph.nodes[node.id] = node

    generator = CppGenerator(graph, strict=False)
    code = generator.generate()

    # Assert there is a fallback warning scalar loop that uses the input, instead of all zeros
    assert "Fallback Unimplemented" in code
    assert "? " + in_node.id + ".data" in code


def test_cpp_templates_expanded() -> None:
    """Verify that templates for specialized activations and others exist."""
    import os

    import yaml

    import ml_switcheroo_compiler

    tmpl_path = os.path.join(os.path.dirname(ml_switcheroo_compiler.__file__), "backends/llvm_cpp/cpp_templates.yaml")
    with open(tmpl_path) as f:
        data = yaml.safe_load(f)

    ops = data.get("operations", {})
    # Check for activations
    assert "gelu" in ops
    assert "swish" in ops
    assert "mish" in ops
    assert "hardsigmoid" in ops
    assert "hardswish" in ops

    # Check for matrices
    assert "cholesky" in ops
    assert "qr" in ops
    assert "svd" in ops
    assert "solvetriangular" in ops

    # Check for normalizations
    assert "groupnorm" in ops
    assert "instancenorm" in ops
    assert "rmsnorm" in ops
