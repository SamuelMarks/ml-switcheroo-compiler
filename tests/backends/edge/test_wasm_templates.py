"""Tests for WebAssembly SIMD and scalar code generation templates."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph


def test_templates_validity() -> None:
    """Test validity and syntax of WASM helper functions and template macros."""
    graph = IRGraph()
    gen = WasmCodeGenerator(graph)

    helpers: list[str] = gen.get_helper_functions()
    assert len(helpers) > 0
    assert any("Scalar Fallback Helpers" in line for line in helpers)


def test_wasm_memory_allocation_template() -> None:
    """Test aligned memory allocation template generation."""
    graph = IRGraph()
    gen = WasmCodeGenerator(graph)

    alloc_str: str = gen._allocate_aligned_memory(size_bytes=1024, alignment=16)
    assert "std::aligned_alloc(16, 1024)" in alloc_str


def test_wasm_type_mapping() -> None:
    """Test type mapping for WASM targets."""
    graph = IRGraph()
    gen = WasmCodeGenerator(graph)

    assert gen._map_type("float32") == "float"
    assert gen._map_type("float64") == "double"
    assert gen._map_type("int32") == "int"
    assert gen._map_type("bool") == "bool"
    assert gen._map_type("unknown_type") == "float"


def test_wasm_striding_logic() -> None:
    """Test striding calculations and indexing expressions for multidimensional tensors."""
    graph = IRGraph()
    gen = WasmCodeGenerator(graph)

    strides, c_code = gen._generate_striding_logic([2, 3, 4])
    assert strides == [12, 4, 1]
    assert "idx" in c_code

    empty_strides, empty_code = gen._generate_striding_logic([])
    assert empty_strides == []
    assert empty_code == "0"


def test_wasm_num_elements() -> None:
    """Test tensor element count calculation."""
    graph = IRGraph()
    gen = WasmCodeGenerator(graph)

    assert gen._num_elements([2, 3, 4]) == 24
    assert gen._num_elements([5]) == 5
    assert gen._num_elements([]) == 1
