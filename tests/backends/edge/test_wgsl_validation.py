"""Tests for WGSL AST specification validation conforming to WebGPU standards."""

import pytest

from ml_switcheroo_compiler.backends.edge.wgsl_ast import (
    WGSLBinding,
    WGSLEmitter,
    WGSLNode,
    WGSLRaw,
    WGSLValidationError,
    WGSLVar,
    validate_binding_limits,
    validate_identifier,
    validate_memory_layout,
    validate_workgroup_size,
)


def test_validate_identifier_comprehensive() -> None:
    """Test identifier validation rules according to WebGPU specification."""
    validate_identifier("tensor_x")
    validate_identifier("_private_buffer")
    validate_identifier("matrix_4x4_f32")

    with pytest.raises(WGSLValidationError, match="must be non-empty string"):
        validate_identifier("")

    with pytest.raises(WGSLValidationError, match="does not match WGSL identifier syntax"):
        validate_identifier("123invalid")

    with pytest.raises(WGSLValidationError, match="does not match WGSL identifier syntax"):
        validate_identifier("invalid-hyphen")

    with pytest.raises(WGSLValidationError, match="reserved WGSL keyword"):
        validate_identifier("array")

    with pytest.raises(WGSLValidationError, match="reserved WGSL keyword"):
        validate_identifier("storage")


def test_validate_memory_layout_qualifiers() -> None:
    """Test memory layout address space and access mode qualification."""
    validate_memory_layout("storage", "read")
    validate_memory_layout("storage", "read_write")
    validate_memory_layout("uniform", "read")
    validate_memory_layout("workgroup", "read_write")
    validate_memory_layout("private")
    validate_memory_layout("function", "read_write")

    with pytest.raises(WGSLValidationError, match="Invalid WGSL address space qualifier"):
        validate_memory_layout("device", "read")

    with pytest.raises(WGSLValidationError, match="Invalid WGSL access mode"):
        validate_memory_layout("storage", "execute")

    with pytest.raises(WGSLValidationError, match="Uniform address space only supports 'read'"):
        validate_memory_layout("uniform", "write")

    with pytest.raises(WGSLValidationError, match="Uniform address space only supports 'read'"):
        validate_memory_layout("uniform", "read_write")

    with pytest.raises(WGSLValidationError, match="Storage buffer address space supports"):
        validate_memory_layout("storage", "write")

    with pytest.raises(WGSLValidationError, match="Workgroup address space only supports 'read_write'"):
        validate_memory_layout("workgroup", "read")


def test_validate_binding_limits() -> None:
    """Test WebGPU guaranteed minimum binding index limits."""
    validate_binding_limits(0, 0)
    validate_binding_limits(3, 63)

    with pytest.raises(WGSLValidationError, match="Bind group index -1 exceeds"):
        validate_binding_limits(-1, 0)

    with pytest.raises(WGSLValidationError, match="Bind group index 4 exceeds"):
        validate_binding_limits(4, 0)

    with pytest.raises(WGSLValidationError, match="Binding index -1 exceeds"):
        validate_binding_limits(0, -1)

    with pytest.raises(WGSLValidationError, match="Binding index 64 exceeds"):
        validate_binding_limits(0, 64)


def test_validate_workgroup_size() -> None:
    """Test WebGPU workgroup dimensionality limits and invocation ceilings."""
    validate_workgroup_size(64, 1, 1)
    validate_workgroup_size(16, 16, 1)
    validate_workgroup_size(8, 8, 4)

    with pytest.raises(WGSLValidationError, match="dimensions must be >= 1"):
        validate_workgroup_size(0, 1, 1)

    with pytest.raises(WGSLValidationError, match="dimension exceeds WebGPU limits"):
        validate_workgroup_size(512, 1, 1)

    with pytest.raises(WGSLValidationError, match="dimension exceeds WebGPU limits"):
        validate_workgroup_size(1, 1, 128)

    with pytest.raises(WGSLValidationError, match="Total workgroup invocations"):
        validate_workgroup_size(16, 17, 1)


def test_wgsl_binding_node() -> None:
    """Test WGSLBinding AST node validation and emission."""
    emitter = WGSLEmitter()

    binding = WGSLBinding(
        group=0,
        binding=1,
        name="input_tensor",
        type_str="array<f32>",
        address_space="storage",
        access_mode="read",
    )
    binding.validate()
    code = emitter.emit(binding)
    assert code == "@group(0) @binding(1) var<storage, read> input_tensor: array<f32>;"

    # Uniform buffer binding without explicit access mode
    uniform_binding = WGSLBinding(
        group=1,
        binding=0,
        name="params",
        type_str="UniformParams",
        address_space="uniform",
        access_mode=None,
    )
    uniform_binding.validate()
    code_uniform = emitter.emit(uniform_binding)
    assert code_uniform == "@group(1) @binding(0) var<uniform> params: UniformParams;"

    # Invalid cases
    with pytest.raises(WGSLValidationError, match="reserved WGSL keyword"):
        WGSLBinding(group=0, binding=0, name="storage", type_str="array<f32>").validate()

    with pytest.raises(WGSLValidationError, match="Bind group index 5 exceeds"):
        WGSLBinding(group=5, binding=0, name="buf", type_str="array<f32>").validate()

    with pytest.raises(WGSLValidationError, match="Uniform address space only supports 'read'"):
        WGSLBinding(group=0, binding=0, name="buf", type_str="array<f32>", address_space="uniform", access_mode="write").validate()


def test_wgsl_raw_validation() -> None:
    """Test WGSLRaw validation against syntax delimiters, workgroup limits, and qualifiers."""
    valid_raw = WGSLRaw(
        """@group(0) @binding(0) var<storage, read> input_buf: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main() { let x = input_buf[0]; return; }"""
    )
    valid_raw.validate()

    # Unbalanced delimiters
    with pytest.raises(WGSLValidationError, match="Unclosed delimiter"):
        WGSLRaw("fn main() { var x = 1;").validate()

    with pytest.raises(WGSLValidationError, match="Unbalanced delimiter"):
        WGSLRaw("fn main() { var x = (1 + 2; }").validate()

    # Workgroup size violation in raw WGSL
    with pytest.raises(WGSLValidationError, match="dimension exceeds WebGPU limits"):
        WGSLRaw("""@compute @workgroup_size(512, 1, 1)
fn main() {}""").validate()

    # Group index violation in raw WGSL
    with pytest.raises(WGSLValidationError, match="Bind group index 4 exceeds"):
        WGSLRaw("@group(4) @binding(0) var<storage> buf: array<f32>;").validate()

    # Binding index violation in raw WGSL
    with pytest.raises(WGSLValidationError, match="Binding index 65 exceeds"):
        WGSLRaw("@group(0) @binding(65) var<storage> buf: array<f32>;").validate()

    # Memory layout qualifier violation in raw WGSL
    with pytest.raises(WGSLValidationError, match="Uniform address space only supports 'read'"):
        WGSLRaw("@group(0) @binding(0) var<uniform, write> buf: array<f32>;").validate()


def test_wgsl_node_recursive_validation() -> None:
    """Test base WGSLNode recursive structural traversal and validation."""

    class ContainerNode(WGSLNode):
        """Test container node containing child AST nodes."""

        def __init__(self, children: list[WGSLNode], single_child: WGSLNode) -> None:
            """Initialize ContainerNode.

            Args:
                children (list[WGSLNode]): List of child nodes.
                single_child (WGSLNode): Single child node.
            """
            self.children = children
            self.single_child = single_child

    valid_container = ContainerNode(
        children=[WGSLVar("var1"), WGSLVar("var2")],
        single_child=WGSLVar("var3"),
    )
    # Add non-node attribute and list with non-node items
    valid_container.scalar_attr = 42
    valid_container.mixed_list = ["string_item", 100]
    valid_container.validate()

    invalid_container = ContainerNode(
        children=[WGSLVar("var1"), WGSLVar("array")],  # 'array' is reserved
        single_child=WGSLVar("var3"),
    )
    with pytest.raises(WGSLValidationError, match="reserved WGSL keyword"):
        invalid_container.validate()
