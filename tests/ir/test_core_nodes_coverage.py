"""Tests for test_core_nodes_coverage."""

from __future__ import annotations

import warnings

import ml_switcheroo_compiler.ir.core as ir_core


class _MockBackendModule:
    """Mock backend module implementing zeros, array, and asarray."""

    def zeros(self, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        """Mock zeros.

        Args:
            shape (tuple[int, ...]): Output shape.

        Returns:
            tuple[str, tuple[int, ...]]: Mock zeros descriptor.
        """
        return ("zeros", shape)

    def array(self, data: object, dtype: object = None) -> tuple[str, object, object]:
        """Mock array.

        Args:
            data (object): Input data.
            dtype (object): Optional data type.

        Returns:
            tuple[str, object, object]: Mock array descriptor.
        """
        return ("array", data, dtype)

    def asarray(self, data: object) -> tuple[str, object]:
        """Mock asarray.

        Args:
            data (object): Input data.

        Returns:
            tuple[str, object]: Mock asarray descriptor.
        """
        return ("asarray", data)


class _MockItemObject:
    """Mock object implementing item()."""

    def item(self) -> float:
        """Return scalar item value.

        Returns:
            float: Scalar float value.
        """
        return 42.5


class _MockDaskDtypeWrapper:
    """Mock dtype object with value attribute."""

    def __init__(self, val: str) -> None:
        """Initialize mock dtype wrapper.

        Args:
            val (str): Type name.
        """
        self.value = val


def _sample_add_fn(x: int, y: int = 10) -> int:
    """Add two integers.

    Args:
        x (int): First int.
        y (int): Second int.

    Returns:
        int: Sum of x and y.
    """
    return x + y


def _sample_shape_fn(a: int, b: int) -> tuple[int, int]:
    """Sample shape function returning tuple.

    Args:
        a (int): First dimension.
        b (int): Second dimension.

    Returns:
        tuple[int, int]: Pair (a, b).
    """
    return (a, b)


def test_ir_core_all_nodes() -> None:
    """Verify all IR node types, IRBlock deprecation, and clone_logical_node in ir.core."""
    node = ir_core.LogicalNode(id="n1", op_type="CustomOp")
    assert node.id == "n1"
    assert node.op_type == "CustomOp"

    # Clone node
    cloned = ir_core.clone_logical_node(node, id="n2")
    assert cloned.id == "n2"
    assert cloned.op_type == "CustomOp"

    # Node with outputs, output_specs, subgraphs
    node_full = ir_core.LogicalNode(
        id="n_full",
        op_type="CompositeOp",
        inputs=["in1"],
        outputs=["out1"],
        attributes={"attr1": 123},
        output_specs=[],
        subgraphs={"body": ir_core.LogicalGraph(name="sub")},
    )
    cloned_full = ir_core.clone_logical_node(node_full, id="n_full_clone")
    assert cloned_full.id == "n_full_clone"
    assert cloned_full.inputs == ["in1"]
    assert cloned_full.outputs == ["out1"]
    assert "body" in cloned_full.subgraphs

    # IRBlock with deprecation warning
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        block = ir_core.IRBlock(id="blk1", nodes=[node], inputs=["in1"], outputs=["out1"])
        assert any(issubclass(w.category, DeprecationWarning) for w in recorded_warnings)
        assert block.id == "blk1"
        assert block.name == "blk1"

    # ZeroTangent and NoTangent
    zt = ir_core.ZeroTangent("zt1", shape_metadata=(2, 3))
    assert zt.id == "zt1"
    assert zt.op_type == "ZeroTangent"

    nt = ir_core.NoTangent("nt1")
    assert nt.id == "nt1"
    assert nt.op_type == "NoTangent"
