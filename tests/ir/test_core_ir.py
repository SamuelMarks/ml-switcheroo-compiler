"""Test module."""

import pytest
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import IRBlock, IRNode, TensorSpec, clone_logical_node


def test_clone_logical_node():
    node = LogicalNode(id="id1", op_type="op1")
    node2 = clone_logical_node(node, op_type="op2")
    assert node2.op_type == "op2"
    assert node2.id == "id1"


def test_irnode():
    n = IRNode("op", "id1", shape_metadata=(1, 2))
    assert n.is_dynamic_shape is False
    assert n.static_shape == (1, 2)
    assert n.rank == 2

    n_dyn = IRNode("op", "id2", shape_metadata=(1, "x"))
    assert n_dyn.is_dynamic_shape is True
    with pytest.raises(ValueError):
        n_dyn.static_shape
    assert n_dyn.rank == 2

    n_none = IRNode("op", "id3", shape_metadata=None)
    assert n_none.is_dynamic_shape is False
    with pytest.raises(ValueError):
        n_none.static_shape
    assert n_none.rank == 0

    n_bad = IRNode("op", "id4", shape_metadata="not a seq")
    assert n_bad.is_dynamic_shape is False
    with pytest.raises(ValueError):
        n_bad.static_shape
    assert n_bad.rank == 0


def test_tensorspec():
    s = TensorSpec((1, 2), DType.Float32)
    assert s.is_dynamic is False
    assert s.static_shape == (1, 2)
    assert s.rank == 2

    s_dyn = TensorSpec((1, "x"), DType.Float32)
    assert s_dyn.is_dynamic is True
    with pytest.raises(ValueError):
        s_dyn.static_shape


def test_irblock():
    b = IRBlock("b1")
    assert b.id == "b1"
    assert b.nodes == []
    assert b.inputs == []
    assert b.outputs == []


def test_tangent_nodes():
    from ml_switcheroo_compiler.ir.core import NoTangent, ZeroTangent

    zt = ZeroTangent("t1")
    assert zt.op_type == "ZeroTangent"
    nt = NoTangent("t2")
    assert nt.op_type == "NoTangent"
