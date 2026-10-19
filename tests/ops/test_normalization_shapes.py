"""Test normalization shape inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.ir.shape_system import SymInt
from ml_switcheroo_compiler.ops.normalization.basic import GroupMean, GroupNorm, GroupVariance

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_group_mean_shapes() -> None:
    """Test GroupMean shape inference over various ranks."""
    gm = GroupMean()

    # 1D
    assert gm.infer_shape((10,)) == (10,)
    assert gm.infer_shape(10) == (10,)

    # 2D
    assert gm.infer_shape((10, 8), num_groups=2) == (10, 2)
    assert gm.infer_shape((10, 8), num_groups=4, keepdims=False) == (10, 4)

    # 3D
    assert gm.infer_shape((10, 8, 5), num_groups=2) == (10, 2, 1)
    assert gm.infer_shape((10, 8, 5), num_groups=4, keepdims=False) == (10, 4)

    # 4D
    assert gm.infer_shape((10, 8, 5, 5), num_groups=2) == (10, 2, 1, 1)

    # 5D
    assert gm.infer_shape((10, 8, 5, 5, 5), num_groups=4) == (10, 4, 1, 1, 1)

    # Symbolic
    sym_n = SymInt("N")
    sym_g = SymInt("G")
    res = cast("EagerCallableProtocol", gm.infer_shape)((sym_n, 8, 5, 5), num_groups=sym_g)
    assert len(res) == 4
    assert id(res[0]) == id(sym_n)
    assert id(res[1]) == id(sym_g)
    assert res[2] == 1
    assert res[3] == 1


def test_group_variance_shapes() -> None:
    """Test GroupVariance shape inference."""
    gv = GroupVariance()

    # 4D
    assert gv.infer_shape((10, 8, 5, 5), num_groups=2) == (10, 2, 1, 1)
    assert gv.infer_shape((10, 8, 5, 5), num_groups=4, keepdims=False) == (10, 4)

    # Symbolic
    sym_n = SymInt("N")
    res = cast("EagerCallableProtocol", gv.infer_shape)((sym_n, 16, 8, 8), num_groups=4)
    assert len(res) == 4
    assert id(res[0]) == id(sym_n)
    assert res[1] == 4


def test_group_norm_shapes() -> None:
    """Test GroupNorm shape inference."""
    gn = GroupNorm()

    assert gn.infer_shape((10, 8, 5, 5)) == (10, 8, 5, 5)
    assert gn.infer_shape(10) == (10,)

    sym_n = SymInt("N")
    assert cast("EagerCallableProtocol", gn.infer_shape)((sym_n, 8)) == (sym_n, 8)
