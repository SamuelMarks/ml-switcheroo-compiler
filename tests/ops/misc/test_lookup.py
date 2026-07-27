# ruff: noqa: D103
"""Tests for lookup ops."""

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.lookup import DenseHashTable, MutableHashTable


def test_lookup() -> None:
    t = Tensor(None, TensorConfig((), DType.Float32, "cpu"))
    mht = MutableHashTable(DType.Int64, DType.Float32, t)
    mht.insert(t, t)
    res = mht.lookup(t)
    assert res is not None

    dht = DenseHashTable(DType.Int64, DType.Float32, t, t, t)
    dht.insert(t, t)
    res2 = dht.lookup(t)
    assert res2 is not None
