"""Tests for lookup and hash table operations."""

from __future__ import annotations

from typing import cast

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.lookup import DenseHashTable, MutableHashTable
from ml_switcheroo_compiler.tracing import global_tracing_state


def test_lookup_eager_and_tracing() -> None:
    """Verify MutableHashTable and DenseHashTable in both eager and tracing modes."""
    orig = config.eager_mode
    config.eager_mode = True
    try:
        default_t = Tensor(np.array(-1.0, dtype=np.float32), TensorConfig((), DType.Float32, "cpu"))
        keys_in = Tensor(np.array([1, 2], dtype=np.int64), TensorConfig((2,), DType.Int64, "cpu"))
        vals_in = Tensor(np.array([10.0, 20.0], dtype=np.float32), TensorConfig((2,), DType.Float32, "cpu"))
        query_keys = Tensor(np.array([2, 99], dtype=np.int64), TensorConfig((2,), DType.Int64, "cpu"))

        mht = MutableHashTable(DType.Int64, DType.Float32, default_t)
        mht.insert(keys_in, vals_in)
        res = cast(Tensor, mht.lookup(query_keys))
        np.testing.assert_allclose(np.asarray(res.data), np.array([20.0, -1.0], dtype=np.float32))

        scalar_k = Tensor(np.array(3, dtype=np.int64), TensorConfig((), DType.Int64, "cpu"))
        scalar_v = Tensor(np.array(30.0, dtype=np.float32), TensorConfig((), DType.Float32, "cpu"))
        mht.insert(scalar_k, scalar_v)
        res_scalar = cast(Tensor, mht.lookup(scalar_k))
        np.testing.assert_allclose(np.asarray(res_scalar.data), np.array(30.0, dtype=np.float32))

        none_key = Tensor(None, TensorConfig((), DType.Int64, "cpu"))
        mht.insert(none_key, none_key)
        res_fallback = cast(Tensor, mht.lookup(none_key))
        assert res_fallback.data == -1.0

        dht = DenseHashTable(DType.Int64, DType.Float32, default_t, default_t, default_t)
        dht.insert(keys_in, vals_in)
        res2 = cast(Tensor, dht.lookup(query_keys))
        np.testing.assert_allclose(np.asarray(res2.data), np.array([20.0, -1.0], dtype=np.float32))
    finally:
        config.eager_mode = orig

    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing("lookup_trace")
        try:
            sym_key = Tensor(None, TensorConfig((2,), DType.Int64, "cpu"))
            sym_res = cast(Tensor, mht.lookup(sym_key))
            assert sym_res.shape == (2,)
        finally:
            global_tracing_state.stop_tracing()
