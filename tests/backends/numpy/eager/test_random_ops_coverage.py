"""Tests for test_random_ops_coverage."""

from __future__ import annotations

import types

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import random_ops as random_mod


def test_random_ops() -> None:
    """Test all random operations and branches in random_ops.

    Returns:
        None
    """
    # Normal with and without config
    norm1 = random_mod._np_normal(np, (2, 3), dtype="float32")
    assert norm1.shape == (2, 3)
    assert numpy_eager_registry.get("Normal")(np, (2, 3)).shape == (2, 3)

    cfg = types.SimpleNamespace(mean=2.0, stddev=0.5)
    norm2 = random_mod._np_normal(np, (4,), config=cfg, dtype="float64")
    assert norm2.shape == (4,)

    # Uniform with config (minval/maxval None and float) and without config
    unif1 = random_mod._np_uniform(np, (2, 2))
    assert unif1.shape == (2, 2)
    assert numpy_eager_registry.get("Uniform")(np, (2, 2)).shape == (2, 2)

    cfg_none = types.SimpleNamespace(minval=None, maxval=None)
    unif_none = random_mod._np_uniform(np, (3,), config=cfg_none)
    assert unif_none.shape == (3,)

    cfg_vals = types.SimpleNamespace(minval=10.0, maxval=20.0)
    unif_vals = random_mod._np_uniform(np, (3,), config=cfg_vals)
    assert np.all(unif_vals >= 10.0) and np.all(unif_vals <= 20.0)

    # StatelessSplit
    # Seed as string
    split_str = random_mod._np_stateless_split(np, "sample_seed", num=3)
    assert split_str.shape == (3, 2)

    # Seed as integer
    split_int = random_mod._np_stateless_split(np, 42, num=2)
    assert split_int.shape == (2, 2)
    assert numpy_eager_registry.get("StatelessSplit")(np, 42, num=2).shape == (2, 2)

    # Seed as non-int convertible
    split_obj = random_mod._np_stateless_split(np, np.array([dict()], dtype=object))
    assert split_obj.shape == (2, 2)

    # Seed empty
    split_empty = random_mod._np_stateless_split(np, [])
    assert split_empty.shape == (2, 2)

    # Lookup: table is dict
    dict_table = {"a": 10, "b": 20}
    dict_res = random_mod._np_lookup(np, dict_table, ["a", "b", "c"], default_value=-1)
    assert np.array_equal(dict_res, [10, 20, -1])

    # Lookup: table has lookup method
    lookup_obj = types.SimpleNamespace(lookup=lambda keys: np.array([100] * len(keys)))
    obj_res = random_mod._np_lookup(np, lookup_obj, [1, 2])
    assert np.array_equal(obj_res, [100, 100])

    # Lookup: table is neither
    fallback_res = random_mod._np_lookup(np, 123, [1, 2], default_value=5)
    assert np.array_equal(fallback_res, [5, 5])
    assert np.array_equal(numpy_eager_registry.get("Lookup")(np, dict_table, ["a"]), [10])
