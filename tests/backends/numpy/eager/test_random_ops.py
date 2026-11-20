"""Behavioral tests for NumPy eager random sampling, stateless seed splitting, and table lookup operations."""

from __future__ import annotations

import types

import numpy as np

from ml_switcheroo_compiler.backends.eager import random_ops as random_mod
from ml_switcheroo_compiler.backends.eager_registry import BackendArg, BackendOperand, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import MockTestStub, extract_test_shape, require_callable


class _NullBoundsConfig(MockTestStub):
    """Configuration stub with `None` bounds to exercise default fallback in `_np_uniform`."""

    minval: float | None = None
    maxval: float | None = None


class _OffsetLookupTable(MockTestStub):
    """Custom lookup table satisfying `LookupTableProtocol`."""

    def lookup(self, keys: BackendArg) -> BackendOperand:
        """Add 100 to every queried key.

        Args:
            keys (BackendArg): Queried key array.

        Returns:
            EagerTensorProtocol: Offset values.
        """
        return np.asarray(keys) + 100


class _NonIntegerSeedToken:
    """Seed object raising `ValueError` on `int(...)` conversion to exercise string-hash fallback."""

    def __int__(self) -> int:
        """Reject integer coercion to force string hashing.

        Returns:
            int: Never returns normally.

        Raises:
            ValueError: Always raised to trigger string-hash fallback.
        """
        raise ValueError("non-integer token")

    def __str__(self) -> str:
        """Return deterministic string token for seed hashing.

        Returns:
            str: Token string.
        """
        return "custom_seed_token"


def test_normal_and_uniform_distributions() -> None:
    """Verify `Normal` and `Uniform` respect target shapes, dtypes, and distribution configs."""
    norm_default = random_mod._np_normal(np, (2, 3), dtype="float32")
    assert norm_default.shape == (2, 3)
    assert norm_default.dtype == np.float32

    reg_normal = numpy_eager_registry.get("Normal")
    assert extract_test_shape(require_callable(reg_normal)(np, (2, 3))) == (2, 3)

    norm_cfg = types.SimpleNamespace(mean=5.0, stddev=0.0)
    norm_det = random_mod._np_normal(np, (4,), config=norm_cfg, dtype="float64")
    assert norm_det.dtype == np.float64
    np.testing.assert_allclose(norm_det, np.full((4,), 5.0, dtype=np.float64))

    unif_default = random_mod._np_uniform(np, (3, 2))
    assert unif_default.shape == (3, 2)
    assert bool(np.all((unif_default >= 0.0) & (unif_default <= 1.0)))

    reg_uniform = numpy_eager_registry.get("Uniform")
    assert extract_test_shape(require_callable(reg_uniform)(np, (2, 2))) == (2, 2)

    unif_null = ensure_ndarray(random_mod._np_uniform(np, (4,), dtype="float32", config=_NullBoundsConfig()))
    assert unif_null.shape == (4,)
    assert bool(np.all((unif_null >= 0.0) & (unif_null <= 1.0)))

    unif_bounded = random_mod._np_uniform(np, (8,), config=types.SimpleNamespace(minval=10.0, maxval=20.0))
    assert bool(np.all((unif_bounded >= 10.0) & (unif_bounded <= 20.0)))


def test_stateless_split_seed_variants() -> None:
    """Verify `StatelessSplit` deterministically splits integer, string, custom-object, and empty seeds."""
    split_str_a = random_mod._np_stateless_split(np, "sample_seed", num=3)
    split_str_b = random_mod._np_stateless_split(np, "sample_seed", num=3)
    assert split_str_a.shape == (3, 2)
    np.testing.assert_array_equal(split_str_a, split_str_b)

    split_int = random_mod._np_stateless_split(np, 42, num=2)
    reg_split = numpy_eager_registry.get("StatelessSplit")
    np.testing.assert_array_equal(split_int, ensure_ndarray(require_callable(reg_split)(np, 42, num=2)))

    split_bad_token = random_mod._np_stateless_split(np, np.array([_NonIntegerSeedToken()]), num=2)
    assert split_bad_token.shape == (2, 2)
    assert split_bad_token.dtype == np.int64

    split_empty = random_mod._np_stateless_split(np, [])
    split_zero = random_mod._np_stateless_split(np, 0, num=None)
    assert split_empty.shape == (2, 2)
    np.testing.assert_array_equal(split_empty, split_zero)


def test_lookup_dict_protocol_and_fallback() -> None:
    """Verify `Lookup` resolves dict mappings, `LookupTableProtocol` tables, and scalar default fallbacks."""
    dict_table: dict[int | str, int | float] = {"a": 10, "b": 20}
    dict_res = random_mod._np_lookup(np, dict_table, ["a", "b", "c"], default_value=-1)
    np.testing.assert_array_equal(dict_res, np.array([10, 20, -1]))

    reg_lookup = numpy_eager_registry.get("Lookup")
    np.testing.assert_array_equal(ensure_ndarray(require_callable(reg_lookup)(np, dict_table, ["a"])), np.array([10]))

    proto_res = random_mod._np_lookup(np, _OffsetLookupTable(), [1, 2, 3])
    np.testing.assert_array_equal(proto_res, np.array([101, 102, 103]))

    fallback_res = random_mod._np_lookup(np, 123, [1, 2], default_value=5)
    np.testing.assert_array_equal(fallback_res, np.array([5, 5], dtype=np.int32))
