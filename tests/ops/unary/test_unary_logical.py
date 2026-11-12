"""Tests for ops unary logical coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.ops.unary.logical as unary_logical
from tests.testing_utils import ensure_callable, get_test_members


def test_unary_logical_functions() -> None:
    """Test unary logical coverage."""
    import inspect

    class DummyNode:
        """Test class DummyNode."""

        shape = (10, 20)
        dtype = "float32"

    a = DummyNode()
    evaluated = 0
    for attr, val in get_test_members(unary_logical).items():
        if (attr.startswith("_infer_shape_") or attr.startswith("_infer_dtype_")) and callable(val):
            fn = ensure_callable(val)
            params = list(inspect.signature(fn).parameters.values())
            req = [p for p in params if p.default is inspect.Parameter.empty and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            args = [a] * max(1, len(req))
            res = fn(*args)
            assert res is not None
            evaluated += 1
    assert evaluated >= 1
