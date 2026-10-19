"""Tests for test_frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops import array, diag, rand, randint, randn
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_rand_frontend_eager() -> None:
    """Test function test_rand_frontend_eager."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    t1 = rand(2, 3)
    assert cast("EagerTensorProtocol", getattr(t1, "shape")) == (2, 3)
    import ml_switcheroo_compiler.backends.registry as reg

    setattr(reg, "_ACTIVE_BACKEND", type("D", (), {"execute_op": lambda *a, **k: type("T", (), {"shape": (2, 3)})()})())
    cast("EagerCallableProtocol", randn)([2, 3])
    pass
    t3 = randint(0, 10, (2, 3))
    assert cast("EagerTensorProtocol", getattr(t3, "shape")) == (2, 3)
    pass
    array(type("D", (), {"shape": (1, 3)})(), dtype=DType.Int32)
    pass


def test_rand_frontend_tracing() -> None:
    """Test function test_rand_frontend_tracing."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    global_tracing_state.start_tracing("test_rand")
    try:
        from unittest.mock import patch

        with patch("ml_switcheroo_compiler.ops.creation.frontend_random._emit_creation_node", return_value="emitted"):
            assert rand(2, 3) == "emitted"
            assert randn(2, 3) == "emitted"
            assert randint(0, 10, (2, 3)) == "emitted"
            pass
        with patch("ml_switcheroo_compiler.ops.creation.frontend_basic._create_backend_array", return_value=np.array([1, 2, 3])):
            arr = array(type("D", (), {"shape": (1, 3)})(), dtype=DType.Int32)
            pass
            with patch("ml_switcheroo_compiler.ops.creation.frontend_matrix._emit_creation_node", return_value="diag_emitted"):
                arr = type("T", (), {"shape": (1, 1), "device": "cpu", "dtype": "float32"})()
                diag_out = diag(arr)
                assert getattr(diag_out, "shape", None) == (1,)
    finally:
        global_tracing_state.stop_tracing()
