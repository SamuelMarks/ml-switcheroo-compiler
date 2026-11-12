"""Tests for test_gru."""

from __future__ import annotations

# ruff: noqa: E501
import sys
import typing

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from tests.testing_utils import ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue

gru_mod = sys.modules["ml_switcheroo_compiler.ops.nn.gru"]


def test_gru_coverage() -> None:
    """Test function test_gru_coverage."""
    config.eager_mode = True
    t_in = Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    t_state = Tensor(np.ones((1, 4)), TensorConfig(shape=(1, 4), dtype=DType("float32"), device=Device("cpu")))
    w = Tensor(np.ones((2, 12)), TensorConfig(shape=(2, 12), dtype=DType("float32"), device=Device("cpu")))
    rw = Tensor(np.ones((4, 12)), TensorConfig(shape=(4, 12), dtype=DType("float32"), device=Device("cpu")))
    b = Tensor(np.ones((12,)), TensorConfig(shape=(12,), dtype=DType("float32"), device=Device("cpu")))

    def mock_matmul(*args: EagerValue) -> EagerValue:
        """Test function mock_matmul.

        Args:
            args: Test parameter `args`.

        Returns:
            EagerValue: Computed output value from `mock_matmul`.
        """
        return Tensor(np.ones((1, 12)), TensorConfig(shape=(1, 12), dtype=DType("float32"), device=Device("cpu")))

    def mock_gates(*args: EagerValue) -> EagerValue:
        """Test function mock_gates.

        Args:
            args: Test parameter `args`.

        Returns:
            EagerValue: Computed output value from `mock_gates`.
        """
        return "hidden"

    orig_matmul = gru_mod.matmul
    orig_gates = gru_mod._compute_gru_gates
    ensure_mutable(gru_mod).matmul = ensure_eager_tensor(mock_matmul)
    ensure_mutable(gru_mod)._compute_gru_gates = ensure_eager_tensor(mock_gates)
    try:
        res = gru_mod.gru_cell(t_in, t_state, w, rw, b)
        assert res == ("hidden", "hidden")
        res = gru_mod.gru_cell(t_in, t_state, w, rw)
        assert res == ("hidden", "hidden")
    finally:
        ensure_mutable(gru_mod).matmul = ensure_eager_tensor(orig_matmul)
        ensure_mutable(gru_mod)._compute_gru_gates = ensure_eager_tensor(orig_gates)

    assert gru_mod.Gru().infer_shape(t_in) is t_in
    assert gru_mod.Gru().infer_shape() == ()

    class MockBackend:
        """Test class MockBackend."""

        def execute_op(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `execute_op`.
            """
            return "gru_res"

    orig_backend = gru_mod.get_active_backend
    ensure_mutable(gru_mod).get_active_backend = ensure_eager_tensor(lambda: MockBackend())
    try:
        assert gru_mod.gru(t_in) == "gru_res"
    finally:
        ensure_mutable(gru_mod).get_active_backend = ensure_eager_tensor(orig_backend)

    config.eager_mode = False
    orig_emit = gru_mod._emit_shape_node

    def dummy_emit(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function dummy_emit.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            EagerValue: Computed output value from `dummy_emit`.
        """
        return "emitted"

    ensure_mutable(gru_mod)._emit_shape_node = ensure_eager_tensor(dummy_emit)
    try:
        assert gru_mod.gru(t_in) == "emitted"
    finally:
        ensure_mutable(gru_mod)._emit_shape_node = ensure_eager_tensor(orig_emit)
        config.eager_mode = True
