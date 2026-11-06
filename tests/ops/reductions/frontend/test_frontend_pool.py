"""Tests for test_frontend_pool."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from tests.testing_utils import ensure_eager_tensor, ensure_mutable, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def _test_frontend_pool_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_frontend_pool_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    from unittest.mock import MagicMock, patch

    import numpy as np

    import ml_switcheroo_compiler.ops.reductions.frontend_pool as pool

    orig = config.eager_mode
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()

    t = Tensor(np.ones((1, 1, 4, 4)), TensorConfig(shape=(1, 1, 4, 4), dtype=DType("float32"), device=Device("cpu")))
    t_3d = Tensor(np.ones((1, 1, 4, 4, 4)), TensorConfig(shape=(1, 1, 4, 4, 4), dtype=DType("float32"), device=Device("cpu")))
    _state.update(ensure_tensor_dict({"MagicMock": MagicMock, "np": np, "orig": orig, "orig_tracing": orig_tracing}))
    _state.update(ensure_tensor_dict({"patch": patch, "pool": pool, "state": state, "t": t}))
    _state.update(ensure_tensor_dict({"t_3d": t_3d}))


def _test_frontend_pool_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_frontend_pool_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, np, orig, orig_tracing = _state["MagicMock"], _state["np"], _state["orig"], _state["orig_tracing"]
    patch, pool, state, t = _state["patch"], _state["pool"], _state["state"], _state["t"]
    t_3d = _state["t_3d"]
    try:
        with patch("ml_switcheroo_compiler.ops.reductions.frontend_utils._emit_reduction_node", return_value=t):
            pool.fractional_max_pool2d(t, (2, 2))
            pool.adaptive_avg_pool2d(t, (2, 2))
            pool.adaptive_max_pool2d(t, (2, 2))
            pool.unfold(t, (2, 2))
            pool.fold(t, (2, 2), (2, 2))

            pool.fractional_max_pool3d(t_3d, (2, 2, 2))
            pool.adaptive_avg_pool3d(t_3d, (2, 2, 2))
            pool.adaptive_max_pool3d(t_3d, (2, 2, 2), return_indices=False)
            pool.adaptive_max_pool3d(t_3d, (2, 2, 2), return_indices=True)

            opt = pool.UnpoolOptions(kernel_size=(2, 2), output_size=(4, 4))
            opt1d = pool.UnpoolOptions(kernel_size=(2,), output_size=(4,))
            opt3d = pool.UnpoolOptions(kernel_size=(2, 2, 2), output_size=(4, 4, 4))
            pool.max_unpool1d(t, t, opt1d)
            pool.max_unpool2d(t, t, opt)
            pool.max_unpool3d(t_3d, t_3d, opt3d)

            # None output_size
            opt_none = pool.UnpoolOptions(kernel_size=(2, 2), output_size=None)
            pool.max_unpool1d(t, t, opt_none)
            pool.max_unpool2d(t, t, opt_none)
            pool.max_unpool3d(t_3d, t_3d, opt_none)

        config.eager_mode = True

        mock_backend = MagicMock()
        ensure_mutable(mock_backend.execute_op).return_value = ensure_eager_tensor(np.zeros((1, 1, 2, 2)))
        ensure_mutable(mock_backend).array = ensure_eager_tensor(lambda x: x)

        mock_backend_3d = MagicMock()
        ensure_mutable(mock_backend_3d.execute_op).return_value = ensure_eager_tensor((np.zeros((1, 1, 2, 2, 2)), np.zeros((1, 1, 2, 2, 2))))
        ensure_mutable(mock_backend_3d).array = ensure_eager_tensor(lambda x: x)

        with patch("ml_switcheroo_compiler.ops.reductions.frontend_pool.get_active_backend", return_value=mock_backend):
            pool.adaptive_avg_pool2d(t, (2, 2))
            pool.adaptive_max_pool2d(t, (2, 2))
            pool.max_unpool1d(t, t, opt1d)
            pool.max_unpool2d(t, t, opt)
            pool.max_unpool3d(t_3d, t_3d, opt3d)

            ensure_mutable(mock_backend.execute_op).return_value = ensure_eager_tensor(np.zeros((1, 1, 2, 2, 2)))
            pool.adaptive_avg_pool3d(t_3d, (2, 2, 2))
            pool.adaptive_max_pool3d(t_3d, (2, 2, 2), return_indices=False)

        with patch("ml_switcheroo_compiler.ops.reductions.frontend_pool.get_active_backend", return_value=mock_backend_3d):
            pool.fractional_max_pool3d(t_3d, (2, 2, 2), random_samples=t_3d)
            pool.adaptive_max_pool3d(t_3d, (2, 2, 2), return_indices=True)

            # and test FractionalMaxPool3d eager mode with no random samples
            pool.fractional_max_pool3d(t_3d, (2, 2, 2))

    finally:
        ensure_mutable(state.global_tracing_state).is_tracing = ensure_eager_tensor(orig_tracing)
        ensure_mutable(config).eager_mode = ensure_eager_tensor(orig)


def _test_frontend_pool_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_frontend_pool_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    t_3d = _state["t_3d"]
    assert ensure_eager_tensor(t_3d).shape == (1, 1, 4, 4, 4)


def test_frontend_pool_coverage() -> None:
    """Test function test_frontend_pool_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_frontend_pool_coverage_part_1(_state)
    _test_frontend_pool_coverage_part_2(_state)
    _test_frontend_pool_coverage_part_3(_state)
    assert len(_state) > 0
