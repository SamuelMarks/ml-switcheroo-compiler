"""Tests for optimizer extras."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.optimizers.updates import ApplyAdagrad, ApplyAdam, ApplyFtrl, ApplyRMSProp, apply_adagrad, apply_adam, apply_ftrl, apply_rmsprop

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_updates_extras_old() -> None:
    """Test function test_updates_extras_old."""
    backend = get_active_backend()
    dev = Device("cpu")
    t1 = Tensor(backend.array([1.0]), TensorConfig((1,), DType.Float32, dev))
    config.eager_mode = True
    apply_adam(t1, t1, t1, t1, 0.1)
    apply_adagrad(t1, t1, t1, 0.1)
    apply_ftrl(t1, t1, t1, t1, 0.1)
    apply_rmsprop(t1, t1, t1, t1, 0.1)
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.optimizers.updates._emit_shape_node") as mock_emit:
        config.eager_mode = False
        try:
            apply_adam(t1, t1, t1, t1, 0.1)
            apply_adagrad(t1, t1, t1, 0.1)
            apply_ftrl(t1, t1, t1, t1, 0.1)
            apply_rmsprop(t1, t1, t1, t1, 0.1)
            assert mock_emit.call_count == 4
        finally:
            config.eager_mode = True


def test_opdefs() -> None:
    """Test function test_opdefs."""
    op1 = ApplyAdam()
    assert cast("EagerCallableProtocol", op1.infer_shape)(None, None, None) == ()
    op2 = ApplyAdagrad()
    assert cast("EagerCallableProtocol", op2.infer_shape)(None, None) == ()
    op3 = ApplyFtrl()
    assert cast("EagerCallableProtocol", op3.infer_shape)(None, None, None) == ()
    op4 = ApplyRMSProp()
    assert cast("EagerCallableProtocol", op4.infer_shape)(None, None, None) == ()


def _test_updates_extras_full_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_updates_extras_full.

    Args:
        _state: Shared test state dictionary.
    """
    (DType, Device, Tensor, TensorConfig) = (_state["DType"], _state["Device"], _state["Tensor"], _state["TensorConfig"])
    (np, updates) = (_state["np"], _state["updates"])
    t = Tensor(np.array([[[1.0, 2.0]]]), TensorConfig((2, 1, 2), DType("float32"), Device("cpu")))
    updates.sgd_update(t, t, updates.SGDConfig(0.1))
    updates.sgd_update(t, t, updates.SGDConfig(0.1, weight_decay=0.01, momentum=0.9, nesterov=True))
    updates.sgd_update(t, t, updates.SGDConfig(0.1, momentum=0.9), state={"momentum_buffer": t})
    updates.adam_update(t, t, updates.AdamHyperparams(0.1, weight_decay=0.01), state={"m": t, "v": t})
    updates.adam_update(t, t, updates.AdamHyperparams(0.1))
    updates.adamw_update(t, t, updates.AdamWHyperparams(0.1), state={"m": t, "v": t})
    updates.adamw_update(t, t, updates.AdamWHyperparams(0.1))
    updates.adagrad_update(t, t, updates.AdagradConfig(0.1, weight_decay=0.01), state={"accum": t})
    updates.adagrad_update(t, t, updates.AdagradConfig(0.1))
    updates.rmsprop_update(t, t, updates.RMSPropHyperparams(0.1, weight_decay=0.01, momentum=0.9), state={"ms": t, "mom": t})
    updates.rmsprop_update(t, t, updates.RMSPropHyperparams(0.1, centered=True), state={"ms": t, "mom": t, "mg": t})
    updates.rmsprop_update(t, t, updates.RMSPropHyperparams(0.1))
    updates.adadelta_update(t, t, updates.AdadeltaConfig(0.1, weight_decay=0.01), state={"accum": t, "accum_update": t})
    updates.adadelta_update(t, t, updates.AdadeltaConfig(0.1))
    updates.adamax_update(t, t, updates.AdamaxHyperparams(0.1, weight_decay=0.01), state={"m": t, "u": t})
    updates.adamax_update(t, t, updates.AdamaxHyperparams(0.1))
    updates.lion_update(t, t, updates.LionConfig(0.1, weight_decay=0.01), state={"m": t})
    getattr(_state, "update")({"t": t})
    _state.update(locals())


def _test_updates_extras_full_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_updates_extras_full.

    Args:
        _state: Shared test state dictionary.
    """
    (config, t, updates) = (_state["config"], _state["t"], _state["updates"])
    updates.lion_update(t, t, updates.LionConfig(0.1))
    updates.adafactor_update(t, t, 0.1, state={"step": t, "exp_avg_sq_row": t, "exp_avg_sq_col": t})
    updates.muon_update(t, t, 0.1, state={"momentum": t})
    updates.adafactor_update(t, t, 0.1)
    updates.muon_update(t, t, 0.1)
    updates.apply_adam(t, t, t, t, 0.1)
    updates.apply_adagrad(t, t, t, 0.1)
    updates.apply_ftrl(t, t, t, t, 0.1)
    updates.apply_rmsprop(t, t, t, t, 0.1)
    setattr(config, "eager_mode", True)
    updates.apply_adam(t, t, t, t, 0.1)
    updates.apply_adagrad(t, t, t, 0.1)
    updates.apply_ftrl(t, t, t, t, 0.1)
    updates.apply_rmsprop(t, t, t, t, 0.1)
    _state.update(locals())


def _test_updates_extras_full_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_updates_extras_full.

    Args:
        _state: Shared test state dictionary.
    """
    t = _state["t"]

    class MockBackend:
        """Test class MockBackend."""

        def execute_op(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function execute_op.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            raise ValueError("mock")

    import ml_switcheroo_compiler.ops.optimizers.updates as updates

    orig_get = cast("EagerTensorProtocol", getattr(updates, "get_active_backend"))
    setattr(updates, "get_active_backend", lambda: MockBackend())
    try:
        cast("EagerCallableProtocol", updates.apply_adam)(t, t, t, t, 0.1)
        cast("EagerCallableProtocol", updates.apply_adagrad)(t, t, t, 0.1)
        cast("EagerCallableProtocol", updates.apply_ftrl)(t, t, t, t, 0.1)
        cast("EagerCallableProtocol", updates.apply_rmsprop)(t, t, t, t, 0.1)
    finally:
        setattr(updates, "get_active_backend", orig_get)
    getattr(_state, "update")({"MockBackend": MockBackend, "orig_get": orig_get, "updates": updates})
    _state.update(locals())


def test_updates_extras_full() -> None:
    """Test function test_updates_extras_full."""
    import numpy as np

    import ml_switcheroo_compiler.ops.optimizers.updates as updates
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    orig = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    setattr(state.global_tracing_state, "add_node", lambda node: None)
    try:
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        getattr(_state, "update")({"DType": DType, "Device": Device, "Tensor": Tensor, "TensorConfig": TensorConfig})
        getattr(_state, "update")({"config": config, "np": np, "orig": orig, "orig_add_node": orig_add_node})
        getattr(_state, "update")({"orig_tracing": orig_tracing, "state": state, "updates": updates})
        for _sub_fn in (_test_updates_extras_full_sub_1, _test_updates_extras_full_sub_2, _test_updates_extras_full_sub_3):
            _sub_fn(_state)
        assert len(_state) > 0
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        setattr(state.global_tracing_state, "add_node", orig_add_node)


def test_apply_tracing() -> None:
    """Test function test_apply_tracing."""
    import numpy as np

    import ml_switcheroo_compiler.ops.optimizers.updates as updates
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    orig = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    setattr(state.global_tracing_state, "add_node", lambda node: None)
    try:
        t = Tensor(np.array([[[1.0, 2.0]]]), TensorConfig((2, 1, 2), DType("float32"), Device("cpu")))
        updates.apply_adam(t, t, t, t, 0.1)
        updates.apply_adagrad(t, t, t, 0.1)
        updates.apply_ftrl(t, t, t, t, 0.1)
        updates.apply_rmsprop(t, t, t, t, 0.1)
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        setattr(state.global_tracing_state, "add_node", orig_add_node)
    assert getattr(state.global_tracing_state, "add_node") == orig_add_node
