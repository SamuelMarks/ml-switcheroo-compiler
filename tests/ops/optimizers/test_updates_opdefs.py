# ruff: noqa: D103
"""Tests for optimizer extras."""

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.optimizers.updates import (
    ApplyAdagrad,
    ApplyAdam,
    ApplyFtrl,
    ApplyRMSProp,
    apply_adagrad,
    apply_adam,
    apply_ftrl,
    apply_rmsprop,
)


def test_updates_extras_old() -> None:
    backend = get_active_backend()
    dev = Device("cpu")
    t1 = Tensor(backend.array([1.0]), TensorConfig((1,), DType.Float32, dev))

    # Eager mode
    config.eager_mode = True
    try:
        apply_adam(t1, t1, t1, t1, 0.1)
    except ValueError:
        pass
    try:
        apply_adagrad(t1, t1, t1, 0.1)
    except ValueError:
        pass
    try:
        apply_ftrl(t1, t1, t1, t1, 0.1)
    except ValueError:
        pass
    try:
        apply_rmsprop(t1, t1, t1, t1, 0.1)
    except ValueError:
        pass

    # Tracing mode
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
    # OpDefs
    op1 = ApplyAdam()
    assert op1.infer_shape(None, None, None) == ()

    op2 = ApplyAdagrad()
    assert op2.infer_shape(None, None) == ()

    op3 = ApplyFtrl()
    assert op3.infer_shape(None, None, None) == ()

    op4 = ApplyRMSProp()
    assert op4.infer_shape(None, None, None) == ()


def test_updates_extras_full():
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
    state.global_tracing_state.add_node = lambda node: None

    try:
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
        updates.lion_update(t, t, updates.LionConfig(0.1))

        updates.adafactor_update(t, t, 0.1, state={"step": t, "exp_avg_sq_row": t, "exp_avg_sq_col": t})
        updates.muon_update(t, t, 0.1, state={"momentum": t})

        # Test missing branches in adafactor/muon (None state)
        updates.adafactor_update(t, t, 0.1)
        updates.muon_update(t, t, 0.1)

        # apply funcs
        updates.apply_adam(t, t, t, t, 0.1)
        updates.apply_adagrad(t, t, t, 0.1)
        updates.apply_ftrl(t, t, t, t, 0.1)
        updates.apply_rmsprop(t, t, t, t, 0.1)

        # Test eager
        config.eager_mode = True
        updates.apply_adam(t, t, t, t, 0.1)
        updates.apply_adagrad(t, t, t, 0.1)
        updates.apply_ftrl(t, t, t, t, 0.1)
        updates.apply_rmsprop(t, t, t, t, 0.1)  # Test eager when backend throws ValueError

        class MockBackend:
            def execute_op(self, *args, **kwargs):
                raise ValueError("mock")

        import ml_switcheroo_compiler.ops.optimizers.updates as updates

        orig_get = updates.get_active_backend
        updates.get_active_backend = lambda: MockBackend()

        # In updates.py it does:
        # if config.eager_mode:
        #     backend = get_active_backend()

        try:
            updates.apply_adam(t, t, t, t, 0.1)
            updates.apply_adagrad(t, t, t, 0.1)
            updates.apply_ftrl(t, t, t, t, 0.1)
            updates.apply_rmsprop(t, t, t, t, 0.1)
        finally:
            updates.get_active_backend = orig_get
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node


def test_apply_tracing():
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
    state.global_tracing_state.add_node = lambda node: None

    try:
        t = Tensor(np.array([[[1.0, 2.0]]]), TensorConfig((2, 1, 2), DType("float32"), Device("cpu")))

        updates.apply_adam(t, t, t, t, 0.1)
        updates.apply_adagrad(t, t, t, 0.1)
        updates.apply_ftrl(t, t, t, t, 0.1)
        updates.apply_rmsprop(t, t, t, t, 0.1)
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node
