"""Tests for test_updates."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.optimizers.updates import (
    AdadeltaConfig,
    AdagradConfig,
    AdamaxHyperparams,
    AdamHyperparams,
    AdamWHyperparams,
    ApplyAdagrad,
    ApplyAdam,
    ApplyFtrl,
    ApplyRMSProp,
    LionConfig,
    RMSPropHyperparams,
    SGDConfig,
    adadelta_update,
    adafactor_update,
    adagrad_update,
    adam_update,
    adamax_update,
    adamw_update,
    apply_adagrad,
    apply_adam,
    apply_ftrl,
    apply_rmsprop,
    lion_update,
    muon_update,
    rmsprop_update,
    sgd_update,
)
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_updates_eager_false() -> None:
    """Test function test_updates_eager_false."""
    from unittest.mock import patch

    import numpy as np

    import ml_switcheroo_compiler.ops.optimizers.updates as upd

    orig = config.eager_mode
    config.eager_mode = False

    try:
        t = Tensor(np.array([1.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))

        with patch("ml_switcheroo_compiler.ops.optimizers.updates._emit_shape_node", return_value=t):
            upd.apply_adam(t, t, t, t, 0.1)
            upd.apply_adagrad(t, t, t, 0.1)
            upd.apply_ftrl(t, t, t, t, 0.1)
            upd.apply_rmsprop(t, t, t, t, 0.1)

            for op_class in [upd.ApplyAdam, upd.ApplyAdagrad, upd.ApplyFtrl, upd.ApplyRMSProp]:
                op = op_class()
                assert op.infer_shape(t, t, t, t, t, t) == (1,)

            assert ensure_callable(upd.LionConfigOp)().infer_shape(t) == (1,)
    finally:
        config.eager_mode = orig


def _test_optimizers_updates_brute_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_optimizers_updates_brute.

    Args:
        _state: Shared test state dictionary.
    """
    config.backend = "numpy"
    config.eager_mode = True

    param = Tensor(np.random.rand(4).astype(np.float32), TensorConfig((4,), "float32", "cpu"))
    grad = Tensor(np.random.rand(4).astype(np.float32), TensorConfig((4,), "float32", "cpu"))

    # configs
    sgd_c = SGDConfig(lr=0.1, momentum=0.9, weight_decay=0.01)
    sgd_nesterov = SGDConfig(lr=0.1, momentum=0.9, weight_decay=0.01, nesterov=True)
    sgd_update(param, grad, sgd_c)
    state_sgd = ensure_tensor_dict({})
    ensure_callable(sgd_update)(param, grad, sgd_c, state_sgd)
    ensure_callable(sgd_update)(param, grad, sgd_c, state_sgd)
    ensure_callable(sgd_update)(param, grad, sgd_nesterov, state_sgd)

    adam_c = AdamHyperparams(lr=0.1, weight_decay=0.01)
    adam_update(param, grad, adam_c)
    state_adam = ensure_tensor_dict({})
    ensure_callable(adam_update)(param, grad, adam_c, state_adam)
    ensure_callable(adam_update)(param, grad, adam_c, state_adam)

    adamw_c = AdamWHyperparams(lr=0.1)
    adamw_update(param, grad, adamw_c)
    state_adamw = ensure_tensor_dict({})
    ensure_callable(adamw_update)(param, grad, adamw_c, state_adamw)
    ensure_callable(adamw_update)(param, grad, adamw_c, state_adamw)

    adagrad_c = AdagradConfig(lr=0.1, weight_decay=0.01)
    _state.update(ensure_tensor_dict({"adagrad_c": adagrad_c, "adam_c": adam_c, "adamw_c": adamw_c, "grad": grad}))
    _state.update(ensure_tensor_dict({"param": param, "sgd_c": sgd_c, "sgd_nesterov": sgd_nesterov, "state_adam": state_adam}))
    _state.update(ensure_tensor_dict({"state_adamw": state_adamw, "state_sgd": state_sgd}))


def _test_optimizers_updates_brute_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_optimizers_updates_brute.

    Args:
        _state: Shared test state dictionary.
    """
    adagrad_c, grad, param = _state["adagrad_c"], _state["grad"], _state["param"]
    ensure_callable(adagrad_update)(param, grad, adagrad_c)
    state_adagrad = ensure_tensor_dict({})
    ensure_callable(adagrad_update)(param, grad, adagrad_c, state_adagrad)
    rmsprop_c = RMSPropHyperparams(lr=0.1, weight_decay=0.01, centered=True, momentum=0.9)
    ensure_callable(rmsprop_update)(param, grad, rmsprop_c)
    state_rmsprop = ensure_tensor_dict({})
    ensure_callable(rmsprop_update)(param, grad, rmsprop_c, state_rmsprop)
    ensure_callable(rmsprop_update)(param, grad, rmsprop_c, state_rmsprop)

    rmsprop_c2 = RMSPropHyperparams(lr=0.1)
    ensure_callable(rmsprop_update)(param, grad, rmsprop_c2)

    adadelta_c = AdadeltaConfig(weight_decay=0.01)
    ensure_callable(adadelta_update)(param, grad, adadelta_c)
    state_adadelta = ensure_tensor_dict({})
    ensure_callable(adadelta_update)(param, grad, adadelta_c, state_adadelta)
    ensure_callable(adadelta_update)(param, grad, adadelta_c, state_adadelta)

    adamax_c = AdamaxHyperparams(lr=0.1, weight_decay=0.01)
    ensure_callable(adamax_update)(param, grad, adamax_c)
    state_adamax = ensure_tensor_dict({})
    ensure_callable(adamax_update)(param, grad, adamax_c, state_adamax)
    ensure_callable(adamax_update)(param, grad, adamax_c, state_adamax)

    lion_c = LionConfig(lr=0.1, weight_decay=0.01)
    ensure_callable(lion_update)(param, grad, lion_c)
    _state.update(ensure_tensor_dict({"adadelta_c": adadelta_c, "adamax_c": adamax_c, "lion_c": lion_c, "rmsprop_c": rmsprop_c}))
    _state.update(ensure_tensor_dict({"rmsprop_c2": rmsprop_c2, "state_adadelta": state_adadelta, "state_adagrad": state_adagrad, "state_adamax": state_adamax}))
    _state.update(ensure_tensor_dict({"state_rmsprop": state_rmsprop}))
    _state.update(ensure_tensor_dict({"adagrad_c": adagrad_c, "grad": grad, "param": param}))


def _test_optimizers_updates_brute_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_optimizers_updates_brute.

    Args:
        _state: Shared test state dictionary.
    """
    grad, lion_c, param = _state["grad"], _state["lion_c"], _state["param"]
    state_lion = ensure_tensor_dict({})
    ensure_callable(lion_update)(param, grad, lion_c, state_lion)

    ensure_callable(adafactor_update)(param, grad, 0.1)
    ensure_callable(adafactor_update)(param, grad, 0.1, {})

    ensure_callable(muon_update)(param, grad, 0.1)
    ensure_callable(muon_update)(param, grad, 0.1, momentum=0.9, state={})
    ensure_callable(apply_adam)(param, param, param, grad, 0.1)
    ensure_callable(apply_adagrad)(param, param, grad, 0.1)
    ensure_callable(apply_ftrl)(param, param, param, grad, 0.1)
    ensure_callable(apply_rmsprop)(param, param, param, grad, 0.1)

    # Tracing
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    _state.update(ensure_tensor_dict({"global_tracing_state": global_tracing_state, "state_lion": state_lion}))
    _state.update(ensure_tensor_dict({"grad": grad, "lion_c": lion_c, "param": param}))


def _test_optimizers_updates_brute_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_optimizers_updates_brute.

    Args:
        _state: Shared test state dictionary.
    """
    global_tracing_state, grad, param = _state["global_tracing_state"], _state["grad"], _state["param"]
    try:
        import sys

        updates_mod = sys.modules["ml_switcheroo_compiler.ops.optimizers.updates"]
        with patch.object(updates_mod, "_emit_shape_node") as mock_emit:
            mock_emit.return_value = param
            ensure_callable(apply_adam)(param, param, param, grad, 0.1)
            ensure_callable(apply_adagrad)(param, param, grad, 0.1)
            ensure_callable(apply_ftrl)(param, param, param, grad, 0.1)
            ensure_callable(apply_rmsprop)(param, param, param, grad, 0.1)

            ApplyAdam().infer_shape(param, param, param)
            ApplyAdagrad().infer_shape(param, param)
            ApplyFtrl().infer_shape(param, param, param)
            ApplyRMSProp().infer_shape(param, param, param)
    finally:
        ensure_mutable(global_tracing_state).is_tracing = ensure_eager_tensor(False)
    assert bool(ensure_eager_tensor(global_tracing_state).is_tracing) is False


def test_optimizers_updates_brute() -> None:
    """Test function test_optimizers_updates_brute."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_optimizers_updates_brute_part_1(_state)
    _test_optimizers_updates_brute_part_2(_state)
    _test_optimizers_updates_brute_part_3(_state)
    _test_optimizers_updates_brute_part_4(_state)
    assert len(_state) > 0
