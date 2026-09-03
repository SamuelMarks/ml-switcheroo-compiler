from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_updates_eager_false():
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

            assert upd.LionConfigOp().infer_shape(t) == (1,)
    finally:
        config.eager_mode = orig


from unittest.mock import patch

import numpy as np

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


def test_optimizers_updates_brute():
    config.backend = "numpy"
    config.eager_mode = True

    param = Tensor(np.random.rand(4).astype(np.float32), TensorConfig((4,), "float32", "cpu"))
    grad = Tensor(np.random.rand(4).astype(np.float32), TensorConfig((4,), "float32", "cpu"))

    # configs
    sgd_c = SGDConfig(lr=0.1, momentum=0.9, weight_decay=0.01)
    sgd_nesterov = SGDConfig(lr=0.1, momentum=0.9, weight_decay=0.01, nesterov=True)
    sgd_update(param, grad, sgd_c)
    state_sgd = {}
    sgd_update(param, grad, sgd_c, state_sgd)
    sgd_update(param, grad, sgd_c, state_sgd)
    sgd_update(param, grad, sgd_nesterov, state_sgd)

    adam_c = AdamHyperparams(lr=0.1, weight_decay=0.01)
    adam_update(param, grad, adam_c)
    state_adam = {}
    adam_update(param, grad, adam_c, state_adam)
    adam_update(param, grad, adam_c, state_adam)

    adamw_c = AdamWHyperparams(lr=0.1)
    adamw_update(param, grad, adamw_c)
    state_adamw = {}
    adamw_update(param, grad, adamw_c, state_adamw)
    adamw_update(param, grad, adamw_c, state_adamw)

    adagrad_c = AdagradConfig(lr=0.1, weight_decay=0.01)
    adagrad_update(param, grad, adagrad_c)
    state_adagrad = {}
    adagrad_update(param, grad, adagrad_c, state_adagrad)

    rmsprop_c = RMSPropHyperparams(lr=0.1, weight_decay=0.01, centered=True, momentum=0.9)
    rmsprop_update(param, grad, rmsprop_c)
    state_rmsprop = {}
    rmsprop_update(param, grad, rmsprop_c, state_rmsprop)
    rmsprop_update(param, grad, rmsprop_c, state_rmsprop)

    rmsprop_c2 = RMSPropHyperparams(lr=0.1)
    rmsprop_update(param, grad, rmsprop_c2)

    adadelta_c = AdadeltaConfig(weight_decay=0.01)
    adadelta_update(param, grad, adadelta_c)
    state_adadelta = {}
    adadelta_update(param, grad, adadelta_c, state_adadelta)
    adadelta_update(param, grad, adadelta_c, state_adadelta)

    adamax_c = AdamaxHyperparams(lr=0.1, weight_decay=0.01)
    adamax_update(param, grad, adamax_c)
    state_adamax = {}
    adamax_update(param, grad, adamax_c, state_adamax)
    adamax_update(param, grad, adamax_c, state_adamax)

    lion_c = LionConfig(lr=0.1, weight_decay=0.01)
    lion_update(param, grad, lion_c)
    state_lion = {}
    lion_update(param, grad, lion_c, state_lion)

    adafactor_update(param, grad, 0.1)
    adafactor_update(param, grad, 0.1, {})

    muon_update(param, grad, 0.1)
    muon_update(param, grad, 0.1, momentum=0.9, state={})

    # Test the apply_* operations
    apply_adam(param, param, param, grad, 0.1)
    apply_adagrad(param, param, grad, 0.1)
    apply_ftrl(param, param, param, grad, 0.1)
    apply_rmsprop(param, param, param, grad, 0.1)

    # Tracing
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    try:
        import sys

        updates_mod = sys.modules["ml_switcheroo_compiler.ops.optimizers.updates"]
        with patch.object(updates_mod, "_emit_shape_node") as mock_emit:
            mock_emit.return_value = param
            apply_adam(param, param, param, grad, 0.1)
            apply_adagrad(param, param, grad, 0.1)
            apply_ftrl(param, param, param, grad, 0.1)
            apply_rmsprop(param, param, param, grad, 0.1)

            ApplyAdam().infer_shape(param, param, param)
            ApplyAdagrad().infer_shape(param, param)
            ApplyFtrl().infer_shape(param, param, param)
            ApplyRMSProp().infer_shape(param, param, param)
    finally:
        global_tracing_state.is_tracing = False
