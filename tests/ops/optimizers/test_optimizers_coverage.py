# ruff: noqa: E501
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
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


def test_optimizers_coverage():
    config.eager_mode = True
    t = Tensor(np.ones((2, 2)), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))
    grad = Tensor(np.ones((2, 2)), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))

    assert sgd_update(t, grad, SGDConfig(lr=0.1, weight_decay=0.1, momentum=0.9, nesterov=True)) is not None
    state = {}
    assert sgd_update(t, grad, SGDConfig(lr=0.1, momentum=0.9, nesterov=False), state) is not None
    assert sgd_update(t, grad, SGDConfig(lr=0.1, momentum=0.9, nesterov=False), state) is not None

    assert adam_update(t, grad, AdamHyperparams(lr=0.1, weight_decay=0.1)) is not None
    assert adamw_update(t, grad, AdamWHyperparams(lr=0.1, weight_decay=0.1)) is not None
    assert adagrad_update(t, grad, AdagradConfig(lr=0.1, weight_decay=0.1)) is not None

    assert rmsprop_update(t, grad, RMSPropHyperparams(lr=0.1, weight_decay=0.1, centered=True, momentum=0.9)) is not None
    state_rms = {}
    assert rmsprop_update(t, grad, RMSPropHyperparams(lr=0.1, weight_decay=0.1, centered=True, momentum=0.9), state_rms) is not None
    assert rmsprop_update(t, grad, RMSPropHyperparams(lr=0.1, weight_decay=0.1, centered=True, momentum=0.9), state_rms) is not None

    assert adadelta_update(t, grad, AdadeltaConfig(lr=0.1, weight_decay=0.1)) is not None
    assert adamax_update(t, grad, AdamaxHyperparams(lr=0.1, weight_decay=0.1)) is not None
    assert lion_update(t, grad, LionConfig(lr=0.1, weight_decay=0.1)) is not None
    assert adafactor_update(t, grad, 0.1) is not None
    assert muon_update(t, grad, 0.1) is not None

    assert ApplyAdam().infer_shape(t, t, t) == (2, 2)
    assert ApplyAdagrad().infer_shape(t, t) == (2, 2)
    assert ApplyFtrl().infer_shape(t, t, t) == (2, 2)
    assert ApplyRMSProp().infer_shape(t, t, t) == (2, 2)

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.side_effect = Exception("ValueError mock")

        original_eager = config.eager_mode
        try:
            config.eager_mode = False
            from ml_switcheroo_compiler.tracing.state import global_tracing_state

            global_tracing_state.is_tracing = True

            class DummyGraph:
                name = "dummy"
                nodes = {}

                def add_node(self, node):
                    pass

            global_tracing_state.active_graph = DummyGraph()

            assert apply_adam(t, t, t, grad, 0.1) is not None
            assert apply_adagrad(t, t, grad, 0.1) is not None
            assert apply_ftrl(t, t, t, grad, 0.1) is not None
            assert apply_rmsprop(t, t, t, grad, 0.1) is not None
        finally:
            config.eager_mode = original_eager
            global_tracing_state.is_tracing = False
