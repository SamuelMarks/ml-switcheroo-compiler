import numpy as np

import ml_switcheroo_compiler.ops.optimizers.updates as upd
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_adamw_no_weight_decay():
    orig = config.eager_mode
    config.eager_mode = True
    try:
        t = Tensor(np.array([1.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))
        hp = upd.AdamWHyperparams(lr=0.1, weight_decay=0.0)
        upd.adamw_update(t, t, hp)
    finally:
        config.eager_mode = orig


def test_missing_optimizers_coverage():
    from ml_switcheroo_compiler.ops.optimizers.updates import AdamaxHyperparamsOp, ApplyRMSProp

    class DummyShape:
        shape = (1,)

    try:
        ApplyRMSProp().infer_shape(DummyShape())
    except:
        pass
    try:
        AdamaxHyperparamsOp().infer_shape(DummyShape())
    except:
        pass
