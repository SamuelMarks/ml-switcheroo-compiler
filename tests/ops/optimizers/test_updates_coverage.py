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
