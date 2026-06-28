from ml_switcheroo_compiler.ops.binary.math import Betainc
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.device import Device
import numpy as np


def test_betainc_infer_shape():
    device = Device("cpu")
    t1 = Tensor(np.ones((2,)), TensorConfig((2,), "float32", device))
    t2 = Tensor(np.ones((2,)), TensorConfig((2,), "float32", device))
    opdef = Betainc()
    # just trigger it
    opdef.infer_shape(t1, t2)
