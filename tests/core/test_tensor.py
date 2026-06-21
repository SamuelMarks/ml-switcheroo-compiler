from ml_switcheroo_compiler.core.tensor import TensorConfig

"""Provides required module functionality."""


def test_tensor_coverage() -> None:
    """Execute the requested function."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor

    config.eager_mode = True

    t = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, Device("cpu")))
    assert t.__len__() == 2
    for _x in t:
        pass

    # Eager getitem
    t[0]

    config.eager_mode = False
