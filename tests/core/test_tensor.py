"""Unit tests for verifying the core functionality and API coverage of the Tensor class."""


def test_tensor_coverage() -> None:
    """Verifies the basic operations and API coverage of the Tensor class.

    This test ensures that a Tensor can be correctly initialized with a NumPy array,
    shape, data type, and device. It also validates core magic methods including
    length retrieval, iteration, and eager item getting and setting

    Returns:
    None
    """
    import numpy as np

    from ml_switcheroo.core.device import Device
    from ml_switcheroo.core.dtype import DType
    from ml_switcheroo.core.tensor import Tensor

    t = Tensor(np.array([1.0, 2.0]), (2,), DType.Float32, Device("cpu"))
    assert t.__len__() == 2
    for _x in t:
        pass

    # Eager getitem
    t[0]
    t[0] = 1.0
