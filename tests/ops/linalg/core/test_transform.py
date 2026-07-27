# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.transform import hadamard_transform


def test_hadamard_transform() -> None:
    config.eager_mode = True
    input_data = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    res = hadamard_transform(t, scale=1.0)
    assert isinstance(res, Tensor)
    assert res.shape == (4,)
    res2 = hadamard_transform(t, scale=0.5)
    assert isinstance(res2, Tensor)
    assert res2.shape == (4,)
