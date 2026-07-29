import numpy as np

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.dataset import ImageDataset
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def create_eager_tensor(data):
    backend = get_active_backend()
    data = backend.array(data)
    return Tensor(data, TensorConfig(data.shape, DType.Float32, Device("cpu")))


def test_image_dataset_missing_branches():
    empty_tensor = create_eager_tensor(np.zeros((0, 1, 1, 1)))
    d = ImageDataset(empty_tensor, target_size=(2, 2), normalize=True).batch(2)
    res = list(d)
    assert len(res) == 0

    t_3d = create_eager_tensor(np.ones((2, 2, 1)))
    d2 = ImageDataset(t_3d, target_size=(2, 2), normalize=False).batch(1)
    res2 = list(d2)
    assert len(res2) == 2
    assert res2[0][0].shape == (1, 2, 1)

    t_4d = create_eager_tensor(np.ones((1, 2, 2, 1)))
    d3 = ImageDataset(t_4d, target_size=None, normalize=False).batch(1)
    res3 = list(d3)
    assert len(res3) == 1
    assert res3[0][0].shape == (1, 2, 2, 1)

    t_norm = create_eager_tensor(np.full((1, 2, 2, 1), 255.0))
    d4 = ImageDataset(t_norm, target_size=None, normalize=True).batch(1)
    res4 = list(d4)
    # The normalization divides by 255
    np.testing.assert_array_equal(res4[0][0].numpy(), np.ones((1, 2, 2, 1)))
