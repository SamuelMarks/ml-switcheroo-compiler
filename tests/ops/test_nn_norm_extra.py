"""Module docstring."""

from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.normalization import (
    BatchNormConfig,
    batch_normalization,
    group_norm,
    instance_norm,
    layer_norm,
    rms_normalization,
)


def test_norm_extra() -> object:
    """Function docstring."""
    device = Device("cpu")
    t1 = Tensor(np.ones((2, 4, 3, 3)), TensorConfig((2, 4, 3, 3), DType.Float32, device))
    t1_fail = Tensor(np.ones((2, 5, 3, 3)), TensorConfig((2, 5, 3, 3), DType.Float32, device))
    scale = Tensor(np.ones((3, 3)), TensorConfig((3, 3), DType.Float32, device))
    offset = Tensor(np.ones((3, 3)), TensorConfig((3, 3), DType.Float32, device))

    scale_gn = Tensor(np.ones((4,)), TensorConfig((4,), DType.Float32, device))
    offset_gn = Tensor(np.ones((4,)), TensorConfig((4,), DType.Float32, device))

    with ConfigContext(eager_mode=True):
        with patch("ml_switcheroo_compiler.ops.divide") as mock_divide:
            with patch("ml_switcheroo_compiler.ops.multiply") as mock_multiply:
                with patch("ml_switcheroo_compiler.ops.add") as mock_add:
                    with patch("ml_switcheroo_compiler.ops.shape.reshape") as mock_reshape:
                        with patch("ml_switcheroo_compiler.ops.reductions.mean") as mock_mean:
                            mock_divide.return_value = t1
                            mock_multiply.return_value = t1
                            mock_add.return_value = t1
                            # need a grouped shape for the mean mock to not fail on numpy
                            # group_shape = (2, 2, 2, 3, 3)
                            mock_reshape.return_value = Tensor(
                                np.ones((2, 2, 2, 3, 3)),
                                TensorConfig((2, 2, 2, 3, 3), DType.Float32, device),
                            )
                            mock_mean.return_value = Tensor(
                                np.ones((2, 2, 2, 1, 1)),
                                TensorConfig((2, 2, 2, 1, 1), DType.Float32, device),
                            )

                            # Layer norm
                            res1 = layer_norm(t1, normalized_shape=(3, 3), scale=scale, offset=offset)
                            assert res1 is not None

                            # Group norm
                            res2 = group_norm(t1, num_groups=2, scale=scale_gn, offset=offset_gn)
                            assert res2 is not None

                            with pytest.raises(ValueError):
                                group_norm(t1_fail, num_groups=2)

                            # Instance norm
                            res3 = instance_norm(t1, scale=scale_gn, offset=offset_gn)
                            assert res3 is not None


def test_norm_extra_batch_rms() -> object:
    """Function docstring."""
    device = Device("cpu")
    t1 = Tensor(np.ones((2, 4, 3, 3)), TensorConfig((2, 4, 3, 3), DType.Float32, device))
    mean = Tensor(np.ones((2, 4, 3, 3)), TensorConfig((2, 4, 3, 3), DType.Float32, device))
    var = Tensor(np.ones((2, 4, 3, 3)), TensorConfig((2, 4, 3, 3), DType.Float32, device))
    scale = Tensor(np.ones((2, 4, 3, 3)), TensorConfig((2, 4, 3, 3), DType.Float32, device))
    offset = Tensor(np.ones((2, 4, 3, 3)), TensorConfig((2, 4, 3, 3), DType.Float32, device))

    with ConfigContext(eager_mode=True):
        with patch("ml_switcheroo_compiler.ops.divide") as mock_divide:
            with patch("ml_switcheroo_compiler.ops.multiply") as mock_multiply:
                with patch("ml_switcheroo_compiler.ops.add") as mock_add:
                    with patch("ml_switcheroo_compiler.ops.reductions.mean") as mock_mean:
                        mock_divide.return_value = t1
                        mock_multiply.return_value = t1
                        mock_add.return_value = t1
                        mock_mean.return_value = t1

                        conf = BatchNormConfig(epsilon=1e-5, scale=scale, offset=offset)
                        res1 = batch_normalization(t1, mean, var, axis=1, config=conf)
                        assert res1 is not None

                        res1_none = batch_normalization(t1, mean, var, axis=1, config=None)
                        assert res1_none is not None

                        scale_rms = Tensor(np.ones((3,)), TensorConfig((3,), DType.Float32, device))
                        res2 = rms_normalization(t1, scale=scale_rms)
                        assert res2 is not None
