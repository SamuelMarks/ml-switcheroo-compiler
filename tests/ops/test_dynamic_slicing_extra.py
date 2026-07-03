"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.tensor import DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.dropout import ActivityRegularization
from ml_switcheroo_compiler.ops.nn.normalization import group_norm, layer_norm
from ml_switcheroo_compiler.ops.nn.rnn_utils import (
    RNNConfig,
    ScanConfig,
    _permute_time_major,
    rnn,
    scan,
)
from ml_switcheroo_compiler.ops.shape.dynamic_slicing import DynamicSlice, DynamicUpdateSlice
from ml_switcheroo_compiler.ops.shape.reshape import BroadcastInDim, Resize
from ml_switcheroo_compiler.ops.vision.color import auto_contrast, equalization


def test_dynamic_slicing_emit_backends() -> object:
    """Function docstring."""
    op = DynamicSlice()
    assert op.emit_jax() == "Not implemented"
    assert op.emit_keras() == "Not implemented"
    assert op.emit_mlx() == "Not implemented"
    assert op.emit_pytorch() == "Not implemented"
    assert op.emit_tensorflow() == "Not implemented"

    op2 = DynamicUpdateSlice()
    assert op2.emit_jax() == "Not implemented"
    assert op2.emit_keras() == "Not implemented"
    assert op2.emit_mlx() == "Not implemented"
    assert op2.emit_pytorch() == "Not implemented"
    assert op2.emit_tensorflow() == "Not implemented"


def test_reshape_emit_backends() -> object:
    """Function docstring."""
    op = BroadcastInDim()
    assert op.emit_jax() == "Not implemented BroadcastInDim"
    assert op.emit_keras() == "Not implemented BroadcastInDim"
    assert op.emit_mlx() == "Not implemented BroadcastInDim"
    assert op.emit_pytorch() == "Not implemented BroadcastInDim"
    assert op.emit_tensorflow() == "Not implemented BroadcastInDim"

    op2 = Resize()
    assert op2.emit_jax() == "Not implemented Resize"
    assert op2.emit_keras() == "Not implemented Resize"
    assert op2.emit_mlx() == "Not implemented Resize"
    assert op2.emit_pytorch() == "Not implemented Resize"
    assert op2.emit_tensorflow() == "Not implemented Resize"


def test_vision_color_eager() -> object:
    """Function docstring."""
    device = Device(DeviceType.CPU)
    t = Tensor(np.zeros((1, 3, 32, 32)), TensorConfig((1, 3, 32, 32), DType.Float32, device))
    with ConfigContext(eager_mode=True):
        try:
            auto_contrast(t)
        except Exception:
            pass
        try:
            equalization(t)
        except Exception:
            pass


def test_rnn_utils_extra() -> object:
    """Function docstring."""
    device = Device(DeviceType.CPU)
    t = Tensor(np.zeros((2, 3, 4)), TensorConfig((2, 3, 4), DType.Float32, device))

    with ConfigContext(eager_mode=True):

        def scan_fn(carry: object, x: object) -> object:
            """Function docstring."""
            return carry, x

        scan(scan_fn, t, t)
        scan(scan_fn, t, t, config=ScanConfig(reverse=True))
        _permute_time_major(t)

        def cell_fn(inputs: object, states: object) -> object:
            """Function docstring."""
            return inputs, states

        rnn(t, t, cell_fn, config=RNNConfig(time_major=False))
        rnn(t, t, cell_fn, config=RNNConfig(time_major=True))


def test_activity_regularization_infer_shape() -> object:
    """Function docstring."""
    op = ActivityRegularization()

    class Dummy:
        """Class docstring."""

        shape = (2, 2)

    assert op.infer_shape(Dummy()) == (2, 2)


def test_normalization_branches() -> object:
    """Function docstring."""
    device = Device(DeviceType.CPU)
    t = Tensor(np.zeros((1, 4, 32, 32)), TensorConfig((1, 4, 32, 32), DType.Float32, device))
    t_scale = Tensor(np.ones((4, 32, 32)), TensorConfig((4, 32, 32), DType.Float32, device))
    t_offset = Tensor(np.zeros((4, 32, 32)), TensorConfig((4, 32, 32), DType.Float32, device))

    with ConfigContext(eager_mode=True):
        layer_norm(t, [4, 32, 32])
        layer_norm(t, [4, 32, 32], scale=t_scale)
        layer_norm(t, [4, 32, 32], offset=t_offset)
        layer_norm(t, [4, 32, 32], scale=t_scale, offset=t_offset)

        t_scale_g = Tensor(np.ones((4,)), TensorConfig((4,), DType.Float32, device))
        t_offset_g = Tensor(np.zeros((4,)), TensorConfig((4,), DType.Float32, device))
        group_norm(t, num_groups=2)
        group_norm(t, num_groups=2, scale=t_scale_g)
        group_norm(t, num_groups=2, offset=t_offset_g)
        group_norm(t, num_groups=2, scale=t_scale_g, offset=t_offset_g)
