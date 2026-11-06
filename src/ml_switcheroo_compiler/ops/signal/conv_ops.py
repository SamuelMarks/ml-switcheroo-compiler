"""2D signal convolution operations."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, ensure_callable, ensure_eager_tensor, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import (
    ShapeProtocol,
    TensorOrScalar,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .common_ops import _calculate_padding, _emit_signal_node


@register_op("Convolve2d")
class Convolve2d(OpDef):
    """2D discrete signal convolution operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Convolve2d` from the first input tensor.

        Args:
            *args (EagerTensorProtocol): Input 2D signal and kernel tensors.
            **kwargs (TensorOrScalar): Convolution mode and boundary attributes.

        Returns:
            TensorOrScalar: Inferred 2D output shape tuple.
        """
        _ = kwargs
        return args[0].shape


def _validate_conv2d_args(in1: Tensor, in2: Tensor) -> None:
    """Validate that input tensors to `convolve2d` have statically known shapes.

    Args:
        in1 (Tensor): First 2D input tensor.
        in2 (Tensor): Second 2D kernel tensor.

    Raises:
        ValueError: If either input shape is `None`.
    """
    raw_in1: TensorOrScalar = in1
    raw_in2: TensorOrScalar = in2
    if not isinstance(raw_in1, ShapeProtocol) or raw_in1.shape is None or not isinstance(raw_in2, ShapeProtocol) or raw_in2.shape is None:
        raise ValueError("Inputs to convolve2d must have statically known shapes.")


def convolve2d(
    in1: Tensor,
    in2: Tensor,
    mode: str = "full",
    boundary: str = "fill",
    fillvalue: float = 0.0,
) -> TensorOrScalar:
    """Convolve two 2-dimensional arrays `in1` and `in2`.

    Args:
        in1 (Tensor): First 2D input signal tensor.
        in2 (Tensor): Second 2D kernel tensor.
        mode (str): Output size mode (`'full'`, `'valid'`, or `'same'`).
        boundary (str): Boundary condition (`'fill'`, `'wrap'`, or `'symm'`).
        fillvalue (float): Value to fill pad regions when `boundary == 'fill'`.

    Returns:
        TensorOrScalar: 2D convolved output `Tensor`.
    """
    _validate_conv2d_args(in1, in2)
    kwargs = _calculate_padding(mode, boundary, fillvalue)

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Convolve2d", in1.data, in2.data, **kwargs)
        return Tensor(data, TensorConfig(extract_shape(data), in1.dtype, in1.device))

    return ensure_eager_tensor(
        ensure_callable(_emit_signal_node)(
            "Convolve2d",
            [in1, in2],
            kwargs,
            in1.shape,
            in1.dtype,
        )
    )
