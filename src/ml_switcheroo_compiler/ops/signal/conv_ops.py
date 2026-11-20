"""2D signal convolution operations."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import (
    get_active_backend,
)
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .common_ops import _calculate_padding, _emit_signal_node


@typing.runtime_checkable
class _OptionalShapeProto(typing.Protocol):
    """Protocol for tensor objects exposing an optional `.shape` attribute."""

    @property
    def shape(self) -> Sequence[int | str | None] | None:
        """Return optional tensor shape.

        Returns:
            Sequence[int | str | None] | None: Tensor shape or `None`.
        """
        ...


@register_op("Convolve2d")
class Convolve2d(OpDef):
    """2D discrete signal convolution operation definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer output shape for `Convolve2d` from the first input tensor.

        Args:
            *args (OpValue): Input 2D signal and kernel tensors.
            **kwargs (OpValue): Convolution mode and boundary attributes.

        Returns:
            OpValue: Inferred 2D output shape tuple.
        """
        _ = kwargs
        return extract_shape(args[0])


def _validate_conv2d_args(in1: Tensor, in2: Tensor) -> None:
    """Validate that input tensors to `convolve2d` have statically known shapes.

    Args:
        in1 (Tensor): First 2D input tensor.
        in2 (Tensor): Second 2D kernel tensor.

    Raises:
        ValueError: If either input shape is `None`.
    """
    raw_in1: OpValue | _OptionalShapeProto = in1
    raw_in2: OpValue | _OptionalShapeProto = in2
    if (
        not isinstance(raw_in1, _OptionalShapeProto)
        or raw_in1.shape is None
        or not isinstance(raw_in2, _OptionalShapeProto)
        or raw_in2.shape is None
    ):
        raise ValueError("Inputs to convolve2d must have statically known shapes.")


def convolve2d(
    in1: Tensor,
    in2: Tensor,
    mode: str = "full",
    boundary: str = "fill",
    fillvalue: float = 0.0,
) -> OpValue:
    """Convolve two 2-dimensional arrays `in1` and `in2`.

    Args:
        in1 (Tensor): First 2D input signal tensor.
        in2 (Tensor): Second 2D kernel tensor.
        mode (str): Output size mode (`'full'`, `'valid'`, or `'same'`).
        boundary (str): Boundary condition (`'fill'`, `'wrap'`, or `'symm'`).
        fillvalue (float): Value to fill pad regions when `boundary == 'fill'`.

    Returns:
        OpValue: 2D convolved output `Tensor`.
    """
    _validate_conv2d_args(in1, in2)
    kwargs = _calculate_padding(mode, boundary, fillvalue)

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Convolve2d", in1.data, in2.data, **kwargs)
        return Tensor(data, TensorConfig(extract_shape(data), in1.dtype, in1.device))

    return _emit_signal_node(
        "Convolve2d",
        [in1, in2],
        kwargs,
        in1.shape,
        in1.dtype,
    )
