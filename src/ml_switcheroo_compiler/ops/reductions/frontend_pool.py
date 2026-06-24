"""Frontend reductions ops."""

from __future__ import annotations


from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.tensor import Tensor


if TYPE_CHECKING:
    pass


from .frontend_utils import _emit_reduction_node


def fractional_max_pool2d(
    operand: Tensor,
    output_size: tuple[int, int],
) -> Tensor:
    """Fractional max pooling 2D.

    Args:
        operand (Tensor): The input tensor.
        output_size (tuple[int, int]): The output size.

    Returns:
        Tensor: The pooled tensor.
    """
    out_shape = list(operand.shape)
    if len(out_shape) >= MAGIC_VAL_2:  # pragma: no branch
        out_shape[-2] = output_size[0]
        out_shape[-1] = output_size[1]
    return _emit_reduction_node(
        "FractionalMaxPool2D",
        [operand],
        {"output_size": output_size},
        tuple(out_shape),
        operand.dtype,
    )


def adaptive_avg_pool2d(
    operand: Tensor,
    output_size: tuple[int, int],
) -> Tensor:
    """Adaptive average pooling 2D.

    Args:
        operand (Tensor): The input tensor.
        output_size (tuple[int, int]): The output size.

    Returns:
        Tensor: The pooled tensor.
    """
    out_shape = list(operand.shape)
    if len(out_shape) >= MAGIC_VAL_2:  # pragma: no branch
        out_shape[-2] = output_size[0]
        out_shape[-1] = output_size[1]
    return _emit_reduction_node(
        "AdaptiveAvgPool2D",
        [operand],
        {"output_size": output_size},
        tuple(out_shape),
        operand.dtype,
    )


def adaptive_max_pool2d(
    operand: Tensor,
    output_size: tuple[int, int],
) -> Tensor:
    """Adaptive max pooling 2D.

    Args:
        operand (Tensor): The input tensor.
        output_size (tuple[int, int]): The output size.

    Returns:
        Tensor: The pooled tensor.
    """
    out_shape = list(operand.shape)
    if len(out_shape) >= MAGIC_VAL_2:  # pragma: no branch
        out_shape[-2] = output_size[0]
        out_shape[-1] = output_size[1]
    return _emit_reduction_node(
        "AdaptiveMaxPool2D",
        [operand],
        {"output_size": output_size},
        tuple(out_shape),
        operand.dtype,
    )


def unfold(
    operand: Tensor,
    kernel_size: tuple[int, int],
) -> Tensor:
    """Unfold (Im2Col) operator.

    Args:
        operand (Tensor): The input tensor.
        kernel_size (tuple[int, int]): The kernel size.

    Returns:
        Tensor: The unfolded tensor.
    """
    return _emit_reduction_node(
        "Unfold", [operand], {"kernel_size": kernel_size}, (), operand.dtype
    )


def fold(
    operand: Tensor,
    output_size: tuple[int, int],
    kernel_size: tuple[int, int],
) -> Tensor:
    """Fold (Col2Im) operator.

    Args:
        operand (Tensor): The input tensor.
        output_size (tuple[int, int]): The output size.
        kernel_size (tuple[int, int]): The kernel size.

    Returns:
        Tensor: The folded tensor.
    """
    return _emit_reduction_node(
        "Fold",
        [operand],
        {"output_size": output_size, "kernel_size": kernel_size},
        (),
        operand.dtype,
    )
