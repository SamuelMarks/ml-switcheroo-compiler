"""Module frontend_pool.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Frontend reductions ops."""


from dataclasses import dataclass

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from .frontend_utils import _emit_reduction_node


@dataclass
class UnpoolOptions:
    """Options for unpooling."""

    kernel_size: int | tuple
    stride: int | tuple | None = None
    padding: int | tuple = 0
    output_size: tuple[object, ...] | None = None


def fractional_max_pool2d(
    operand: Tensor,
    output_size: tuple[int, int],
) -> object:
    """Fractional max pooling 2D.

    Args:
        operand (Tensor): The input tensor.
        output_size (tuple[int, int]): The output size.

    Returns:
        Tensor: The pooled tensor.
    """
    out_shape: object = list(operand.shape)
    if len(out_shape) >= MAGIC_VAL_2:
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
) -> object:
    """Adaptive average pooling 2D.

    Args:
        operand (Tensor): The input tensor.
        output_size (tuple[int, int]): The output size.

    Returns:
        Tensor: The pooled tensor.
    """
    out_shape: object = list(operand.shape)
    if len(out_shape) >= MAGIC_VAL_2:
        out_shape[-2] = output_size[0]
        out_shape[-1] = output_size[1]

    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op(
            "AdaptiveAvgPool2D",
            operand.data,
            output_size=output_size,
        )
        return Tensor(backend.array(data), TensorConfig(tuple(out_shape), operand.dtype, operand.device))

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
) -> object:
    """Adaptive max pooling 2D.

    Args:
        operand (Tensor): The input tensor.
        output_size (tuple[int, int]): The output size.

    Returns:
        Tensor: The pooled tensor.
    """
    out_shape: object = list(operand.shape)
    if len(out_shape) >= MAGIC_VAL_2:
        out_shape[-2] = output_size[0]
        out_shape[-1] = output_size[1]

    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op(
            "AdaptiveMaxPool2D",
            operand.data,
            output_size=output_size,
        )
        return Tensor(backend.array(data), TensorConfig(tuple(out_shape), operand.dtype, operand.device))

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
) -> object:
    """Unfold (Im2Col) operator.

    Args:
        operand (Tensor): The input tensor.
        kernel_size (tuple[int, int]): The kernel size.

    Returns:
        Tensor: The unfolded tensor.
    """
    return _emit_reduction_node("Unfold", [operand], {"kernel_size": kernel_size}, (), operand.dtype)


def fold(
    operand: Tensor,
    output_size: tuple[int, int],
    kernel_size: tuple[int, int],
) -> object:
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


def fractional_max_pool3d(
    operand: Tensor,
    output_size: tuple[int, int, int],
    output_ratio: object = None,
    random_samples: object = None,
) -> object:
    """Fractional max pooling 3D.

    Args:
        operand (Tensor): The operand parameter.
        output_size (tuple): The output_size parameter.
        output_ratio (tuple): The output_ratio parameter.
        random_samples (Tensor): The random_samples parameter.

    Returns:
        tuple: Result.
    """
    out_shape: object = list(operand.shape)
    if len(out_shape) >= 3:
        out_shape[-3] = output_size[0]
        out_shape[-2] = output_size[1]
        out_shape[-1] = output_size[2]

    if config.eager_mode:
        backend: object = get_active_backend()
        data, indices = backend.execute_op(
            "FractionalMaxPool3D",
            operand.data,
            output_size=output_size,
            output_ratio=output_ratio,
            random_samples=random_samples.data if random_samples is not None else None,
        )
        return Tensor(backend.array(data), TensorConfig(tuple(out_shape), operand.dtype, operand.device)), Tensor(backend.array(indices), TensorConfig(tuple(out_shape), "int64", operand.device))

    pooled: object = _emit_reduction_node(
        "FractionalMaxPool3D",
        [operand],
        {"output_size": output_size, "output_ratio": output_ratio, "random_samples": random_samples},
        tuple(out_shape),
        operand.dtype,
    )
    indices_tensor: object = _emit_reduction_node(
        "FractionalMaxPool3D_Indices",
        [operand],
        {"output_size": output_size, "output_ratio": output_ratio, "random_samples": random_samples},
        tuple(out_shape),
        "int64",
    )
    return pooled, indices_tensor


def adaptive_avg_pool3d(
    operand: Tensor,
    output_size: tuple[int, int, int],
) -> object:
    """Adaptive average pooling 3D.

    Args:
        operand (Tensor): The input tensor.
        output_size (tuple[int, int, int]): The output size.

    Returns:
        Tensor: The pooled tensor.
    """
    out_shape: object = list(operand.shape)
    if len(out_shape) >= 3:
        out_shape[-3] = output_size[0]
        out_shape[-2] = output_size[1]
        out_shape[-1] = output_size[2]

    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op(
            "AdaptiveAvgPool3D",
            operand.data,
            output_size=output_size,
        )
        return Tensor(backend.array(data), TensorConfig(tuple(out_shape), operand.dtype, operand.device))

    return _emit_reduction_node(
        "AdaptiveAvgPool3D",
        [operand],
        {"output_size": output_size},
        tuple(out_shape),
        operand.dtype,
    )


def adaptive_max_pool3d(
    operand: Tensor,
    output_size: tuple[int, int, int],
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Adaptive max pooling 3D.

    Args:
        operand (Tensor): The input tensor.
        output_size (tuple[int, int, int]): The output size.
        return_indices (bool): Whether to return indices.

    Returns:
        Tensor | tuple[Tensor, Tensor]: The pooled tensor, or a tuple of (pooled, indices).
    """
    out_shape: object = list(operand.shape)
    if len(out_shape) >= 3:
        out_shape[-3] = output_size[0]
        out_shape[-2] = output_size[1]
        out_shape[-1] = output_size[2]

    if config.eager_mode:
        backend: object = get_active_backend()
        if return_indices:
            data, indices = backend.execute_op(
                "AdaptiveMaxPool3D",
                operand.data,
                output_size=output_size,
                return_indices=True,
            )
            return Tensor(backend.array(data), TensorConfig(tuple(out_shape), operand.dtype, operand.device)), Tensor(backend.array(indices), TensorConfig(tuple(out_shape), "int64", operand.device))
        else:
            data: object = backend.execute_op(
                "AdaptiveMaxPool3D",
                operand.data,
                output_size=output_size,
                return_indices=False,
            )
            return Tensor(backend.array(data), TensorConfig(tuple(out_shape), operand.dtype, operand.device))

    pooled: object = _emit_reduction_node(
        "AdaptiveMaxPool3D",
        [operand],
        {"output_size": output_size, "return_indices": return_indices},
        tuple(out_shape),
        operand.dtype,
    )
    if return_indices:
        indices_tensor: object = _emit_reduction_node(
            "AdaptiveMaxPool3D_Indices",
            [operand],
            {"output_size": output_size, "return_indices": return_indices},
            tuple(out_shape),
            "int64",
        )
        return pooled, indices_tensor
    return pooled


def max_unpool1d(
    operand: Tensor,
    indices: Tensor,
    options: UnpoolOptions,
) -> object:
    """Max unpooling 1D.

    Args:
        operand (Tensor): The input tensor.
        indices (Tensor): The indices.
        options (UnpoolOptions): Options.

    Returns:
        Tensor: Unpooled tensor.
    """
    out_shape: object = list(operand.shape)
    if options.output_size is not None:
        out_shape[-1] = options.output_size[0]

    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op(
            "MaxUnpool1D",
            operand.data,
            indices=indices.data,
            kernel_size=options.kernel_size,
            stride=options.stride,
            padding=options.padding,
            output_size=options.output_size,
        )
        return Tensor(backend.array(data), TensorConfig(tuple(out_shape), operand.dtype, operand.device))

    return _emit_reduction_node(
        "MaxUnpool1D",
        [operand, indices],
        {
            "kernel_size": options.kernel_size,
            "stride": options.stride,
            "padding": options.padding,
            "output_size": options.output_size,
        },
        tuple(out_shape),
        operand.dtype,
    )


def max_unpool2d(
    operand: Tensor,
    indices: Tensor,
    options: UnpoolOptions,
) -> object:
    """Max unpooling 2D.

    Args:
        operand (Tensor): The input tensor.
        indices (Tensor): The indices.
        options (UnpoolOptions): Options.

    Returns:
        Tensor: Unpooled tensor.
    """
    out_shape: object = list(operand.shape)
    if options.output_size is not None:
        out_shape[-2] = options.output_size[0]
        out_shape[-1] = options.output_size[1]

    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op(
            "MaxUnpool2D",
            operand.data,
            indices=indices.data,
            kernel_size=options.kernel_size,
            stride=options.stride,
            padding=options.padding,
            output_size=options.output_size,
        )
        return Tensor(backend.array(data), TensorConfig(tuple(out_shape), operand.dtype, operand.device))

    return _emit_reduction_node(
        "MaxUnpool2D",
        [operand, indices],
        {
            "kernel_size": options.kernel_size,
            "stride": options.stride,
            "padding": options.padding,
            "output_size": options.output_size,
        },
        tuple(out_shape),
        operand.dtype,
    )


def max_unpool3d(
    operand: Tensor,
    indices: Tensor,
    options: UnpoolOptions,
) -> object:
    """Max unpooling 3D.

    Args:
        operand (Tensor): The input tensor.
        indices (Tensor): The indices.
        options (UnpoolOptions): Options.

    Returns:
        Tensor: Unpooled tensor.
    """
    out_shape: object = list(operand.shape)
    if options.output_size is not None:
        out_shape[-3] = options.output_size[0]
        out_shape[-2] = options.output_size[1]
        out_shape[-1] = options.output_size[2]

    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op(
            "MaxUnpool3D",
            operand.data,
            indices=indices.data,
            kernel_size=options.kernel_size,
            stride=options.stride,
            padding=options.padding,
            output_size=options.output_size,
        )
        return Tensor(backend.array(data), TensorConfig(tuple(out_shape), operand.dtype, operand.device))

    return _emit_reduction_node(
        "MaxUnpool3D",
        [operand, indices],
        {
            "kernel_size": options.kernel_size,
            "stride": options.stride,
            "padding": options.padding,
            "output_size": options.output_size,
        },
        tuple(out_shape),
        operand.dtype,
    )
