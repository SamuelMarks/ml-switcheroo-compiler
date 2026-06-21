"""Shape operations for Tensor objects."""

from __future__ import annotations
# pylint: disable=duplicate-code


from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

if TYPE_CHECKING:
    from collections.abc import Sequence


def split(
    input: Tensor,
    split_size_or_sections: int | Sequence[int],
    dim: int = 0,
) -> Sequence[Tensor]:
    """Splits the input tensor into multiple sub-tensors.

    Args:
        input (Tensor): The input tensor to split
        split_size_or_sections (int | Sequence[int]): Size of a single chunk or list of
        sizes for each chunk
        dim (int): The dimension along which to split. Defaults to 0

    Returns:
    Sequence[Tensor]: A sequence of sub-tensors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        datas = backend.execute_op("Split", input.data, split_size_or_sections, axis=dim)
        return tuple(Tensor(d, TensorConfig(d.shape, input.dtype, input.device)) for d in datas)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return (
        _emit_shape_node(
            "Split",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        ),
    )


def unstack(input: Tensor, dim: int = 0) -> Sequence[Tensor]:
    """Unstacks the input tensor along a specified dimension into a sequence of tensors.

    Args:
        input (Tensor): The input tensor to unstack
        dim (int): The dimension along which to unstack. Defaults to 0

    Returns:
    Sequence[Tensor]: A sequence of unstacked tensors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        datas = (
            backend.execute_op("Unstack", input.data, axis=dim)
            if hasattr(backend, "unstack")
            else backend.execute_op("Moveaxis", input.data, dim, 0)
        )
        return tuple(Tensor(d, TensorConfig(d.shape, input.dtype, input.device)) for d in datas)
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return (
        _emit_shape_node(
            "Unstack",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        ),
    )


def array_split(
    ary: Tensor,
    indices_or_sections: int | Sequence[int],
    axis: int = 0,
) -> Sequence[Tensor]:
    """Split an array into multiple sub-arrays.

    Args:
        ary (Tensor): The input tensor to split
        indices_or_sections (int | Sequence[int]): Size of a single chunk or list of
        sizes for each chunk
        axis (int): The dimension along which to split. Defaults to 0

    Returns:
        Sequence[Tensor]: A sequence of sub-tensors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        datas = backend.execute_op("ArraySplit", ary.data, indices_or_sections, axis=axis)
        return tuple(Tensor(d, TensorConfig(d.shape, ary.dtype, ary.device)) for d in datas)
    return (
        _emit_shape_node(
            "ArraySplit",
            [ary],
            {"indices_or_sections": indices_or_sections, "axis": axis},
            ary.shape,
            ary.dtype,
        ),
    )


def vsplit(ary: Tensor, indices_or_sections: int | Sequence[int]) -> Sequence[Tensor]:
    """Split an array into multiple sub-arrays vertically (row-wise).

    Args:
        ary (Tensor): The input tensor to split
        indices_or_sections (int | Sequence[int]): Size of a single chunk or list of
        sizes for each chunk

    Returns:
        Sequence[Tensor]: A sequence of sub-tensors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        datas = backend.execute_op("Vsplit", ary.data, indices_or_sections)
        return tuple(Tensor(d, TensorConfig(d.shape, ary.dtype, ary.device)) for d in datas)
    return (
        _emit_shape_node(
            "Vsplit",
            [ary],
            {"indices_or_sections": indices_or_sections},
            ary.shape,
            ary.dtype,
        ),
    )


def hsplit(ary: Tensor, indices_or_sections: int | Sequence[int]) -> Sequence[Tensor]:
    """Split an array into multiple sub-arrays horizontally (column-wise).

    Args:
        ary (Tensor): The input tensor to split
        indices_or_sections (int | Sequence[int]): Size of a single chunk or list of
        sizes for each chunk

    Returns:
        Sequence[Tensor]: A sequence of sub-tensors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        datas = backend.execute_op("Hsplit", ary.data, indices_or_sections)
        return tuple(Tensor(d, TensorConfig(d.shape, ary.dtype, ary.device)) for d in datas)
    return (
        _emit_shape_node(
            "Hsplit",
            [ary],
            {"indices_or_sections": indices_or_sections},
            ary.shape,
            ary.dtype,
        ),
    )


def dsplit(ary: Tensor, indices_or_sections: int | Sequence[int]) -> Sequence[Tensor]:
    """Split array into multiple sub-arrays along the 3rd axis (depth).

    Args:
        ary (Tensor): The input tensor to split
        indices_or_sections (int | Sequence[int]): Size of a single chunk or list of
        sizes for each chunk

    Returns:
        Sequence[Tensor]: A sequence of sub-tensors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        datas = backend.execute_op("Dsplit", ary.data, indices_or_sections)
        return tuple(Tensor(d, TensorConfig(d.shape, ary.dtype, ary.device)) for d in datas)
    return (
        _emit_shape_node(
            "Dsplit",
            [ary],
            {"indices_or_sections": indices_or_sections},
            ary.shape,
            ary.dtype,
        ),
    )
