"""Scatter shape operations."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.registry import get_util

_emit_shape_node = get_util("_emit_shape_node")


def scatter(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:  # pragma: no cover
    """Scatters values from a source tensor into the input tensor along a specified.

    dimension

    Args:
        input (Tensor): The destination tensor
        dim (int): The axis along which to index
        index (Tensor): The indices of elements to scatter
        src (Tensor): The source tensor containing values to scatter

    Returns:
    Tensor: The updated tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Scatter", input.data, index.data, src.data, dim=dim)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs = [input, index, src]
    attributes = {"dim": dim}
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Scatter",
        inputs,
        attributes,
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def scatter_nd(indices: Tensor, updates: Tensor, shape: Sequence[int]) -> Tensor:  # pragma: no cover
    """Scatters updates into a new tensor of specified shape using indices.

    Args:
        indices (Tensor): The index tensor
        updates (Tensor): The updates to scatter
        shape (Sequence[int]): The shape of the output tensor

    Returns:
    Tensor: The output tensor with scattered updates

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("ScatterNd", indices.data, updates.data, shape=shape)
        return Tensor(backend.array(data), TensorConfig(tuple(shape), updates.dtype, updates.device))
    inputs = [indices, updates]
    attributes = {"shape": shape}
    return _emit_shape_node(
        "ScatterNd",
        inputs,
        attributes,
        tuple(shape),
        inputs[1].dtype if len(inputs) > 1 else DType.Float32,
    )


def scatter_add(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:  # pragma: no cover
    """Adds values from a source tensor to the input tensor at specified indices along a.

    dimension

    Args:
        input (Tensor): The destination tensor
        dim (int): The axis along which to index
        index (Tensor): The indices of elements to add
        src (Tensor): The source tensor containing values to add

    Returns:
    Tensor: The updated tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("ScatterAdd", input.data, index.data, src.data, dim=dim)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs = [input, index, src]
    attributes = {"dim": dim}
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "ScatterAdd",
        inputs,
        attributes,
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def tensor_scatter_update(tensor: Tensor, indices: Tensor, updates: Tensor) -> Tensor:  # pragma: no cover
    """Updates the value of a tensor at given indices.

    Args:
        tensor (Tensor): The input tensor
        indices (Tensor): The indices to update
        updates (Tensor): The updates to apply

    Returns:
    Tensor: The updated tensor
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("TensorScatterUpdate", tensor.data, indices.data, updates.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
    inputs = [tensor, indices, updates]
    # shape calculation placeholder
    out_shape = tensor.shape
    return _emit_shape_node(
        "TensorScatterUpdate",
        inputs,
        {},
        out_shape,
        tensor.dtype,
    )


def tensor_scatter_max(tensor: Tensor, indices: Tensor, updates: Tensor) -> Tensor:  # pragma: no cover
    """Updates a tensor at given indices with the maximum of the current value and the update.

    Args:
        tensor (Tensor): The input tensor
        indices (Tensor): The indices to update
        updates (Tensor): The updates to apply

    Returns:
    Tensor: The updated tensor
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("TensorScatterMax", tensor.data, indices.data, updates.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
    inputs = [tensor, indices, updates]
    # shape calculation placeholder
    out_shape = tensor.shape
    return _emit_shape_node(
        "TensorScatterMax",
        inputs,
        {},
        out_shape,
        tensor.dtype,
    )


def tensor_scatter_min(tensor: Tensor, indices: Tensor, updates: Tensor) -> Tensor:  # pragma: no cover
    """Updates a tensor at given indices with the minimum of the current value and the update.

    Args:
        tensor (Tensor): The input tensor
        indices (Tensor): The indices to update
        updates (Tensor): The updates to apply

    Returns:
    Tensor: The updated tensor
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("TensorScatterMin", tensor.data, indices.data, updates.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
    inputs = [tensor, indices, updates]
    # shape calculation placeholder
    out_shape = tensor.shape
    return _emit_shape_node(
        "TensorScatterMin",
        inputs,
        {},
        out_shape,
        tensor.dtype,
    )


def tensor_scatter_add(tensor: Tensor, indices: Tensor, updates: Tensor) -> Tensor:  # pragma: no cover
    """Adds updates to a tensor at given indices.

    Args:
        tensor (Tensor): The input tensor
        indices (Tensor): The indices to update
        updates (Tensor): The updates to apply

    Returns:
    Tensor: The updated tensor
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("TensorScatterAdd", tensor.data, indices.data, updates.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
    inputs = [tensor, indices, updates]
    # shape calculation placeholder
    out_shape = tensor.shape
    return _emit_shape_node(
        "TensorScatterAdd",
        inputs,
        {},
        out_shape,
        tensor.dtype,
    )
