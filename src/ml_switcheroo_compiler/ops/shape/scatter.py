"""Module scatter.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Scatter shape operations."""


from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.registry import get_util

_emit_shape_node = get_util("_emit_shape_node")


def scatter(input: Tensor, axis: int, index: Tensor, src: Tensor):
    """Scatter values from a source tensor into the input tensor along a specified.

    Args:
        input (Tensor): The input parameter.
        axis (int): The axis parameter.
        index (Tensor): The index parameter.
        src (Tensor): The src parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Scatter", input.data, index.data, src.data, axis=axis)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs = [input, index, src]
    attributes = {"axis": axis}
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Scatter",
        inputs,
        attributes,
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def scatter_nd(indices: Tensor, updates: Tensor, shape: Sequence[int]):
    """Scatter updates into a new tensor of specified shape using indices.

    Args:
        indices (Tensor): The indices parameter.
        updates (Tensor): The updates parameter.
        shape (Sequence): The shape parameter.

    Returns:
        Tensor: Result.
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


def scatter_add(input: Tensor, axis: int, index: Tensor, src: Tensor):
    """Add values from a source tensor to the input tensor at specified indices along a.

    Args:
        input (Tensor): The input parameter.
        axis (int): The axis parameter.
        index (Tensor): The index parameter.
        src (Tensor): The src parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("ScatterAdd", input.data, index.data, src.data, axis=axis)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs = [input, index, src]
    attributes = {"axis": axis}
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "ScatterAdd",
        inputs,
        attributes,
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def tensor_scatter_update(tensor: Tensor, indices: Tensor, updates: Tensor):
    """Update the value of a tensor at given indices.

    Args:
        tensor (Tensor): The tensor parameter.
        indices (Tensor): The indices parameter.
        updates (Tensor): The updates parameter.

    Returns:
        Tensor: Result.
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


def tensor_scatter_max(tensor: Tensor, indices: Tensor, updates: Tensor):
    """Update a tensor at given indices with the maximum of the current value and the update.

    Args:
        tensor (Tensor): The tensor parameter.
        indices (Tensor): The indices parameter.
        updates (Tensor): The updates parameter.

    Returns:
        Tensor: Result.
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


def tensor_scatter_min(tensor: Tensor, indices: Tensor, updates: Tensor):
    """Update a tensor at given indices with the minimum of the current value and the update.

    Args:
        tensor (Tensor): The tensor parameter.
        indices (Tensor): The indices parameter.
        updates (Tensor): The updates parameter.

    Returns:
        Tensor: Result.
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


def tensor_scatter_add(tensor: Tensor, indices: Tensor, updates: Tensor):
    """Add updates to a tensor at given indices.

    Args:
        tensor (Tensor): The tensor parameter.
        indices (Tensor): The indices parameter.
        updates (Tensor): The updates parameter.

    Returns:
        Tensor: Result.
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
