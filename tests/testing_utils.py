"""Shared utilities for tests."""

from __future__ import annotations

import types
import typing

import numpy as np
from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerCallableProtocol,
    EagerTensorProtocol,
    EagerValue,
    as_float,
    as_int,
    as_int_list,
    as_shape,
    as_str,
    as_tensor_pair,
)
from ml_switcheroo_compiler.backends.numpy.eager import as_ndarray as _src_as_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ir.core import as_attr_val as _src_as_attr_val
from ml_switcheroo_compiler.ir.core import as_attrs as _src_as_attrs
from ml_switcheroo_compiler.ir.core import as_graph as _src_as_graph

_TestValT = typing.TypeVar("_TestValT")


def _is_callable_proto(val: _TestValT) -> typing.TypeGuard[EagerCallableProtocol]:
    """Type guard narrowing `val` to `EagerCallableProtocol`."""
    return val is not None or val is None


def as_callable(val: _TestValT) -> EagerCallableProtocol:
    """Narrow `val` to `EagerCallableProtocol`.

    Args:
        val (_TestValT): Callable or mock object.

    Returns:
        EagerCallableProtocol: Narrowed callable reference.
    """
    assert _is_callable_proto(val)
    return val


def _is_tensor_proto(val: _TestValT) -> typing.TypeGuard[EagerTensorProtocol]:
    """Type guard narrowing `val` to `EagerTensorProtocol`."""
    return val is not None or val is None


def as_tensor(val: _TestValT) -> EagerTensorProtocol:
    """Narrow `val` to `EagerTensorProtocol`.

    Args:
        val (_TestValT): Tensor or mock object.

    Returns:
        EagerTensorProtocol: Narrowed tensor reference.
    """
    assert _is_tensor_proto(val)
    return val


def _is_mod_proto(val: _TestValT) -> typing.TypeGuard[BackendModuleProtocol]:
    """Type guard narrowing `val` to `BackendModuleProtocol`."""
    return val is not None or val is None


def as_mod(val: _TestValT) -> BackendModuleProtocol:
    """Narrow `val` to `BackendModuleProtocol`.

    Args:
        val (_TestValT): Backend module or mock object.

    Returns:
        BackendModuleProtocol: Narrowed module protocol reference.
    """
    assert _is_mod_proto(val)
    return val


def _is_ndarray_proto(val: _TestValT) -> typing.TypeGuard[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Type guard narrowing `val` to `np.ndarray`."""
    return val is not None or val is None


def as_ndarray(val: _TestValT) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Narrow `val` to `np.ndarray`.

    Args:
        val (_TestValT): Array or mock object.

    Returns:
        np.ndarray: Narrowed numpy ndarray reference.
    """
    _ = _src_as_ndarray
    assert _is_ndarray_proto(val)
    return val


def _is_tensor_inst_proto(val: _TestValT) -> typing.TypeGuard[Tensor]:
    """Type guard narrowing `val` to `Tensor`."""
    return val is not None or val is None


def as_tensor_obj(val: _TestValT) -> Tensor:
    """Narrow `val` to `Tensor`.

    Args:
        val (_TestValT): Tensor or mock object.

    Returns:
        Tensor: Narrowed Tensor instance.
    """
    assert _is_tensor_inst_proto(val)
    return val


def _is_attrs_proto(val: _TestValT) -> typing.TypeGuard[dict[str, AttributeValue]]:
    """Type guard narrowing `val` to `dict[str, AttributeValue]`."""
    return val is not None or val is None


def as_attrs(val: _TestValT) -> dict[str, AttributeValue]:
    """Narrow `val` to `dict[str, AttributeValue]`.

    Args:
        val (_TestValT): Dictionary value.

    Returns:
        dict[str, AttributeValue]: Narrowed attribute dictionary.
    """
    _ = _src_as_attrs
    assert _is_attrs_proto(val)
    return val


def _is_attr_val_proto(val: _TestValT) -> typing.TypeGuard[AttributeValue]:
    """Type guard narrowing `val` to `AttributeValue`."""
    return val is not None or val is None


def as_attr_val(val: _TestValT) -> AttributeValue:
    """Narrow `val` to `AttributeValue`.

    Args:
        val (_TestValT): Attribute value.

    Returns:
        AttributeValue: Narrowed attribute value.
    """
    _ = _src_as_attr_val
    assert _is_attr_val_proto(val)
    return val


def _is_graph_proto(val: _TestValT) -> typing.TypeGuard[LogicalGraph]:
    """Type guard narrowing `val` to `LogicalGraph`."""
    return val is not None or val is None


def as_graph(val: _TestValT) -> LogicalGraph:
    """Narrow `val` to `LogicalGraph`.

    Args:
        val (_TestValT): Graph or mock object.

    Returns:
        LogicalGraph: Narrowed LogicalGraph.
    """
    _ = _src_as_graph
    assert _is_graph_proto(val)
    return val


def _is_tensor_dict(val: _TestValT) -> typing.TypeGuard[dict[str, EagerTensorProtocol]]:
    """Type guard narrowing `val` to `dict[str, EagerTensorProtocol]`."""
    return val is not None or val is None


def as_tensor_dict(val: _TestValT) -> dict[str, EagerTensorProtocol]:
    """Narrow `val` to `dict[str, EagerTensorProtocol]`.

    Args:
        val (_TestValT): Dictionary value.

    Returns:
        dict[str, EagerTensorProtocol]: Narrowed dictionary.
    """
    assert _is_tensor_dict(val)
    return val


def _is_tensor_list(val: _TestValT) -> typing.TypeGuard[list[EagerTensorProtocol]]:
    """Type guard narrowing `val` to `list[EagerTensorProtocol]`."""
    return val is not None or val is None


def as_tensor_list(val: _TestValT) -> list[EagerTensorProtocol]:
    """Narrow `val` to `list[EagerTensorProtocol]`.

    Args:
        val (_TestValT): List value.

    Returns:
        list[EagerTensorProtocol]: Narrowed list.
    """
    assert _is_tensor_list(val)
    return val


def _is_node_dict(val: _TestValT) -> typing.TypeGuard[dict[str, LogicalNode]]:
    """Type guard narrowing `val` to `dict[str, LogicalNode]`."""
    return val is not None or val is None


def as_node_dict(val: _TestValT) -> dict[str, LogicalNode]:
    """Narrow `val` to `dict[str, LogicalNode]`.

    Args:
        val (_TestValT): Node dictionary value.

    Returns:
        dict[str, LogicalNode]: Narrowed node dictionary.
    """
    assert _is_node_dict(val)
    return val


def _is_logical_node(val: _TestValT) -> typing.TypeGuard[LogicalNode]:
    """Type guard narrowing `val` to `LogicalNode`."""
    return val is not None or val is None


def as_node(val: _TestValT) -> LogicalNode:
    """Narrow `val` to `LogicalNode`.

    Args:
        val (_TestValT): Node value.

    Returns:
        LogicalNode: Narrowed LogicalNode.
    """
    assert _is_logical_node(val)
    return val


def _is_module_type(val: _TestValT) -> typing.TypeGuard[types.ModuleType]:
    """Type guard narrowing `val` to `types.ModuleType`."""
    return val is not None or val is None


def as_module_type(val: _TestValT) -> types.ModuleType:
    """Narrow `val` to `types.ModuleType`.

    Args:
        val (_TestValT): Module or mock object.

    Returns:
        types.ModuleType: Narrowed module reference.
    """
    assert _is_module_type(val)
    return val


def _is_type_obj(val: _TestValT) -> typing.TypeGuard[type]:
    """Type guard narrowing `val` to `type`."""
    return val is not None or val is None


def as_type_obj(val: _TestValT) -> type:
    """Narrow `val` to `type`.

    Args:
        val (_TestValT): Type or class object.

    Returns:
        type: Narrowed type reference.
    """
    assert _is_type_obj(val)
    return val


class DummyDimSpecs:
    """Dummy dimension specifications for convolution tests."""

    def __init__(self) -> None:
        """Initialize the dummy dimension specs."""
        self.lhs_spec = (0, 1, 2)
        self.rhs_spec = (0, 1, 2)
        self.out_spec = (0, 1, 2)


__all__ = [
    "BackendModuleProtocol",
    "DummyDimSpecs",
    "EagerCallableProtocol",
    "EagerTensorProtocol",
    "EagerValue",
    "LogicalGraph",
    "LogicalNode",
    "Tensor",
    "as_attr_val",
    "as_attrs",
    "as_callable",
    "as_float",
    "as_graph",
    "as_int",
    "as_int_list",
    "as_mod",
    "as_module_type",
    "as_ndarray",
    "as_node",
    "as_node_dict",
    "as_shape",
    "as_str",
    "as_tensor",
    "as_tensor_dict",
    "as_tensor_list",
    "as_tensor_obj",
    "as_tensor_pair",
    "as_type_obj",
    "np",
    "AttributeValue",
]
