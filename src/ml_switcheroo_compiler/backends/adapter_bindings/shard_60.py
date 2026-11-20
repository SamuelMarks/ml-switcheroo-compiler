"""Static structural protocol bindings shard 60 (_np_tensor_scatter_sub..unbind)."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable

from ml_switcheroo_compiler.backends.eager_registry.protocols import EagerTensorProtocol
from ml_switcheroo_compiler.backends.eager_registry.types_base import BackendOperand

MemberReader = Callable[
    [typing.Union[types.ModuleType, BackendOperand]],  # noqa: UP007
    tuple[bool, typing.Optional[EagerTensorProtocol]],  # noqa: UP045
]


@typing.runtime_checkable
class _HasMember60_00(typing.Protocol):
    """Structural protocol for objects exposing `_np_tensor_scatter_sub`."""

    _np_tensor_scatter_sub: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_01(typing.Protocol):
    """Structural protocol for objects exposing `_tensor_scatter_sub`."""

    _tensor_scatter_sub: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_02(typing.Protocol):
    """Structural protocol for objects exposing `all_reduce`."""

    all_reduce: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_03(typing.Protocol):
    """Structural protocol for objects exposing `as_tensor`."""

    as_tensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_04(typing.Protocol):
    """Structural protocol for objects exposing `broadcast`."""

    broadcast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_05(typing.Protocol):
    """Structural protocol for objects exposing `index_put_`."""

    index_put_: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_06(typing.Protocol):
    """Structural protocol for objects exposing `jax_all_gather`."""

    jax_all_gather: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_07(typing.Protocol):
    """Structural protocol for objects exposing `jax_all_reduce`."""

    jax_all_reduce: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_08(typing.Protocol):
    """Structural protocol for objects exposing `jax_broadcast`."""

    jax_broadcast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_09(typing.Protocol):
    """Structural protocol for objects exposing `long`."""

    long: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_10(typing.Protocol):
    """Structural protocol for objects exposing `max_memory_allocated`."""

    max_memory_allocated: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_11(typing.Protocol):
    """Structural protocol for objects exposing `newaxis`."""

    newaxis: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_12(typing.Protocol):
    """Structural protocol for objects exposing `pytorch_all_gather`."""

    pytorch_all_gather: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_13(typing.Protocol):
    """Structural protocol for objects exposing `pytorch_all_reduce`."""

    pytorch_all_reduce: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_14(typing.Protocol):
    """Structural protocol for objects exposing `pytorch_broadcast`."""

    pytorch_broadcast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_15(typing.Protocol):
    """Structural protocol for objects exposing `pytorch_reduce_scatter`."""

    pytorch_reduce_scatter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_16(typing.Protocol):
    """Structural protocol for objects exposing `scatter_reduce_`."""

    scatter_reduce_: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_17(typing.Protocol):
    """Structural protocol for objects exposing `set_active_backend`."""

    set_active_backend: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_18(typing.Protocol):
    """Structural protocol for objects exposing `stride`."""

    stride: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_19(typing.Protocol):
    """Structural protocol for objects exposing `tensor_split`."""

    tensor_split: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_20(typing.Protocol):
    """Structural protocol for objects exposing `ufunc`."""

    ufunc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_21(typing.Protocol):
    """Structural protocol for objects exposing `unbind`."""

    unbind: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember60_22(typing.Protocol):
    """Structural protocol for objects exposing `equal`."""

    equal: EagerTensorProtocol


SHARD_60_READERS: dict[str, MemberReader] = {
    "_np_tensor_scatter_sub": (
        lambda t: (True, t._np_tensor_scatter_sub) if isinstance(t, _HasMember60_00) else (False, None)
    ),
    "_tensor_scatter_sub": lambda t: (True, t._tensor_scatter_sub) if isinstance(t, _HasMember60_01) else (False, None),
    "all_reduce": lambda t: (True, t.all_reduce) if isinstance(t, _HasMember60_02) else (False, None),
    "as_tensor": lambda t: (True, t.as_tensor) if isinstance(t, _HasMember60_03) else (False, None),
    "broadcast": lambda t: (True, t.broadcast) if isinstance(t, _HasMember60_04) else (False, None),
    "index_put_": lambda t: (True, t.index_put_) if isinstance(t, _HasMember60_05) else (False, None),
    "jax_all_gather": lambda t: (True, t.jax_all_gather) if isinstance(t, _HasMember60_06) else (False, None),
    "jax_all_reduce": lambda t: (True, t.jax_all_reduce) if isinstance(t, _HasMember60_07) else (False, None),
    "jax_broadcast": lambda t: (True, t.jax_broadcast) if isinstance(t, _HasMember60_08) else (False, None),
    "long": lambda t: (True, t.long) if isinstance(t, _HasMember60_09) else (False, None),
    "max_memory_allocated": (
        lambda t: (True, t.max_memory_allocated) if isinstance(t, _HasMember60_10) else (False, None)
    ),
    "newaxis": lambda t: (True, t.newaxis) if isinstance(t, _HasMember60_11) else (False, None),
    "pytorch_all_gather": lambda t: (True, t.pytorch_all_gather) if isinstance(t, _HasMember60_12) else (False, None),
    "pytorch_all_reduce": lambda t: (True, t.pytorch_all_reduce) if isinstance(t, _HasMember60_13) else (False, None),
    "pytorch_broadcast": lambda t: (True, t.pytorch_broadcast) if isinstance(t, _HasMember60_14) else (False, None),
    "pytorch_reduce_scatter": (
        lambda t: (True, t.pytorch_reduce_scatter) if isinstance(t, _HasMember60_15) else (False, None)
    ),
    "scatter_reduce_": lambda t: (True, t.scatter_reduce_) if isinstance(t, _HasMember60_16) else (False, None),
    "set_active_backend": lambda t: (True, t.set_active_backend) if isinstance(t, _HasMember60_17) else (False, None),
    "stride": lambda t: (True, t.stride) if isinstance(t, _HasMember60_18) else (False, None),
    "tensor_split": lambda t: (True, t.tensor_split) if isinstance(t, _HasMember60_19) else (False, None),
    "ufunc": lambda t: (True, t.ufunc) if isinstance(t, _HasMember60_20) else (False, None),
    "unbind": lambda t: (True, t.unbind) if isinstance(t, _HasMember60_21) else (False, None),
    "equal": lambda t: (True, t.equal) if isinstance(t, _HasMember60_22) else (False, None),
}
