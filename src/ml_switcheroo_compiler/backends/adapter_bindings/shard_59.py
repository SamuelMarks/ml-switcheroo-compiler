"""Static structural protocol bindings shard 59 (AVG.._np_tensor_scatter)."""

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
class _HasMember59_00(typing.Protocol):
    """Structural protocol for objects exposing `AVG`."""

    AVG: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_01(typing.Protocol):
    """Structural protocol for objects exposing `MAX`."""

    MAX: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_02(typing.Protocol):
    """Structural protocol for objects exposing `MIN`."""

    MIN: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_03(typing.Protocol):
    """Structural protocol for objects exposing `PRODUCT`."""

    PRODUCT: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_04(typing.Protocol):
    """Structural protocol for objects exposing `SUM`."""

    SUM: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_05(typing.Protocol):
    """Structural protocol for objects exposing `T`."""

    T: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_06(typing.Protocol):
    """Structural protocol for objects exposing `_find_spec`."""

    _find_spec: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_07(typing.Protocol):
    """Structural protocol for objects exposing `_global_adaptive_pool_mock`."""

    _global_adaptive_pool_mock: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_08(typing.Protocol):
    """Structural protocol for objects exposing `_np_all_gather`."""

    _np_all_gather: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_09(typing.Protocol):
    """Structural protocol for objects exposing `_np_all_reduce`."""

    _np_all_reduce: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_10(typing.Protocol):
    """Structural protocol for objects exposing `_np_all_to_all`."""

    _np_all_to_all: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_11(typing.Protocol):
    """Structural protocol for objects exposing `_np_logdet`."""

    _np_logdet: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_12(typing.Protocol):
    """Structural protocol for objects exposing `_np_pmax`."""

    _np_pmax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_13(typing.Protocol):
    """Structural protocol for objects exposing `_np_pmin`."""

    _np_pmin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_14(typing.Protocol):
    """Structural protocol for objects exposing `_np_qr`."""

    _np_qr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_15(typing.Protocol):
    """Structural protocol for objects exposing `_np_rem_2`."""

    _np_rem_2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_16(typing.Protocol):
    """Structural protocol for objects exposing `_np_scatter_nd_add`."""

    _np_scatter_nd_add: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_17(typing.Protocol):
    """Structural protocol for objects exposing `_np_scatter_nd_sub`."""

    _np_scatter_nd_sub: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_18(typing.Protocol):
    """Structural protocol for objects exposing `_np_scatter_nd_update`."""

    _np_scatter_nd_update: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_19(typing.Protocol):
    """Structural protocol for objects exposing `_np_scatter_update`."""

    _np_scatter_update: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_20(typing.Protocol):
    """Structural protocol for objects exposing `_np_slogdet`."""

    _np_slogdet: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_21(typing.Protocol):
    """Structural protocol for objects exposing `_np_svd`."""

    _np_svd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember59_22(typing.Protocol):
    """Structural protocol for objects exposing `_np_tensor_scatter`."""

    _np_tensor_scatter: EagerTensorProtocol


SHARD_59_READERS: dict[str, MemberReader] = {
    "AVG": lambda t: (True, t.AVG) if isinstance(t, _HasMember59_00) else (False, None),
    "MAX": lambda t: (True, t.MAX) if isinstance(t, _HasMember59_01) else (False, None),
    "MIN": lambda t: (True, t.MIN) if isinstance(t, _HasMember59_02) else (False, None),
    "PRODUCT": lambda t: (True, t.PRODUCT) if isinstance(t, _HasMember59_03) else (False, None),
    "SUM": lambda t: (True, t.SUM) if isinstance(t, _HasMember59_04) else (False, None),
    "T": lambda t: (True, t.T) if isinstance(t, _HasMember59_05) else (False, None),
    "_find_spec": lambda t: (True, t._find_spec) if isinstance(t, _HasMember59_06) else (False, None),
    "_global_adaptive_pool_mock": (
        lambda t: (True, t._global_adaptive_pool_mock) if isinstance(t, _HasMember59_07) else (False, None)
    ),
    "_np_all_gather": lambda t: (True, t._np_all_gather) if isinstance(t, _HasMember59_08) else (False, None),
    "_np_all_reduce": lambda t: (True, t._np_all_reduce) if isinstance(t, _HasMember59_09) else (False, None),
    "_np_all_to_all": lambda t: (True, t._np_all_to_all) if isinstance(t, _HasMember59_10) else (False, None),
    "_np_logdet": lambda t: (True, t._np_logdet) if isinstance(t, _HasMember59_11) else (False, None),
    "_np_pmax": lambda t: (True, t._np_pmax) if isinstance(t, _HasMember59_12) else (False, None),
    "_np_pmin": lambda t: (True, t._np_pmin) if isinstance(t, _HasMember59_13) else (False, None),
    "_np_qr": lambda t: (True, t._np_qr) if isinstance(t, _HasMember59_14) else (False, None),
    "_np_rem_2": lambda t: (True, t._np_rem_2) if isinstance(t, _HasMember59_15) else (False, None),
    "_np_scatter_nd_add": lambda t: (True, t._np_scatter_nd_add) if isinstance(t, _HasMember59_16) else (False, None),
    "_np_scatter_nd_sub": lambda t: (True, t._np_scatter_nd_sub) if isinstance(t, _HasMember59_17) else (False, None),
    "_np_scatter_nd_update": (
        lambda t: (True, t._np_scatter_nd_update) if isinstance(t, _HasMember59_18) else (False, None)
    ),
    "_np_scatter_update": lambda t: (True, t._np_scatter_update) if isinstance(t, _HasMember59_19) else (False, None),
    "_np_slogdet": lambda t: (True, t._np_slogdet) if isinstance(t, _HasMember59_20) else (False, None),
    "_np_svd": lambda t: (True, t._np_svd) if isinstance(t, _HasMember59_21) else (False, None),
    "_np_tensor_scatter": lambda t: (True, t._np_tensor_scatter) if isinstance(t, _HasMember59_22) else (False, None),
}
