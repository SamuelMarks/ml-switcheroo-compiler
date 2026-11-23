"""Static structural protocol bindings shard 36 (my_util..newFunctionWithName_)."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable

from ml_switcheroo_compiler.backends.eager_registry.protocols import EagerTensorProtocol
from ml_switcheroo_compiler.backends.eager_registry.types_base import BackendOperand

from ._support import register_dynamic_protocols

MemberReader = Callable[
    [typing.Union[types.ModuleType, BackendOperand]],  # noqa: UP007
    tuple[bool, typing.Optional[EagerTensorProtocol]],  # noqa: UP045
]


@typing.runtime_checkable
class _HasMember36_00(typing.Protocol):
    """Structural protocol for objects exposing `my_util`."""

    my_util: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_01(typing.Protocol):
    """Structural protocol for objects exposing `my_util_fn`."""

    my_util_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_02(typing.Protocol):
    """Structural protocol for objects exposing `my_vjp`."""

    my_vjp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_03(typing.Protocol):
    """Structural protocol for objects exposing `name`."""

    name: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_04(typing.Protocol):
    """Structural protocol for objects exposing `named_modules`."""

    named_modules: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_05(typing.Protocol):
    """Structural protocol for objects exposing `nan`."""

    nan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_06(typing.Protocol):
    """Structural protocol for objects exposing `nan_to_num`."""

    nan_to_num: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_07(typing.Protocol):
    """Structural protocol for objects exposing `nanargmax`."""

    nanargmax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_08(typing.Protocol):
    """Structural protocol for objects exposing `nanargmin`."""

    nanargmin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_09(typing.Protocol):
    """Structural protocol for objects exposing `nancumprod`."""

    nancumprod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_10(typing.Protocol):
    """Structural protocol for objects exposing `nancumsum`."""

    nancumsum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_11(typing.Protocol):
    """Structural protocol for objects exposing `nanmax`."""

    nanmax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_12(typing.Protocol):
    """Structural protocol for objects exposing `nanmean`."""

    nanmean: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_13(typing.Protocol):
    """Structural protocol for objects exposing `nanmedian`."""

    nanmedian: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_14(typing.Protocol):
    """Structural protocol for objects exposing `nanmin`."""

    nanmin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_15(typing.Protocol):
    """Structural protocol for objects exposing `nanpercentile`."""

    nanpercentile: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_16(typing.Protocol):
    """Structural protocol for objects exposing `nanprod`."""

    nanprod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_17(typing.Protocol):
    """Structural protocol for objects exposing `nanquantile`."""

    nanquantile: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_18(typing.Protocol):
    """Structural protocol for objects exposing `nanstd`."""

    nanstd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_19(typing.Protocol):
    """Structural protocol for objects exposing `nansum`."""

    nansum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_20(typing.Protocol):
    """Structural protocol for objects exposing `nanvar`."""

    nanvar: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_21(typing.Protocol):
    """Structural protocol for objects exposing `narrow`."""

    narrow: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_22(typing.Protocol):
    """Structural protocol for objects exposing `nbytes`."""

    nbytes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_23(typing.Protocol):
    """Structural protocol for objects exposing `ncclAllGather`."""

    ncclAllGather: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_24(typing.Protocol):
    """Structural protocol for objects exposing `ncclAllReduce`."""

    ncclAllReduce: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_25(typing.Protocol):
    """Structural protocol for objects exposing `ncclAllToAll`."""

    ncclAllToAll: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_26(typing.Protocol):
    """Structural protocol for objects exposing `ncclBroadcast`."""

    ncclBroadcast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_27(typing.Protocol):
    """Structural protocol for objects exposing `ncclCommDestroy`."""

    ncclCommDestroy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_28(typing.Protocol):
    """Structural protocol for objects exposing `ncclCommInitRank`."""

    ncclCommInitRank: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_29(typing.Protocol):
    """Structural protocol for objects exposing `ncclGetUniqueId`."""

    ncclGetUniqueId: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_30(typing.Protocol):
    """Structural protocol for objects exposing `ncclGroupEnd`."""

    ncclGroupEnd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_31(typing.Protocol):
    """Structural protocol for objects exposing `ncclGroupStart`."""

    ncclGroupStart: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_32(typing.Protocol):
    """Structural protocol for objects exposing `ncclRecv`."""

    ncclRecv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_33(typing.Protocol):
    """Structural protocol for objects exposing `ncclReduce`."""

    ncclReduce: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_34(typing.Protocol):
    """Structural protocol for objects exposing `ncclReduceScatter`."""

    ncclReduceScatter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_35(typing.Protocol):
    """Structural protocol for objects exposing `ncclSend`."""

    ncclSend: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_36(typing.Protocol):
    """Structural protocol for objects exposing `ndarray`."""

    ndarray: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_37(typing.Protocol):
    """Structural protocol for objects exposing `ndim`."""

    ndim: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_38(typing.Protocol):
    """Structural protocol for objects exposing `ndtri`."""

    ndtri: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_39(typing.Protocol):
    """Structural protocol for objects exposing `ne`."""

    ne: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_40(typing.Protocol):
    """Structural protocol for objects exposing `neg`."""

    neg: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_41(typing.Protocol):
    """Structural protocol for objects exposing `negate`."""

    negate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_42(typing.Protocol):
    """Structural protocol for objects exposing `negative`."""

    negative: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_43(typing.Protocol):
    """Structural protocol for objects exposing `newBufferWithLength_options_`."""

    newBufferWithLength_options_: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember36_44(typing.Protocol):
    """Structural protocol for objects exposing `newFunctionWithName_`."""

    newFunctionWithName_: EagerTensorProtocol


SHARD_36_READERS: dict[str, MemberReader] = {
    "my_util": lambda t: (True, t.my_util) if isinstance(t, _HasMember36_00) else (False, None),
    "my_util_fn": lambda t: (True, t.my_util_fn) if isinstance(t, _HasMember36_01) else (False, None),
    "my_vjp": lambda t: (True, t.my_vjp) if isinstance(t, _HasMember36_02) else (False, None),
    "name": lambda t: (True, t.name) if isinstance(t, _HasMember36_03) else (False, None),
    "named_modules": lambda t: (True, t.named_modules) if isinstance(t, _HasMember36_04) else (False, None),
    "nan": lambda t: (True, t.nan) if isinstance(t, _HasMember36_05) else (False, None),
    "nan_to_num": lambda t: (True, t.nan_to_num) if isinstance(t, _HasMember36_06) else (False, None),
    "nanargmax": lambda t: (True, t.nanargmax) if isinstance(t, _HasMember36_07) else (False, None),
    "nanargmin": lambda t: (True, t.nanargmin) if isinstance(t, _HasMember36_08) else (False, None),
    "nancumprod": lambda t: (True, t.nancumprod) if isinstance(t, _HasMember36_09) else (False, None),
    "nancumsum": lambda t: (True, t.nancumsum) if isinstance(t, _HasMember36_10) else (False, None),
    "nanmax": lambda t: (True, t.nanmax) if isinstance(t, _HasMember36_11) else (False, None),
    "nanmean": lambda t: (True, t.nanmean) if isinstance(t, _HasMember36_12) else (False, None),
    "nanmedian": lambda t: (True, t.nanmedian) if isinstance(t, _HasMember36_13) else (False, None),
    "nanmin": lambda t: (True, t.nanmin) if isinstance(t, _HasMember36_14) else (False, None),
    "nanpercentile": lambda t: (True, t.nanpercentile) if isinstance(t, _HasMember36_15) else (False, None),
    "nanprod": lambda t: (True, t.nanprod) if isinstance(t, _HasMember36_16) else (False, None),
    "nanquantile": lambda t: (True, t.nanquantile) if isinstance(t, _HasMember36_17) else (False, None),
    "nanstd": lambda t: (True, t.nanstd) if isinstance(t, _HasMember36_18) else (False, None),
    "nansum": lambda t: (True, t.nansum) if isinstance(t, _HasMember36_19) else (False, None),
    "nanvar": lambda t: (True, t.nanvar) if isinstance(t, _HasMember36_20) else (False, None),
    "narrow": lambda t: (True, t.narrow) if isinstance(t, _HasMember36_21) else (False, None),
    "nbytes": lambda t: (True, t.nbytes) if isinstance(t, _HasMember36_22) else (False, None),
    "ncclAllGather": lambda t: (True, t.ncclAllGather) if isinstance(t, _HasMember36_23) else (False, None),
    "ncclAllReduce": lambda t: (True, t.ncclAllReduce) if isinstance(t, _HasMember36_24) else (False, None),
    "ncclAllToAll": lambda t: (True, t.ncclAllToAll) if isinstance(t, _HasMember36_25) else (False, None),
    "ncclBroadcast": lambda t: (True, t.ncclBroadcast) if isinstance(t, _HasMember36_26) else (False, None),
    "ncclCommDestroy": lambda t: (True, t.ncclCommDestroy) if isinstance(t, _HasMember36_27) else (False, None),
    "ncclCommInitRank": lambda t: (True, t.ncclCommInitRank) if isinstance(t, _HasMember36_28) else (False, None),
    "ncclGetUniqueId": lambda t: (True, t.ncclGetUniqueId) if isinstance(t, _HasMember36_29) else (False, None),
    "ncclGroupEnd": lambda t: (True, t.ncclGroupEnd) if isinstance(t, _HasMember36_30) else (False, None),
    "ncclGroupStart": lambda t: (True, t.ncclGroupStart) if isinstance(t, _HasMember36_31) else (False, None),
    "ncclRecv": lambda t: (True, t.ncclRecv) if isinstance(t, _HasMember36_32) else (False, None),
    "ncclReduce": lambda t: (True, t.ncclReduce) if isinstance(t, _HasMember36_33) else (False, None),
    "ncclReduceScatter": lambda t: (True, t.ncclReduceScatter) if isinstance(t, _HasMember36_34) else (False, None),
    "ncclSend": lambda t: (True, t.ncclSend) if isinstance(t, _HasMember36_35) else (False, None),
    "ndarray": lambda t: (True, t.ndarray) if isinstance(t, _HasMember36_36) else (False, None),
    "ndim": lambda t: (True, t.ndim) if isinstance(t, _HasMember36_37) else (False, None),
    "ndtri": lambda t: (True, t.ndtri) if isinstance(t, _HasMember36_38) else (False, None),
    "ne": lambda t: (True, t.ne) if isinstance(t, _HasMember36_39) else (False, None),
    "neg": lambda t: (True, t.neg) if isinstance(t, _HasMember36_40) else (False, None),
    "negate": lambda t: (True, t.negate) if isinstance(t, _HasMember36_41) else (False, None),
    "negative": lambda t: (True, t.negative) if isinstance(t, _HasMember36_42) else (False, None),
    "newBufferWithLength_options_": lambda t: (
        (True, t.newBufferWithLength_options_) if isinstance(t, _HasMember36_43) else (False, None)
    ),
    "newFunctionWithName_": lambda t: (
        (True, t.newFunctionWithName_) if isinstance(t, _HasMember36_44) else (False, None)
    ),
}

register_dynamic_protocols(
    (
        _HasMember36_00,
        _HasMember36_01,
        _HasMember36_02,
        _HasMember36_03,
        _HasMember36_04,
        _HasMember36_05,
        _HasMember36_06,
        _HasMember36_07,
        _HasMember36_08,
        _HasMember36_09,
        _HasMember36_10,
        _HasMember36_11,
        _HasMember36_12,
        _HasMember36_13,
        _HasMember36_14,
        _HasMember36_15,
        _HasMember36_16,
        _HasMember36_17,
        _HasMember36_18,
        _HasMember36_19,
        _HasMember36_20,
        _HasMember36_21,
        _HasMember36_22,
        _HasMember36_23,
        _HasMember36_24,
        _HasMember36_25,
        _HasMember36_26,
        _HasMember36_27,
        _HasMember36_28,
        _HasMember36_29,
        _HasMember36_30,
        _HasMember36_31,
        _HasMember36_32,
        _HasMember36_33,
        _HasMember36_34,
        _HasMember36_35,
        _HasMember36_36,
        _HasMember36_37,
        _HasMember36_38,
        _HasMember36_39,
        _HasMember36_40,
        _HasMember36_41,
        _HasMember36_42,
        _HasMember36_43,
        _HasMember36_44,
    )
)
