"""Static structural protocol bindings shard 28 (iscomplex..kwarg_map)."""

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
class _HasMember28_00(typing.Protocol):
    """Structural protocol for objects exposing `iscomplex`."""

    iscomplex: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_01(typing.Protocol):
    """Structural protocol for objects exposing `iscomplexobj`."""

    iscomplexobj: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_02(typing.Protocol):
    """Structural protocol for objects exposing `isfinite`."""

    isfinite: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_03(typing.Protocol):
    """Structural protocol for objects exposing `isin`."""

    isin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_04(typing.Protocol):
    """Structural protocol for objects exposing `isinf`."""

    isinf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_05(typing.Protocol):
    """Structural protocol for objects exposing `isnan`."""

    isnan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_06(typing.Protocol):
    """Structural protocol for objects exposing `isneginf`."""

    isneginf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_07(typing.Protocol):
    """Structural protocol for objects exposing `isnull`."""

    isnull: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_08(typing.Protocol):
    """Structural protocol for objects exposing `isposinf`."""

    isposinf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_09(typing.Protocol):
    """Structural protocol for objects exposing `isreal`."""

    isreal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_10(typing.Protocol):
    """Structural protocol for objects exposing `isrealobj`."""

    isrealobj: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_11(typing.Protocol):
    """Structural protocol for objects exposing `isscalar`."""

    isscalar: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_12(typing.Protocol):
    """Structural protocol for objects exposing `issubdtype`."""

    issubdtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_13(typing.Protocol):
    """Structural protocol for objects exposing `istft`."""

    istft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_14(typing.Protocol):
    """Structural protocol for objects exposing `isvalid`."""

    isvalid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_15(typing.Protocol):
    """Structural protocol for objects exposing `item`."""

    item: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_16(typing.Protocol):
    """Structural protocol for objects exposing `items`."""

    items: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_17(typing.Protocol):
    """Structural protocol for objects exposing `itemsize`."""

    itemsize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_18(typing.Protocol):
    """Structural protocol for objects exposing `iterable`."""

    iterable: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_19(typing.Protocol):
    """Structural protocol for objects exposing `ix`."""

    ix: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_20(typing.Protocol):
    """Structural protocol for objects exposing `ix_`."""

    ix_: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_21(typing.Protocol):
    """Structural protocol for objects exposing `jax`."""

    jax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_22(typing.Protocol):
    """Structural protocol for objects exposing `jit`."""

    jit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_23(typing.Protocol):
    """Structural protocol for objects exposing `join`."""

    join: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_24(typing.Protocol):
    """Structural protocol for objects exposing `join_server`."""

    join_server: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_25(typing.Protocol):
    """Structural protocol for objects exposing `joining`."""

    joining: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_26(typing.Protocol):
    """Structural protocol for objects exposing `jv`."""

    jv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_27(typing.Protocol):
    """Structural protocol for objects exposing `jvp`."""

    jvp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_28(typing.Protocol):
    """Structural protocol for objects exposing `jvp_bad`."""

    jvp_bad: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_29(typing.Protocol):
    """Structural protocol for objects exposing `jvp_style1`."""

    jvp_style1: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_30(typing.Protocol):
    """Structural protocol for objects exposing `jvp_style1_err`."""

    jvp_style1_err: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_31(typing.Protocol):
    """Structural protocol for objects exposing `jvp_style2`."""

    jvp_style2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_32(typing.Protocol):
    """Structural protocol for objects exposing `k0`."""

    k0: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_33(typing.Protocol):
    """Structural protocol for objects exposing `k1`."""

    k1: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_34(typing.Protocol):
    """Structural protocol for objects exposing `keras`."""

    keras: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_35(typing.Protocol):
    """Structural protocol for objects exposing `keras_ops`."""

    keras_ops: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_36(typing.Protocol):
    """Structural protocol for objects exposing `kernels`."""

    kernels: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_37(typing.Protocol):
    """Structural protocol for objects exposing `key`."""

    key: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_38(typing.Protocol):
    """Structural protocol for objects exposing `kind`."""

    kind: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_39(typing.Protocol):
    """Structural protocol for objects exposing `kl_div_loss`."""

    kl_div_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_40(typing.Protocol):
    """Structural protocol for objects exposing `kld`."""

    kld: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_41(typing.Protocol):
    """Structural protocol for objects exposing `kron`."""

    kron: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_42(typing.Protocol):
    """Structural protocol for objects exposing `kronecker`."""

    kronecker: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_43(typing.Protocol):
    """Structural protocol for objects exposing `kullback_leibler_divergence`."""

    kullback_leibler_divergence: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember28_44(typing.Protocol):
    """Structural protocol for objects exposing `kwarg_map`."""

    kwarg_map: EagerTensorProtocol


SHARD_28_READERS: dict[str, MemberReader] = {
    "iscomplex": lambda t: (True, t.iscomplex) if isinstance(t, _HasMember28_00) else (False, None),
    "iscomplexobj": lambda t: (True, t.iscomplexobj) if isinstance(t, _HasMember28_01) else (False, None),
    "isfinite": lambda t: (True, t.isfinite) if isinstance(t, _HasMember28_02) else (False, None),
    "isin": lambda t: (True, t.isin) if isinstance(t, _HasMember28_03) else (False, None),
    "isinf": lambda t: (True, t.isinf) if isinstance(t, _HasMember28_04) else (False, None),
    "isnan": lambda t: (True, t.isnan) if isinstance(t, _HasMember28_05) else (False, None),
    "isneginf": lambda t: (True, t.isneginf) if isinstance(t, _HasMember28_06) else (False, None),
    "isnull": lambda t: (True, t.isnull) if isinstance(t, _HasMember28_07) else (False, None),
    "isposinf": lambda t: (True, t.isposinf) if isinstance(t, _HasMember28_08) else (False, None),
    "isreal": lambda t: (True, t.isreal) if isinstance(t, _HasMember28_09) else (False, None),
    "isrealobj": lambda t: (True, t.isrealobj) if isinstance(t, _HasMember28_10) else (False, None),
    "isscalar": lambda t: (True, t.isscalar) if isinstance(t, _HasMember28_11) else (False, None),
    "issubdtype": lambda t: (True, t.issubdtype) if isinstance(t, _HasMember28_12) else (False, None),
    "istft": lambda t: (True, t.istft) if isinstance(t, _HasMember28_13) else (False, None),
    "isvalid": lambda t: (True, t.isvalid) if isinstance(t, _HasMember28_14) else (False, None),
    "item": lambda t: (True, t.item) if isinstance(t, _HasMember28_15) else (False, None),
    "items": lambda t: (True, t.items) if isinstance(t, _HasMember28_16) else (False, None),
    "itemsize": lambda t: (True, t.itemsize) if isinstance(t, _HasMember28_17) else (False, None),
    "iterable": lambda t: (True, t.iterable) if isinstance(t, _HasMember28_18) else (False, None),
    "ix": lambda t: (True, t.ix) if isinstance(t, _HasMember28_19) else (False, None),
    "ix_": lambda t: (True, t.ix_) if isinstance(t, _HasMember28_20) else (False, None),
    "jax": lambda t: (True, t.jax) if isinstance(t, _HasMember28_21) else (False, None),
    "jit": lambda t: (True, t.jit) if isinstance(t, _HasMember28_22) else (False, None),
    "join": lambda t: (True, t.join) if isinstance(t, _HasMember28_23) else (False, None),
    "join_server": lambda t: (True, t.join_server) if isinstance(t, _HasMember28_24) else (False, None),
    "joining": lambda t: (True, t.joining) if isinstance(t, _HasMember28_25) else (False, None),
    "jv": lambda t: (True, t.jv) if isinstance(t, _HasMember28_26) else (False, None),
    "jvp": lambda t: (True, t.jvp) if isinstance(t, _HasMember28_27) else (False, None),
    "jvp_bad": lambda t: (True, t.jvp_bad) if isinstance(t, _HasMember28_28) else (False, None),
    "jvp_style1": lambda t: (True, t.jvp_style1) if isinstance(t, _HasMember28_29) else (False, None),
    "jvp_style1_err": lambda t: (True, t.jvp_style1_err) if isinstance(t, _HasMember28_30) else (False, None),
    "jvp_style2": lambda t: (True, t.jvp_style2) if isinstance(t, _HasMember28_31) else (False, None),
    "k0": lambda t: (True, t.k0) if isinstance(t, _HasMember28_32) else (False, None),
    "k1": lambda t: (True, t.k1) if isinstance(t, _HasMember28_33) else (False, None),
    "keras": lambda t: (True, t.keras) if isinstance(t, _HasMember28_34) else (False, None),
    "keras_ops": lambda t: (True, t.keras_ops) if isinstance(t, _HasMember28_35) else (False, None),
    "kernels": lambda t: (True, t.kernels) if isinstance(t, _HasMember28_36) else (False, None),
    "key": lambda t: (True, t.key) if isinstance(t, _HasMember28_37) else (False, None),
    "kind": lambda t: (True, t.kind) if isinstance(t, _HasMember28_38) else (False, None),
    "kl_div_loss": lambda t: (True, t.kl_div_loss) if isinstance(t, _HasMember28_39) else (False, None),
    "kld": lambda t: (True, t.kld) if isinstance(t, _HasMember28_40) else (False, None),
    "kron": lambda t: (True, t.kron) if isinstance(t, _HasMember28_41) else (False, None),
    "kronecker": lambda t: (True, t.kronecker) if isinstance(t, _HasMember28_42) else (False, None),
    "kullback_leibler_divergence": lambda t: (
        (True, t.kullback_leibler_divergence) if isinstance(t, _HasMember28_43) else (False, None)
    ),
    "kwarg_map": lambda t: (True, t.kwarg_map) if isinstance(t, _HasMember28_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember28_00,
        _HasMember28_01,
        _HasMember28_02,
        _HasMember28_03,
        _HasMember28_04,
        _HasMember28_05,
        _HasMember28_06,
        _HasMember28_07,
        _HasMember28_08,
        _HasMember28_09,
        _HasMember28_10,
        _HasMember28_11,
        _HasMember28_12,
        _HasMember28_13,
        _HasMember28_14,
        _HasMember28_15,
        _HasMember28_16,
        _HasMember28_17,
        _HasMember28_18,
        _HasMember28_19,
        _HasMember28_20,
        _HasMember28_21,
        _HasMember28_22,
        _HasMember28_23,
        _HasMember28_24,
        _HasMember28_25,
        _HasMember28_26,
        _HasMember28_27,
        _HasMember28_28,
        _HasMember28_29,
        _HasMember28_30,
        _HasMember28_31,
        _HasMember28_32,
        _HasMember28_33,
        _HasMember28_34,
        _HasMember28_35,
        _HasMember28_36,
        _HasMember28_37,
        _HasMember28_38,
        _HasMember28_39,
        _HasMember28_40,
        _HasMember28_41,
        _HasMember28_42,
        _HasMember28_43,
        _HasMember28_44,
    )
)
