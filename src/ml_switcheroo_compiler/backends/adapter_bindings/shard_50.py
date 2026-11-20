"""Static structural protocol bindings shard 50 (tree_util..uint32)."""

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
class _HasMember50_00(typing.Protocol):
    """Structural protocol for objects exposing `tree_util`."""

    tree_util: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_01(typing.Protocol):
    """Structural protocol for objects exposing `tri`."""

    tri: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_02(typing.Protocol):
    """Structural protocol for objects exposing `triangular`."""

    triangular: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_03(typing.Protocol):
    """Structural protocol for objects exposing `triangular_solve`."""

    triangular_solve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_04(typing.Protocol):
    """Structural protocol for objects exposing `triangularsolve`."""

    triangularsolve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_05(typing.Protocol):
    """Structural protocol for objects exposing `tridiagonal`."""

    tridiagonal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_06(typing.Protocol):
    """Structural protocol for objects exposing `tridiagonal_matmul`."""

    tridiagonal_matmul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_07(typing.Protocol):
    """Structural protocol for objects exposing `tridiagonal_solve`."""

    tridiagonal_solve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_08(typing.Protocol):
    """Structural protocol for objects exposing `tridiagonalmatmul`."""

    tridiagonalmatmul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_09(typing.Protocol):
    """Structural protocol for objects exposing `tridiagonalsolve`."""

    tridiagonalsolve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_10(typing.Protocol):
    """Structural protocol for objects exposing `trigger_drain`."""

    trigger_drain: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_11(typing.Protocol):
    """Structural protocol for objects exposing `trigger_low_buffer`."""

    trigger_low_buffer: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_12(typing.Protocol):
    """Structural protocol for objects exposing `trigger_message`."""

    trigger_message: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_13(typing.Protocol):
    """Structural protocol for objects exposing `triinv`."""

    triinv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_14(typing.Protocol):
    """Structural protocol for objects exposing `tril`."""

    tril: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_15(typing.Protocol):
    """Structural protocol for objects exposing `tril_indices`."""

    tril_indices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_16(typing.Protocol):
    """Structural protocol for objects exposing `tril_indices_from`."""

    tril_indices_from: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_17(typing.Protocol):
    """Structural protocol for objects exposing `trilindices`."""

    trilindices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_18(typing.Protocol):
    """Structural protocol for objects exposing `trilindicesfrom`."""

    trilindicesfrom: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_19(typing.Protocol):
    """Structural protocol for objects exposing `trim_zeros`."""

    trim_zeros: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_20(typing.Protocol):
    """Structural protocol for objects exposing `trimzeros`."""

    trimzeros: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_21(typing.Protocol):
    """Structural protocol for objects exposing `triplet_loss`."""

    triplet_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_22(typing.Protocol):
    """Structural protocol for objects exposing `triu`."""

    triu: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_23(typing.Protocol):
    """Structural protocol for objects exposing `triu_indices`."""

    triu_indices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_24(typing.Protocol):
    """Structural protocol for objects exposing `triu_indices_from`."""

    triu_indices_from: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_25(typing.Protocol):
    """Structural protocol for objects exposing `triuindices`."""

    triuindices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_26(typing.Protocol):
    """Structural protocol for objects exposing `triuindicesfrom`."""

    triuindicesfrom: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_27(typing.Protocol):
    """Structural protocol for objects exposing `true_divide`."""

    true_divide: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_28(typing.Protocol):
    """Structural protocol for objects exposing `true_fn`."""

    true_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_29(typing.Protocol):
    """Structural protocol for objects exposing `truediv`."""

    truediv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_30(typing.Protocol):
    """Structural protocol for objects exposing `truedivide`."""

    truedivide: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_31(typing.Protocol):
    """Structural protocol for objects exposing `trunc`."""

    trunc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_32(typing.Protocol):
    """Structural protocol for objects exposing `truncate`."""

    truncate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_33(typing.Protocol):
    """Structural protocol for objects exposing `truncatediv`."""

    truncatediv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_34(typing.Protocol):
    """Structural protocol for objects exposing `truncatemod`."""

    truncatemod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_35(typing.Protocol):
    """Structural protocol for objects exposing `tversky_loss`."""

    tversky_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_36(typing.Protocol):
    """Structural protocol for objects exposing `type`."""

    type: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_37(typing.Protocol):
    """Structural protocol for objects exposing `type_annotation`."""

    type_annotation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_38(typing.Protocol):
    """Structural protocol for objects exposing `type_inference`."""

    type_inference: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_39(typing.Protocol):
    """Structural protocol for objects exposing `type_str`."""

    type_str: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_40(typing.Protocol):
    """Structural protocol for objects exposing `types_mapping_proxy`."""

    types_mapping_proxy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_41(typing.Protocol):
    """Structural protocol for objects exposing `typing`."""

    typing: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_42(typing.Protocol):
    """Structural protocol for objects exposing `uint`."""

    uint: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_43(typing.Protocol):
    """Structural protocol for objects exposing `uint16`."""

    uint16: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember50_44(typing.Protocol):
    """Structural protocol for objects exposing `uint32`."""

    uint32: EagerTensorProtocol


SHARD_50_READERS: dict[str, MemberReader] = {
    "tree_util": lambda t: (True, t.tree_util) if isinstance(t, _HasMember50_00) else (False, None),
    "tri": lambda t: (True, t.tri) if isinstance(t, _HasMember50_01) else (False, None),
    "triangular": lambda t: (True, t.triangular) if isinstance(t, _HasMember50_02) else (False, None),
    "triangular_solve": lambda t: (True, t.triangular_solve) if isinstance(t, _HasMember50_03) else (False, None),
    "triangularsolve": lambda t: (True, t.triangularsolve) if isinstance(t, _HasMember50_04) else (False, None),
    "tridiagonal": lambda t: (True, t.tridiagonal) if isinstance(t, _HasMember50_05) else (False, None),
    "tridiagonal_matmul": lambda t: (True, t.tridiagonal_matmul) if isinstance(t, _HasMember50_06) else (False, None),
    "tridiagonal_solve": lambda t: (True, t.tridiagonal_solve) if isinstance(t, _HasMember50_07) else (False, None),
    "tridiagonalmatmul": lambda t: (True, t.tridiagonalmatmul) if isinstance(t, _HasMember50_08) else (False, None),
    "tridiagonalsolve": lambda t: (True, t.tridiagonalsolve) if isinstance(t, _HasMember50_09) else (False, None),
    "trigger_drain": lambda t: (True, t.trigger_drain) if isinstance(t, _HasMember50_10) else (False, None),
    "trigger_low_buffer": lambda t: (True, t.trigger_low_buffer) if isinstance(t, _HasMember50_11) else (False, None),
    "trigger_message": lambda t: (True, t.trigger_message) if isinstance(t, _HasMember50_12) else (False, None),
    "triinv": lambda t: (True, t.triinv) if isinstance(t, _HasMember50_13) else (False, None),
    "tril": lambda t: (True, t.tril) if isinstance(t, _HasMember50_14) else (False, None),
    "tril_indices": lambda t: (True, t.tril_indices) if isinstance(t, _HasMember50_15) else (False, None),
    "tril_indices_from": lambda t: (True, t.tril_indices_from) if isinstance(t, _HasMember50_16) else (False, None),
    "trilindices": lambda t: (True, t.trilindices) if isinstance(t, _HasMember50_17) else (False, None),
    "trilindicesfrom": lambda t: (True, t.trilindicesfrom) if isinstance(t, _HasMember50_18) else (False, None),
    "trim_zeros": lambda t: (True, t.trim_zeros) if isinstance(t, _HasMember50_19) else (False, None),
    "trimzeros": lambda t: (True, t.trimzeros) if isinstance(t, _HasMember50_20) else (False, None),
    "triplet_loss": lambda t: (True, t.triplet_loss) if isinstance(t, _HasMember50_21) else (False, None),
    "triu": lambda t: (True, t.triu) if isinstance(t, _HasMember50_22) else (False, None),
    "triu_indices": lambda t: (True, t.triu_indices) if isinstance(t, _HasMember50_23) else (False, None),
    "triu_indices_from": lambda t: (True, t.triu_indices_from) if isinstance(t, _HasMember50_24) else (False, None),
    "triuindices": lambda t: (True, t.triuindices) if isinstance(t, _HasMember50_25) else (False, None),
    "triuindicesfrom": lambda t: (True, t.triuindicesfrom) if isinstance(t, _HasMember50_26) else (False, None),
    "true_divide": lambda t: (True, t.true_divide) if isinstance(t, _HasMember50_27) else (False, None),
    "true_fn": lambda t: (True, t.true_fn) if isinstance(t, _HasMember50_28) else (False, None),
    "truediv": lambda t: (True, t.truediv) if isinstance(t, _HasMember50_29) else (False, None),
    "truedivide": lambda t: (True, t.truedivide) if isinstance(t, _HasMember50_30) else (False, None),
    "trunc": lambda t: (True, t.trunc) if isinstance(t, _HasMember50_31) else (False, None),
    "truncate": lambda t: (True, t.truncate) if isinstance(t, _HasMember50_32) else (False, None),
    "truncatediv": lambda t: (True, t.truncatediv) if isinstance(t, _HasMember50_33) else (False, None),
    "truncatemod": lambda t: (True, t.truncatemod) if isinstance(t, _HasMember50_34) else (False, None),
    "tversky_loss": lambda t: (True, t.tversky_loss) if isinstance(t, _HasMember50_35) else (False, None),
    "type": lambda t: (True, t.type) if isinstance(t, _HasMember50_36) else (False, None),
    "type_annotation": lambda t: (True, t.type_annotation) if isinstance(t, _HasMember50_37) else (False, None),
    "type_inference": lambda t: (True, t.type_inference) if isinstance(t, _HasMember50_38) else (False, None),
    "type_str": lambda t: (True, t.type_str) if isinstance(t, _HasMember50_39) else (False, None),
    "types_mapping_proxy": lambda t: (True, t.types_mapping_proxy) if isinstance(t, _HasMember50_40) else (False, None),
    "typing": lambda t: (True, t.typing) if isinstance(t, _HasMember50_41) else (False, None),
    "uint": lambda t: (True, t.uint) if isinstance(t, _HasMember50_42) else (False, None),
    "uint16": lambda t: (True, t.uint16) if isinstance(t, _HasMember50_43) else (False, None),
    "uint32": lambda t: (True, t.uint32) if isinstance(t, _HasMember50_44) else (False, None),
}
