"""Static structural protocol bindings shard 10 (axis..bitwiseand)."""

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
class _HasMember10_00(typing.Protocol):
    """Structural protocol for objects exposing `axis`."""

    axis: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_01(typing.Protocol):
    """Structural protocol for objects exposing `axis_translation_pass`."""

    axis_translation_pass: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_02(typing.Protocol):
    """Structural protocol for objects exposing `b`."""

    b: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_03(typing.Protocol):
    """Structural protocol for objects exposing `backend`."""

    backend: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_04(typing.Protocol):
    """Structural protocol for objects exposing `backends`."""

    backends: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_05(typing.Protocol):
    """Structural protocol for objects exposing `bad_ast_func`."""

    bad_ast_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_06(typing.Protocol):
    """Structural protocol for objects exposing `bad_func`."""

    bad_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_07(typing.Protocol):
    """Structural protocol for objects exposing `bad_sig`."""

    bad_sig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_08(typing.Protocol):
    """Structural protocol for objects exposing `ball`."""

    ball: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_09(typing.Protocol):
    """Structural protocol for objects exposing `band_part`."""

    band_part: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_10(typing.Protocol):
    """Structural protocol for objects exposing `banded_triangular_solve`."""

    banded_triangular_solve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_11(typing.Protocol):
    """Structural protocol for objects exposing `bandpart`."""

    bandpart: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_12(typing.Protocol):
    """Structural protocol for objects exposing `bartlett`."""

    bartlett: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_13(typing.Protocol):
    """Structural protocol for objects exposing `base`."""

    base: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_14(typing.Protocol):
    """Structural protocol for objects exposing `bessel_i0`."""

    bessel_i0: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_15(typing.Protocol):
    """Structural protocol for objects exposing `bessel_i0e`."""

    bessel_i0e: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_16(typing.Protocol):
    """Structural protocol for objects exposing `bessel_i1`."""

    bessel_i1: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_17(typing.Protocol):
    """Structural protocol for objects exposing `bessel_i1e`."""

    bessel_i1e: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_18(typing.Protocol):
    """Structural protocol for objects exposing `bessel_j0`."""

    bessel_j0: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_19(typing.Protocol):
    """Structural protocol for objects exposing `bessel_j1`."""

    bessel_j1: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_20(typing.Protocol):
    """Structural protocol for objects exposing `bessel_y0`."""

    bessel_y0: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_21(typing.Protocol):
    """Structural protocol for objects exposing `bessel_y1`."""

    bessel_y1: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_22(typing.Protocol):
    """Structural protocol for objects exposing `beta`."""

    beta: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_23(typing.Protocol):
    """Structural protocol for objects exposing `betainc`."""

    betainc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_24(typing.Protocol):
    """Structural protocol for objects exposing `bfloat16`."""

    bfloat16: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_25(typing.Protocol):
    """Structural protocol for objects exposing `binary`."""

    binary: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_26(typing.Protocol):
    """Structural protocol for objects exposing `binary_crossentropy`."""

    binary_crossentropy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_27(typing.Protocol):
    """Structural protocol for objects exposing `binary_record_options`."""

    binary_record_options: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_28(typing.Protocol):
    """Structural protocol for objects exposing `binary_record_writer`."""

    binary_record_writer: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_29(typing.Protocol):
    """Structural protocol for objects exposing `binaryrecordoptions`."""

    binaryrecordoptions: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_30(typing.Protocol):
    """Structural protocol for objects exposing `binaryrecordwriter`."""

    binaryrecordwriter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_31(typing.Protocol):
    """Structural protocol for objects exposing `bincount`."""

    bincount: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_32(typing.Protocol):
    """Structural protocol for objects exposing `binding`."""

    binding: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_33(typing.Protocol):
    """Structural protocol for objects exposing `binomial`."""

    binomial: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_34(typing.Protocol):
    """Structural protocol for objects exposing `bit_wise_and`."""

    bit_wise_and: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_35(typing.Protocol):
    """Structural protocol for objects exposing `bit_wise_not`."""

    bit_wise_not: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_36(typing.Protocol):
    """Structural protocol for objects exposing `bit_wise_or`."""

    bit_wise_or: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_37(typing.Protocol):
    """Structural protocol for objects exposing `bit_wise_xor`."""

    bit_wise_xor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_38(typing.Protocol):
    """Structural protocol for objects exposing `bitcast`."""

    bitcast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_39(typing.Protocol):
    """Structural protocol for objects exposing `bitwise_and`."""

    bitwise_and: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_40(typing.Protocol):
    """Structural protocol for objects exposing `bitwise_count`."""

    bitwise_count: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_41(typing.Protocol):
    """Structural protocol for objects exposing `bitwise_not`."""

    bitwise_not: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_42(typing.Protocol):
    """Structural protocol for objects exposing `bitwise_or`."""

    bitwise_or: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_43(typing.Protocol):
    """Structural protocol for objects exposing `bitwise_xor`."""

    bitwise_xor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember10_44(typing.Protocol):
    """Structural protocol for objects exposing `bitwiseand`."""

    bitwiseand: EagerTensorProtocol


SHARD_10_READERS: dict[str, MemberReader] = {
    "axis": lambda t: (True, t.axis) if isinstance(t, _HasMember10_00) else (False, None),
    "axis_translation_pass": lambda t: (
        (True, t.axis_translation_pass) if isinstance(t, _HasMember10_01) else (False, None)
    ),
    "b": lambda t: (True, t.b) if isinstance(t, _HasMember10_02) else (False, None),
    "backend": lambda t: (True, t.backend) if isinstance(t, _HasMember10_03) else (False, None),
    "backends": lambda t: (True, t.backends) if isinstance(t, _HasMember10_04) else (False, None),
    "bad_ast_func": lambda t: (True, t.bad_ast_func) if isinstance(t, _HasMember10_05) else (False, None),
    "bad_func": lambda t: (True, t.bad_func) if isinstance(t, _HasMember10_06) else (False, None),
    "bad_sig": lambda t: (True, t.bad_sig) if isinstance(t, _HasMember10_07) else (False, None),
    "ball": lambda t: (True, t.ball) if isinstance(t, _HasMember10_08) else (False, None),
    "band_part": lambda t: (True, t.band_part) if isinstance(t, _HasMember10_09) else (False, None),
    "banded_triangular_solve": lambda t: (
        (True, t.banded_triangular_solve) if isinstance(t, _HasMember10_10) else (False, None)
    ),
    "bandpart": lambda t: (True, t.bandpart) if isinstance(t, _HasMember10_11) else (False, None),
    "bartlett": lambda t: (True, t.bartlett) if isinstance(t, _HasMember10_12) else (False, None),
    "base": lambda t: (True, t.base) if isinstance(t, _HasMember10_13) else (False, None),
    "bessel_i0": lambda t: (True, t.bessel_i0) if isinstance(t, _HasMember10_14) else (False, None),
    "bessel_i0e": lambda t: (True, t.bessel_i0e) if isinstance(t, _HasMember10_15) else (False, None),
    "bessel_i1": lambda t: (True, t.bessel_i1) if isinstance(t, _HasMember10_16) else (False, None),
    "bessel_i1e": lambda t: (True, t.bessel_i1e) if isinstance(t, _HasMember10_17) else (False, None),
    "bessel_j0": lambda t: (True, t.bessel_j0) if isinstance(t, _HasMember10_18) else (False, None),
    "bessel_j1": lambda t: (True, t.bessel_j1) if isinstance(t, _HasMember10_19) else (False, None),
    "bessel_y0": lambda t: (True, t.bessel_y0) if isinstance(t, _HasMember10_20) else (False, None),
    "bessel_y1": lambda t: (True, t.bessel_y1) if isinstance(t, _HasMember10_21) else (False, None),
    "beta": lambda t: (True, t.beta) if isinstance(t, _HasMember10_22) else (False, None),
    "betainc": lambda t: (True, t.betainc) if isinstance(t, _HasMember10_23) else (False, None),
    "bfloat16": lambda t: (True, t.bfloat16) if isinstance(t, _HasMember10_24) else (False, None),
    "binary": lambda t: (True, t.binary) if isinstance(t, _HasMember10_25) else (False, None),
    "binary_crossentropy": lambda t: (True, t.binary_crossentropy) if isinstance(t, _HasMember10_26) else (False, None),
    "binary_record_options": lambda t: (
        (True, t.binary_record_options) if isinstance(t, _HasMember10_27) else (False, None)
    ),
    "binary_record_writer": lambda t: (
        (True, t.binary_record_writer) if isinstance(t, _HasMember10_28) else (False, None)
    ),
    "binaryrecordoptions": lambda t: (True, t.binaryrecordoptions) if isinstance(t, _HasMember10_29) else (False, None),
    "binaryrecordwriter": lambda t: (True, t.binaryrecordwriter) if isinstance(t, _HasMember10_30) else (False, None),
    "bincount": lambda t: (True, t.bincount) if isinstance(t, _HasMember10_31) else (False, None),
    "binding": lambda t: (True, t.binding) if isinstance(t, _HasMember10_32) else (False, None),
    "binomial": lambda t: (True, t.binomial) if isinstance(t, _HasMember10_33) else (False, None),
    "bit_wise_and": lambda t: (True, t.bit_wise_and) if isinstance(t, _HasMember10_34) else (False, None),
    "bit_wise_not": lambda t: (True, t.bit_wise_not) if isinstance(t, _HasMember10_35) else (False, None),
    "bit_wise_or": lambda t: (True, t.bit_wise_or) if isinstance(t, _HasMember10_36) else (False, None),
    "bit_wise_xor": lambda t: (True, t.bit_wise_xor) if isinstance(t, _HasMember10_37) else (False, None),
    "bitcast": lambda t: (True, t.bitcast) if isinstance(t, _HasMember10_38) else (False, None),
    "bitwise_and": lambda t: (True, t.bitwise_and) if isinstance(t, _HasMember10_39) else (False, None),
    "bitwise_count": lambda t: (True, t.bitwise_count) if isinstance(t, _HasMember10_40) else (False, None),
    "bitwise_not": lambda t: (True, t.bitwise_not) if isinstance(t, _HasMember10_41) else (False, None),
    "bitwise_or": lambda t: (True, t.bitwise_or) if isinstance(t, _HasMember10_42) else (False, None),
    "bitwise_xor": lambda t: (True, t.bitwise_xor) if isinstance(t, _HasMember10_43) else (False, None),
    "bitwiseand": lambda t: (True, t.bitwiseand) if isinstance(t, _HasMember10_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember10_00,
        _HasMember10_01,
        _HasMember10_02,
        _HasMember10_03,
        _HasMember10_04,
        _HasMember10_05,
        _HasMember10_06,
        _HasMember10_07,
        _HasMember10_08,
        _HasMember10_09,
        _HasMember10_10,
        _HasMember10_11,
        _HasMember10_12,
        _HasMember10_13,
        _HasMember10_14,
        _HasMember10_15,
        _HasMember10_16,
        _HasMember10_17,
        _HasMember10_18,
        _HasMember10_19,
        _HasMember10_20,
        _HasMember10_21,
        _HasMember10_22,
        _HasMember10_23,
        _HasMember10_24,
        _HasMember10_25,
        _HasMember10_26,
        _HasMember10_27,
        _HasMember10_28,
        _HasMember10_29,
        _HasMember10_30,
        _HasMember10_31,
        _HasMember10_32,
        _HasMember10_33,
        _HasMember10_34,
        _HasMember10_35,
        _HasMember10_36,
        _HasMember10_37,
        _HasMember10_38,
        _HasMember10_39,
        _HasMember10_40,
        _HasMember10_41,
        _HasMember10_42,
        _HasMember10_43,
        _HasMember10_44,
    )
)
