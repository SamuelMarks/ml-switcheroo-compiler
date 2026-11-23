"""Static structural protocol bindings shard 51 (uint64..vander)."""

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
class _HasMember51_00(typing.Protocol):
    """Structural protocol for objects exposing `uint64`."""

    uint64: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_01(typing.Protocol):
    """Structural protocol for objects exposing `uint8`."""

    uint8: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_02(typing.Protocol):
    """Structural protocol for objects exposing `unary`."""

    unary: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_03(typing.Protocol):
    """Structural protocol for objects exposing `unflatten`."""

    unflatten: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_04(typing.Protocol):
    """Structural protocol for objects exposing `unfold`."""

    unfold: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_05(typing.Protocol):
    """Structural protocol for objects exposing `uniform`."""

    uniform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_06(typing.Protocol):
    """Structural protocol for objects exposing `union1d`."""

    union1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_07(typing.Protocol):
    """Structural protocol for objects exposing `unique`."""

    unique: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_08(typing.Protocol):
    """Structural protocol for objects exposing `uniqueall`."""

    uniqueall: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_09(typing.Protocol):
    """Structural protocol for objects exposing `uniquecounts`."""

    uniquecounts: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_10(typing.Protocol):
    """Structural protocol for objects exposing `uniqueinverse`."""

    uniqueinverse: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_11(typing.Protocol):
    """Structural protocol for objects exposing `uniquevalues`."""

    uniquevalues: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_12(typing.Protocol):
    """Structural protocol for objects exposing `unknown_model`."""

    unknown_model: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_13(typing.Protocol):
    """Structural protocol for objects exposing `unpack4_fn`."""

    unpack4_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_14(typing.Protocol):
    """Structural protocol for objects exposing `unpack4_jvp`."""

    unpack4_jvp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_15(typing.Protocol):
    """Structural protocol for objects exposing `unpack_fn`."""

    unpack_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_16(typing.Protocol):
    """Structural protocol for objects exposing `unpack_jvp`."""

    unpack_jvp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_17(typing.Protocol):
    """Structural protocol for objects exposing `unpackbits`."""

    unpackbits: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_18(typing.Protocol):
    """Structural protocol for objects exposing `unravel_index`."""

    unravel_index: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_19(typing.Protocol):
    """Structural protocol for objects exposing `unravelindex`."""

    unravelindex: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_20(typing.Protocol):
    """Structural protocol for objects exposing `unsorted_segment_max`."""

    unsorted_segment_max: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_21(typing.Protocol):
    """Structural protocol for objects exposing `unsorted_segment_mean`."""

    unsorted_segment_mean: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_22(typing.Protocol):
    """Structural protocol for objects exposing `unsorted_segment_min`."""

    unsorted_segment_min: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_23(typing.Protocol):
    """Structural protocol for objects exposing `unsorted_segment_prod`."""

    unsorted_segment_prod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_24(typing.Protocol):
    """Structural protocol for objects exposing `unsorted_segment_sqrt_n`."""

    unsorted_segment_sqrt_n: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_25(typing.Protocol):
    """Structural protocol for objects exposing `unsorted_segment_sum`."""

    unsorted_segment_sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_26(typing.Protocol):
    """Structural protocol for objects exposing `unsqueeze`."""

    unsqueeze: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_27(typing.Protocol):
    """Structural protocol for objects exposing `unstack`."""

    unstack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_28(typing.Protocol):
    """Structural protocol for objects exposing `unwrap`."""

    unwrap: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_29(typing.Protocol):
    """Structural protocol for objects exposing `unzip`."""

    unzip: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_30(typing.Protocol):
    """Structural protocol for objects exposing `updateslice`."""

    updateslice: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_31(typing.Protocol):
    """Structural protocol for objects exposing `upper`."""

    upper: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_32(typing.Protocol):
    """Structural protocol for objects exposing `usm_type`."""

    usm_type: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_33(typing.Protocol):
    """Structural protocol for objects exposing `utf8_capitalize`."""

    utf8_capitalize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_34(typing.Protocol):
    """Structural protocol for objects exposing `utf8_length`."""

    utf8_length: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_35(typing.Protocol):
    """Structural protocol for objects exposing `utf8_lower`."""

    utf8_lower: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_36(typing.Protocol):
    """Structural protocol for objects exposing `utf8_title`."""

    utf8_title: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_37(typing.Protocol):
    """Structural protocol for objects exposing `utf8_upper`."""

    utf8_upper: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_38(typing.Protocol):
    """Structural protocol for objects exposing `utils`."""

    utils: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_39(typing.Protocol):
    """Structural protocol for objects exposing `val`."""

    val: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_40(typing.Protocol):
    """Structural protocol for objects exposing `value`."""

    value: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_41(typing.Protocol):
    """Structural protocol for objects exposing `value_and_grad`."""

    value_and_grad: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_42(typing.Protocol):
    """Structural protocol for objects exposing `value_counts`."""

    value_counts: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_43(typing.Protocol):
    """Structural protocol for objects exposing `values`."""

    values: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember51_44(typing.Protocol):
    """Structural protocol for objects exposing `vander`."""

    vander: EagerTensorProtocol


SHARD_51_READERS: dict[str, MemberReader] = {
    "uint64": lambda t: (True, t.uint64) if isinstance(t, _HasMember51_00) else (False, None),
    "uint8": lambda t: (True, t.uint8) if isinstance(t, _HasMember51_01) else (False, None),
    "unary": lambda t: (True, t.unary) if isinstance(t, _HasMember51_02) else (False, None),
    "unflatten": lambda t: (True, t.unflatten) if isinstance(t, _HasMember51_03) else (False, None),
    "unfold": lambda t: (True, t.unfold) if isinstance(t, _HasMember51_04) else (False, None),
    "uniform": lambda t: (True, t.uniform) if isinstance(t, _HasMember51_05) else (False, None),
    "union1d": lambda t: (True, t.union1d) if isinstance(t, _HasMember51_06) else (False, None),
    "unique": lambda t: (True, t.unique) if isinstance(t, _HasMember51_07) else (False, None),
    "uniqueall": lambda t: (True, t.uniqueall) if isinstance(t, _HasMember51_08) else (False, None),
    "uniquecounts": lambda t: (True, t.uniquecounts) if isinstance(t, _HasMember51_09) else (False, None),
    "uniqueinverse": lambda t: (True, t.uniqueinverse) if isinstance(t, _HasMember51_10) else (False, None),
    "uniquevalues": lambda t: (True, t.uniquevalues) if isinstance(t, _HasMember51_11) else (False, None),
    "unknown_model": lambda t: (True, t.unknown_model) if isinstance(t, _HasMember51_12) else (False, None),
    "unpack4_fn": lambda t: (True, t.unpack4_fn) if isinstance(t, _HasMember51_13) else (False, None),
    "unpack4_jvp": lambda t: (True, t.unpack4_jvp) if isinstance(t, _HasMember51_14) else (False, None),
    "unpack_fn": lambda t: (True, t.unpack_fn) if isinstance(t, _HasMember51_15) else (False, None),
    "unpack_jvp": lambda t: (True, t.unpack_jvp) if isinstance(t, _HasMember51_16) else (False, None),
    "unpackbits": lambda t: (True, t.unpackbits) if isinstance(t, _HasMember51_17) else (False, None),
    "unravel_index": lambda t: (True, t.unravel_index) if isinstance(t, _HasMember51_18) else (False, None),
    "unravelindex": lambda t: (True, t.unravelindex) if isinstance(t, _HasMember51_19) else (False, None),
    "unsorted_segment_max": lambda t: (
        (True, t.unsorted_segment_max) if isinstance(t, _HasMember51_20) else (False, None)
    ),
    "unsorted_segment_mean": lambda t: (
        (True, t.unsorted_segment_mean) if isinstance(t, _HasMember51_21) else (False, None)
    ),
    "unsorted_segment_min": lambda t: (
        (True, t.unsorted_segment_min) if isinstance(t, _HasMember51_22) else (False, None)
    ),
    "unsorted_segment_prod": lambda t: (
        (True, t.unsorted_segment_prod) if isinstance(t, _HasMember51_23) else (False, None)
    ),
    "unsorted_segment_sqrt_n": lambda t: (
        (True, t.unsorted_segment_sqrt_n) if isinstance(t, _HasMember51_24) else (False, None)
    ),
    "unsorted_segment_sum": lambda t: (
        (True, t.unsorted_segment_sum) if isinstance(t, _HasMember51_25) else (False, None)
    ),
    "unsqueeze": lambda t: (True, t.unsqueeze) if isinstance(t, _HasMember51_26) else (False, None),
    "unstack": lambda t: (True, t.unstack) if isinstance(t, _HasMember51_27) else (False, None),
    "unwrap": lambda t: (True, t.unwrap) if isinstance(t, _HasMember51_28) else (False, None),
    "unzip": lambda t: (True, t.unzip) if isinstance(t, _HasMember51_29) else (False, None),
    "updateslice": lambda t: (True, t.updateslice) if isinstance(t, _HasMember51_30) else (False, None),
    "upper": lambda t: (True, t.upper) if isinstance(t, _HasMember51_31) else (False, None),
    "usm_type": lambda t: (True, t.usm_type) if isinstance(t, _HasMember51_32) else (False, None),
    "utf8_capitalize": lambda t: (True, t.utf8_capitalize) if isinstance(t, _HasMember51_33) else (False, None),
    "utf8_length": lambda t: (True, t.utf8_length) if isinstance(t, _HasMember51_34) else (False, None),
    "utf8_lower": lambda t: (True, t.utf8_lower) if isinstance(t, _HasMember51_35) else (False, None),
    "utf8_title": lambda t: (True, t.utf8_title) if isinstance(t, _HasMember51_36) else (False, None),
    "utf8_upper": lambda t: (True, t.utf8_upper) if isinstance(t, _HasMember51_37) else (False, None),
    "utils": lambda t: (True, t.utils) if isinstance(t, _HasMember51_38) else (False, None),
    "val": lambda t: (True, t.val) if isinstance(t, _HasMember51_39) else (False, None),
    "value": lambda t: (True, t.value) if isinstance(t, _HasMember51_40) else (False, None),
    "value_and_grad": lambda t: (True, t.value_and_grad) if isinstance(t, _HasMember51_41) else (False, None),
    "value_counts": lambda t: (True, t.value_counts) if isinstance(t, _HasMember51_42) else (False, None),
    "values": lambda t: (True, t.values) if isinstance(t, _HasMember51_43) else (False, None),
    "vander": lambda t: (True, t.vander) if isinstance(t, _HasMember51_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember51_00,
        _HasMember51_01,
        _HasMember51_02,
        _HasMember51_03,
        _HasMember51_04,
        _HasMember51_05,
        _HasMember51_06,
        _HasMember51_07,
        _HasMember51_08,
        _HasMember51_09,
        _HasMember51_10,
        _HasMember51_11,
        _HasMember51_12,
        _HasMember51_13,
        _HasMember51_14,
        _HasMember51_15,
        _HasMember51_16,
        _HasMember51_17,
        _HasMember51_18,
        _HasMember51_19,
        _HasMember51_20,
        _HasMember51_21,
        _HasMember51_22,
        _HasMember51_23,
        _HasMember51_24,
        _HasMember51_25,
        _HasMember51_26,
        _HasMember51_27,
        _HasMember51_28,
        _HasMember51_29,
        _HasMember51_30,
        _HasMember51_31,
        _HasMember51_32,
        _HasMember51_33,
        _HasMember51_34,
        _HasMember51_35,
        _HasMember51_36,
        _HasMember51_37,
        _HasMember51_38,
        _HasMember51_39,
        _HasMember51_40,
        _HasMember51_41,
        _HasMember51_42,
        _HasMember51_43,
        _HasMember51_44,
    )
)
