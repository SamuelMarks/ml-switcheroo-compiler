"""Static structural protocol bindings shard 47 (splitting..sub)."""

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
class _HasMember47_00(typing.Protocol):
    """Structural protocol for objects exposing `splitting`."""

    splitting: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_01(typing.Protocol):
    """Structural protocol for objects exposing `sqrt`."""

    sqrt: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_02(typing.Protocol):
    """Structural protocol for objects exposing `sqrtm`."""

    sqrtm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_03(typing.Protocol):
    """Structural protocol for objects exposing `square`."""

    square: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_04(typing.Protocol):
    """Structural protocol for objects exposing `square_fn`."""

    square_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_05(typing.Protocol):
    """Structural protocol for objects exposing `square_scaled`."""

    square_scaled: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_06(typing.Protocol):
    """Structural protocol for objects exposing `square_scaled_jvp`."""

    square_scaled_jvp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_07(typing.Protocol):
    """Structural protocol for objects exposing `squared_difference`."""

    squared_difference: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_08(typing.Protocol):
    """Structural protocol for objects exposing `squareplus`."""

    squareplus: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_09(typing.Protocol):
    """Structural protocol for objects exposing `squeeze`."""

    squeeze: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_10(typing.Protocol):
    """Structural protocol for objects exposing `stack`."""

    stack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_11(typing.Protocol):
    """Structural protocol for objects exposing `standard_fn`."""

    standard_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_12(typing.Protocol):
    """Structural protocol for objects exposing `start`."""

    start: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_13(typing.Protocol):
    """Structural protocol for objects exposing `state`."""

    state: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_14(typing.Protocol):
    """Structural protocol for objects exposing `state_dict`."""

    state_dict: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_15(typing.Protocol):
    """Structural protocol for objects exposing `stateless_random_uniform`."""

    stateless_random_uniform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_16(typing.Protocol):
    """Structural protocol for objects exposing `stateless_split`."""

    stateless_split: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_17(typing.Protocol):
    """Structural protocol for objects exposing `stats`."""

    stats: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_18(typing.Protocol):
    """Structural protocol for objects exposing `std`."""

    std: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_19(typing.Protocol):
    """Structural protocol for objects exposing `std_func`."""

    std_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_20(typing.Protocol):
    """Structural protocol for objects exposing `stddev`."""

    stddev: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_21(typing.Protocol):
    """Structural protocol for objects exposing `stderr`."""

    stderr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_22(typing.Protocol):
    """Structural protocol for objects exposing `stdout`."""

    stdout: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_23(typing.Protocol):
    """Structural protocol for objects exposing `step`."""

    step: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_24(typing.Protocol):
    """Structural protocol for objects exposing `stft`."""

    stft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_25(typing.Protocol):
    """Structural protocol for objects exposing `stop_gradient`."""

    stop_gradient: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_26(typing.Protocol):
    """Structural protocol for objects exposing `str`."""

    str: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_27(typing.Protocol):
    """Structural protocol for objects exposing `strided_slice`."""

    strided_slice: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_28(typing.Protocol):
    """Structural protocol for objects exposing `strides`."""

    strides: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_29(typing.Protocol):
    """Structural protocol for objects exposing `string_join`."""

    string_join: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_30(typing.Protocol):
    """Structural protocol for objects exposing `string_length`."""

    string_length: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_31(typing.Protocol):
    """Structural protocol for objects exposing `string_lower`."""

    string_lower: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_32(typing.Protocol):
    """Structural protocol for objects exposing `string_split`."""

    string_split: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_33(typing.Protocol):
    """Structural protocol for objects exposing `string_substr`."""

    string_substr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_34(typing.Protocol):
    """Structural protocol for objects exposing `string_to_hash`."""

    string_to_hash: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_35(typing.Protocol):
    """Structural protocol for objects exposing `string_to_number`."""

    string_to_number: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_36(typing.Protocol):
    """Structural protocol for objects exposing `string_upper`."""

    string_upper: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_37(typing.Protocol):
    """Structural protocol for objects exposing `stringlower`."""

    stringlower: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_38(typing.Protocol):
    """Structural protocol for objects exposing `stringsplit`."""

    stringsplit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_39(typing.Protocol):
    """Structural protocol for objects exposing `stringsubstr`."""

    stringsubstr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_40(typing.Protocol):
    """Structural protocol for objects exposing `stringtohash`."""

    stringtohash: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_41(typing.Protocol):
    """Structural protocol for objects exposing `stringtonumber`."""

    stringtonumber: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_42(typing.Protocol):
    """Structural protocol for objects exposing `stringupper`."""

    stringupper: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_43(typing.Protocol):
    """Structural protocol for objects exposing `struct_field`."""

    struct_field: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember47_44(typing.Protocol):
    """Structural protocol for objects exposing `sub`."""

    sub: EagerTensorProtocol


SHARD_47_READERS: dict[str, MemberReader] = {
    "splitting": lambda t: (True, t.splitting) if isinstance(t, _HasMember47_00) else (False, None),
    "sqrt": lambda t: (True, t.sqrt) if isinstance(t, _HasMember47_01) else (False, None),
    "sqrtm": lambda t: (True, t.sqrtm) if isinstance(t, _HasMember47_02) else (False, None),
    "square": lambda t: (True, t.square) if isinstance(t, _HasMember47_03) else (False, None),
    "square_fn": lambda t: (True, t.square_fn) if isinstance(t, _HasMember47_04) else (False, None),
    "square_scaled": lambda t: (True, t.square_scaled) if isinstance(t, _HasMember47_05) else (False, None),
    "square_scaled_jvp": lambda t: (True, t.square_scaled_jvp) if isinstance(t, _HasMember47_06) else (False, None),
    "squared_difference": lambda t: (True, t.squared_difference) if isinstance(t, _HasMember47_07) else (False, None),
    "squareplus": lambda t: (True, t.squareplus) if isinstance(t, _HasMember47_08) else (False, None),
    "squeeze": lambda t: (True, t.squeeze) if isinstance(t, _HasMember47_09) else (False, None),
    "stack": lambda t: (True, t.stack) if isinstance(t, _HasMember47_10) else (False, None),
    "standard_fn": lambda t: (True, t.standard_fn) if isinstance(t, _HasMember47_11) else (False, None),
    "start": lambda t: (True, t.start) if isinstance(t, _HasMember47_12) else (False, None),
    "state": lambda t: (True, t.state) if isinstance(t, _HasMember47_13) else (False, None),
    "state_dict": lambda t: (True, t.state_dict) if isinstance(t, _HasMember47_14) else (False, None),
    "stateless_random_uniform": lambda t: (
        (True, t.stateless_random_uniform) if isinstance(t, _HasMember47_15) else (False, None)
    ),
    "stateless_split": lambda t: (True, t.stateless_split) if isinstance(t, _HasMember47_16) else (False, None),
    "stats": lambda t: (True, t.stats) if isinstance(t, _HasMember47_17) else (False, None),
    "std": lambda t: (True, t.std) if isinstance(t, _HasMember47_18) else (False, None),
    "std_func": lambda t: (True, t.std_func) if isinstance(t, _HasMember47_19) else (False, None),
    "stddev": lambda t: (True, t.stddev) if isinstance(t, _HasMember47_20) else (False, None),
    "stderr": lambda t: (True, t.stderr) if isinstance(t, _HasMember47_21) else (False, None),
    "stdout": lambda t: (True, t.stdout) if isinstance(t, _HasMember47_22) else (False, None),
    "step": lambda t: (True, t.step) if isinstance(t, _HasMember47_23) else (False, None),
    "stft": lambda t: (True, t.stft) if isinstance(t, _HasMember47_24) else (False, None),
    "stop_gradient": lambda t: (True, t.stop_gradient) if isinstance(t, _HasMember47_25) else (False, None),
    "str": lambda t: (True, t.str) if isinstance(t, _HasMember47_26) else (False, None),
    "strided_slice": lambda t: (True, t.strided_slice) if isinstance(t, _HasMember47_27) else (False, None),
    "strides": lambda t: (True, t.strides) if isinstance(t, _HasMember47_28) else (False, None),
    "string_join": lambda t: (True, t.string_join) if isinstance(t, _HasMember47_29) else (False, None),
    "string_length": lambda t: (True, t.string_length) if isinstance(t, _HasMember47_30) else (False, None),
    "string_lower": lambda t: (True, t.string_lower) if isinstance(t, _HasMember47_31) else (False, None),
    "string_split": lambda t: (True, t.string_split) if isinstance(t, _HasMember47_32) else (False, None),
    "string_substr": lambda t: (True, t.string_substr) if isinstance(t, _HasMember47_33) else (False, None),
    "string_to_hash": lambda t: (True, t.string_to_hash) if isinstance(t, _HasMember47_34) else (False, None),
    "string_to_number": lambda t: (True, t.string_to_number) if isinstance(t, _HasMember47_35) else (False, None),
    "string_upper": lambda t: (True, t.string_upper) if isinstance(t, _HasMember47_36) else (False, None),
    "stringlower": lambda t: (True, t.stringlower) if isinstance(t, _HasMember47_37) else (False, None),
    "stringsplit": lambda t: (True, t.stringsplit) if isinstance(t, _HasMember47_38) else (False, None),
    "stringsubstr": lambda t: (True, t.stringsubstr) if isinstance(t, _HasMember47_39) else (False, None),
    "stringtohash": lambda t: (True, t.stringtohash) if isinstance(t, _HasMember47_40) else (False, None),
    "stringtonumber": lambda t: (True, t.stringtonumber) if isinstance(t, _HasMember47_41) else (False, None),
    "stringupper": lambda t: (True, t.stringupper) if isinstance(t, _HasMember47_42) else (False, None),
    "struct_field": lambda t: (True, t.struct_field) if isinstance(t, _HasMember47_43) else (False, None),
    "sub": lambda t: (True, t.sub) if isinstance(t, _HasMember47_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember47_00,
        _HasMember47_01,
        _HasMember47_02,
        _HasMember47_03,
        _HasMember47_04,
        _HasMember47_05,
        _HasMember47_06,
        _HasMember47_07,
        _HasMember47_08,
        _HasMember47_09,
        _HasMember47_10,
        _HasMember47_11,
        _HasMember47_12,
        _HasMember47_13,
        _HasMember47_14,
        _HasMember47_15,
        _HasMember47_16,
        _HasMember47_17,
        _HasMember47_18,
        _HasMember47_19,
        _HasMember47_20,
        _HasMember47_21,
        _HasMember47_22,
        _HasMember47_23,
        _HasMember47_24,
        _HasMember47_25,
        _HasMember47_26,
        _HasMember47_27,
        _HasMember47_28,
        _HasMember47_29,
        _HasMember47_30,
        _HasMember47_31,
        _HasMember47_32,
        _HasMember47_33,
        _HasMember47_34,
        _HasMember47_35,
        _HasMember47_36,
        _HasMember47_37,
        _HasMember47_38,
        _HasMember47_39,
        _HasMember47_40,
        _HasMember47_41,
        _HasMember47_42,
        _HasMember47_43,
        _HasMember47_44,
    )
)
