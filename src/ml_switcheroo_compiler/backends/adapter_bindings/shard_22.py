"""Static structural protocol bindings shard 22 (flatnonzero..fractionalavgpool)."""

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
class _HasMember22_00(typing.Protocol):
    """Structural protocol for objects exposing `flatnonzero`."""

    flatnonzero: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_01(typing.Protocol):
    """Structural protocol for objects exposing `flatten`."""

    flatten: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_02(typing.Protocol):
    """Structural protocol for objects exposing `flip`."""

    flip: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_03(typing.Protocol):
    """Structural protocol for objects exposing `flip_left_right`."""

    flip_left_right: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_04(typing.Protocol):
    """Structural protocol for objects exposing `flip_up_down`."""

    flip_up_down: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_05(typing.Protocol):
    """Structural protocol for objects exposing `fliplr`."""

    fliplr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_06(typing.Protocol):
    """Structural protocol for objects exposing `flipud`."""

    flipud: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_07(typing.Protocol):
    """Structural protocol for objects exposing `float16`."""

    float16: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_08(typing.Protocol):
    """Structural protocol for objects exposing `float32`."""

    float32: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_09(typing.Protocol):
    """Structural protocol for objects exposing `float64`."""

    float64: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_10(typing.Protocol):
    """Structural protocol for objects exposing `float8_e4m3fn`."""

    float8_e4m3fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_11(typing.Protocol):
    """Structural protocol for objects exposing `float8_e5m2`."""

    float8_e5m2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_12(typing.Protocol):
    """Structural protocol for objects exposing `float_power`."""

    float_power: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_13(typing.Protocol):
    """Structural protocol for objects exposing `floor`."""

    floor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_14(typing.Protocol):
    """Structural protocol for objects exposing `floor_divide`."""

    floor_divide: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_15(typing.Protocol):
    """Structural protocol for objects exposing `floordiv`."""

    floordiv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_16(typing.Protocol):
    """Structural protocol for objects exposing `floordivide`."""

    floordivide: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_17(typing.Protocol):
    """Structural protocol for objects exposing `flush`."""

    flush: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_18(typing.Protocol):
    """Structural protocol for objects exposing `fmax`."""

    fmax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_19(typing.Protocol):
    """Structural protocol for objects exposing `fmin`."""

    fmin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_20(typing.Protocol):
    """Structural protocol for objects exposing `fmod`."""

    fmod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_21(typing.Protocol):
    """Structural protocol for objects exposing `fn`."""

    fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_22(typing.Protocol):
    """Structural protocol for objects exposing `fn_aux`."""

    fn_aux: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_23(typing.Protocol):
    """Structural protocol for objects exposing `fn_hints`."""

    fn_hints: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_24(typing.Protocol):
    """Structural protocol for objects exposing `fn_jvp_hasattr_graph`."""

    fn_jvp_hasattr_graph: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_25(typing.Protocol):
    """Structural protocol for objects exposing `fn_jvp_rule_graph`."""

    fn_jvp_rule_graph: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_26(typing.Protocol):
    """Structural protocol for objects exposing `fn_jvp_sub`."""

    fn_jvp_sub: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_27(typing.Protocol):
    """Structural protocol for objects exposing `fn_no_ret`."""

    fn_no_ret: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_28(typing.Protocol):
    """Structural protocol for objects exposing `fn_str_ret`."""

    fn_str_ret: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_29(typing.Protocol):
    """Structural protocol for objects exposing `fn_vjp`."""

    fn_vjp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_30(typing.Protocol):
    """Structural protocol for objects exposing `fn_vjp_graph`."""

    fn_vjp_graph: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_31(typing.Protocol):
    """Structural protocol for objects exposing `fn_vjp_hasattr`."""

    fn_vjp_hasattr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_32(typing.Protocol):
    """Structural protocol for objects exposing `fn_vjp_nofwd`."""

    fn_vjp_nofwd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_33(typing.Protocol):
    """Structural protocol for objects exposing `fn_vjp_nosub`."""

    fn_vjp_nosub: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_34(typing.Protocol):
    """Structural protocol for objects exposing `fn_with_aux`."""

    fn_with_aux: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_35(typing.Protocol):
    """Structural protocol for objects exposing `fn_with_aux_jvp`."""

    fn_with_aux_jvp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_36(typing.Protocol):
    """Structural protocol for objects exposing `foo`."""

    foo: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_37(typing.Protocol):
    """Structural protocol for objects exposing `foreign_call`."""

    foreign_call: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_38(typing.Protocol):
    """Structural protocol for objects exposing `foreigncall`."""

    foreigncall: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_39(typing.Protocol):
    """Structural protocol for objects exposing `fori_loop`."""

    fori_loop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_40(typing.Protocol):
    """Structural protocol for objects exposing `formatter`."""

    formatter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_41(typing.Protocol):
    """Structural protocol for objects exposing `forward`."""

    forward: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_42(typing.Protocol):
    """Structural protocol for objects exposing `fractional_avg_pool`."""

    fractional_avg_pool: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_43(typing.Protocol):
    """Structural protocol for objects exposing `fractional_max_pool`."""

    fractional_max_pool: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember22_44(typing.Protocol):
    """Structural protocol for objects exposing `fractionalavgpool`."""

    fractionalavgpool: EagerTensorProtocol


SHARD_22_READERS: dict[str, MemberReader] = {
    "flatnonzero": lambda t: (True, t.flatnonzero) if isinstance(t, _HasMember22_00) else (False, None),
    "flatten": lambda t: (True, t.flatten) if isinstance(t, _HasMember22_01) else (False, None),
    "flip": lambda t: (True, t.flip) if isinstance(t, _HasMember22_02) else (False, None),
    "flip_left_right": lambda t: (True, t.flip_left_right) if isinstance(t, _HasMember22_03) else (False, None),
    "flip_up_down": lambda t: (True, t.flip_up_down) if isinstance(t, _HasMember22_04) else (False, None),
    "fliplr": lambda t: (True, t.fliplr) if isinstance(t, _HasMember22_05) else (False, None),
    "flipud": lambda t: (True, t.flipud) if isinstance(t, _HasMember22_06) else (False, None),
    "float16": lambda t: (True, t.float16) if isinstance(t, _HasMember22_07) else (False, None),
    "float32": lambda t: (True, t.float32) if isinstance(t, _HasMember22_08) else (False, None),
    "float64": lambda t: (True, t.float64) if isinstance(t, _HasMember22_09) else (False, None),
    "float8_e4m3fn": lambda t: (True, t.float8_e4m3fn) if isinstance(t, _HasMember22_10) else (False, None),
    "float8_e5m2": lambda t: (True, t.float8_e5m2) if isinstance(t, _HasMember22_11) else (False, None),
    "float_power": lambda t: (True, t.float_power) if isinstance(t, _HasMember22_12) else (False, None),
    "floor": lambda t: (True, t.floor) if isinstance(t, _HasMember22_13) else (False, None),
    "floor_divide": lambda t: (True, t.floor_divide) if isinstance(t, _HasMember22_14) else (False, None),
    "floordiv": lambda t: (True, t.floordiv) if isinstance(t, _HasMember22_15) else (False, None),
    "floordivide": lambda t: (True, t.floordivide) if isinstance(t, _HasMember22_16) else (False, None),
    "flush": lambda t: (True, t.flush) if isinstance(t, _HasMember22_17) else (False, None),
    "fmax": lambda t: (True, t.fmax) if isinstance(t, _HasMember22_18) else (False, None),
    "fmin": lambda t: (True, t.fmin) if isinstance(t, _HasMember22_19) else (False, None),
    "fmod": lambda t: (True, t.fmod) if isinstance(t, _HasMember22_20) else (False, None),
    "fn": lambda t: (True, t.fn) if isinstance(t, _HasMember22_21) else (False, None),
    "fn_aux": lambda t: (True, t.fn_aux) if isinstance(t, _HasMember22_22) else (False, None),
    "fn_hints": lambda t: (True, t.fn_hints) if isinstance(t, _HasMember22_23) else (False, None),
    "fn_jvp_hasattr_graph": lambda t: (
        (True, t.fn_jvp_hasattr_graph) if isinstance(t, _HasMember22_24) else (False, None)
    ),
    "fn_jvp_rule_graph": lambda t: (True, t.fn_jvp_rule_graph) if isinstance(t, _HasMember22_25) else (False, None),
    "fn_jvp_sub": lambda t: (True, t.fn_jvp_sub) if isinstance(t, _HasMember22_26) else (False, None),
    "fn_no_ret": lambda t: (True, t.fn_no_ret) if isinstance(t, _HasMember22_27) else (False, None),
    "fn_str_ret": lambda t: (True, t.fn_str_ret) if isinstance(t, _HasMember22_28) else (False, None),
    "fn_vjp": lambda t: (True, t.fn_vjp) if isinstance(t, _HasMember22_29) else (False, None),
    "fn_vjp_graph": lambda t: (True, t.fn_vjp_graph) if isinstance(t, _HasMember22_30) else (False, None),
    "fn_vjp_hasattr": lambda t: (True, t.fn_vjp_hasattr) if isinstance(t, _HasMember22_31) else (False, None),
    "fn_vjp_nofwd": lambda t: (True, t.fn_vjp_nofwd) if isinstance(t, _HasMember22_32) else (False, None),
    "fn_vjp_nosub": lambda t: (True, t.fn_vjp_nosub) if isinstance(t, _HasMember22_33) else (False, None),
    "fn_with_aux": lambda t: (True, t.fn_with_aux) if isinstance(t, _HasMember22_34) else (False, None),
    "fn_with_aux_jvp": lambda t: (True, t.fn_with_aux_jvp) if isinstance(t, _HasMember22_35) else (False, None),
    "foo": lambda t: (True, t.foo) if isinstance(t, _HasMember22_36) else (False, None),
    "foreign_call": lambda t: (True, t.foreign_call) if isinstance(t, _HasMember22_37) else (False, None),
    "foreigncall": lambda t: (True, t.foreigncall) if isinstance(t, _HasMember22_38) else (False, None),
    "fori_loop": lambda t: (True, t.fori_loop) if isinstance(t, _HasMember22_39) else (False, None),
    "formatter": lambda t: (True, t.formatter) if isinstance(t, _HasMember22_40) else (False, None),
    "forward": lambda t: (True, t.forward) if isinstance(t, _HasMember22_41) else (False, None),
    "fractional_avg_pool": lambda t: (True, t.fractional_avg_pool) if isinstance(t, _HasMember22_42) else (False, None),
    "fractional_max_pool": lambda t: (True, t.fractional_max_pool) if isinstance(t, _HasMember22_43) else (False, None),
    "fractionalavgpool": lambda t: (True, t.fractionalavgpool) if isinstance(t, _HasMember22_44) else (False, None),
}
