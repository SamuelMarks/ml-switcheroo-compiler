"""Static structural protocol bindings shard 27 (init..isclose)."""

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
class _HasMember27_00(typing.Protocol):
    """Structural protocol for objects exposing `init`."""

    init: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_01(typing.Protocol):
    """Structural protocol for objects exposing `init_ctx`."""

    init_ctx: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_02(typing.Protocol):
    """Structural protocol for objects exposing `init_method`."""

    init_method: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_03(typing.Protocol):
    """Structural protocol for objects exposing `init_val`."""

    init_val: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_04(typing.Protocol):
    """Structural protocol for objects exposing `initialize`."""

    initialize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_05(typing.Protocol):
    """Structural protocol for objects exposing `initialize_distributed`."""

    initialize_distributed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_06(typing.Protocol):
    """Structural protocol for objects exposing `inner`."""

    inner: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_07(typing.Protocol):
    """Structural protocol for objects exposing `input_specs`."""

    input_specs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_08(typing.Protocol):
    """Structural protocol for objects exposing `inputs`."""

    inputs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_09(typing.Protocol):
    """Structural protocol for objects exposing `inputs_nelem`."""

    inputs_nelem: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_10(typing.Protocol):
    """Structural protocol for objects exposing `insert`."""

    insert: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_11(typing.Protocol):
    """Structural protocol for objects exposing `install_profiler_backend_stubs`."""

    install_profiler_backend_stubs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_12(typing.Protocol):
    """Structural protocol for objects exposing `int16`."""

    int16: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_13(typing.Protocol):
    """Structural protocol for objects exposing `int32`."""

    int32: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_14(typing.Protocol):
    """Structural protocol for objects exposing `int64`."""

    int64: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_15(typing.Protocol):
    """Structural protocol for objects exposing `int8`."""

    int8: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_16(typing.Protocol):
    """Structural protocol for objects exposing `integer`."""

    integer: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_17(typing.Protocol):
    """Structural protocol for objects exposing `interp`."""

    interp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_18(typing.Protocol):
    """Structural protocol for objects exposing `interpolation`."""

    interpolation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_19(typing.Protocol):
    """Structural protocol for objects exposing `intersect1d`."""

    intersect1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_20(typing.Protocol):
    """Structural protocol for objects exposing `intopk`."""

    intopk: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_21(typing.Protocol):
    """Structural protocol for objects exposing `inv`."""

    inv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_22(typing.Protocol):
    """Structural protocol for objects exposing `invert`."""

    invert: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_23(typing.Protocol):
    """Structural protocol for objects exposing `invert_permutation`."""

    invert_permutation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_24(typing.Protocol):
    """Structural protocol for objects exposing `io`."""

    io: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_25(typing.Protocol):
    """Structural protocol for objects exposing `iou`."""

    iou: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_26(typing.Protocol):
    """Structural protocol for objects exposing `irfft`."""

    irfft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_27(typing.Protocol):
    """Structural protocol for objects exposing `irfft2`."""

    irfft2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_28(typing.Protocol):
    """Structural protocol for objects exposing `irfftn`."""

    irfftn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_29(typing.Protocol):
    """Structural protocol for objects exposing `irfftnd`."""

    irfftnd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_30(typing.Protocol):
    """Structural protocol for objects exposing `is_alive`."""

    is_alive: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_31(typing.Protocol):
    """Structural protocol for objects exposing `is_available`."""

    is_available: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_32(typing.Protocol):
    """Structural protocol for objects exposing `is_dynamic_shape`."""

    is_dynamic_shape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_33(typing.Protocol):
    """Structural protocol for objects exposing `is_finite`."""

    is_finite: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_34(typing.Protocol):
    """Structural protocol for objects exposing `is_fusion_valid`."""

    is_fusion_valid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_35(typing.Protocol):
    """Structural protocol for objects exposing `is_hip`."""

    is_hip: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_36(typing.Protocol):
    """Structural protocol for objects exposing `is_inf`."""

    is_inf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_37(typing.Protocol):
    """Structural protocol for objects exposing `is_method`."""

    is_method: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_38(typing.Protocol):
    """Structural protocol for objects exposing `is_nan`."""

    is_nan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_39(typing.Protocol):
    """Structural protocol for objects exposing `is_nested`."""

    is_nested: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_40(typing.Protocol):
    """Structural protocol for objects exposing `is_non_decreasing`."""

    is_non_decreasing: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_41(typing.Protocol):
    """Structural protocol for objects exposing `is_null`."""

    is_null: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_42(typing.Protocol):
    """Structural protocol for objects exposing `is_strictly_increasing`."""

    is_strictly_increasing: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_43(typing.Protocol):
    """Structural protocol for objects exposing `is_valid`."""

    is_valid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember27_44(typing.Protocol):
    """Structural protocol for objects exposing `isclose`."""

    isclose: EagerTensorProtocol


SHARD_27_READERS: dict[str, MemberReader] = {
    "init": lambda t: (True, t.init) if isinstance(t, _HasMember27_00) else (False, None),
    "init_ctx": lambda t: (True, t.init_ctx) if isinstance(t, _HasMember27_01) else (False, None),
    "init_method": lambda t: (True, t.init_method) if isinstance(t, _HasMember27_02) else (False, None),
    "init_val": lambda t: (True, t.init_val) if isinstance(t, _HasMember27_03) else (False, None),
    "initialize": lambda t: (True, t.initialize) if isinstance(t, _HasMember27_04) else (False, None),
    "initialize_distributed": lambda t: (
        (True, t.initialize_distributed) if isinstance(t, _HasMember27_05) else (False, None)
    ),
    "inner": lambda t: (True, t.inner) if isinstance(t, _HasMember27_06) else (False, None),
    "input_specs": lambda t: (True, t.input_specs) if isinstance(t, _HasMember27_07) else (False, None),
    "inputs": lambda t: (True, t.inputs) if isinstance(t, _HasMember27_08) else (False, None),
    "inputs_nelem": lambda t: (True, t.inputs_nelem) if isinstance(t, _HasMember27_09) else (False, None),
    "insert": lambda t: (True, t.insert) if isinstance(t, _HasMember27_10) else (False, None),
    "install_profiler_backend_stubs": lambda t: (
        (True, t.install_profiler_backend_stubs) if isinstance(t, _HasMember27_11) else (False, None)
    ),
    "int16": lambda t: (True, t.int16) if isinstance(t, _HasMember27_12) else (False, None),
    "int32": lambda t: (True, t.int32) if isinstance(t, _HasMember27_13) else (False, None),
    "int64": lambda t: (True, t.int64) if isinstance(t, _HasMember27_14) else (False, None),
    "int8": lambda t: (True, t.int8) if isinstance(t, _HasMember27_15) else (False, None),
    "integer": lambda t: (True, t.integer) if isinstance(t, _HasMember27_16) else (False, None),
    "interp": lambda t: (True, t.interp) if isinstance(t, _HasMember27_17) else (False, None),
    "interpolation": lambda t: (True, t.interpolation) if isinstance(t, _HasMember27_18) else (False, None),
    "intersect1d": lambda t: (True, t.intersect1d) if isinstance(t, _HasMember27_19) else (False, None),
    "intopk": lambda t: (True, t.intopk) if isinstance(t, _HasMember27_20) else (False, None),
    "inv": lambda t: (True, t.inv) if isinstance(t, _HasMember27_21) else (False, None),
    "invert": lambda t: (True, t.invert) if isinstance(t, _HasMember27_22) else (False, None),
    "invert_permutation": lambda t: (True, t.invert_permutation) if isinstance(t, _HasMember27_23) else (False, None),
    "io": lambda t: (True, t.io) if isinstance(t, _HasMember27_24) else (False, None),
    "iou": lambda t: (True, t.iou) if isinstance(t, _HasMember27_25) else (False, None),
    "irfft": lambda t: (True, t.irfft) if isinstance(t, _HasMember27_26) else (False, None),
    "irfft2": lambda t: (True, t.irfft2) if isinstance(t, _HasMember27_27) else (False, None),
    "irfftn": lambda t: (True, t.irfftn) if isinstance(t, _HasMember27_28) else (False, None),
    "irfftnd": lambda t: (True, t.irfftnd) if isinstance(t, _HasMember27_29) else (False, None),
    "is_alive": lambda t: (True, t.is_alive) if isinstance(t, _HasMember27_30) else (False, None),
    "is_available": lambda t: (True, t.is_available) if isinstance(t, _HasMember27_31) else (False, None),
    "is_dynamic_shape": lambda t: (True, t.is_dynamic_shape) if isinstance(t, _HasMember27_32) else (False, None),
    "is_finite": lambda t: (True, t.is_finite) if isinstance(t, _HasMember27_33) else (False, None),
    "is_fusion_valid": lambda t: (True, t.is_fusion_valid) if isinstance(t, _HasMember27_34) else (False, None),
    "is_hip": lambda t: (True, t.is_hip) if isinstance(t, _HasMember27_35) else (False, None),
    "is_inf": lambda t: (True, t.is_inf) if isinstance(t, _HasMember27_36) else (False, None),
    "is_method": lambda t: (True, t.is_method) if isinstance(t, _HasMember27_37) else (False, None),
    "is_nan": lambda t: (True, t.is_nan) if isinstance(t, _HasMember27_38) else (False, None),
    "is_nested": lambda t: (True, t.is_nested) if isinstance(t, _HasMember27_39) else (False, None),
    "is_non_decreasing": lambda t: (True, t.is_non_decreasing) if isinstance(t, _HasMember27_40) else (False, None),
    "is_null": lambda t: (True, t.is_null) if isinstance(t, _HasMember27_41) else (False, None),
    "is_strictly_increasing": lambda t: (
        (True, t.is_strictly_increasing) if isinstance(t, _HasMember27_42) else (False, None)
    ),
    "is_valid": lambda t: (True, t.is_valid) if isinstance(t, _HasMember27_43) else (False, None),
    "isclose": lambda t: (True, t.isclose) if isinstance(t, _HasMember27_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember27_00,
        _HasMember27_01,
        _HasMember27_02,
        _HasMember27_03,
        _HasMember27_04,
        _HasMember27_05,
        _HasMember27_06,
        _HasMember27_07,
        _HasMember27_08,
        _HasMember27_09,
        _HasMember27_10,
        _HasMember27_11,
        _HasMember27_12,
        _HasMember27_13,
        _HasMember27_14,
        _HasMember27_15,
        _HasMember27_16,
        _HasMember27_17,
        _HasMember27_18,
        _HasMember27_19,
        _HasMember27_20,
        _HasMember27_21,
        _HasMember27_22,
        _HasMember27_23,
        _HasMember27_24,
        _HasMember27_25,
        _HasMember27_26,
        _HasMember27_27,
        _HasMember27_28,
        _HasMember27_29,
        _HasMember27_30,
        _HasMember27_31,
        _HasMember27_32,
        _HasMember27_33,
        _HasMember27_34,
        _HasMember27_35,
        _HasMember27_36,
        _HasMember27_37,
        _HasMember27_38,
        _HasMember27_39,
        _HasMember27_40,
        _HasMember27_41,
        _HasMember27_42,
        _HasMember27_43,
        _HasMember27_44,
    )
)
