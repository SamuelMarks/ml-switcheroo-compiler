"""Static structural protocol bindings shard 42 (requires_grad..run_all_to_all)."""

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
class _HasMember42_00(typing.Protocol):
    """Structural protocol for objects exposing `requires_grad`."""

    requires_grad: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_01(typing.Protocol):
    """Structural protocol for objects exposing `reset_global_tracing_state`."""

    reset_global_tracing_state: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_02(typing.Protocol):
    """Structural protocol for objects exposing `reset_peak_memory`."""

    reset_peak_memory: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_03(typing.Protocol):
    """Structural protocol for objects exposing `reset_peak_memory_stats`."""

    reset_peak_memory_stats: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_04(typing.Protocol):
    """Structural protocol for objects exposing `reset_wasm_templates`."""

    reset_wasm_templates: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_05(typing.Protocol):
    """Structural protocol for objects exposing `reshape`."""

    reshape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_06(typing.Protocol):
    """Structural protocol for objects exposing `resize`."""

    resize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_07(typing.Protocol):
    """Structural protocol for objects exposing `resize_bicubic`."""

    resize_bicubic: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_08(typing.Protocol):
    """Structural protocol for objects exposing `resize_bilinear`."""

    resize_bilinear: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_09(typing.Protocol):
    """Structural protocol for objects exposing `resize_lanczos3`."""

    resize_lanczos3: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_10(typing.Protocol):
    """Structural protocol for objects exposing `resize_nearest`."""

    resize_nearest: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_11(typing.Protocol):
    """Structural protocol for objects exposing `result_type`."""

    result_type: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_12(typing.Protocol):
    """Structural protocol for objects exposing `returncode`."""

    returncode: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_13(typing.Protocol):
    """Structural protocol for objects exposing `rfft`."""

    rfft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_14(typing.Protocol):
    """Structural protocol for objects exposing `rfft2`."""

    rfft2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_15(typing.Protocol):
    """Structural protocol for objects exposing `rfftfreq`."""

    rfftfreq: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_16(typing.Protocol):
    """Structural protocol for objects exposing `rfftn`."""

    rfftn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_17(typing.Protocol):
    """Structural protocol for objects exposing `rfftnd`."""

    rfftnd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_18(typing.Protocol):
    """Structural protocol for objects exposing `rgb_to_grayscale`."""

    rgb_to_grayscale: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_19(typing.Protocol):
    """Structural protocol for objects exposing `rgb_to_hsv`."""

    rgb_to_hsv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_20(typing.Protocol):
    """Structural protocol for objects exposing `rgb_to_yiq`."""

    rgb_to_yiq: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_21(typing.Protocol):
    """Structural protocol for objects exposing `rgb_to_yuv`."""

    rgb_to_yuv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_22(typing.Protocol):
    """Structural protocol for objects exposing `rgbtograyscale`."""

    rgbtograyscale: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_23(typing.Protocol):
    """Structural protocol for objects exposing `rhs_spec`."""

    rhs_spec: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_24(typing.Protocol):
    """Structural protocol for objects exposing `right`."""

    right: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_25(typing.Protocol):
    """Structural protocol for objects exposing `right_shift`."""

    right_shift: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_26(typing.Protocol):
    """Structural protocol for objects exposing `rightshift`."""

    rightshift: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_27(typing.Protocol):
    """Structural protocol for objects exposing `rint`."""

    rint: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_28(typing.Protocol):
    """Structural protocol for objects exposing `rnguniform`."""

    rnguniform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_29(typing.Protocol):
    """Structural protocol for objects exposing `rnn`."""

    rnn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_30(typing.Protocol):
    """Structural protocol for objects exposing `rnn_step`."""

    rnn_step: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_31(typing.Protocol):
    """Structural protocol for objects exposing `rnnconfig`."""

    rnnconfig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_32(typing.Protocol):
    """Structural protocol for objects exposing `roll`."""

    roll: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_33(typing.Protocol):
    """Structural protocol for objects exposing `rot90`."""

    rot90: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_34(typing.Protocol):
    """Structural protocol for objects exposing `round`."""

    round: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_35(typing.Protocol):
    """Structural protocol for objects exposing `row_splits`."""

    row_splits: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_36(typing.Protocol):
    """Structural protocol for objects exposing `rsqrt`."""

    rsqrt: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_37(typing.Protocol):
    """Structural protocol for objects exposing `rule_packed`."""

    rule_packed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_38(typing.Protocol):
    """Structural protocol for objects exposing `rule_single`."""

    rule_single: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_39(typing.Protocol):
    """Structural protocol for objects exposing `rule_unpacked`."""

    rule_unpacked: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_40(typing.Protocol):
    """Structural protocol for objects exposing `run`."""

    run: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_41(typing.Protocol):
    """Structural protocol for objects exposing `run_adagrad`."""

    run_adagrad: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_42(typing.Protocol):
    """Structural protocol for objects exposing `run_adam`."""

    run_adam: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_43(typing.Protocol):
    """Structural protocol for objects exposing `run_all_reduce`."""

    run_all_reduce: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember42_44(typing.Protocol):
    """Structural protocol for objects exposing `run_all_to_all`."""

    run_all_to_all: EagerTensorProtocol


SHARD_42_READERS: dict[str, MemberReader] = {
    "requires_grad": lambda t: (True, t.requires_grad) if isinstance(t, _HasMember42_00) else (False, None),
    "reset_global_tracing_state": lambda t: (
        (True, t.reset_global_tracing_state) if isinstance(t, _HasMember42_01) else (False, None)
    ),
    "reset_peak_memory": lambda t: (True, t.reset_peak_memory) if isinstance(t, _HasMember42_02) else (False, None),
    "reset_peak_memory_stats": lambda t: (
        (True, t.reset_peak_memory_stats) if isinstance(t, _HasMember42_03) else (False, None)
    ),
    "reset_wasm_templates": lambda t: (
        (True, t.reset_wasm_templates) if isinstance(t, _HasMember42_04) else (False, None)
    ),
    "reshape": lambda t: (True, t.reshape) if isinstance(t, _HasMember42_05) else (False, None),
    "resize": lambda t: (True, t.resize) if isinstance(t, _HasMember42_06) else (False, None),
    "resize_bicubic": lambda t: (True, t.resize_bicubic) if isinstance(t, _HasMember42_07) else (False, None),
    "resize_bilinear": lambda t: (True, t.resize_bilinear) if isinstance(t, _HasMember42_08) else (False, None),
    "resize_lanczos3": lambda t: (True, t.resize_lanczos3) if isinstance(t, _HasMember42_09) else (False, None),
    "resize_nearest": lambda t: (True, t.resize_nearest) if isinstance(t, _HasMember42_10) else (False, None),
    "result_type": lambda t: (True, t.result_type) if isinstance(t, _HasMember42_11) else (False, None),
    "returncode": lambda t: (True, t.returncode) if isinstance(t, _HasMember42_12) else (False, None),
    "rfft": lambda t: (True, t.rfft) if isinstance(t, _HasMember42_13) else (False, None),
    "rfft2": lambda t: (True, t.rfft2) if isinstance(t, _HasMember42_14) else (False, None),
    "rfftfreq": lambda t: (True, t.rfftfreq) if isinstance(t, _HasMember42_15) else (False, None),
    "rfftn": lambda t: (True, t.rfftn) if isinstance(t, _HasMember42_16) else (False, None),
    "rfftnd": lambda t: (True, t.rfftnd) if isinstance(t, _HasMember42_17) else (False, None),
    "rgb_to_grayscale": lambda t: (True, t.rgb_to_grayscale) if isinstance(t, _HasMember42_18) else (False, None),
    "rgb_to_hsv": lambda t: (True, t.rgb_to_hsv) if isinstance(t, _HasMember42_19) else (False, None),
    "rgb_to_yiq": lambda t: (True, t.rgb_to_yiq) if isinstance(t, _HasMember42_20) else (False, None),
    "rgb_to_yuv": lambda t: (True, t.rgb_to_yuv) if isinstance(t, _HasMember42_21) else (False, None),
    "rgbtograyscale": lambda t: (True, t.rgbtograyscale) if isinstance(t, _HasMember42_22) else (False, None),
    "rhs_spec": lambda t: (True, t.rhs_spec) if isinstance(t, _HasMember42_23) else (False, None),
    "right": lambda t: (True, t.right) if isinstance(t, _HasMember42_24) else (False, None),
    "right_shift": lambda t: (True, t.right_shift) if isinstance(t, _HasMember42_25) else (False, None),
    "rightshift": lambda t: (True, t.rightshift) if isinstance(t, _HasMember42_26) else (False, None),
    "rint": lambda t: (True, t.rint) if isinstance(t, _HasMember42_27) else (False, None),
    "rnguniform": lambda t: (True, t.rnguniform) if isinstance(t, _HasMember42_28) else (False, None),
    "rnn": lambda t: (True, t.rnn) if isinstance(t, _HasMember42_29) else (False, None),
    "rnn_step": lambda t: (True, t.rnn_step) if isinstance(t, _HasMember42_30) else (False, None),
    "rnnconfig": lambda t: (True, t.rnnconfig) if isinstance(t, _HasMember42_31) else (False, None),
    "roll": lambda t: (True, t.roll) if isinstance(t, _HasMember42_32) else (False, None),
    "rot90": lambda t: (True, t.rot90) if isinstance(t, _HasMember42_33) else (False, None),
    "round": lambda t: (True, t.round) if isinstance(t, _HasMember42_34) else (False, None),
    "row_splits": lambda t: (True, t.row_splits) if isinstance(t, _HasMember42_35) else (False, None),
    "rsqrt": lambda t: (True, t.rsqrt) if isinstance(t, _HasMember42_36) else (False, None),
    "rule_packed": lambda t: (True, t.rule_packed) if isinstance(t, _HasMember42_37) else (False, None),
    "rule_single": lambda t: (True, t.rule_single) if isinstance(t, _HasMember42_38) else (False, None),
    "rule_unpacked": lambda t: (True, t.rule_unpacked) if isinstance(t, _HasMember42_39) else (False, None),
    "run": lambda t: (True, t.run) if isinstance(t, _HasMember42_40) else (False, None),
    "run_adagrad": lambda t: (True, t.run_adagrad) if isinstance(t, _HasMember42_41) else (False, None),
    "run_adam": lambda t: (True, t.run_adam) if isinstance(t, _HasMember42_42) else (False, None),
    "run_all_reduce": lambda t: (True, t.run_all_reduce) if isinstance(t, _HasMember42_43) else (False, None),
    "run_all_to_all": lambda t: (True, t.run_all_to_all) if isinstance(t, _HasMember42_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember42_00,
        _HasMember42_01,
        _HasMember42_02,
        _HasMember42_03,
        _HasMember42_04,
        _HasMember42_05,
        _HasMember42_06,
        _HasMember42_07,
        _HasMember42_08,
        _HasMember42_09,
        _HasMember42_10,
        _HasMember42_11,
        _HasMember42_12,
        _HasMember42_13,
        _HasMember42_14,
        _HasMember42_15,
        _HasMember42_16,
        _HasMember42_17,
        _HasMember42_18,
        _HasMember42_19,
        _HasMember42_20,
        _HasMember42_21,
        _HasMember42_22,
        _HasMember42_23,
        _HasMember42_24,
        _HasMember42_25,
        _HasMember42_26,
        _HasMember42_27,
        _HasMember42_28,
        _HasMember42_29,
        _HasMember42_30,
        _HasMember42_31,
        _HasMember42_32,
        _HasMember42_33,
        _HasMember42_34,
        _HasMember42_35,
        _HasMember42_36,
        _HasMember42_37,
        _HasMember42_38,
        _HasMember42_39,
        _HasMember42_40,
        _HasMember42_41,
        _HasMember42_42,
        _HasMember42_43,
        _HasMember42_44,
    )
)
