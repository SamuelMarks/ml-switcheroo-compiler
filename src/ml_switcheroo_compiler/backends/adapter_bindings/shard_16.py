"""Static structural protocol bindings shard 16 (decode_jpeg..dispatch_infer_shape)."""

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
class _HasMember16_00(typing.Protocol):
    """Structural protocol for objects exposing `decode_jpeg`."""

    decode_jpeg: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_01(typing.Protocol):
    """Structural protocol for objects exposing `decode_png`."""

    decode_png: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_02(typing.Protocol):
    """Structural protocol for objects exposing `decodebase64`."""

    decodebase64: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_03(typing.Protocol):
    """Structural protocol for objects exposing `decodecsv`."""

    decodecsv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_04(typing.Protocol):
    """Structural protocol for objects exposing `decodeimage`."""

    decodeimage: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_05(typing.Protocol):
    """Structural protocol for objects exposing `default_backend`."""

    default_backend: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_06(typing.Protocol):
    """Structural protocol for objects exposing `default_kwargs`."""

    default_kwargs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_07(typing.Protocol):
    """Structural protocol for objects exposing `default_rng`."""

    default_rng: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_08(typing.Protocol):
    """Structural protocol for objects exposing `deg2rad`."""

    deg2rad: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_09(typing.Protocol):
    """Structural protocol for objects exposing `degeneration`."""

    degeneration: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_10(typing.Protocol):
    """Structural protocol for objects exposing `degrees`."""

    degrees: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_11(typing.Protocol):
    """Structural protocol for objects exposing `delete`."""

    delete: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_12(typing.Protocol):
    """Structural protocol for objects exposing `depthwise_conv`."""

    depthwise_conv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_13(typing.Protocol):
    """Structural protocol for objects exposing `depthwise_conv1d`."""

    depthwise_conv1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_14(typing.Protocol):
    """Structural protocol for objects exposing `depthwise_conv2d`."""

    depthwise_conv2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_15(typing.Protocol):
    """Structural protocol for objects exposing `descriptive`."""

    descriptive: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_16(typing.Protocol):
    """Structural protocol for objects exposing `descriptive2`."""

    descriptive2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_17(typing.Protocol):
    """Structural protocol for objects exposing `deserialize`."""

    deserialize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_18(typing.Protocol):
    """Structural protocol for objects exposing `det`."""

    det: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_19(typing.Protocol):
    """Structural protocol for objects exposing `detach`."""

    detach: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_20(typing.Protocol):
    """Structural protocol for objects exposing `device`."""

    device: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_21(typing.Protocol):
    """Structural protocol for objects exposing `device_holder_protocol`."""

    device_holder_protocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_22(typing.Protocol):
    """Structural protocol for objects exposing `deviceholderprotocol`."""

    deviceholderprotocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_23(typing.Protocol):
    """Structural protocol for objects exposing `devices`."""

    devices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_24(typing.Protocol):
    """Structural protocol for objects exposing `diag`."""

    diag: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_25(typing.Protocol):
    """Structural protocol for objects exposing `diag_indices`."""

    diag_indices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_26(typing.Protocol):
    """Structural protocol for objects exposing `diag_indices_from`."""

    diag_indices_from: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_27(typing.Protocol):
    """Structural protocol for objects exposing `diag_part`."""

    diag_part: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_28(typing.Protocol):
    """Structural protocol for objects exposing `diagflat`."""

    diagflat: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_29(typing.Protocol):
    """Structural protocol for objects exposing `diagindices`."""

    diagindices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_30(typing.Protocol):
    """Structural protocol for objects exposing `diagindicesfrom`."""

    diagindicesfrom: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_31(typing.Protocol):
    """Structural protocol for objects exposing `diagonal`."""

    diagonal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_32(typing.Protocol):
    """Structural protocol for objects exposing `dice_loss`."""

    dice_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_33(typing.Protocol):
    """Structural protocol for objects exposing `dictionary_encode`."""

    dictionary_encode: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_34(typing.Protocol):
    """Structural protocol for objects exposing `diff`."""

    diff: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_35(typing.Protocol):
    """Structural protocol for objects exposing `digamma`."""

    digamma: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_36(typing.Protocol):
    """Structural protocol for objects exposing `digitize`."""

    digitize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_37(typing.Protocol):
    """Structural protocol for objects exposing `direct_func`."""

    direct_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_38(typing.Protocol):
    """Structural protocol for objects exposing `direct_target`."""

    direct_target: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_39(typing.Protocol):
    """Structural protocol for objects exposing `dirichlet`."""

    dirichlet: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_40(typing.Protocol):
    """Structural protocol for objects exposing `disable_jit`."""

    disable_jit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_41(typing.Protocol):
    """Structural protocol for objects exposing `disabled_test_evaluator_handle_custom_ops`."""

    disabled_test_evaluator_handle_custom_ops: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_42(typing.Protocol):
    """Structural protocol for objects exposing `dispatchThreads_threadsPerThreadgroup_`."""

    dispatchThreads_threadsPerThreadgroup_: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_43(typing.Protocol):
    """Structural protocol for objects exposing `dispatch_eager`."""

    dispatch_eager: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember16_44(typing.Protocol):
    """Structural protocol for objects exposing `dispatch_infer_shape`."""

    dispatch_infer_shape: EagerTensorProtocol


SHARD_16_READERS: dict[str, MemberReader] = {
    "decode_jpeg": lambda t: (True, t.decode_jpeg) if isinstance(t, _HasMember16_00) else (False, None),
    "decode_png": lambda t: (True, t.decode_png) if isinstance(t, _HasMember16_01) else (False, None),
    "decodebase64": lambda t: (True, t.decodebase64) if isinstance(t, _HasMember16_02) else (False, None),
    "decodecsv": lambda t: (True, t.decodecsv) if isinstance(t, _HasMember16_03) else (False, None),
    "decodeimage": lambda t: (True, t.decodeimage) if isinstance(t, _HasMember16_04) else (False, None),
    "default_backend": lambda t: (True, t.default_backend) if isinstance(t, _HasMember16_05) else (False, None),
    "default_kwargs": lambda t: (True, t.default_kwargs) if isinstance(t, _HasMember16_06) else (False, None),
    "default_rng": lambda t: (True, t.default_rng) if isinstance(t, _HasMember16_07) else (False, None),
    "deg2rad": lambda t: (True, t.deg2rad) if isinstance(t, _HasMember16_08) else (False, None),
    "degeneration": lambda t: (True, t.degeneration) if isinstance(t, _HasMember16_09) else (False, None),
    "degrees": lambda t: (True, t.degrees) if isinstance(t, _HasMember16_10) else (False, None),
    "delete": lambda t: (True, t.delete) if isinstance(t, _HasMember16_11) else (False, None),
    "depthwise_conv": lambda t: (True, t.depthwise_conv) if isinstance(t, _HasMember16_12) else (False, None),
    "depthwise_conv1d": lambda t: (True, t.depthwise_conv1d) if isinstance(t, _HasMember16_13) else (False, None),
    "depthwise_conv2d": lambda t: (True, t.depthwise_conv2d) if isinstance(t, _HasMember16_14) else (False, None),
    "descriptive": lambda t: (True, t.descriptive) if isinstance(t, _HasMember16_15) else (False, None),
    "descriptive2": lambda t: (True, t.descriptive2) if isinstance(t, _HasMember16_16) else (False, None),
    "deserialize": lambda t: (True, t.deserialize) if isinstance(t, _HasMember16_17) else (False, None),
    "det": lambda t: (True, t.det) if isinstance(t, _HasMember16_18) else (False, None),
    "detach": lambda t: (True, t.detach) if isinstance(t, _HasMember16_19) else (False, None),
    "device": lambda t: (True, t.device) if isinstance(t, _HasMember16_20) else (False, None),
    "device_holder_protocol": lambda t: (
        (True, t.device_holder_protocol) if isinstance(t, _HasMember16_21) else (False, None)
    ),
    "deviceholderprotocol": lambda t: (
        (True, t.deviceholderprotocol) if isinstance(t, _HasMember16_22) else (False, None)
    ),
    "devices": lambda t: (True, t.devices) if isinstance(t, _HasMember16_23) else (False, None),
    "diag": lambda t: (True, t.diag) if isinstance(t, _HasMember16_24) else (False, None),
    "diag_indices": lambda t: (True, t.diag_indices) if isinstance(t, _HasMember16_25) else (False, None),
    "diag_indices_from": lambda t: (True, t.diag_indices_from) if isinstance(t, _HasMember16_26) else (False, None),
    "diag_part": lambda t: (True, t.diag_part) if isinstance(t, _HasMember16_27) else (False, None),
    "diagflat": lambda t: (True, t.diagflat) if isinstance(t, _HasMember16_28) else (False, None),
    "diagindices": lambda t: (True, t.diagindices) if isinstance(t, _HasMember16_29) else (False, None),
    "diagindicesfrom": lambda t: (True, t.diagindicesfrom) if isinstance(t, _HasMember16_30) else (False, None),
    "diagonal": lambda t: (True, t.diagonal) if isinstance(t, _HasMember16_31) else (False, None),
    "dice_loss": lambda t: (True, t.dice_loss) if isinstance(t, _HasMember16_32) else (False, None),
    "dictionary_encode": lambda t: (True, t.dictionary_encode) if isinstance(t, _HasMember16_33) else (False, None),
    "diff": lambda t: (True, t.diff) if isinstance(t, _HasMember16_34) else (False, None),
    "digamma": lambda t: (True, t.digamma) if isinstance(t, _HasMember16_35) else (False, None),
    "digitize": lambda t: (True, t.digitize) if isinstance(t, _HasMember16_36) else (False, None),
    "direct_func": lambda t: (True, t.direct_func) if isinstance(t, _HasMember16_37) else (False, None),
    "direct_target": lambda t: (True, t.direct_target) if isinstance(t, _HasMember16_38) else (False, None),
    "dirichlet": lambda t: (True, t.dirichlet) if isinstance(t, _HasMember16_39) else (False, None),
    "disable_jit": lambda t: (True, t.disable_jit) if isinstance(t, _HasMember16_40) else (False, None),
    "disabled_test_evaluator_handle_custom_ops": lambda t: (
        (True, t.disabled_test_evaluator_handle_custom_ops) if isinstance(t, _HasMember16_41) else (False, None)
    ),
    "dispatchThreads_threadsPerThreadgroup_": lambda t: (
        (True, t.dispatchThreads_threadsPerThreadgroup_) if isinstance(t, _HasMember16_42) else (False, None)
    ),
    "dispatch_eager": lambda t: (True, t.dispatch_eager) if isinstance(t, _HasMember16_43) else (False, None),
    "dispatch_infer_shape": lambda t: (
        (True, t.dispatch_infer_shape) if isinstance(t, _HasMember16_44) else (False, None)
    ),
}

register_dynamic_protocols(
    (
        _HasMember16_00,
        _HasMember16_01,
        _HasMember16_02,
        _HasMember16_03,
        _HasMember16_04,
        _HasMember16_05,
        _HasMember16_06,
        _HasMember16_07,
        _HasMember16_08,
        _HasMember16_09,
        _HasMember16_10,
        _HasMember16_11,
        _HasMember16_12,
        _HasMember16_13,
        _HasMember16_14,
        _HasMember16_15,
        _HasMember16_16,
        _HasMember16_17,
        _HasMember16_18,
        _HasMember16_19,
        _HasMember16_20,
        _HasMember16_21,
        _HasMember16_22,
        _HasMember16_23,
        _HasMember16_24,
        _HasMember16_25,
        _HasMember16_26,
        _HasMember16_27,
        _HasMember16_28,
        _HasMember16_29,
        _HasMember16_30,
        _HasMember16_31,
        _HasMember16_32,
        _HasMember16_33,
        _HasMember16_34,
        _HasMember16_35,
        _HasMember16_36,
        _HasMember16_37,
        _HasMember16_38,
        _HasMember16_39,
        _HasMember16_40,
        _HasMember16_41,
        _HasMember16_42,
        _HasMember16_43,
        _HasMember16_44,
    )
)
