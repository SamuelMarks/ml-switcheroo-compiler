"""Static structural protocol bindings shard 04 (READ_VARIABLE_OP..SobolSample)."""

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
class _HasMember04_00(typing.Protocol):
    """Structural protocol for objects exposing `READ_VARIABLE_OP`."""

    READ_VARIABLE_OP: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_01(typing.Protocol):
    """Structural protocol for objects exposing `RNNConfig`."""

    RNNConfig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_02(typing.Protocol):
    """Structural protocol for objects exposing `RaggedAdd`."""

    RaggedAdd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_03(typing.Protocol):
    """Structural protocol for objects exposing `RaggedConstant`."""

    RaggedConstant: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_04(typing.Protocol):
    """Structural protocol for objects exposing `RaggedCrossHashed`."""

    RaggedCrossHashed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_05(typing.Protocol):
    """Structural protocol for objects exposing `RaggedDot`."""

    RaggedDot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_06(typing.Protocol):
    """Structural protocol for objects exposing `RaggedDynamicBroadcast`."""

    RaggedDynamicBroadcast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_07(typing.Protocol):
    """Structural protocol for objects exposing `RaggedGather`."""

    RaggedGather: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_08(typing.Protocol):
    """Structural protocol for objects exposing `RaggedMatMul`."""

    RaggedMatMul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_09(typing.Protocol):
    """Structural protocol for objects exposing `RaggedOperandProtocol`."""

    RaggedOperandProtocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_10(typing.Protocol):
    """Structural protocol for objects exposing `RaggedRange`."""

    RaggedRange: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_11(typing.Protocol):
    """Structural protocol for objects exposing `RaggedRowSplitsToSegmentIds`."""

    RaggedRowSplitsToSegmentIds: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_12(typing.Protocol):
    """Structural protocol for objects exposing `RaggedSegmentIdsToRowSplits`."""

    RaggedSegmentIdsToRowSplits: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_13(typing.Protocol):
    """Structural protocol for objects exposing `RaggedStack`."""

    RaggedStack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_14(typing.Protocol):
    """Structural protocol for objects exposing `RaggedStackDynamicPartitions`."""

    RaggedStackDynamicPartitions: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_15(typing.Protocol):
    """Structural protocol for objects exposing `RaggedTensorToDense`."""

    RaggedTensorToDense: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_16(typing.Protocol):
    """Structural protocol for objects exposing `Random`."""

    Random: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_17(typing.Protocol):
    """Structural protocol for objects exposing `RawConv2D`."""

    RawConv2D: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_18(typing.Protocol):
    """Structural protocol for objects exposing `RawMatMul`."""

    RawMatMul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_19(typing.Protocol):
    """Structural protocol for objects exposing `RawMerge`."""

    RawMerge: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_20(typing.Protocol):
    """Structural protocol for objects exposing `RawOp`."""

    RawOp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_21(typing.Protocol):
    """Structural protocol for objects exposing `RawSwitch`."""

    RawSwitch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_22(typing.Protocol):
    """Structural protocol for objects exposing `ReadFile`."""

    ReadFile: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_23(typing.Protocol):
    """Structural protocol for objects exposing `ReadVariable`."""

    ReadVariable: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_24(typing.Protocol):
    """Structural protocol for objects exposing `RecordBatch`."""

    RecordBatch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_25(typing.Protocol):
    """Structural protocol for objects exposing `Resize`."""

    Resize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_26(typing.Protocol):
    """Structural protocol for objects exposing `Rfft`."""

    Rfft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_27(typing.Protocol):
    """Structural protocol for objects exposing `Rfft2`."""

    Rfft2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_28(typing.Protocol):
    """Structural protocol for objects exposing `Rfftfreq`."""

    Rfftfreq: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_29(typing.Protocol):
    """Structural protocol for objects exposing `Rfftn`."""

    Rfftn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_30(typing.Protocol):
    """Structural protocol for objects exposing `Rfftnd`."""

    Rfftnd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_31(typing.Protocol):
    """Structural protocol for objects exposing `RgbToGrayscale`."""

    RgbToGrayscale: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_32(typing.Protocol):
    """Structural protocol for objects exposing `Roll`."""

    Roll: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_33(typing.Protocol):
    """Structural protocol for objects exposing `Save`."""

    Save: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_34(typing.Protocol):
    """Structural protocol for objects exposing `SaveGguf`."""

    SaveGguf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_35(typing.Protocol):
    """Structural protocol for objects exposing `Savez`."""

    Savez: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_36(typing.Protocol):
    """Structural protocol for objects exposing `SavezCompressed`."""

    SavezCompressed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_37(typing.Protocol):
    """Structural protocol for objects exposing `ScanOp`."""

    ScanOp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_38(typing.Protocol):
    """Structural protocol for objects exposing `SerializeTensor`."""

    SerializeTensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_39(typing.Protocol):
    """Structural protocol for objects exposing `SerializeToString`."""

    SerializeToString: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_40(typing.Protocol):
    """Structural protocol for objects exposing `ShapeMetadataProtocol`."""

    ShapeMetadataProtocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_41(typing.Protocol):
    """Structural protocol for objects exposing `ShapeProtocol`."""

    ShapeProtocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_42(typing.Protocol):
    """Structural protocol for objects exposing `SignalingHandler`."""

    SignalingHandler: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_43(typing.Protocol):
    """Structural protocol for objects exposing `SignalingServer`."""

    SignalingServer: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember04_44(typing.Protocol):
    """Structural protocol for objects exposing `SobolSample`."""

    SobolSample: EagerTensorProtocol


SHARD_04_READERS: dict[str, MemberReader] = {
    "READ_VARIABLE_OP": lambda t: (True, t.READ_VARIABLE_OP) if isinstance(t, _HasMember04_00) else (False, None),
    "RNNConfig": lambda t: (True, t.RNNConfig) if isinstance(t, _HasMember04_01) else (False, None),
    "RaggedAdd": lambda t: (True, t.RaggedAdd) if isinstance(t, _HasMember04_02) else (False, None),
    "RaggedConstant": lambda t: (True, t.RaggedConstant) if isinstance(t, _HasMember04_03) else (False, None),
    "RaggedCrossHashed": lambda t: (True, t.RaggedCrossHashed) if isinstance(t, _HasMember04_04) else (False, None),
    "RaggedDot": lambda t: (True, t.RaggedDot) if isinstance(t, _HasMember04_05) else (False, None),
    "RaggedDynamicBroadcast": lambda t: (
        (True, t.RaggedDynamicBroadcast) if isinstance(t, _HasMember04_06) else (False, None)
    ),
    "RaggedGather": lambda t: (True, t.RaggedGather) if isinstance(t, _HasMember04_07) else (False, None),
    "RaggedMatMul": lambda t: (True, t.RaggedMatMul) if isinstance(t, _HasMember04_08) else (False, None),
    "RaggedOperandProtocol": lambda t: (
        (True, t.RaggedOperandProtocol) if isinstance(t, _HasMember04_09) else (False, None)
    ),
    "RaggedRange": lambda t: (True, t.RaggedRange) if isinstance(t, _HasMember04_10) else (False, None),
    "RaggedRowSplitsToSegmentIds": lambda t: (
        (True, t.RaggedRowSplitsToSegmentIds) if isinstance(t, _HasMember04_11) else (False, None)
    ),
    "RaggedSegmentIdsToRowSplits": lambda t: (
        (True, t.RaggedSegmentIdsToRowSplits) if isinstance(t, _HasMember04_12) else (False, None)
    ),
    "RaggedStack": lambda t: (True, t.RaggedStack) if isinstance(t, _HasMember04_13) else (False, None),
    "RaggedStackDynamicPartitions": lambda t: (
        (True, t.RaggedStackDynamicPartitions) if isinstance(t, _HasMember04_14) else (False, None)
    ),
    "RaggedTensorToDense": lambda t: (True, t.RaggedTensorToDense) if isinstance(t, _HasMember04_15) else (False, None),
    "Random": lambda t: (True, t.Random) if isinstance(t, _HasMember04_16) else (False, None),
    "RawConv2D": lambda t: (True, t.RawConv2D) if isinstance(t, _HasMember04_17) else (False, None),
    "RawMatMul": lambda t: (True, t.RawMatMul) if isinstance(t, _HasMember04_18) else (False, None),
    "RawMerge": lambda t: (True, t.RawMerge) if isinstance(t, _HasMember04_19) else (False, None),
    "RawOp": lambda t: (True, t.RawOp) if isinstance(t, _HasMember04_20) else (False, None),
    "RawSwitch": lambda t: (True, t.RawSwitch) if isinstance(t, _HasMember04_21) else (False, None),
    "ReadFile": lambda t: (True, t.ReadFile) if isinstance(t, _HasMember04_22) else (False, None),
    "ReadVariable": lambda t: (True, t.ReadVariable) if isinstance(t, _HasMember04_23) else (False, None),
    "RecordBatch": lambda t: (True, t.RecordBatch) if isinstance(t, _HasMember04_24) else (False, None),
    "Resize": lambda t: (True, t.Resize) if isinstance(t, _HasMember04_25) else (False, None),
    "Rfft": lambda t: (True, t.Rfft) if isinstance(t, _HasMember04_26) else (False, None),
    "Rfft2": lambda t: (True, t.Rfft2) if isinstance(t, _HasMember04_27) else (False, None),
    "Rfftfreq": lambda t: (True, t.Rfftfreq) if isinstance(t, _HasMember04_28) else (False, None),
    "Rfftn": lambda t: (True, t.Rfftn) if isinstance(t, _HasMember04_29) else (False, None),
    "Rfftnd": lambda t: (True, t.Rfftnd) if isinstance(t, _HasMember04_30) else (False, None),
    "RgbToGrayscale": lambda t: (True, t.RgbToGrayscale) if isinstance(t, _HasMember04_31) else (False, None),
    "Roll": lambda t: (True, t.Roll) if isinstance(t, _HasMember04_32) else (False, None),
    "Save": lambda t: (True, t.Save) if isinstance(t, _HasMember04_33) else (False, None),
    "SaveGguf": lambda t: (True, t.SaveGguf) if isinstance(t, _HasMember04_34) else (False, None),
    "Savez": lambda t: (True, t.Savez) if isinstance(t, _HasMember04_35) else (False, None),
    "SavezCompressed": lambda t: (True, t.SavezCompressed) if isinstance(t, _HasMember04_36) else (False, None),
    "ScanOp": lambda t: (True, t.ScanOp) if isinstance(t, _HasMember04_37) else (False, None),
    "SerializeTensor": lambda t: (True, t.SerializeTensor) if isinstance(t, _HasMember04_38) else (False, None),
    "SerializeToString": lambda t: (True, t.SerializeToString) if isinstance(t, _HasMember04_39) else (False, None),
    "ShapeMetadataProtocol": lambda t: (
        (True, t.ShapeMetadataProtocol) if isinstance(t, _HasMember04_40) else (False, None)
    ),
    "ShapeProtocol": lambda t: (True, t.ShapeProtocol) if isinstance(t, _HasMember04_41) else (False, None),
    "SignalingHandler": lambda t: (True, t.SignalingHandler) if isinstance(t, _HasMember04_42) else (False, None),
    "SignalingServer": lambda t: (True, t.SignalingServer) if isinstance(t, _HasMember04_43) else (False, None),
    "SobolSample": lambda t: (True, t.SobolSample) if isinstance(t, _HasMember04_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember04_00,
        _HasMember04_01,
        _HasMember04_02,
        _HasMember04_03,
        _HasMember04_04,
        _HasMember04_05,
        _HasMember04_06,
        _HasMember04_07,
        _HasMember04_08,
        _HasMember04_09,
        _HasMember04_10,
        _HasMember04_11,
        _HasMember04_12,
        _HasMember04_13,
        _HasMember04_14,
        _HasMember04_15,
        _HasMember04_16,
        _HasMember04_17,
        _HasMember04_18,
        _HasMember04_19,
        _HasMember04_20,
        _HasMember04_21,
        _HasMember04_22,
        _HasMember04_23,
        _HasMember04_24,
        _HasMember04_25,
        _HasMember04_26,
        _HasMember04_27,
        _HasMember04_28,
        _HasMember04_29,
        _HasMember04_30,
        _HasMember04_31,
        _HasMember04_32,
        _HasMember04_33,
        _HasMember04_34,
        _HasMember04_35,
        _HasMember04_36,
        _HasMember04_37,
        _HasMember04_38,
        _HasMember04_39,
        _HasMember04_40,
        _HasMember04_41,
        _HasMember04_42,
        _HasMember04_43,
        _HasMember04_44,
    )
)
