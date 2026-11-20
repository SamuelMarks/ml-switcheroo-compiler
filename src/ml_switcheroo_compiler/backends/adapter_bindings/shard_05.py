"""Static structural protocol bindings shard 05 (Solarize..WelchConfig)."""

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
class _HasMember05_00(typing.Protocol):
    """Structural protocol for objects exposing `Solarize`."""

    Solarize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_01(typing.Protocol):
    """Structural protocol for objects exposing `SparseConcat`."""

    SparseConcat: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_02(typing.Protocol):
    """Structural protocol for objects exposing `SparseDenseMatMul`."""

    SparseDenseMatMul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_03(typing.Protocol):
    """Structural protocol for objects exposing `SparseGenerator`."""

    SparseGenerator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_04(typing.Protocol):
    """Structural protocol for objects exposing `SparseMapValues`."""

    SparseMapValues: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_05(typing.Protocol):
    """Structural protocol for objects exposing `SparseOperandProtocol`."""

    SparseOperandProtocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_06(typing.Protocol):
    """Structural protocol for objects exposing `SparsePlus`."""

    SparsePlus: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_07(typing.Protocol):
    """Structural protocol for objects exposing `SparseProfiler`."""

    SparseProfiler: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_08(typing.Protocol):
    """Structural protocol for objects exposing `SparseReduceMax`."""

    SparseReduceMax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_09(typing.Protocol):
    """Structural protocol for objects exposing `SparseReshape`."""

    SparseReshape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_10(typing.Protocol):
    """Structural protocol for objects exposing `SparseSampledAdd`."""

    SparseSampledAdd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_11(typing.Protocol):
    """Structural protocol for objects exposing `SparseSegmentSum`."""

    SparseSegmentSum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_12(typing.Protocol):
    """Structural protocol for objects exposing `SparseSigmoid`."""

    SparseSigmoid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_13(typing.Protocol):
    """Structural protocol for objects exposing `SparseSplit`."""

    SparseSplit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_14(typing.Protocol):
    """Structural protocol for objects exposing `SparseToDense`."""

    SparseToDense: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_15(typing.Protocol):
    """Structural protocol for objects exposing `SparseTranspose`."""

    SparseTranspose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_16(typing.Protocol):
    """Structural protocol for objects exposing `Sqrtm`."""

    Sqrtm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_17(typing.Protocol):
    """Structural protocol for objects exposing `Squeeze`."""

    Squeeze: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_18(typing.Protocol):
    """Structural protocol for objects exposing `StableHLOCodeGenerator`."""

    StableHLOCodeGenerator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_19(typing.Protocol):
    """Structural protocol for objects exposing `Stft`."""

    Stft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_20(typing.Protocol):
    """Structural protocol for objects exposing `Stream`."""

    Stream: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_21(typing.Protocol):
    """Structural protocol for objects exposing `Swapaxes`."""

    Swapaxes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_22(typing.Protocol):
    """Structural protocol for objects exposing `SwitchOp`."""

    SwitchOp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_23(typing.Protocol):
    """Structural protocol for objects exposing `Table`."""

    Table: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_24(typing.Protocol):
    """Structural protocol for objects exposing `Tensor`."""

    Tensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_25(typing.Protocol):
    """Structural protocol for objects exposing `TensorArrayOperandProtocol`."""

    TensorArrayOperandProtocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_26(typing.Protocol):
    """Structural protocol for objects exposing `TensorArrayRead`."""

    TensorArrayRead: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_27(typing.Protocol):
    """Structural protocol for objects exposing `TensorArrayStack`."""

    TensorArrayStack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_28(typing.Protocol):
    """Structural protocol for objects exposing `TensorArrayWrite`."""

    TensorArrayWrite: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_29(typing.Protocol):
    """Structural protocol for objects exposing `TensorFlowProfiler`."""

    TensorFlowProfiler: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_30(typing.Protocol):
    """Structural protocol for objects exposing `TensorOperandProtocol`."""

    TensorOperandProtocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_31(typing.Protocol):
    """Structural protocol for objects exposing `Trace`."""

    Trace: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_32(typing.Protocol):
    """Structural protocol for objects exposing `TransformerEncoderLayer`."""

    TransformerEncoderLayer: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_33(typing.Protocol):
    """Structural protocol for objects exposing `Transpose`."""

    Transpose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_34(typing.Protocol):
    """Structural protocol for objects exposing `VALID_KINDS`."""

    VALID_KINDS: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_35(typing.Protocol):
    """Structural protocol for objects exposing `VALID_OPS`."""

    VALID_OPS: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_36(typing.Protocol):
    """Structural protocol for objects exposing `Vecdot`."""

    Vecdot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_37(typing.Protocol):
    """Structural protocol for objects exposing `Vsplit`."""

    Vsplit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_38(typing.Protocol):
    """Structural protocol for objects exposing `Vstack`."""

    Vstack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_39(typing.Protocol):
    """Structural protocol for objects exposing `WasmCodeGenerator`."""

    WasmCodeGenerator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_40(typing.Protocol):
    """Structural protocol for objects exposing `WebGLCodeGenerator`."""

    WebGLCodeGenerator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_41(typing.Protocol):
    """Structural protocol for objects exposing `WebGPUCodeGenerator`."""

    WebGPUCodeGenerator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_42(typing.Protocol):
    """Structural protocol for objects exposing `WebRTCDataChannelStream`."""

    WebRTCDataChannelStream: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_43(typing.Protocol):
    """Structural protocol for objects exposing `Welch`."""

    Welch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember05_44(typing.Protocol):
    """Structural protocol for objects exposing `WelchConfig`."""

    WelchConfig: EagerTensorProtocol


SHARD_05_READERS: dict[str, MemberReader] = {
    "Solarize": lambda t: (True, t.Solarize) if isinstance(t, _HasMember05_00) else (False, None),
    "SparseConcat": lambda t: (True, t.SparseConcat) if isinstance(t, _HasMember05_01) else (False, None),
    "SparseDenseMatMul": lambda t: (True, t.SparseDenseMatMul) if isinstance(t, _HasMember05_02) else (False, None),
    "SparseGenerator": lambda t: (True, t.SparseGenerator) if isinstance(t, _HasMember05_03) else (False, None),
    "SparseMapValues": lambda t: (True, t.SparseMapValues) if isinstance(t, _HasMember05_04) else (False, None),
    "SparseOperandProtocol": lambda t: (
        (True, t.SparseOperandProtocol) if isinstance(t, _HasMember05_05) else (False, None)
    ),
    "SparsePlus": lambda t: (True, t.SparsePlus) if isinstance(t, _HasMember05_06) else (False, None),
    "SparseProfiler": lambda t: (True, t.SparseProfiler) if isinstance(t, _HasMember05_07) else (False, None),
    "SparseReduceMax": lambda t: (True, t.SparseReduceMax) if isinstance(t, _HasMember05_08) else (False, None),
    "SparseReshape": lambda t: (True, t.SparseReshape) if isinstance(t, _HasMember05_09) else (False, None),
    "SparseSampledAdd": lambda t: (True, t.SparseSampledAdd) if isinstance(t, _HasMember05_10) else (False, None),
    "SparseSegmentSum": lambda t: (True, t.SparseSegmentSum) if isinstance(t, _HasMember05_11) else (False, None),
    "SparseSigmoid": lambda t: (True, t.SparseSigmoid) if isinstance(t, _HasMember05_12) else (False, None),
    "SparseSplit": lambda t: (True, t.SparseSplit) if isinstance(t, _HasMember05_13) else (False, None),
    "SparseToDense": lambda t: (True, t.SparseToDense) if isinstance(t, _HasMember05_14) else (False, None),
    "SparseTranspose": lambda t: (True, t.SparseTranspose) if isinstance(t, _HasMember05_15) else (False, None),
    "Sqrtm": lambda t: (True, t.Sqrtm) if isinstance(t, _HasMember05_16) else (False, None),
    "Squeeze": lambda t: (True, t.Squeeze) if isinstance(t, _HasMember05_17) else (False, None),
    "StableHLOCodeGenerator": lambda t: (
        (True, t.StableHLOCodeGenerator) if isinstance(t, _HasMember05_18) else (False, None)
    ),
    "Stft": lambda t: (True, t.Stft) if isinstance(t, _HasMember05_19) else (False, None),
    "Stream": lambda t: (True, t.Stream) if isinstance(t, _HasMember05_20) else (False, None),
    "Swapaxes": lambda t: (True, t.Swapaxes) if isinstance(t, _HasMember05_21) else (False, None),
    "SwitchOp": lambda t: (True, t.SwitchOp) if isinstance(t, _HasMember05_22) else (False, None),
    "Table": lambda t: (True, t.Table) if isinstance(t, _HasMember05_23) else (False, None),
    "Tensor": lambda t: (True, t.Tensor) if isinstance(t, _HasMember05_24) else (False, None),
    "TensorArrayOperandProtocol": lambda t: (
        (True, t.TensorArrayOperandProtocol) if isinstance(t, _HasMember05_25) else (False, None)
    ),
    "TensorArrayRead": lambda t: (True, t.TensorArrayRead) if isinstance(t, _HasMember05_26) else (False, None),
    "TensorArrayStack": lambda t: (True, t.TensorArrayStack) if isinstance(t, _HasMember05_27) else (False, None),
    "TensorArrayWrite": lambda t: (True, t.TensorArrayWrite) if isinstance(t, _HasMember05_28) else (False, None),
    "TensorFlowProfiler": lambda t: (True, t.TensorFlowProfiler) if isinstance(t, _HasMember05_29) else (False, None),
    "TensorOperandProtocol": lambda t: (
        (True, t.TensorOperandProtocol) if isinstance(t, _HasMember05_30) else (False, None)
    ),
    "Trace": lambda t: (True, t.Trace) if isinstance(t, _HasMember05_31) else (False, None),
    "TransformerEncoderLayer": lambda t: (
        (True, t.TransformerEncoderLayer) if isinstance(t, _HasMember05_32) else (False, None)
    ),
    "Transpose": lambda t: (True, t.Transpose) if isinstance(t, _HasMember05_33) else (False, None),
    "VALID_KINDS": lambda t: (True, t.VALID_KINDS) if isinstance(t, _HasMember05_34) else (False, None),
    "VALID_OPS": lambda t: (True, t.VALID_OPS) if isinstance(t, _HasMember05_35) else (False, None),
    "Vecdot": lambda t: (True, t.Vecdot) if isinstance(t, _HasMember05_36) else (False, None),
    "Vsplit": lambda t: (True, t.Vsplit) if isinstance(t, _HasMember05_37) else (False, None),
    "Vstack": lambda t: (True, t.Vstack) if isinstance(t, _HasMember05_38) else (False, None),
    "WasmCodeGenerator": lambda t: (True, t.WasmCodeGenerator) if isinstance(t, _HasMember05_39) else (False, None),
    "WebGLCodeGenerator": lambda t: (True, t.WebGLCodeGenerator) if isinstance(t, _HasMember05_40) else (False, None),
    "WebGPUCodeGenerator": lambda t: (True, t.WebGPUCodeGenerator) if isinstance(t, _HasMember05_41) else (False, None),
    "WebRTCDataChannelStream": lambda t: (
        (True, t.WebRTCDataChannelStream) if isinstance(t, _HasMember05_42) else (False, None)
    ),
    "Welch": lambda t: (True, t.Welch) if isinstance(t, _HasMember05_43) else (False, None),
    "WelchConfig": lambda t: (True, t.WelchConfig) if isinstance(t, _HasMember05_44) else (False, None),
}
