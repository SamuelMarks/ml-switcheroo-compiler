"""Static structural protocol bindings shard 55 (visit_MultiHeadAttention..visit_SegmentedMm)."""

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
class _HasMember55_00(typing.Protocol):
    """Structural protocol for objects exposing `visit_MultiHeadAttention`."""

    visit_MultiHeadAttention: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_01(typing.Protocol):
    """Structural protocol for objects exposing `visit_NanToNum`."""

    visit_NanToNum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_02(typing.Protocol):
    """Structural protocol for objects exposing `visit_Neg`."""

    visit_Neg: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_03(typing.Protocol):
    """Structural protocol for objects exposing `visit_NonMaxSuppression`."""

    visit_NonMaxSuppression: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_04(typing.Protocol):
    """Structural protocol for objects exposing `visit_Num`."""

    visit_Num: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_05(typing.Protocol):
    """Structural protocol for objects exposing `visit_Ones`."""

    visit_Ones: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_06(typing.Protocol):
    """Structural protocol for objects exposing `visit_OpNode`."""

    visit_OpNode: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_07(typing.Protocol):
    """Structural protocol for objects exposing `visit_OtherOp`."""

    visit_OtherOp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_08(typing.Protocol):
    """Structural protocol for objects exposing `visit_Pad`."""

    visit_Pad: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_09(typing.Protocol):
    """Structural protocol for objects exposing `visit_PadNone`."""

    visit_PadNone: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_10(typing.Protocol):
    """Structural protocol for objects exposing `visit_Parameter`."""

    visit_Parameter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_11(typing.Protocol):
    """Structural protocol for objects exposing `visit_Partition`."""

    visit_Partition: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_12(typing.Protocol):
    """Structural protocol for objects exposing `visit_PerspectiveTransform`."""

    visit_PerspectiveTransform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_13(typing.Protocol):
    """Structural protocol for objects exposing `visit_Pmap`."""

    visit_Pmap: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_14(typing.Protocol):
    """Structural protocol for objects exposing `visit_Polar`."""

    visit_Polar: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_15(typing.Protocol):
    """Structural protocol for objects exposing `visit_PowerIteration`."""

    visit_PowerIteration: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_16(typing.Protocol):
    """Structural protocol for objects exposing `visit_Prod`."""

    visit_Prod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_17(typing.Protocol):
    """Structural protocol for objects exposing `visit_PutAlongAxis`."""

    visit_PutAlongAxis: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_18(typing.Protocol):
    """Structural protocol for objects exposing `visit_QrVjp`."""

    visit_QrVjp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_19(typing.Protocol):
    """Structural protocol for objects exposing `visit_Quantize`."""

    visit_Quantize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_20(typing.Protocol):
    """Structural protocol for objects exposing `visit_QuantizedMatmul`."""

    visit_QuantizedMatmul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_21(typing.Protocol):
    """Structural protocol for objects exposing `visit_RMSNorm`."""

    visit_RMSNorm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_22(typing.Protocol):
    """Structural protocol for objects exposing `visit_RaggedDot`."""

    visit_RaggedDot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_23(typing.Protocol):
    """Structural protocol for objects exposing `visit_Raise`."""

    visit_Raise: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_24(typing.Protocol):
    """Structural protocol for objects exposing `visit_RandomCategorical`."""

    visit_RandomCategorical: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_25(typing.Protocol):
    """Structural protocol for objects exposing `visit_Recv`."""

    visit_Recv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_26(typing.Protocol):
    """Structural protocol for objects exposing `visit_ReduceScatter`."""

    visit_ReduceScatter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_27(typing.Protocol):
    """Structural protocol for objects exposing `visit_ReduceSum`."""

    visit_ReduceSum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_28(typing.Protocol):
    """Structural protocol for objects exposing `visit_Relu`."""

    visit_Relu: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_29(typing.Protocol):
    """Structural protocol for objects exposing `visit_Repeat`."""

    visit_Repeat: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_30(typing.Protocol):
    """Structural protocol for objects exposing `visit_ResizeBicubic`."""

    visit_ResizeBicubic: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_31(typing.Protocol):
    """Structural protocol for objects exposing `visit_ResizeLanczos3`."""

    visit_ResizeLanczos3: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_32(typing.Protocol):
    """Structural protocol for objects exposing `visit_Return`."""

    visit_Return: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_33(typing.Protocol):
    """Structural protocol for objects exposing `visit_Roll`."""

    visit_Roll: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_34(typing.Protocol):
    """Structural protocol for objects exposing `visit_Rope`."""

    visit_Rope: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_35(typing.Protocol):
    """Structural protocol for objects exposing `visit_Round`."""

    visit_Round: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_36(typing.Protocol):
    """Structural protocol for objects exposing `visit_ScaledDotProductAttention`."""

    visit_ScaledDotProductAttention: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_37(typing.Protocol):
    """Structural protocol for objects exposing `visit_Scan`."""

    visit_Scan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_38(typing.Protocol):
    """Structural protocol for objects exposing `visit_ScatterND`."""

    visit_ScatterND: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_39(typing.Protocol):
    """Structural protocol for objects exposing `visit_Schur`."""

    visit_Schur: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_40(typing.Protocol):
    """Structural protocol for objects exposing `visit_SegmentMax`."""

    visit_SegmentMax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_41(typing.Protocol):
    """Structural protocol for objects exposing `visit_SegmentMin`."""

    visit_SegmentMin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_42(typing.Protocol):
    """Structural protocol for objects exposing `visit_SegmentProd`."""

    visit_SegmentProd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_43(typing.Protocol):
    """Structural protocol for objects exposing `visit_SegmentSum`."""

    visit_SegmentSum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember55_44(typing.Protocol):
    """Structural protocol for objects exposing `visit_SegmentedMm`."""

    visit_SegmentedMm: EagerTensorProtocol


SHARD_55_READERS: dict[str, MemberReader] = {
    "visit_MultiHeadAttention": lambda t: (
        (True, t.visit_MultiHeadAttention) if isinstance(t, _HasMember55_00) else (False, None)
    ),
    "visit_NanToNum": lambda t: (True, t.visit_NanToNum) if isinstance(t, _HasMember55_01) else (False, None),
    "visit_Neg": lambda t: (True, t.visit_Neg) if isinstance(t, _HasMember55_02) else (False, None),
    "visit_NonMaxSuppression": lambda t: (
        (True, t.visit_NonMaxSuppression) if isinstance(t, _HasMember55_03) else (False, None)
    ),
    "visit_Num": lambda t: (True, t.visit_Num) if isinstance(t, _HasMember55_04) else (False, None),
    "visit_Ones": lambda t: (True, t.visit_Ones) if isinstance(t, _HasMember55_05) else (False, None),
    "visit_OpNode": lambda t: (True, t.visit_OpNode) if isinstance(t, _HasMember55_06) else (False, None),
    "visit_OtherOp": lambda t: (True, t.visit_OtherOp) if isinstance(t, _HasMember55_07) else (False, None),
    "visit_Pad": lambda t: (True, t.visit_Pad) if isinstance(t, _HasMember55_08) else (False, None),
    "visit_PadNone": lambda t: (True, t.visit_PadNone) if isinstance(t, _HasMember55_09) else (False, None),
    "visit_Parameter": lambda t: (True, t.visit_Parameter) if isinstance(t, _HasMember55_10) else (False, None),
    "visit_Partition": lambda t: (True, t.visit_Partition) if isinstance(t, _HasMember55_11) else (False, None),
    "visit_PerspectiveTransform": lambda t: (
        (True, t.visit_PerspectiveTransform) if isinstance(t, _HasMember55_12) else (False, None)
    ),
    "visit_Pmap": lambda t: (True, t.visit_Pmap) if isinstance(t, _HasMember55_13) else (False, None),
    "visit_Polar": lambda t: (True, t.visit_Polar) if isinstance(t, _HasMember55_14) else (False, None),
    "visit_PowerIteration": lambda t: (
        (True, t.visit_PowerIteration) if isinstance(t, _HasMember55_15) else (False, None)
    ),
    "visit_Prod": lambda t: (True, t.visit_Prod) if isinstance(t, _HasMember55_16) else (False, None),
    "visit_PutAlongAxis": lambda t: (True, t.visit_PutAlongAxis) if isinstance(t, _HasMember55_17) else (False, None),
    "visit_QrVjp": lambda t: (True, t.visit_QrVjp) if isinstance(t, _HasMember55_18) else (False, None),
    "visit_Quantize": lambda t: (True, t.visit_Quantize) if isinstance(t, _HasMember55_19) else (False, None),
    "visit_QuantizedMatmul": lambda t: (
        (True, t.visit_QuantizedMatmul) if isinstance(t, _HasMember55_20) else (False, None)
    ),
    "visit_RMSNorm": lambda t: (True, t.visit_RMSNorm) if isinstance(t, _HasMember55_21) else (False, None),
    "visit_RaggedDot": lambda t: (True, t.visit_RaggedDot) if isinstance(t, _HasMember55_22) else (False, None),
    "visit_Raise": lambda t: (True, t.visit_Raise) if isinstance(t, _HasMember55_23) else (False, None),
    "visit_RandomCategorical": lambda t: (
        (True, t.visit_RandomCategorical) if isinstance(t, _HasMember55_24) else (False, None)
    ),
    "visit_Recv": lambda t: (True, t.visit_Recv) if isinstance(t, _HasMember55_25) else (False, None),
    "visit_ReduceScatter": lambda t: (True, t.visit_ReduceScatter) if isinstance(t, _HasMember55_26) else (False, None),
    "visit_ReduceSum": lambda t: (True, t.visit_ReduceSum) if isinstance(t, _HasMember55_27) else (False, None),
    "visit_Relu": lambda t: (True, t.visit_Relu) if isinstance(t, _HasMember55_28) else (False, None),
    "visit_Repeat": lambda t: (True, t.visit_Repeat) if isinstance(t, _HasMember55_29) else (False, None),
    "visit_ResizeBicubic": lambda t: (True, t.visit_ResizeBicubic) if isinstance(t, _HasMember55_30) else (False, None),
    "visit_ResizeLanczos3": lambda t: (
        (True, t.visit_ResizeLanczos3) if isinstance(t, _HasMember55_31) else (False, None)
    ),
    "visit_Return": lambda t: (True, t.visit_Return) if isinstance(t, _HasMember55_32) else (False, None),
    "visit_Roll": lambda t: (True, t.visit_Roll) if isinstance(t, _HasMember55_33) else (False, None),
    "visit_Rope": lambda t: (True, t.visit_Rope) if isinstance(t, _HasMember55_34) else (False, None),
    "visit_Round": lambda t: (True, t.visit_Round) if isinstance(t, _HasMember55_35) else (False, None),
    "visit_ScaledDotProductAttention": lambda t: (
        (True, t.visit_ScaledDotProductAttention) if isinstance(t, _HasMember55_36) else (False, None)
    ),
    "visit_Scan": lambda t: (True, t.visit_Scan) if isinstance(t, _HasMember55_37) else (False, None),
    "visit_ScatterND": lambda t: (True, t.visit_ScatterND) if isinstance(t, _HasMember55_38) else (False, None),
    "visit_Schur": lambda t: (True, t.visit_Schur) if isinstance(t, _HasMember55_39) else (False, None),
    "visit_SegmentMax": lambda t: (True, t.visit_SegmentMax) if isinstance(t, _HasMember55_40) else (False, None),
    "visit_SegmentMin": lambda t: (True, t.visit_SegmentMin) if isinstance(t, _HasMember55_41) else (False, None),
    "visit_SegmentProd": lambda t: (True, t.visit_SegmentProd) if isinstance(t, _HasMember55_42) else (False, None),
    "visit_SegmentSum": lambda t: (True, t.visit_SegmentSum) if isinstance(t, _HasMember55_43) else (False, None),
    "visit_SegmentedMm": lambda t: (True, t.visit_SegmentedMm) if isinstance(t, _HasMember55_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember55_00,
        _HasMember55_01,
        _HasMember55_02,
        _HasMember55_03,
        _HasMember55_04,
        _HasMember55_05,
        _HasMember55_06,
        _HasMember55_07,
        _HasMember55_08,
        _HasMember55_09,
        _HasMember55_10,
        _HasMember55_11,
        _HasMember55_12,
        _HasMember55_13,
        _HasMember55_14,
        _HasMember55_15,
        _HasMember55_16,
        _HasMember55_17,
        _HasMember55_18,
        _HasMember55_19,
        _HasMember55_20,
        _HasMember55_21,
        _HasMember55_22,
        _HasMember55_23,
        _HasMember55_24,
        _HasMember55_25,
        _HasMember55_26,
        _HasMember55_27,
        _HasMember55_28,
        _HasMember55_29,
        _HasMember55_30,
        _HasMember55_31,
        _HasMember55_32,
        _HasMember55_33,
        _HasMember55_34,
        _HasMember55_35,
        _HasMember55_36,
        _HasMember55_37,
        _HasMember55_38,
        _HasMember55_39,
        _HasMember55_40,
        _HasMember55_41,
        _HasMember55_42,
        _HasMember55_43,
        _HasMember55_44,
    )
)
