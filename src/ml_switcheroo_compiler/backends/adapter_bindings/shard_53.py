"""Static structural protocol bindings shard 53 (visit_AutoContrast..visit_FillNone)."""

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
class _HasMember53_00(typing.Protocol):
    """Structural protocol for objects exposing `visit_AutoContrast`."""

    visit_AutoContrast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_01(typing.Protocol):
    """Structural protocol for objects exposing `visit_AvgPool`."""

    visit_AvgPool: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_02(typing.Protocol):
    """Structural protocol for objects exposing `visit_AvgPool2D`."""

    visit_AvgPool2D: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_03(typing.Protocol):
    """Structural protocol for objects exposing `visit_AvgPool2DDynamic`."""

    visit_AvgPool2DDynamic: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_04(typing.Protocol):
    """Structural protocol for objects exposing `visit_AvgPool3D`."""

    visit_AvgPool3D: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_05(typing.Protocol):
    """Structural protocol for objects exposing `visit_AxisIndex`."""

    visit_AxisIndex: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_06(typing.Protocol):
    """Structural protocol for objects exposing `visit_BandPart`."""

    visit_BandPart: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_07(typing.Protocol):
    """Structural protocol for objects exposing `visit_BandedTriangularSolve`."""

    visit_BandedTriangularSolve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_08(typing.Protocol):
    """Structural protocol for objects exposing `visit_BatchMatMul`."""

    visit_BatchMatMul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_09(typing.Protocol):
    """Structural protocol for objects exposing `visit_BatchNorm`."""

    visit_BatchNorm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_10(typing.Protocol):
    """Structural protocol for objects exposing `visit_BinaryOperation`."""

    visit_BinaryOperation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_11(typing.Protocol):
    """Structural protocol for objects exposing `visit_BlockMaskedMm`."""

    visit_BlockMaskedMm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_12(typing.Protocol):
    """Structural protocol for objects exposing `visit_Broadcast`."""

    visit_Broadcast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_13(typing.Protocol):
    """Structural protocol for objects exposing `visit_BroadcastArrays`."""

    visit_BroadcastArrays: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_14(typing.Protocol):
    """Structural protocol for objects exposing `visit_Call`."""

    visit_Call: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_15(typing.Protocol):
    """Structural protocol for objects exposing `visit_Cartesian`."""

    visit_Cartesian: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_16(typing.Protocol):
    """Structural protocol for objects exposing `visit_Ceil`."""

    visit_Ceil: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_17(typing.Protocol):
    """Structural protocol for objects exposing `visit_Cholesky`."""

    visit_Cholesky: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_18(typing.Protocol):
    """Structural protocol for objects exposing `visit_CholeskyVjp`."""

    visit_CholeskyVjp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_19(typing.Protocol):
    """Structural protocol for objects exposing `visit_ClassDef`."""

    visit_ClassDef: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_20(typing.Protocol):
    """Structural protocol for objects exposing `visit_Combinations`."""

    visit_Combinations: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_21(typing.Protocol):
    """Structural protocol for objects exposing `visit_Concatenate`."""

    visit_Concatenate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_22(typing.Protocol):
    """Structural protocol for objects exposing `visit_Cond`."""

    visit_Cond: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_23(typing.Protocol):
    """Structural protocol for objects exposing `visit_ConstantOfShape`."""

    visit_ConstantOfShape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_24(typing.Protocol):
    """Structural protocol for objects exposing `visit_Conv1D`."""

    visit_Conv1D: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_25(typing.Protocol):
    """Structural protocol for objects exposing `visit_Conv2D`."""

    visit_Conv2D: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_26(typing.Protocol):
    """Structural protocol for objects exposing `visit_Conv2DSharedHalo`."""

    visit_Conv2DSharedHalo: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_27(typing.Protocol):
    """Structural protocol for objects exposing `visit_Conv3D`."""

    visit_Conv3D: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_28(typing.Protocol):
    """Structural protocol for objects exposing `visit_ConvTranspose`."""

    visit_ConvTranspose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_29(typing.Protocol):
    """Structural protocol for objects exposing `visit_Cos`."""

    visit_Cos: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_30(typing.Protocol):
    """Structural protocol for objects exposing `visit_CscAdd`."""

    visit_CscAdd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_31(typing.Protocol):
    """Structural protocol for objects exposing `visit_CscSub`."""

    visit_CscSub: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_32(typing.Protocol):
    """Structural protocol for objects exposing `visit_CsrAdd`."""

    visit_CsrAdd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_33(typing.Protocol):
    """Structural protocol for objects exposing `visit_CsrSub`."""

    visit_CsrSub: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_34(typing.Protocol):
    """Structural protocol for objects exposing `visit_DenseSpmm`."""

    visit_DenseSpmm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_35(typing.Protocol):
    """Structural protocol for objects exposing `visit_DepthwiseConv2D`."""

    visit_DepthwiseConv2D: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_36(typing.Protocol):
    """Structural protocol for objects exposing `visit_Div`."""

    visit_Div: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_37(typing.Protocol):
    """Structural protocol for objects exposing `visit_DropNone`."""

    visit_DropNone: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_38(typing.Protocol):
    """Structural protocol for objects exposing `visit_DynamicSlice`."""

    visit_DynamicSlice: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_39(typing.Protocol):
    """Structural protocol for objects exposing `visit_DynamicUpdateSlice`."""

    visit_DynamicUpdateSlice: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_40(typing.Protocol):
    """Structural protocol for objects exposing `visit_Einsum`."""

    visit_Einsum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_41(typing.Protocol):
    """Structural protocol for objects exposing `visit_ElasticTransform`."""

    visit_ElasticTransform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_42(typing.Protocol):
    """Structural protocol for objects exposing `visit_Exp`."""

    visit_Exp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_43(typing.Protocol):
    """Structural protocol for objects exposing `visit_ExtractBoundingBoxes`."""

    visit_ExtractBoundingBoxes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember53_44(typing.Protocol):
    """Structural protocol for objects exposing `visit_FillNone`."""

    visit_FillNone: EagerTensorProtocol


SHARD_53_READERS: dict[str, MemberReader] = {
    "visit_AutoContrast": lambda t: (True, t.visit_AutoContrast) if isinstance(t, _HasMember53_00) else (False, None),
    "visit_AvgPool": lambda t: (True, t.visit_AvgPool) if isinstance(t, _HasMember53_01) else (False, None),
    "visit_AvgPool2D": lambda t: (True, t.visit_AvgPool2D) if isinstance(t, _HasMember53_02) else (False, None),
    "visit_AvgPool2DDynamic": lambda t: (
        (True, t.visit_AvgPool2DDynamic) if isinstance(t, _HasMember53_03) else (False, None)
    ),
    "visit_AvgPool3D": lambda t: (True, t.visit_AvgPool3D) if isinstance(t, _HasMember53_04) else (False, None),
    "visit_AxisIndex": lambda t: (True, t.visit_AxisIndex) if isinstance(t, _HasMember53_05) else (False, None),
    "visit_BandPart": lambda t: (True, t.visit_BandPart) if isinstance(t, _HasMember53_06) else (False, None),
    "visit_BandedTriangularSolve": lambda t: (
        (True, t.visit_BandedTriangularSolve) if isinstance(t, _HasMember53_07) else (False, None)
    ),
    "visit_BatchMatMul": lambda t: (True, t.visit_BatchMatMul) if isinstance(t, _HasMember53_08) else (False, None),
    "visit_BatchNorm": lambda t: (True, t.visit_BatchNorm) if isinstance(t, _HasMember53_09) else (False, None),
    "visit_BinaryOperation": lambda t: (
        (True, t.visit_BinaryOperation) if isinstance(t, _HasMember53_10) else (False, None)
    ),
    "visit_BlockMaskedMm": lambda t: (True, t.visit_BlockMaskedMm) if isinstance(t, _HasMember53_11) else (False, None),
    "visit_Broadcast": lambda t: (True, t.visit_Broadcast) if isinstance(t, _HasMember53_12) else (False, None),
    "visit_BroadcastArrays": lambda t: (
        (True, t.visit_BroadcastArrays) if isinstance(t, _HasMember53_13) else (False, None)
    ),
    "visit_Call": lambda t: (True, t.visit_Call) if isinstance(t, _HasMember53_14) else (False, None),
    "visit_Cartesian": lambda t: (True, t.visit_Cartesian) if isinstance(t, _HasMember53_15) else (False, None),
    "visit_Ceil": lambda t: (True, t.visit_Ceil) if isinstance(t, _HasMember53_16) else (False, None),
    "visit_Cholesky": lambda t: (True, t.visit_Cholesky) if isinstance(t, _HasMember53_17) else (False, None),
    "visit_CholeskyVjp": lambda t: (True, t.visit_CholeskyVjp) if isinstance(t, _HasMember53_18) else (False, None),
    "visit_ClassDef": lambda t: (True, t.visit_ClassDef) if isinstance(t, _HasMember53_19) else (False, None),
    "visit_Combinations": lambda t: (True, t.visit_Combinations) if isinstance(t, _HasMember53_20) else (False, None),
    "visit_Concatenate": lambda t: (True, t.visit_Concatenate) if isinstance(t, _HasMember53_21) else (False, None),
    "visit_Cond": lambda t: (True, t.visit_Cond) if isinstance(t, _HasMember53_22) else (False, None),
    "visit_ConstantOfShape": lambda t: (
        (True, t.visit_ConstantOfShape) if isinstance(t, _HasMember53_23) else (False, None)
    ),
    "visit_Conv1D": lambda t: (True, t.visit_Conv1D) if isinstance(t, _HasMember53_24) else (False, None),
    "visit_Conv2D": lambda t: (True, t.visit_Conv2D) if isinstance(t, _HasMember53_25) else (False, None),
    "visit_Conv2DSharedHalo": lambda t: (
        (True, t.visit_Conv2DSharedHalo) if isinstance(t, _HasMember53_26) else (False, None)
    ),
    "visit_Conv3D": lambda t: (True, t.visit_Conv3D) if isinstance(t, _HasMember53_27) else (False, None),
    "visit_ConvTranspose": lambda t: (True, t.visit_ConvTranspose) if isinstance(t, _HasMember53_28) else (False, None),
    "visit_Cos": lambda t: (True, t.visit_Cos) if isinstance(t, _HasMember53_29) else (False, None),
    "visit_CscAdd": lambda t: (True, t.visit_CscAdd) if isinstance(t, _HasMember53_30) else (False, None),
    "visit_CscSub": lambda t: (True, t.visit_CscSub) if isinstance(t, _HasMember53_31) else (False, None),
    "visit_CsrAdd": lambda t: (True, t.visit_CsrAdd) if isinstance(t, _HasMember53_32) else (False, None),
    "visit_CsrSub": lambda t: (True, t.visit_CsrSub) if isinstance(t, _HasMember53_33) else (False, None),
    "visit_DenseSpmm": lambda t: (True, t.visit_DenseSpmm) if isinstance(t, _HasMember53_34) else (False, None),
    "visit_DepthwiseConv2D": lambda t: (
        (True, t.visit_DepthwiseConv2D) if isinstance(t, _HasMember53_35) else (False, None)
    ),
    "visit_Div": lambda t: (True, t.visit_Div) if isinstance(t, _HasMember53_36) else (False, None),
    "visit_DropNone": lambda t: (True, t.visit_DropNone) if isinstance(t, _HasMember53_37) else (False, None),
    "visit_DynamicSlice": lambda t: (True, t.visit_DynamicSlice) if isinstance(t, _HasMember53_38) else (False, None),
    "visit_DynamicUpdateSlice": lambda t: (
        (True, t.visit_DynamicUpdateSlice) if isinstance(t, _HasMember53_39) else (False, None)
    ),
    "visit_Einsum": lambda t: (True, t.visit_Einsum) if isinstance(t, _HasMember53_40) else (False, None),
    "visit_ElasticTransform": lambda t: (
        (True, t.visit_ElasticTransform) if isinstance(t, _HasMember53_41) else (False, None)
    ),
    "visit_Exp": lambda t: (True, t.visit_Exp) if isinstance(t, _HasMember53_42) else (False, None),
    "visit_ExtractBoundingBoxes": lambda t: (
        (True, t.visit_ExtractBoundingBoxes) if isinstance(t, _HasMember53_43) else (False, None)
    ),
    "visit_FillNone": lambda t: (True, t.visit_FillNone) if isinstance(t, _HasMember53_44) else (False, None),
}
