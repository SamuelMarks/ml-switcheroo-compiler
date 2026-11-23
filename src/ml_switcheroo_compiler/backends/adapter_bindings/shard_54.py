"""Static structural protocol bindings shard 54 (visit_Flatten..visit_Mul)."""

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
class _HasMember54_00(typing.Protocol):
    """Structural protocol for objects exposing `visit_Flatten`."""

    visit_Flatten: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_01(typing.Protocol):
    """Structural protocol for objects exposing `visit_Float`."""

    visit_Float: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_02(typing.Protocol):
    """Structural protocol for objects exposing `visit_Floor`."""

    visit_Floor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_03(typing.Protocol):
    """Structural protocol for objects exposing `visit_Fold`."""

    visit_Fold: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_04(typing.Protocol):
    """Structural protocol for objects exposing `visit_For`."""

    visit_For: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_05(typing.Protocol):
    """Structural protocol for objects exposing `visit_ForiLoop`."""

    visit_ForiLoop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_06(typing.Protocol):
    """Structural protocol for objects exposing `visit_FromRegular`."""

    visit_FromRegular: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_07(typing.Protocol):
    """Structural protocol for objects exposing `visit_Full`."""

    visit_Full: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_08(typing.Protocol):
    """Structural protocol for objects exposing `visit_FunctionDef`."""

    visit_FunctionDef: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_09(typing.Protocol):
    """Structural protocol for objects exposing `visit_GatherMm`."""

    visit_GatherMm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_10(typing.Protocol):
    """Structural protocol for objects exposing `visit_GatherND`."""

    visit_GatherND: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_11(typing.Protocol):
    """Structural protocol for objects exposing `visit_GatherQMM`."""

    visit_GatherQMM: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_12(typing.Protocol):
    """Structural protocol for objects exposing `visit_GaussianBlur`."""

    visit_GaussianBlur: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_13(typing.Protocol):
    """Structural protocol for objects exposing `visit_GetItem`."""

    visit_GetItem: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_14(typing.Protocol):
    """Structural protocol for objects exposing `visit_GnnSpmmAttention`."""

    visit_GnnSpmmAttention: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_15(typing.Protocol):
    """Structural protocol for objects exposing `visit_GraphNormAdjacency`."""

    visit_GraphNormAdjacency: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_16(typing.Protocol):
    """Structural protocol for objects exposing `visit_GroupNorm`."""

    visit_GroupNorm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_17(typing.Protocol):
    """Structural protocol for objects exposing `visit_If`."""

    visit_If: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_18(typing.Protocol):
    """Structural protocol for objects exposing `visit_IfExp`."""

    visit_IfExp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_19(typing.Protocol):
    """Structural protocol for objects exposing `visit_Integer`."""

    visit_Integer: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_20(typing.Protocol):
    """Structural protocol for objects exposing `visit_IoU`."""

    visit_IoU: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_21(typing.Protocol):
    """Structural protocol for objects exposing `visit_Istft`."""

    visit_Istft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_22(typing.Protocol):
    """Structural protocol for objects exposing `visit_LayerNorm`."""

    visit_LayerNorm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_23(typing.Protocol):
    """Structural protocol for objects exposing `visit_Linear`."""

    visit_Linear: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_24(typing.Protocol):
    """Structural protocol for objects exposing `visit_ListComp`."""

    visit_ListComp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_25(typing.Protocol):
    """Structural protocol for objects exposing `visit_Log`."""

    visit_Log: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_26(typing.Protocol):
    """Structural protocol for objects exposing `visit_LogSoftmax`."""

    visit_LogSoftmax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_27(typing.Protocol):
    """Structural protocol for objects exposing `visit_Loop`."""

    visit_Loop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_28(typing.Protocol):
    """Structural protocol for objects exposing `visit_LuVjp`."""

    visit_LuVjp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_29(typing.Protocol):
    """Structural protocol for objects exposing `visit_Map`."""

    visit_Map: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_30(typing.Protocol):
    """Structural protocol for objects exposing `visit_MatMul`."""

    visit_MatMul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_31(typing.Protocol):
    """Structural protocol for objects exposing `visit_MatrixExponential`."""

    visit_MatrixExponential: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_32(typing.Protocol):
    """Structural protocol for objects exposing `visit_Max`."""

    visit_Max: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_33(typing.Protocol):
    """Structural protocol for objects exposing `visit_MaxPool`."""

    visit_MaxPool: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_34(typing.Protocol):
    """Structural protocol for objects exposing `visit_MaxPool2D`."""

    visit_MaxPool2D: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_35(typing.Protocol):
    """Structural protocol for objects exposing `visit_MaxPool2DDynamic`."""

    visit_MaxPool2DDynamic: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_36(typing.Protocol):
    """Structural protocol for objects exposing `visit_MaxPool3D`."""

    visit_MaxPool3D: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_37(typing.Protocol):
    """Structural protocol for objects exposing `visit_Mean`."""

    visit_Mean: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_38(typing.Protocol):
    """Structural protocol for objects exposing `visit_MedianFilter`."""

    visit_MedianFilter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_39(typing.Protocol):
    """Structural protocol for objects exposing `visit_MelFilterbank`."""

    visit_MelFilterbank: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_40(typing.Protocol):
    """Structural protocol for objects exposing `visit_Meshgrid`."""

    visit_Meshgrid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_41(typing.Protocol):
    """Structural protocol for objects exposing `visit_Mfcc`."""

    visit_Mfcc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_42(typing.Protocol):
    """Structural protocol for objects exposing `visit_Min`."""

    visit_Min: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_43(typing.Protocol):
    """Structural protocol for objects exposing `visit_Moveaxis`."""

    visit_Moveaxis: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember54_44(typing.Protocol):
    """Structural protocol for objects exposing `visit_Mul`."""

    visit_Mul: EagerTensorProtocol


SHARD_54_READERS: dict[str, MemberReader] = {
    "visit_Flatten": lambda t: (True, t.visit_Flatten) if isinstance(t, _HasMember54_00) else (False, None),
    "visit_Float": lambda t: (True, t.visit_Float) if isinstance(t, _HasMember54_01) else (False, None),
    "visit_Floor": lambda t: (True, t.visit_Floor) if isinstance(t, _HasMember54_02) else (False, None),
    "visit_Fold": lambda t: (True, t.visit_Fold) if isinstance(t, _HasMember54_03) else (False, None),
    "visit_For": lambda t: (True, t.visit_For) if isinstance(t, _HasMember54_04) else (False, None),
    "visit_ForiLoop": lambda t: (True, t.visit_ForiLoop) if isinstance(t, _HasMember54_05) else (False, None),
    "visit_FromRegular": lambda t: (True, t.visit_FromRegular) if isinstance(t, _HasMember54_06) else (False, None),
    "visit_Full": lambda t: (True, t.visit_Full) if isinstance(t, _HasMember54_07) else (False, None),
    "visit_FunctionDef": lambda t: (True, t.visit_FunctionDef) if isinstance(t, _HasMember54_08) else (False, None),
    "visit_GatherMm": lambda t: (True, t.visit_GatherMm) if isinstance(t, _HasMember54_09) else (False, None),
    "visit_GatherND": lambda t: (True, t.visit_GatherND) if isinstance(t, _HasMember54_10) else (False, None),
    "visit_GatherQMM": lambda t: (True, t.visit_GatherQMM) if isinstance(t, _HasMember54_11) else (False, None),
    "visit_GaussianBlur": lambda t: (True, t.visit_GaussianBlur) if isinstance(t, _HasMember54_12) else (False, None),
    "visit_GetItem": lambda t: (True, t.visit_GetItem) if isinstance(t, _HasMember54_13) else (False, None),
    "visit_GnnSpmmAttention": lambda t: (
        (True, t.visit_GnnSpmmAttention) if isinstance(t, _HasMember54_14) else (False, None)
    ),
    "visit_GraphNormAdjacency": lambda t: (
        (True, t.visit_GraphNormAdjacency) if isinstance(t, _HasMember54_15) else (False, None)
    ),
    "visit_GroupNorm": lambda t: (True, t.visit_GroupNorm) if isinstance(t, _HasMember54_16) else (False, None),
    "visit_If": lambda t: (True, t.visit_If) if isinstance(t, _HasMember54_17) else (False, None),
    "visit_IfExp": lambda t: (True, t.visit_IfExp) if isinstance(t, _HasMember54_18) else (False, None),
    "visit_Integer": lambda t: (True, t.visit_Integer) if isinstance(t, _HasMember54_19) else (False, None),
    "visit_IoU": lambda t: (True, t.visit_IoU) if isinstance(t, _HasMember54_20) else (False, None),
    "visit_Istft": lambda t: (True, t.visit_Istft) if isinstance(t, _HasMember54_21) else (False, None),
    "visit_LayerNorm": lambda t: (True, t.visit_LayerNorm) if isinstance(t, _HasMember54_22) else (False, None),
    "visit_Linear": lambda t: (True, t.visit_Linear) if isinstance(t, _HasMember54_23) else (False, None),
    "visit_ListComp": lambda t: (True, t.visit_ListComp) if isinstance(t, _HasMember54_24) else (False, None),
    "visit_Log": lambda t: (True, t.visit_Log) if isinstance(t, _HasMember54_25) else (False, None),
    "visit_LogSoftmax": lambda t: (True, t.visit_LogSoftmax) if isinstance(t, _HasMember54_26) else (False, None),
    "visit_Loop": lambda t: (True, t.visit_Loop) if isinstance(t, _HasMember54_27) else (False, None),
    "visit_LuVjp": lambda t: (True, t.visit_LuVjp) if isinstance(t, _HasMember54_28) else (False, None),
    "visit_Map": lambda t: (True, t.visit_Map) if isinstance(t, _HasMember54_29) else (False, None),
    "visit_MatMul": lambda t: (True, t.visit_MatMul) if isinstance(t, _HasMember54_30) else (False, None),
    "visit_MatrixExponential": lambda t: (
        (True, t.visit_MatrixExponential) if isinstance(t, _HasMember54_31) else (False, None)
    ),
    "visit_Max": lambda t: (True, t.visit_Max) if isinstance(t, _HasMember54_32) else (False, None),
    "visit_MaxPool": lambda t: (True, t.visit_MaxPool) if isinstance(t, _HasMember54_33) else (False, None),
    "visit_MaxPool2D": lambda t: (True, t.visit_MaxPool2D) if isinstance(t, _HasMember54_34) else (False, None),
    "visit_MaxPool2DDynamic": lambda t: (
        (True, t.visit_MaxPool2DDynamic) if isinstance(t, _HasMember54_35) else (False, None)
    ),
    "visit_MaxPool3D": lambda t: (True, t.visit_MaxPool3D) if isinstance(t, _HasMember54_36) else (False, None),
    "visit_Mean": lambda t: (True, t.visit_Mean) if isinstance(t, _HasMember54_37) else (False, None),
    "visit_MedianFilter": lambda t: (True, t.visit_MedianFilter) if isinstance(t, _HasMember54_38) else (False, None),
    "visit_MelFilterbank": lambda t: (True, t.visit_MelFilterbank) if isinstance(t, _HasMember54_39) else (False, None),
    "visit_Meshgrid": lambda t: (True, t.visit_Meshgrid) if isinstance(t, _HasMember54_40) else (False, None),
    "visit_Mfcc": lambda t: (True, t.visit_Mfcc) if isinstance(t, _HasMember54_41) else (False, None),
    "visit_Min": lambda t: (True, t.visit_Min) if isinstance(t, _HasMember54_42) else (False, None),
    "visit_Moveaxis": lambda t: (True, t.visit_Moveaxis) if isinstance(t, _HasMember54_43) else (False, None),
    "visit_Mul": lambda t: (True, t.visit_Mul) if isinstance(t, _HasMember54_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember54_00,
        _HasMember54_01,
        _HasMember54_02,
        _HasMember54_03,
        _HasMember54_04,
        _HasMember54_05,
        _HasMember54_06,
        _HasMember54_07,
        _HasMember54_08,
        _HasMember54_09,
        _HasMember54_10,
        _HasMember54_11,
        _HasMember54_12,
        _HasMember54_13,
        _HasMember54_14,
        _HasMember54_15,
        _HasMember54_16,
        _HasMember54_17,
        _HasMember54_18,
        _HasMember54_19,
        _HasMember54_20,
        _HasMember54_21,
        _HasMember54_22,
        _HasMember54_23,
        _HasMember54_24,
        _HasMember54_25,
        _HasMember54_26,
        _HasMember54_27,
        _HasMember54_28,
        _HasMember54_29,
        _HasMember54_30,
        _HasMember54_31,
        _HasMember54_32,
        _HasMember54_33,
        _HasMember54_34,
        _HasMember54_35,
        _HasMember54_36,
        _HasMember54_37,
        _HasMember54_38,
        _HasMember54_39,
        _HasMember54_40,
        _HasMember54_41,
        _HasMember54_42,
        _HasMember54_43,
        _HasMember54_44,
    )
)
