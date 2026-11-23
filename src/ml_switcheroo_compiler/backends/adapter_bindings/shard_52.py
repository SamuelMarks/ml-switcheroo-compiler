"""Static structural protocol bindings shard 52 (var..visit_AugMix)."""

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
class _HasMember52_00(typing.Protocol):
    """Structural protocol for objects exposing `var`."""

    var: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_01(typing.Protocol):
    """Structural protocol for objects exposing `var_map`."""

    var_map: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_02(typing.Protocol):
    """Structural protocol for objects exposing `variance`."""

    variance: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_03(typing.Protocol):
    """Structural protocol for objects exposing `vdot`."""

    vdot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_04(typing.Protocol):
    """Structural protocol for objects exposing `vecdot`."""

    vecdot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_05(typing.Protocol):
    """Structural protocol for objects exposing `vector_norm`."""

    vector_norm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_06(typing.Protocol):
    """Structural protocol for objects exposing `vectorize`."""

    vectorize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_07(typing.Protocol):
    """Structural protocol for objects exposing `vectorized_map`."""

    vectorized_map: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_08(typing.Protocol):
    """Structural protocol for objects exposing `vectornorm`."""

    vectornorm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_09(typing.Protocol):
    """Structural protocol for objects exposing `view_as_complex`."""

    view_as_complex: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_10(typing.Protocol):
    """Structural protocol for objects exposing `view_as_real`."""

    view_as_real: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_11(typing.Protocol):
    """Structural protocol for objects exposing `visible`."""

    visible: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_12(typing.Protocol):
    """Structural protocol for objects exposing `vision`."""

    vision: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_13(typing.Protocol):
    """Structural protocol for objects exposing `visit`."""

    visit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_14(typing.Protocol):
    """Structural protocol for objects exposing `visit_Abs`."""

    visit_Abs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_15(typing.Protocol):
    """Structural protocol for objects exposing `visit_AccumulateN`."""

    visit_AccumulateN: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_16(typing.Protocol):
    """Structural protocol for objects exposing `visit_AdaptiveAvgPool2D`."""

    visit_AdaptiveAvgPool2D: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_17(typing.Protocol):
    """Structural protocol for objects exposing `visit_Add`."""

    visit_Add: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_18(typing.Protocol):
    """Structural protocol for objects exposing `visit_AddN`."""

    visit_AddN: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_19(typing.Protocol):
    """Structural protocol for objects exposing `visit_AdjustBrightness`."""

    visit_AdjustBrightness: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_20(typing.Protocol):
    """Structural protocol for objects exposing `visit_AdjustContrast`."""

    visit_AdjustContrast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_21(typing.Protocol):
    """Structural protocol for objects exposing `visit_AdjustHue`."""

    visit_AdjustHue: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_22(typing.Protocol):
    """Structural protocol for objects exposing `visit_AdjustSaturation`."""

    visit_AdjustSaturation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_23(typing.Protocol):
    """Structural protocol for objects exposing `visit_AffineGenerator`."""

    visit_AffineGenerator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_24(typing.Protocol):
    """Structural protocol for objects exposing `visit_AffineGrid`."""

    visit_AffineGrid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_25(typing.Protocol):
    """Structural protocol for objects exposing `visit_AffineTransform`."""

    visit_AffineTransform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_26(typing.Protocol):
    """Structural protocol for objects exposing `visit_AllGather`."""

    visit_AllGather: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_27(typing.Protocol):
    """Structural protocol for objects exposing `visit_AllReduce`."""

    visit_AllReduce: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_28(typing.Protocol):
    """Structural protocol for objects exposing `visit_AllToAll`."""

    visit_AllToAll: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_29(typing.Protocol):
    """Structural protocol for objects exposing `visit_ApproxMaxK`."""

    visit_ApproxMaxK: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_30(typing.Protocol):
    """Structural protocol for objects exposing `visit_ApproxMaxKIndices`."""

    visit_ApproxMaxKIndices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_31(typing.Protocol):
    """Structural protocol for objects exposing `visit_ApproxMinK`."""

    visit_ApproxMinK: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_32(typing.Protocol):
    """Structural protocol for objects exposing `visit_ApproxMinKIndices`."""

    visit_ApproxMinKIndices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_33(typing.Protocol):
    """Structural protocol for objects exposing `visit_ArgSort`."""

    visit_ArgSort: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_34(typing.Protocol):
    """Structural protocol for objects exposing `visit_Argpartition`."""

    visit_Argpartition: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_35(typing.Protocol):
    """Structural protocol for objects exposing `visit_Argwhere`."""

    visit_Argwhere: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_36(typing.Protocol):
    """Structural protocol for objects exposing `visit_AsString`."""

    visit_AsString: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_37(typing.Protocol):
    """Structural protocol for objects exposing `visit_Assert`."""

    visit_Assert: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_38(typing.Protocol):
    """Structural protocol for objects exposing `visit_Assign`."""

    visit_Assign: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_39(typing.Protocol):
    """Structural protocol for objects exposing `visit_AssignAdd`."""

    visit_AssignAdd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_40(typing.Protocol):
    """Structural protocol for objects exposing `visit_AssignSub`."""

    visit_AssignSub: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_41(typing.Protocol):
    """Structural protocol for objects exposing `visit_AssociativeScan`."""

    visit_AssociativeScan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_42(typing.Protocol):
    """Structural protocol for objects exposing `visit_AsyncFunctionDef`."""

    visit_AsyncFunctionDef: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_43(typing.Protocol):
    """Structural protocol for objects exposing `visit_Attention`."""

    visit_Attention: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember52_44(typing.Protocol):
    """Structural protocol for objects exposing `visit_AugMix`."""

    visit_AugMix: EagerTensorProtocol


SHARD_52_READERS: dict[str, MemberReader] = {
    "var": lambda t: (True, t.var) if isinstance(t, _HasMember52_00) else (False, None),
    "var_map": lambda t: (True, t.var_map) if isinstance(t, _HasMember52_01) else (False, None),
    "variance": lambda t: (True, t.variance) if isinstance(t, _HasMember52_02) else (False, None),
    "vdot": lambda t: (True, t.vdot) if isinstance(t, _HasMember52_03) else (False, None),
    "vecdot": lambda t: (True, t.vecdot) if isinstance(t, _HasMember52_04) else (False, None),
    "vector_norm": lambda t: (True, t.vector_norm) if isinstance(t, _HasMember52_05) else (False, None),
    "vectorize": lambda t: (True, t.vectorize) if isinstance(t, _HasMember52_06) else (False, None),
    "vectorized_map": lambda t: (True, t.vectorized_map) if isinstance(t, _HasMember52_07) else (False, None),
    "vectornorm": lambda t: (True, t.vectornorm) if isinstance(t, _HasMember52_08) else (False, None),
    "view_as_complex": lambda t: (True, t.view_as_complex) if isinstance(t, _HasMember52_09) else (False, None),
    "view_as_real": lambda t: (True, t.view_as_real) if isinstance(t, _HasMember52_10) else (False, None),
    "visible": lambda t: (True, t.visible) if isinstance(t, _HasMember52_11) else (False, None),
    "vision": lambda t: (True, t.vision) if isinstance(t, _HasMember52_12) else (False, None),
    "visit": lambda t: (True, t.visit) if isinstance(t, _HasMember52_13) else (False, None),
    "visit_Abs": lambda t: (True, t.visit_Abs) if isinstance(t, _HasMember52_14) else (False, None),
    "visit_AccumulateN": lambda t: (True, t.visit_AccumulateN) if isinstance(t, _HasMember52_15) else (False, None),
    "visit_AdaptiveAvgPool2D": lambda t: (
        (True, t.visit_AdaptiveAvgPool2D) if isinstance(t, _HasMember52_16) else (False, None)
    ),
    "visit_Add": lambda t: (True, t.visit_Add) if isinstance(t, _HasMember52_17) else (False, None),
    "visit_AddN": lambda t: (True, t.visit_AddN) if isinstance(t, _HasMember52_18) else (False, None),
    "visit_AdjustBrightness": lambda t: (
        (True, t.visit_AdjustBrightness) if isinstance(t, _HasMember52_19) else (False, None)
    ),
    "visit_AdjustContrast": lambda t: (
        (True, t.visit_AdjustContrast) if isinstance(t, _HasMember52_20) else (False, None)
    ),
    "visit_AdjustHue": lambda t: (True, t.visit_AdjustHue) if isinstance(t, _HasMember52_21) else (False, None),
    "visit_AdjustSaturation": lambda t: (
        (True, t.visit_AdjustSaturation) if isinstance(t, _HasMember52_22) else (False, None)
    ),
    "visit_AffineGenerator": lambda t: (
        (True, t.visit_AffineGenerator) if isinstance(t, _HasMember52_23) else (False, None)
    ),
    "visit_AffineGrid": lambda t: (True, t.visit_AffineGrid) if isinstance(t, _HasMember52_24) else (False, None),
    "visit_AffineTransform": lambda t: (
        (True, t.visit_AffineTransform) if isinstance(t, _HasMember52_25) else (False, None)
    ),
    "visit_AllGather": lambda t: (True, t.visit_AllGather) if isinstance(t, _HasMember52_26) else (False, None),
    "visit_AllReduce": lambda t: (True, t.visit_AllReduce) if isinstance(t, _HasMember52_27) else (False, None),
    "visit_AllToAll": lambda t: (True, t.visit_AllToAll) if isinstance(t, _HasMember52_28) else (False, None),
    "visit_ApproxMaxK": lambda t: (True, t.visit_ApproxMaxK) if isinstance(t, _HasMember52_29) else (False, None),
    "visit_ApproxMaxKIndices": lambda t: (
        (True, t.visit_ApproxMaxKIndices) if isinstance(t, _HasMember52_30) else (False, None)
    ),
    "visit_ApproxMinK": lambda t: (True, t.visit_ApproxMinK) if isinstance(t, _HasMember52_31) else (False, None),
    "visit_ApproxMinKIndices": lambda t: (
        (True, t.visit_ApproxMinKIndices) if isinstance(t, _HasMember52_32) else (False, None)
    ),
    "visit_ArgSort": lambda t: (True, t.visit_ArgSort) if isinstance(t, _HasMember52_33) else (False, None),
    "visit_Argpartition": lambda t: (True, t.visit_Argpartition) if isinstance(t, _HasMember52_34) else (False, None),
    "visit_Argwhere": lambda t: (True, t.visit_Argwhere) if isinstance(t, _HasMember52_35) else (False, None),
    "visit_AsString": lambda t: (True, t.visit_AsString) if isinstance(t, _HasMember52_36) else (False, None),
    "visit_Assert": lambda t: (True, t.visit_Assert) if isinstance(t, _HasMember52_37) else (False, None),
    "visit_Assign": lambda t: (True, t.visit_Assign) if isinstance(t, _HasMember52_38) else (False, None),
    "visit_AssignAdd": lambda t: (True, t.visit_AssignAdd) if isinstance(t, _HasMember52_39) else (False, None),
    "visit_AssignSub": lambda t: (True, t.visit_AssignSub) if isinstance(t, _HasMember52_40) else (False, None),
    "visit_AssociativeScan": lambda t: (
        (True, t.visit_AssociativeScan) if isinstance(t, _HasMember52_41) else (False, None)
    ),
    "visit_AsyncFunctionDef": lambda t: (
        (True, t.visit_AsyncFunctionDef) if isinstance(t, _HasMember52_42) else (False, None)
    ),
    "visit_Attention": lambda t: (True, t.visit_Attention) if isinstance(t, _HasMember52_43) else (False, None),
    "visit_AugMix": lambda t: (True, t.visit_AugMix) if isinstance(t, _HasMember52_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember52_00,
        _HasMember52_01,
        _HasMember52_02,
        _HasMember52_03,
        _HasMember52_04,
        _HasMember52_05,
        _HasMember52_06,
        _HasMember52_07,
        _HasMember52_08,
        _HasMember52_09,
        _HasMember52_10,
        _HasMember52_11,
        _HasMember52_12,
        _HasMember52_13,
        _HasMember52_14,
        _HasMember52_15,
        _HasMember52_16,
        _HasMember52_17,
        _HasMember52_18,
        _HasMember52_19,
        _HasMember52_20,
        _HasMember52_21,
        _HasMember52_22,
        _HasMember52_23,
        _HasMember52_24,
        _HasMember52_25,
        _HasMember52_26,
        _HasMember52_27,
        _HasMember52_28,
        _HasMember52_29,
        _HasMember52_30,
        _HasMember52_31,
        _HasMember52_32,
        _HasMember52_33,
        _HasMember52_34,
        _HasMember52_35,
        _HasMember52_36,
        _HasMember52_37,
        _HasMember52_38,
        _HasMember52_39,
        _HasMember52_40,
        _HasMember52_41,
        _HasMember52_42,
        _HasMember52_43,
        _HasMember52_44,
    )
)
