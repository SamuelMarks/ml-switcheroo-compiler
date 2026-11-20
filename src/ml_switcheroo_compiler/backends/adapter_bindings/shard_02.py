"""Static structural protocol bindings shard 02 (Flipud..LinearOperatorLowerTriangular)."""

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
class _HasMember02_00(typing.Protocol):
    """Structural protocol for objects exposing `Flipud`."""

    Flipud: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_01(typing.Protocol):
    """Structural protocol for objects exposing `ForeignCall`."""

    ForeignCall: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_02(typing.Protocol):
    """Structural protocol for objects exposing `Fromfile`."""

    Fromfile: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_03(typing.Protocol):
    """Structural protocol for objects exposing `Fromfunction`."""

    Fromfunction: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_04(typing.Protocol):
    """Structural protocol for objects exposing `Fromiter`."""

    Fromiter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_05(typing.Protocol):
    """Structural protocol for objects exposing `Fromstring`."""

    Fromstring: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_06(typing.Protocol):
    """Structural protocol for objects exposing `GenericConvConfig`."""

    GenericConvConfig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_07(typing.Protocol):
    """Structural protocol for objects exposing `Graph`."""

    Graph: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_08(typing.Protocol):
    """Structural protocol for objects exposing `Hfft`."""

    Hfft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_09(typing.Protocol):
    """Structural protocol for objects exposing `Hsplit`."""

    Hsplit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_10(typing.Protocol):
    """Structural protocol for objects exposing `Hstack`."""

    Hstack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_11(typing.Protocol):
    """Structural protocol for objects exposing `Ifft`."""

    Ifft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_12(typing.Protocol):
    """Structural protocol for objects exposing `Ifft2`."""

    Ifft2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_13(typing.Protocol):
    """Structural protocol for objects exposing `Ifftn`."""

    Ifftn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_14(typing.Protocol):
    """Structural protocol for objects exposing `Ifftnd`."""

    Ifftnd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_15(typing.Protocol):
    """Structural protocol for objects exposing `Ifftshift`."""

    Ifftshift: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_16(typing.Protocol):
    """Structural protocol for objects exposing `Ihfft`."""

    Ihfft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_17(typing.Protocol):
    """Structural protocol for objects exposing `InTopK`."""

    InTopK: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_18(typing.Protocol):
    """Structural protocol for objects exposing `Insert`."""

    Insert: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_19(typing.Protocol):
    """Structural protocol for objects exposing `Invert`."""

    Invert: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_20(typing.Protocol):
    """Structural protocol for objects exposing `Iou`."""

    Iou: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_21(typing.Protocol):
    """Structural protocol for objects exposing `Irfft`."""

    Irfft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_22(typing.Protocol):
    """Structural protocol for objects exposing `Irfft2`."""

    Irfft2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_23(typing.Protocol):
    """Structural protocol for objects exposing `Irfftn`."""

    Irfftn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_24(typing.Protocol):
    """Structural protocol for objects exposing `Irfftnd`."""

    Irfftnd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_25(typing.Protocol):
    """Structural protocol for objects exposing `Istft`."""

    Istft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_26(typing.Protocol):
    """Structural protocol for objects exposing `JAXProfiler`."""

    JAXProfiler: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_27(typing.Protocol):
    """Structural protocol for objects exposing `KerasProfiler`."""

    KerasProfiler: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_28(typing.Protocol):
    """Structural protocol for objects exposing `LLVMCPPRunner`."""

    LLVMCPPRunner: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_29(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperator`."""

    LinearOperator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_30(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorAdjoint`."""

    LinearOperatorAdjoint: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_31(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorBlockDiag`."""

    LinearOperatorBlockDiag: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_32(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorBlockLowerTriangular`."""

    LinearOperatorBlockLowerTriangular: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_33(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorCirculant`."""

    LinearOperatorCirculant: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_34(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorCirculant2D`."""

    LinearOperatorCirculant2D: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_35(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorCirculant3D`."""

    LinearOperatorCirculant3D: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_36(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorComposition`."""

    LinearOperatorComposition: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_37(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorDiag`."""

    LinearOperatorDiag: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_38(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorFullMatrix`."""

    LinearOperatorFullMatrix: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_39(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorHouseholder`."""

    LinearOperatorHouseholder: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_40(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorIdentity`."""

    LinearOperatorIdentity: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_41(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorInversion`."""

    LinearOperatorInversion: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_42(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorKronecker`."""

    LinearOperatorKronecker: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_43(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorLowRankUpdate`."""

    LinearOperatorLowRankUpdate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember02_44(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorLowerTriangular`."""

    LinearOperatorLowerTriangular: EagerTensorProtocol


SHARD_02_READERS: dict[str, MemberReader] = {
    "Flipud": lambda t: (True, t.Flipud) if isinstance(t, _HasMember02_00) else (False, None),
    "ForeignCall": lambda t: (True, t.ForeignCall) if isinstance(t, _HasMember02_01) else (False, None),
    "Fromfile": lambda t: (True, t.Fromfile) if isinstance(t, _HasMember02_02) else (False, None),
    "Fromfunction": lambda t: (True, t.Fromfunction) if isinstance(t, _HasMember02_03) else (False, None),
    "Fromiter": lambda t: (True, t.Fromiter) if isinstance(t, _HasMember02_04) else (False, None),
    "Fromstring": lambda t: (True, t.Fromstring) if isinstance(t, _HasMember02_05) else (False, None),
    "GenericConvConfig": lambda t: (True, t.GenericConvConfig) if isinstance(t, _HasMember02_06) else (False, None),
    "Graph": lambda t: (True, t.Graph) if isinstance(t, _HasMember02_07) else (False, None),
    "Hfft": lambda t: (True, t.Hfft) if isinstance(t, _HasMember02_08) else (False, None),
    "Hsplit": lambda t: (True, t.Hsplit) if isinstance(t, _HasMember02_09) else (False, None),
    "Hstack": lambda t: (True, t.Hstack) if isinstance(t, _HasMember02_10) else (False, None),
    "Ifft": lambda t: (True, t.Ifft) if isinstance(t, _HasMember02_11) else (False, None),
    "Ifft2": lambda t: (True, t.Ifft2) if isinstance(t, _HasMember02_12) else (False, None),
    "Ifftn": lambda t: (True, t.Ifftn) if isinstance(t, _HasMember02_13) else (False, None),
    "Ifftnd": lambda t: (True, t.Ifftnd) if isinstance(t, _HasMember02_14) else (False, None),
    "Ifftshift": lambda t: (True, t.Ifftshift) if isinstance(t, _HasMember02_15) else (False, None),
    "Ihfft": lambda t: (True, t.Ihfft) if isinstance(t, _HasMember02_16) else (False, None),
    "InTopK": lambda t: (True, t.InTopK) if isinstance(t, _HasMember02_17) else (False, None),
    "Insert": lambda t: (True, t.Insert) if isinstance(t, _HasMember02_18) else (False, None),
    "Invert": lambda t: (True, t.Invert) if isinstance(t, _HasMember02_19) else (False, None),
    "Iou": lambda t: (True, t.Iou) if isinstance(t, _HasMember02_20) else (False, None),
    "Irfft": lambda t: (True, t.Irfft) if isinstance(t, _HasMember02_21) else (False, None),
    "Irfft2": lambda t: (True, t.Irfft2) if isinstance(t, _HasMember02_22) else (False, None),
    "Irfftn": lambda t: (True, t.Irfftn) if isinstance(t, _HasMember02_23) else (False, None),
    "Irfftnd": lambda t: (True, t.Irfftnd) if isinstance(t, _HasMember02_24) else (False, None),
    "Istft": lambda t: (True, t.Istft) if isinstance(t, _HasMember02_25) else (False, None),
    "JAXProfiler": lambda t: (True, t.JAXProfiler) if isinstance(t, _HasMember02_26) else (False, None),
    "KerasProfiler": lambda t: (True, t.KerasProfiler) if isinstance(t, _HasMember02_27) else (False, None),
    "LLVMCPPRunner": lambda t: (True, t.LLVMCPPRunner) if isinstance(t, _HasMember02_28) else (False, None),
    "LinearOperator": lambda t: (True, t.LinearOperator) if isinstance(t, _HasMember02_29) else (False, None),
    "LinearOperatorAdjoint": lambda t: (
        (True, t.LinearOperatorAdjoint) if isinstance(t, _HasMember02_30) else (False, None)
    ),
    "LinearOperatorBlockDiag": lambda t: (
        (True, t.LinearOperatorBlockDiag) if isinstance(t, _HasMember02_31) else (False, None)
    ),
    "LinearOperatorBlockLowerTriangular": lambda t: (
        (True, t.LinearOperatorBlockLowerTriangular) if isinstance(t, _HasMember02_32) else (False, None)
    ),
    "LinearOperatorCirculant": lambda t: (
        (True, t.LinearOperatorCirculant) if isinstance(t, _HasMember02_33) else (False, None)
    ),
    "LinearOperatorCirculant2D": lambda t: (
        (True, t.LinearOperatorCirculant2D) if isinstance(t, _HasMember02_34) else (False, None)
    ),
    "LinearOperatorCirculant3D": lambda t: (
        (True, t.LinearOperatorCirculant3D) if isinstance(t, _HasMember02_35) else (False, None)
    ),
    "LinearOperatorComposition": lambda t: (
        (True, t.LinearOperatorComposition) if isinstance(t, _HasMember02_36) else (False, None)
    ),
    "LinearOperatorDiag": lambda t: (True, t.LinearOperatorDiag) if isinstance(t, _HasMember02_37) else (False, None),
    "LinearOperatorFullMatrix": lambda t: (
        (True, t.LinearOperatorFullMatrix) if isinstance(t, _HasMember02_38) else (False, None)
    ),
    "LinearOperatorHouseholder": lambda t: (
        (True, t.LinearOperatorHouseholder) if isinstance(t, _HasMember02_39) else (False, None)
    ),
    "LinearOperatorIdentity": lambda t: (
        (True, t.LinearOperatorIdentity) if isinstance(t, _HasMember02_40) else (False, None)
    ),
    "LinearOperatorInversion": lambda t: (
        (True, t.LinearOperatorInversion) if isinstance(t, _HasMember02_41) else (False, None)
    ),
    "LinearOperatorKronecker": lambda t: (
        (True, t.LinearOperatorKronecker) if isinstance(t, _HasMember02_42) else (False, None)
    ),
    "LinearOperatorLowRankUpdate": lambda t: (
        (True, t.LinearOperatorLowRankUpdate) if isinstance(t, _HasMember02_43) else (False, None)
    ),
    "LinearOperatorLowerTriangular": lambda t: (
        (True, t.LinearOperatorLowerTriangular) if isinstance(t, _HasMember02_44) else (False, None)
    ),
}
