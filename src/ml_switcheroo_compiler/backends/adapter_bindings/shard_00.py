"""Static structural protocol bindings shard 00 (ASSIGN_VARIABLE_OP..CtcLoss)."""

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
class _HasMember00_00(typing.Protocol):
    """Structural protocol for objects exposing `ASSIGN_VARIABLE_OP`."""

    ASSIGN_VARIABLE_OP: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_01(typing.Protocol):
    """Structural protocol for objects exposing `AdaptiveLogSoftmaxWithLoss`."""

    AdaptiveLogSoftmaxWithLoss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_02(typing.Protocol):
    """Structural protocol for objects exposing `Add`."""

    Add: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_03(typing.Protocol):
    """Structural protocol for objects exposing `AdjustBrightness`."""

    AdjustBrightness: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_04(typing.Protocol):
    """Structural protocol for objects exposing `AdjustContrast`."""

    AdjustContrast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_05(typing.Protocol):
    """Structural protocol for objects exposing `AdjustHue`."""

    AdjustHue: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_06(typing.Protocol):
    """Structural protocol for objects exposing `AdjustSaturation`."""

    AdjustSaturation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_07(typing.Protocol):
    """Structural protocol for objects exposing `AffineConfig`."""

    AffineConfig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_08(typing.Protocol):
    """Structural protocol for objects exposing `AffineGenerator`."""

    AffineGenerator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_09(typing.Protocol):
    """Structural protocol for objects exposing `AffineGrid`."""

    AffineGrid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_10(typing.Protocol):
    """Structural protocol for objects exposing `AffineTransform`."""

    AffineTransform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_11(typing.Protocol):
    """Structural protocol for objects exposing `Append`."""

    Append: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_12(typing.Protocol):
    """Structural protocol for objects exposing `Array`."""

    Array: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_13(typing.Protocol):
    """Structural protocol for objects exposing `AsStringConfig`."""

    AsStringConfig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_14(typing.Protocol):
    """Structural protocol for objects exposing `AssertOp`."""

    AssertOp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_15(typing.Protocol):
    """Structural protocol for objects exposing `AssignVariable`."""

    AssignVariable: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_16(typing.Protocol):
    """Structural protocol for objects exposing `AssociativeScan`."""

    AssociativeScan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_17(typing.Protocol):
    """Structural protocol for objects exposing `AsyncIOFuture`."""

    AsyncIOFuture: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_18(typing.Protocol):
    """Structural protocol for objects exposing `Atleast1d`."""

    Atleast1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_19(typing.Protocol):
    """Structural protocol for objects exposing `Atleast2d`."""

    Atleast2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_20(typing.Protocol):
    """Structural protocol for objects exposing `Atleast3d`."""

    Atleast3d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_21(typing.Protocol):
    """Structural protocol for objects exposing `AugMix`."""

    AugMix: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_22(typing.Protocol):
    """Structural protocol for objects exposing `AutoContrast`."""

    AutoContrast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_23(typing.Protocol):
    """Structural protocol for objects exposing `BenchmarkOrchestrator`."""

    BenchmarkOrchestrator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_24(typing.Protocol):
    """Structural protocol for objects exposing `BenchmarkPlan`."""

    BenchmarkPlan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_25(typing.Protocol):
    """Structural protocol for objects exposing `BenchmarkRunResult`."""

    BenchmarkRunResult: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_26(typing.Protocol):
    """Structural protocol for objects exposing `BinaryRecordOptions`."""

    BinaryRecordOptions: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_27(typing.Protocol):
    """Structural protocol for objects exposing `BinaryRecordWriter`."""

    BinaryRecordWriter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_28(typing.Protocol):
    """Structural protocol for objects exposing `Block`."""

    Block: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_29(typing.Protocol):
    """Structural protocol for objects exposing `BooleanMask`."""

    BooleanMask: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_30(typing.Protocol):
    """Structural protocol for objects exposing `COOTensor`."""

    COOTensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_31(typing.Protocol):
    """Structural protocol for objects exposing `CategoricalGeneralizedCrossEntropy`."""

    CategoricalGeneralizedCrossEntropy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_32(typing.Protocol):
    """Structural protocol for objects exposing `CircleLoss`."""

    CircleLoss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_33(typing.Protocol):
    """Structural protocol for objects exposing `ColumnStack`."""

    ColumnStack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_34(typing.Protocol):
    """Structural protocol for objects exposing `ConvGeneralDilated`."""

    ConvGeneralDilated: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_35(typing.Protocol):
    """Structural protocol for objects exposing `ConvGeneralDilatedLocal`."""

    ConvGeneralDilatedLocal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_36(typing.Protocol):
    """Structural protocol for objects exposing `ConvGeneralDilatedPatches`."""

    ConvGeneralDilatedPatches: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_37(typing.Protocol):
    """Structural protocol for objects exposing `ConvTranspose`."""

    ConvTranspose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_38(typing.Protocol):
    """Structural protocol for objects exposing `ConvWithGeneralPadding`."""

    ConvWithGeneralPadding: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_39(typing.Protocol):
    """Structural protocol for objects exposing `Convolve2d`."""

    Convolve2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_40(typing.Protocol):
    """Structural protocol for objects exposing `CookieJar`."""

    CookieJar: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_41(typing.Protocol):
    """Structural protocol for objects exposing `CppGenerator`."""

    CppGenerator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_42(typing.Protocol):
    """Structural protocol for objects exposing `CppProfiler`."""

    CppProfiler: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_43(typing.Protocol):
    """Structural protocol for objects exposing `CreateToken`."""

    CreateToken: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember00_44(typing.Protocol):
    """Structural protocol for objects exposing `CtcLoss`."""

    CtcLoss: EagerTensorProtocol


SHARD_00_READERS: dict[str, MemberReader] = {
    "ASSIGN_VARIABLE_OP": lambda t: (True, t.ASSIGN_VARIABLE_OP) if isinstance(t, _HasMember00_00) else (False, None),
    "AdaptiveLogSoftmaxWithLoss": lambda t: (
        (True, t.AdaptiveLogSoftmaxWithLoss) if isinstance(t, _HasMember00_01) else (False, None)
    ),
    "Add": lambda t: (True, t.Add) if isinstance(t, _HasMember00_02) else (False, None),
    "AdjustBrightness": lambda t: (True, t.AdjustBrightness) if isinstance(t, _HasMember00_03) else (False, None),
    "AdjustContrast": lambda t: (True, t.AdjustContrast) if isinstance(t, _HasMember00_04) else (False, None),
    "AdjustHue": lambda t: (True, t.AdjustHue) if isinstance(t, _HasMember00_05) else (False, None),
    "AdjustSaturation": lambda t: (True, t.AdjustSaturation) if isinstance(t, _HasMember00_06) else (False, None),
    "AffineConfig": lambda t: (True, t.AffineConfig) if isinstance(t, _HasMember00_07) else (False, None),
    "AffineGenerator": lambda t: (True, t.AffineGenerator) if isinstance(t, _HasMember00_08) else (False, None),
    "AffineGrid": lambda t: (True, t.AffineGrid) if isinstance(t, _HasMember00_09) else (False, None),
    "AffineTransform": lambda t: (True, t.AffineTransform) if isinstance(t, _HasMember00_10) else (False, None),
    "Append": lambda t: (True, t.Append) if isinstance(t, _HasMember00_11) else (False, None),
    "Array": lambda t: (True, t.Array) if isinstance(t, _HasMember00_12) else (False, None),
    "AsStringConfig": lambda t: (True, t.AsStringConfig) if isinstance(t, _HasMember00_13) else (False, None),
    "AssertOp": lambda t: (True, t.AssertOp) if isinstance(t, _HasMember00_14) else (False, None),
    "AssignVariable": lambda t: (True, t.AssignVariable) if isinstance(t, _HasMember00_15) else (False, None),
    "AssociativeScan": lambda t: (True, t.AssociativeScan) if isinstance(t, _HasMember00_16) else (False, None),
    "AsyncIOFuture": lambda t: (True, t.AsyncIOFuture) if isinstance(t, _HasMember00_17) else (False, None),
    "Atleast1d": lambda t: (True, t.Atleast1d) if isinstance(t, _HasMember00_18) else (False, None),
    "Atleast2d": lambda t: (True, t.Atleast2d) if isinstance(t, _HasMember00_19) else (False, None),
    "Atleast3d": lambda t: (True, t.Atleast3d) if isinstance(t, _HasMember00_20) else (False, None),
    "AugMix": lambda t: (True, t.AugMix) if isinstance(t, _HasMember00_21) else (False, None),
    "AutoContrast": lambda t: (True, t.AutoContrast) if isinstance(t, _HasMember00_22) else (False, None),
    "BenchmarkOrchestrator": lambda t: (
        (True, t.BenchmarkOrchestrator) if isinstance(t, _HasMember00_23) else (False, None)
    ),
    "BenchmarkPlan": lambda t: (True, t.BenchmarkPlan) if isinstance(t, _HasMember00_24) else (False, None),
    "BenchmarkRunResult": lambda t: (True, t.BenchmarkRunResult) if isinstance(t, _HasMember00_25) else (False, None),
    "BinaryRecordOptions": lambda t: (True, t.BinaryRecordOptions) if isinstance(t, _HasMember00_26) else (False, None),
    "BinaryRecordWriter": lambda t: (True, t.BinaryRecordWriter) if isinstance(t, _HasMember00_27) else (False, None),
    "Block": lambda t: (True, t.Block) if isinstance(t, _HasMember00_28) else (False, None),
    "BooleanMask": lambda t: (True, t.BooleanMask) if isinstance(t, _HasMember00_29) else (False, None),
    "COOTensor": lambda t: (True, t.COOTensor) if isinstance(t, _HasMember00_30) else (False, None),
    "CategoricalGeneralizedCrossEntropy": lambda t: (
        (True, t.CategoricalGeneralizedCrossEntropy) if isinstance(t, _HasMember00_31) else (False, None)
    ),
    "CircleLoss": lambda t: (True, t.CircleLoss) if isinstance(t, _HasMember00_32) else (False, None),
    "ColumnStack": lambda t: (True, t.ColumnStack) if isinstance(t, _HasMember00_33) else (False, None),
    "ConvGeneralDilated": lambda t: (True, t.ConvGeneralDilated) if isinstance(t, _HasMember00_34) else (False, None),
    "ConvGeneralDilatedLocal": lambda t: (
        (True, t.ConvGeneralDilatedLocal) if isinstance(t, _HasMember00_35) else (False, None)
    ),
    "ConvGeneralDilatedPatches": lambda t: (
        (True, t.ConvGeneralDilatedPatches) if isinstance(t, _HasMember00_36) else (False, None)
    ),
    "ConvTranspose": lambda t: (True, t.ConvTranspose) if isinstance(t, _HasMember00_37) else (False, None),
    "ConvWithGeneralPadding": lambda t: (
        (True, t.ConvWithGeneralPadding) if isinstance(t, _HasMember00_38) else (False, None)
    ),
    "Convolve2d": lambda t: (True, t.Convolve2d) if isinstance(t, _HasMember00_39) else (False, None),
    "CookieJar": lambda t: (True, t.CookieJar) if isinstance(t, _HasMember00_40) else (False, None),
    "CppGenerator": lambda t: (True, t.CppGenerator) if isinstance(t, _HasMember00_41) else (False, None),
    "CppProfiler": lambda t: (True, t.CppProfiler) if isinstance(t, _HasMember00_42) else (False, None),
    "CreateToken": lambda t: (True, t.CreateToken) if isinstance(t, _HasMember00_43) else (False, None),
    "CtcLoss": lambda t: (True, t.CtcLoss) if isinstance(t, _HasMember00_44) else (False, None),
}
