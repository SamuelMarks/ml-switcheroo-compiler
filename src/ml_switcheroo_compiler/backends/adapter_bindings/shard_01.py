"""Static structural protocol bindings shard 01 (CupyProfiler..Fliplr)."""

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
class _HasMember01_00(typing.Protocol):
    """Structural protocol for objects exposing `CupyProfiler`."""

    CupyProfiler: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_01(typing.Protocol):
    """Structural protocol for objects exposing `CustomLinearSolve`."""

    CustomLinearSolve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_02(typing.Protocol):
    """Structural protocol for objects exposing `CustomRoot`."""

    CustomRoot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_03(typing.Protocol):
    """Structural protocol for objects exposing `DTypeHolderProtocol`."""

    DTypeHolderProtocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_04(typing.Protocol):
    """Structural protocol for objects exposing `DaskProfiler`."""

    DaskProfiler: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_05(typing.Protocol):
    """Structural protocol for objects exposing `DataHolderProtocol`."""

    DataHolderProtocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_06(typing.Protocol):
    """Structural protocol for objects exposing `DebugInfs`."""

    DebugInfs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_07(typing.Protocol):
    """Structural protocol for objects exposing `DebugNans`."""

    DebugNans: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_08(typing.Protocol):
    """Structural protocol for objects exposing `DecodeBase64`."""

    DecodeBase64: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_09(typing.Protocol):
    """Structural protocol for objects exposing `DecodeCsv`."""

    DecodeCsv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_10(typing.Protocol):
    """Structural protocol for objects exposing `DecodeImage`."""

    DecodeImage: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_11(typing.Protocol):
    """Structural protocol for objects exposing `Delete`."""

    Delete: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_12(typing.Protocol):
    """Structural protocol for objects exposing `Device`."""

    Device: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_13(typing.Protocol):
    """Structural protocol for objects exposing `DeviceHolderProtocol`."""

    DeviceHolderProtocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_14(typing.Protocol):
    """Structural protocol for objects exposing `DiagIndices`."""

    DiagIndices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_15(typing.Protocol):
    """Structural protocol for objects exposing `DiagIndicesFrom`."""

    DiagIndicesFrom: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_16(typing.Protocol):
    """Structural protocol for objects exposing `Diagflat`."""

    Diagflat: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_17(typing.Protocol):
    """Structural protocol for objects exposing `Dot`."""

    Dot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_18(typing.Protocol):
    """Structural protocol for objects exposing `DotGeneral`."""

    DotGeneral: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_19(typing.Protocol):
    """Structural protocol for objects exposing `Dsplit`."""

    Dsplit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_20(typing.Protocol):
    """Structural protocol for objects exposing `Dstack`."""

    Dstack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_21(typing.Protocol):
    """Structural protocol for objects exposing `DummyBackend`."""

    DummyBackend: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_22(typing.Protocol):
    """Structural protocol for objects exposing `EdgeProfiler`."""

    EdgeProfiler: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_23(typing.Protocol):
    """Structural protocol for objects exposing `Einsum`."""

    Einsum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_24(typing.Protocol):
    """Structural protocol for objects exposing `ElasticTransform`."""

    ElasticTransform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_25(typing.Protocol):
    """Structural protocol for objects exposing `EncodeBase64`."""

    EncodeBase64: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_26(typing.Protocol):
    """Structural protocol for objects exposing `Environment`."""

    Environment: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_27(typing.Protocol):
    """Structural protocol for objects exposing `Equalization`."""

    Equalization: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_28(typing.Protocol):
    """Structural protocol for objects exposing `Event`."""

    Event: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_29(typing.Protocol):
    """Structural protocol for objects exposing `ExpandDims`."""

    ExpandDims: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_30(typing.Protocol):
    """Structural protocol for objects exposing `ExportArchive`."""

    ExportArchive: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_31(typing.Protocol):
    """Structural protocol for objects exposing `ExtractBoundingBoxes`."""

    ExtractBoundingBoxes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_32(typing.Protocol):
    """Structural protocol for objects exposing `ExtractPatchesOptions`."""

    ExtractPatchesOptions: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_33(typing.Protocol):
    """Structural protocol for objects exposing `FallbackBackend`."""

    FallbackBackend: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_34(typing.Protocol):
    """Structural protocol for objects exposing `Fft`."""

    Fft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_35(typing.Protocol):
    """Structural protocol for objects exposing `Fft2`."""

    Fft2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_36(typing.Protocol):
    """Structural protocol for objects exposing `Fftconvolve`."""

    Fftconvolve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_37(typing.Protocol):
    """Structural protocol for objects exposing `Fftfreq`."""

    Fftfreq: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_38(typing.Protocol):
    """Structural protocol for objects exposing `Fftn`."""

    Fftn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_39(typing.Protocol):
    """Structural protocol for objects exposing `Fftnd`."""

    Fftnd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_40(typing.Protocol):
    """Structural protocol for objects exposing `Fftshift`."""

    Fftshift: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_41(typing.Protocol):
    """Structural protocol for objects exposing `FillDiagonal`."""

    FillDiagonal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_42(typing.Protocol):
    """Structural protocol for objects exposing `FilterState`."""

    FilterState: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_43(typing.Protocol):
    """Structural protocol for objects exposing `Flip`."""

    Flip: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember01_44(typing.Protocol):
    """Structural protocol for objects exposing `Fliplr`."""

    Fliplr: EagerTensorProtocol


SHARD_01_READERS: dict[str, MemberReader] = {
    "CupyProfiler": lambda t: (True, t.CupyProfiler) if isinstance(t, _HasMember01_00) else (False, None),
    "CustomLinearSolve": lambda t: (True, t.CustomLinearSolve) if isinstance(t, _HasMember01_01) else (False, None),
    "CustomRoot": lambda t: (True, t.CustomRoot) if isinstance(t, _HasMember01_02) else (False, None),
    "DTypeHolderProtocol": lambda t: (True, t.DTypeHolderProtocol) if isinstance(t, _HasMember01_03) else (False, None),
    "DaskProfiler": lambda t: (True, t.DaskProfiler) if isinstance(t, _HasMember01_04) else (False, None),
    "DataHolderProtocol": lambda t: (True, t.DataHolderProtocol) if isinstance(t, _HasMember01_05) else (False, None),
    "DebugInfs": lambda t: (True, t.DebugInfs) if isinstance(t, _HasMember01_06) else (False, None),
    "DebugNans": lambda t: (True, t.DebugNans) if isinstance(t, _HasMember01_07) else (False, None),
    "DecodeBase64": lambda t: (True, t.DecodeBase64) if isinstance(t, _HasMember01_08) else (False, None),
    "DecodeCsv": lambda t: (True, t.DecodeCsv) if isinstance(t, _HasMember01_09) else (False, None),
    "DecodeImage": lambda t: (True, t.DecodeImage) if isinstance(t, _HasMember01_10) else (False, None),
    "Delete": lambda t: (True, t.Delete) if isinstance(t, _HasMember01_11) else (False, None),
    "Device": lambda t: (True, t.Device) if isinstance(t, _HasMember01_12) else (False, None),
    "DeviceHolderProtocol": lambda t: (
        (True, t.DeviceHolderProtocol) if isinstance(t, _HasMember01_13) else (False, None)
    ),
    "DiagIndices": lambda t: (True, t.DiagIndices) if isinstance(t, _HasMember01_14) else (False, None),
    "DiagIndicesFrom": lambda t: (True, t.DiagIndicesFrom) if isinstance(t, _HasMember01_15) else (False, None),
    "Diagflat": lambda t: (True, t.Diagflat) if isinstance(t, _HasMember01_16) else (False, None),
    "Dot": lambda t: (True, t.Dot) if isinstance(t, _HasMember01_17) else (False, None),
    "DotGeneral": lambda t: (True, t.DotGeneral) if isinstance(t, _HasMember01_18) else (False, None),
    "Dsplit": lambda t: (True, t.Dsplit) if isinstance(t, _HasMember01_19) else (False, None),
    "Dstack": lambda t: (True, t.Dstack) if isinstance(t, _HasMember01_20) else (False, None),
    "DummyBackend": lambda t: (True, t.DummyBackend) if isinstance(t, _HasMember01_21) else (False, None),
    "EdgeProfiler": lambda t: (True, t.EdgeProfiler) if isinstance(t, _HasMember01_22) else (False, None),
    "Einsum": lambda t: (True, t.Einsum) if isinstance(t, _HasMember01_23) else (False, None),
    "ElasticTransform": lambda t: (True, t.ElasticTransform) if isinstance(t, _HasMember01_24) else (False, None),
    "EncodeBase64": lambda t: (True, t.EncodeBase64) if isinstance(t, _HasMember01_25) else (False, None),
    "Environment": lambda t: (True, t.Environment) if isinstance(t, _HasMember01_26) else (False, None),
    "Equalization": lambda t: (True, t.Equalization) if isinstance(t, _HasMember01_27) else (False, None),
    "Event": lambda t: (True, t.Event) if isinstance(t, _HasMember01_28) else (False, None),
    "ExpandDims": lambda t: (True, t.ExpandDims) if isinstance(t, _HasMember01_29) else (False, None),
    "ExportArchive": lambda t: (True, t.ExportArchive) if isinstance(t, _HasMember01_30) else (False, None),
    "ExtractBoundingBoxes": lambda t: (
        (True, t.ExtractBoundingBoxes) if isinstance(t, _HasMember01_31) else (False, None)
    ),
    "ExtractPatchesOptions": lambda t: (
        (True, t.ExtractPatchesOptions) if isinstance(t, _HasMember01_32) else (False, None)
    ),
    "FallbackBackend": lambda t: (True, t.FallbackBackend) if isinstance(t, _HasMember01_33) else (False, None),
    "Fft": lambda t: (True, t.Fft) if isinstance(t, _HasMember01_34) else (False, None),
    "Fft2": lambda t: (True, t.Fft2) if isinstance(t, _HasMember01_35) else (False, None),
    "Fftconvolve": lambda t: (True, t.Fftconvolve) if isinstance(t, _HasMember01_36) else (False, None),
    "Fftfreq": lambda t: (True, t.Fftfreq) if isinstance(t, _HasMember01_37) else (False, None),
    "Fftn": lambda t: (True, t.Fftn) if isinstance(t, _HasMember01_38) else (False, None),
    "Fftnd": lambda t: (True, t.Fftnd) if isinstance(t, _HasMember01_39) else (False, None),
    "Fftshift": lambda t: (True, t.Fftshift) if isinstance(t, _HasMember01_40) else (False, None),
    "FillDiagonal": lambda t: (True, t.FillDiagonal) if isinstance(t, _HasMember01_41) else (False, None),
    "FilterState": lambda t: (True, t.FilterState) if isinstance(t, _HasMember01_42) else (False, None),
    "Flip": lambda t: (True, t.Flip) if isinstance(t, _HasMember01_43) else (False, None),
    "Fliplr": lambda t: (True, t.Fliplr) if isinstance(t, _HasMember01_44) else (False, None),
}
