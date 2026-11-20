"""Static structural protocol bindings shard 03 (LinearOperatorOperandProtocol..PyTorchProfiler)."""

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
class _HasMember03_00(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorOperandProtocol`."""

    LinearOperatorOperandProtocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_01(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorPermutation`."""

    LinearOperatorPermutation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_02(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorScaledIdentity`."""

    LinearOperatorScaledIdentity: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_03(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorToeplitz`."""

    LinearOperatorToeplitz: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_04(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorTridiag`."""

    LinearOperatorTridiag: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_05(typing.Protocol):
    """Structural protocol for objects exposing `LinearOperatorZeros`."""

    LinearOperatorZeros: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_06(typing.Protocol):
    """Structural protocol for objects exposing `Load`."""

    Load: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_07(typing.Protocol):
    """Structural protocol for objects exposing `LogPoissonLoss`."""

    LogPoissonLoss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_08(typing.Protocol):
    """Structural protocol for objects exposing `MLIRBytecodeEncoder`."""

    MLIRBytecodeEncoder: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_09(typing.Protocol):
    """Structural protocol for objects exposing `MLXProfiler`."""

    MLXProfiler: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_10(typing.Protocol):
    """Structural protocol for objects exposing `MTLCreateSystemDefaultDevice`."""

    MTLCreateSystemDefaultDevice: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_11(typing.Protocol):
    """Structural protocol for objects exposing `MY_VAR`."""

    MY_VAR: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_12(typing.Protocol):
    """Structural protocol for objects exposing `MapFlatValues`."""

    MapFlatValues: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_13(typing.Protocol):
    """Structural protocol for objects exposing `Matmul`."""

    Matmul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_14(typing.Protocol):
    """Structural protocol for objects exposing `MatrixRank`."""

    MatrixRank: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_15(typing.Protocol):
    """Structural protocol for objects exposing `MatrixTranspose`."""

    MatrixTranspose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_16(typing.Protocol):
    """Structural protocol for objects exposing `MockDescriptiveFunc`."""

    MockDescriptiveFunc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_17(typing.Protocol):
    """Structural protocol for objects exposing `MockDistributionsFunc`."""

    MockDistributionsFunc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_18(typing.Protocol):
    """Structural protocol for objects exposing `Module`."""

    Module: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_19(typing.Protocol):
    """Structural protocol for objects exposing `Moveaxis`."""

    Moveaxis: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_20(typing.Protocol):
    """Structural protocol for objects exposing `Nms`."""

    Nms: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_21(typing.Protocol):
    """Structural protocol for objects exposing `NonExistentOp`."""

    NonExistentOp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_22(typing.Protocol):
    """Structural protocol for objects exposing `NonExistentOp123`."""

    NonExistentOp123: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_23(typing.Protocol):
    """Structural protocol for objects exposing `NonblockingLoad`."""

    NonblockingLoad: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_24(typing.Protocol):
    """Structural protocol for objects exposing `NonblockingSave`."""

    NonblockingSave: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_25(typing.Protocol):
    """Structural protocol for objects exposing `NormConfig`."""

    NormConfig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_26(typing.Protocol):
    """Structural protocol for objects exposing `Normalization`."""

    Normalization: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_27(typing.Protocol):
    """Structural protocol for objects exposing `NumbaGenerator`."""

    NumbaGenerator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_28(typing.Protocol):
    """Structural protocol for objects exposing `NumbaProfiler`."""

    NumbaProfiler: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_29(typing.Protocol):
    """Structural protocol for objects exposing `NumpyProfiler`."""

    NumpyProfiler: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_30(typing.Protocol):
    """Structural protocol for objects exposing `ONNXCodeGenerator`."""

    ONNXCodeGenerator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_31(typing.Protocol):
    """Structural protocol for objects exposing `Op1`."""

    Op1: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_32(typing.Protocol):
    """Structural protocol for objects exposing `OpDef`."""

    OpDef: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_33(typing.Protocol):
    """Structural protocol for objects exposing `OutputShapeHolderProtocol`."""

    OutputShapeHolderProtocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_34(typing.Protocol):
    """Structural protocol for objects exposing `ParseTensor`."""

    ParseTensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_35(typing.Protocol):
    """Structural protocol for objects exposing `PassManager`."""

    PassManager: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_36(typing.Protocol):
    """Structural protocol for objects exposing `Pdot`."""

    Pdot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_37(typing.Protocol):
    """Structural protocol for objects exposing `Permute`."""

    Permute: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_38(typing.Protocol):
    """Structural protocol for objects exposing `PerspectiveTransform`."""

    PerspectiveTransform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_39(typing.Protocol):
    """Structural protocol for objects exposing `Pmean`."""

    Pmean: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_40(typing.Protocol):
    """Structural protocol for objects exposing `Posterize`."""

    Posterize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_41(typing.Protocol):
    """Structural protocol for objects exposing `Psum`."""

    Psum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_42(typing.Protocol):
    """Structural protocol for objects exposing `PurePythonGenerator`."""

    PurePythonGenerator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_43(typing.Protocol):
    """Structural protocol for objects exposing `PurePythonTensor`."""

    PurePythonTensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember03_44(typing.Protocol):
    """Structural protocol for objects exposing `PyTorchProfiler`."""

    PyTorchProfiler: EagerTensorProtocol


SHARD_03_READERS: dict[str, MemberReader] = {
    "LinearOperatorOperandProtocol": lambda t: (
        (True, t.LinearOperatorOperandProtocol) if isinstance(t, _HasMember03_00) else (False, None)
    ),
    "LinearOperatorPermutation": lambda t: (
        (True, t.LinearOperatorPermutation) if isinstance(t, _HasMember03_01) else (False, None)
    ),
    "LinearOperatorScaledIdentity": lambda t: (
        (True, t.LinearOperatorScaledIdentity) if isinstance(t, _HasMember03_02) else (False, None)
    ),
    "LinearOperatorToeplitz": lambda t: (
        (True, t.LinearOperatorToeplitz) if isinstance(t, _HasMember03_03) else (False, None)
    ),
    "LinearOperatorTridiag": lambda t: (
        (True, t.LinearOperatorTridiag) if isinstance(t, _HasMember03_04) else (False, None)
    ),
    "LinearOperatorZeros": lambda t: (True, t.LinearOperatorZeros) if isinstance(t, _HasMember03_05) else (False, None),
    "Load": lambda t: (True, t.Load) if isinstance(t, _HasMember03_06) else (False, None),
    "LogPoissonLoss": lambda t: (True, t.LogPoissonLoss) if isinstance(t, _HasMember03_07) else (False, None),
    "MLIRBytecodeEncoder": lambda t: (True, t.MLIRBytecodeEncoder) if isinstance(t, _HasMember03_08) else (False, None),
    "MLXProfiler": lambda t: (True, t.MLXProfiler) if isinstance(t, _HasMember03_09) else (False, None),
    "MTLCreateSystemDefaultDevice": lambda t: (
        (True, t.MTLCreateSystemDefaultDevice) if isinstance(t, _HasMember03_10) else (False, None)
    ),
    "MY_VAR": lambda t: (True, t.MY_VAR) if isinstance(t, _HasMember03_11) else (False, None),
    "MapFlatValues": lambda t: (True, t.MapFlatValues) if isinstance(t, _HasMember03_12) else (False, None),
    "Matmul": lambda t: (True, t.Matmul) if isinstance(t, _HasMember03_13) else (False, None),
    "MatrixRank": lambda t: (True, t.MatrixRank) if isinstance(t, _HasMember03_14) else (False, None),
    "MatrixTranspose": lambda t: (True, t.MatrixTranspose) if isinstance(t, _HasMember03_15) else (False, None),
    "MockDescriptiveFunc": lambda t: (True, t.MockDescriptiveFunc) if isinstance(t, _HasMember03_16) else (False, None),
    "MockDistributionsFunc": lambda t: (
        (True, t.MockDistributionsFunc) if isinstance(t, _HasMember03_17) else (False, None)
    ),
    "Module": lambda t: (True, t.Module) if isinstance(t, _HasMember03_18) else (False, None),
    "Moveaxis": lambda t: (True, t.Moveaxis) if isinstance(t, _HasMember03_19) else (False, None),
    "Nms": lambda t: (True, t.Nms) if isinstance(t, _HasMember03_20) else (False, None),
    "NonExistentOp": lambda t: (True, t.NonExistentOp) if isinstance(t, _HasMember03_21) else (False, None),
    "NonExistentOp123": lambda t: (True, t.NonExistentOp123) if isinstance(t, _HasMember03_22) else (False, None),
    "NonblockingLoad": lambda t: (True, t.NonblockingLoad) if isinstance(t, _HasMember03_23) else (False, None),
    "NonblockingSave": lambda t: (True, t.NonblockingSave) if isinstance(t, _HasMember03_24) else (False, None),
    "NormConfig": lambda t: (True, t.NormConfig) if isinstance(t, _HasMember03_25) else (False, None),
    "Normalization": lambda t: (True, t.Normalization) if isinstance(t, _HasMember03_26) else (False, None),
    "NumbaGenerator": lambda t: (True, t.NumbaGenerator) if isinstance(t, _HasMember03_27) else (False, None),
    "NumbaProfiler": lambda t: (True, t.NumbaProfiler) if isinstance(t, _HasMember03_28) else (False, None),
    "NumpyProfiler": lambda t: (True, t.NumpyProfiler) if isinstance(t, _HasMember03_29) else (False, None),
    "ONNXCodeGenerator": lambda t: (True, t.ONNXCodeGenerator) if isinstance(t, _HasMember03_30) else (False, None),
    "Op1": lambda t: (True, t.Op1) if isinstance(t, _HasMember03_31) else (False, None),
    "OpDef": lambda t: (True, t.OpDef) if isinstance(t, _HasMember03_32) else (False, None),
    "OutputShapeHolderProtocol": lambda t: (
        (True, t.OutputShapeHolderProtocol) if isinstance(t, _HasMember03_33) else (False, None)
    ),
    "ParseTensor": lambda t: (True, t.ParseTensor) if isinstance(t, _HasMember03_34) else (False, None),
    "PassManager": lambda t: (True, t.PassManager) if isinstance(t, _HasMember03_35) else (False, None),
    "Pdot": lambda t: (True, t.Pdot) if isinstance(t, _HasMember03_36) else (False, None),
    "Permute": lambda t: (True, t.Permute) if isinstance(t, _HasMember03_37) else (False, None),
    "PerspectiveTransform": lambda t: (
        (True, t.PerspectiveTransform) if isinstance(t, _HasMember03_38) else (False, None)
    ),
    "Pmean": lambda t: (True, t.Pmean) if isinstance(t, _HasMember03_39) else (False, None),
    "Posterize": lambda t: (True, t.Posterize) if isinstance(t, _HasMember03_40) else (False, None),
    "Psum": lambda t: (True, t.Psum) if isinstance(t, _HasMember03_41) else (False, None),
    "PurePythonGenerator": lambda t: (True, t.PurePythonGenerator) if isinstance(t, _HasMember03_42) else (False, None),
    "PurePythonTensor": lambda t: (True, t.PurePythonTensor) if isinstance(t, _HasMember03_43) else (False, None),
    "PyTorchProfiler": lambda t: (True, t.PyTorchProfiler) if isinstance(t, _HasMember03_44) else (False, None),
}
