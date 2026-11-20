"""Static structural protocol bindings shard 56 (visit_Send..visit_Vmap)."""

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
class _HasMember56_00(typing.Protocol):
    """Structural protocol for objects exposing `visit_Send`."""

    visit_Send: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_01(typing.Protocol):
    """Structural protocol for objects exposing `visit_SharedMemoryTreeReduce`."""

    visit_SharedMemoryTreeReduce: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_02(typing.Protocol):
    """Structural protocol for objects exposing `visit_Sigmoid`."""

    visit_Sigmoid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_03(typing.Protocol):
    """Structural protocol for objects exposing `visit_Sin`."""

    visit_Sin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_04(typing.Protocol):
    """Structural protocol for objects exposing `visit_Slice`."""

    visit_Slice: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_05(typing.Protocol):
    """Structural protocol for objects exposing `visit_Softmax`."""

    visit_Softmax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_06(typing.Protocol):
    """Structural protocol for objects exposing `visit_SparseDropout`."""

    visit_SparseDropout: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_07(typing.Protocol):
    """Structural protocol for objects exposing `visit_SparseMask`."""

    visit_SparseMask: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_08(typing.Protocol):
    """Structural protocol for objects exposing `visit_SparseSoftmax`."""

    visit_SparseSoftmax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_09(typing.Protocol):
    """Structural protocol for objects exposing `visit_Spgemm`."""

    visit_Spgemm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_10(typing.Protocol):
    """Structural protocol for objects exposing `visit_Spmm`."""

    visit_Spmm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_11(typing.Protocol):
    """Structural protocol for objects exposing `visit_Sqrt`."""

    visit_Sqrt: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_12(typing.Protocol):
    """Structural protocol for objects exposing `visit_Stack`."""

    visit_Stack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_13(typing.Protocol):
    """Structural protocol for objects exposing `visit_Sub`."""

    visit_Sub: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_14(typing.Protocol):
    """Structural protocol for objects exposing `visit_SubclassOp`."""

    visit_SubclassOp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_15(typing.Protocol):
    """Structural protocol for objects exposing `visit_Subscript`."""

    visit_Subscript: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_16(typing.Protocol):
    """Structural protocol for objects exposing `visit_Sum`."""

    visit_Sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_17(typing.Protocol):
    """Structural protocol for objects exposing `visit_Svd`."""

    visit_Svd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_18(typing.Protocol):
    """Structural protocol for objects exposing `visit_SvdVjp`."""

    visit_SvdVjp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_19(typing.Protocol):
    """Structural protocol for objects exposing `visit_Switch`."""

    visit_Switch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_20(typing.Protocol):
    """Structural protocol for objects exposing `visit_Tanh`."""

    visit_Tanh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_21(typing.Protocol):
    """Structural protocol for objects exposing `visit_TensorScatterAdd`."""

    visit_TensorScatterAdd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_22(typing.Protocol):
    """Structural protocol for objects exposing `visit_TensorScatterMax`."""

    visit_TensorScatterMax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_23(typing.Protocol):
    """Structural protocol for objects exposing `visit_TensorScatterMin`."""

    visit_TensorScatterMin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_24(typing.Protocol):
    """Structural protocol for objects exposing `visit_TensorScatterUpdate`."""

    visit_TensorScatterUpdate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_25(typing.Protocol):
    """Structural protocol for objects exposing `visit_TestOp`."""

    visit_TestOp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_26(typing.Protocol):
    """Structural protocol for objects exposing `visit_Tile`."""

    visit_Tile: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_27(typing.Protocol):
    """Structural protocol for objects exposing `visit_TiledMatMul2D`."""

    visit_TiledMatMul2D: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_28(typing.Protocol):
    """Structural protocol for objects exposing `visit_TimeDistributed`."""

    visit_TimeDistributed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_29(typing.Protocol):
    """Structural protocol for objects exposing `visit_ToRegular`."""

    visit_ToRegular: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_30(typing.Protocol):
    """Structural protocol for objects exposing `visit_TopK`."""

    visit_TopK: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_31(typing.Protocol):
    """Structural protocol for objects exposing `visit_Transpose`."""

    visit_Transpose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_32(typing.Protocol):
    """Structural protocol for objects exposing `visit_TriInv`."""

    visit_TriInv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_33(typing.Protocol):
    """Structural protocol for objects exposing `visit_Trig`."""

    visit_Trig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_34(typing.Protocol):
    """Structural protocol for objects exposing `visit_TruncateDiv`."""

    visit_TruncateDiv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_35(typing.Protocol):
    """Structural protocol for objects exposing `visit_TruncateMod`."""

    visit_TruncateMod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_36(typing.Protocol):
    """Structural protocol for objects exposing `visit_Try`."""

    visit_Try: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_37(typing.Protocol):
    """Structural protocol for objects exposing `visit_Unflatten`."""

    visit_Unflatten: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_38(typing.Protocol):
    """Structural protocol for objects exposing `visit_UnsortedSegmentMax`."""

    visit_UnsortedSegmentMax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_39(typing.Protocol):
    """Structural protocol for objects exposing `visit_UnsortedSegmentMin`."""

    visit_UnsortedSegmentMin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_40(typing.Protocol):
    """Structural protocol for objects exposing `visit_UnsortedSegmentProd`."""

    visit_UnsortedSegmentProd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_41(typing.Protocol):
    """Structural protocol for objects exposing `visit_UnsortedSegmentSum`."""

    visit_UnsortedSegmentSum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_42(typing.Protocol):
    """Structural protocol for objects exposing `visit_Unzip`."""

    visit_Unzip: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_43(typing.Protocol):
    """Structural protocol for objects exposing `visit_VisitorOp`."""

    visit_VisitorOp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember56_44(typing.Protocol):
    """Structural protocol for objects exposing `visit_Vmap`."""

    visit_Vmap: EagerTensorProtocol


SHARD_56_READERS: dict[str, MemberReader] = {
    "visit_Send": lambda t: (True, t.visit_Send) if isinstance(t, _HasMember56_00) else (False, None),
    "visit_SharedMemoryTreeReduce": lambda t: (
        (True, t.visit_SharedMemoryTreeReduce) if isinstance(t, _HasMember56_01) else (False, None)
    ),
    "visit_Sigmoid": lambda t: (True, t.visit_Sigmoid) if isinstance(t, _HasMember56_02) else (False, None),
    "visit_Sin": lambda t: (True, t.visit_Sin) if isinstance(t, _HasMember56_03) else (False, None),
    "visit_Slice": lambda t: (True, t.visit_Slice) if isinstance(t, _HasMember56_04) else (False, None),
    "visit_Softmax": lambda t: (True, t.visit_Softmax) if isinstance(t, _HasMember56_05) else (False, None),
    "visit_SparseDropout": lambda t: (True, t.visit_SparseDropout) if isinstance(t, _HasMember56_06) else (False, None),
    "visit_SparseMask": lambda t: (True, t.visit_SparseMask) if isinstance(t, _HasMember56_07) else (False, None),
    "visit_SparseSoftmax": lambda t: (True, t.visit_SparseSoftmax) if isinstance(t, _HasMember56_08) else (False, None),
    "visit_Spgemm": lambda t: (True, t.visit_Spgemm) if isinstance(t, _HasMember56_09) else (False, None),
    "visit_Spmm": lambda t: (True, t.visit_Spmm) if isinstance(t, _HasMember56_10) else (False, None),
    "visit_Sqrt": lambda t: (True, t.visit_Sqrt) if isinstance(t, _HasMember56_11) else (False, None),
    "visit_Stack": lambda t: (True, t.visit_Stack) if isinstance(t, _HasMember56_12) else (False, None),
    "visit_Sub": lambda t: (True, t.visit_Sub) if isinstance(t, _HasMember56_13) else (False, None),
    "visit_SubclassOp": lambda t: (True, t.visit_SubclassOp) if isinstance(t, _HasMember56_14) else (False, None),
    "visit_Subscript": lambda t: (True, t.visit_Subscript) if isinstance(t, _HasMember56_15) else (False, None),
    "visit_Sum": lambda t: (True, t.visit_Sum) if isinstance(t, _HasMember56_16) else (False, None),
    "visit_Svd": lambda t: (True, t.visit_Svd) if isinstance(t, _HasMember56_17) else (False, None),
    "visit_SvdVjp": lambda t: (True, t.visit_SvdVjp) if isinstance(t, _HasMember56_18) else (False, None),
    "visit_Switch": lambda t: (True, t.visit_Switch) if isinstance(t, _HasMember56_19) else (False, None),
    "visit_Tanh": lambda t: (True, t.visit_Tanh) if isinstance(t, _HasMember56_20) else (False, None),
    "visit_TensorScatterAdd": lambda t: (
        (True, t.visit_TensorScatterAdd) if isinstance(t, _HasMember56_21) else (False, None)
    ),
    "visit_TensorScatterMax": lambda t: (
        (True, t.visit_TensorScatterMax) if isinstance(t, _HasMember56_22) else (False, None)
    ),
    "visit_TensorScatterMin": lambda t: (
        (True, t.visit_TensorScatterMin) if isinstance(t, _HasMember56_23) else (False, None)
    ),
    "visit_TensorScatterUpdate": lambda t: (
        (True, t.visit_TensorScatterUpdate) if isinstance(t, _HasMember56_24) else (False, None)
    ),
    "visit_TestOp": lambda t: (True, t.visit_TestOp) if isinstance(t, _HasMember56_25) else (False, None),
    "visit_Tile": lambda t: (True, t.visit_Tile) if isinstance(t, _HasMember56_26) else (False, None),
    "visit_TiledMatMul2D": lambda t: (True, t.visit_TiledMatMul2D) if isinstance(t, _HasMember56_27) else (False, None),
    "visit_TimeDistributed": lambda t: (
        (True, t.visit_TimeDistributed) if isinstance(t, _HasMember56_28) else (False, None)
    ),
    "visit_ToRegular": lambda t: (True, t.visit_ToRegular) if isinstance(t, _HasMember56_29) else (False, None),
    "visit_TopK": lambda t: (True, t.visit_TopK) if isinstance(t, _HasMember56_30) else (False, None),
    "visit_Transpose": lambda t: (True, t.visit_Transpose) if isinstance(t, _HasMember56_31) else (False, None),
    "visit_TriInv": lambda t: (True, t.visit_TriInv) if isinstance(t, _HasMember56_32) else (False, None),
    "visit_Trig": lambda t: (True, t.visit_Trig) if isinstance(t, _HasMember56_33) else (False, None),
    "visit_TruncateDiv": lambda t: (True, t.visit_TruncateDiv) if isinstance(t, _HasMember56_34) else (False, None),
    "visit_TruncateMod": lambda t: (True, t.visit_TruncateMod) if isinstance(t, _HasMember56_35) else (False, None),
    "visit_Try": lambda t: (True, t.visit_Try) if isinstance(t, _HasMember56_36) else (False, None),
    "visit_Unflatten": lambda t: (True, t.visit_Unflatten) if isinstance(t, _HasMember56_37) else (False, None),
    "visit_UnsortedSegmentMax": lambda t: (
        (True, t.visit_UnsortedSegmentMax) if isinstance(t, _HasMember56_38) else (False, None)
    ),
    "visit_UnsortedSegmentMin": lambda t: (
        (True, t.visit_UnsortedSegmentMin) if isinstance(t, _HasMember56_39) else (False, None)
    ),
    "visit_UnsortedSegmentProd": lambda t: (
        (True, t.visit_UnsortedSegmentProd) if isinstance(t, _HasMember56_40) else (False, None)
    ),
    "visit_UnsortedSegmentSum": lambda t: (
        (True, t.visit_UnsortedSegmentSum) if isinstance(t, _HasMember56_41) else (False, None)
    ),
    "visit_Unzip": lambda t: (True, t.visit_Unzip) if isinstance(t, _HasMember56_42) else (False, None),
    "visit_VisitorOp": lambda t: (True, t.visit_VisitorOp) if isinstance(t, _HasMember56_43) else (False, None),
    "visit_Vmap": lambda t: (True, t.visit_Vmap) if isinstance(t, _HasMember56_44) else (False, None),
}
