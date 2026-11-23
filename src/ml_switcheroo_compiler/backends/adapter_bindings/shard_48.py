"""Static structural protocol bindings shard 48 (subtract..tensorscatterupdate)."""

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
class _HasMember48_00(typing.Protocol):
    """Structural protocol for objects exposing `subtract`."""

    subtract: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_01(typing.Protocol):
    """Structural protocol for objects exposing `suggest_closest_symbol`."""

    suggest_closest_symbol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_02(typing.Protocol):
    """Structural protocol for objects exposing `sum`."""

    sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_03(typing.Protocol):
    """Structural protocol for objects exposing `svd`."""

    svd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_04(typing.Protocol):
    """Structural protocol for objects exposing `svdvals`."""

    svdvals: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_05(typing.Protocol):
    """Structural protocol for objects exposing `swapaxes`."""

    swapaxes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_06(typing.Protocol):
    """Structural protocol for objects exposing `switch`."""

    switch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_07(typing.Protocol):
    """Structural protocol for objects exposing `switch_case`."""

    switch_case: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_08(typing.Protocol):
    """Structural protocol for objects exposing `switch_op`."""

    switch_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_09(typing.Protocol):
    """Structural protocol for objects exposing `switchop`."""

    switchop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_10(typing.Protocol):
    """Structural protocol for objects exposing `sycl_device`."""

    sycl_device: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_11(typing.Protocol):
    """Structural protocol for objects exposing `sycl_queue`."""

    sycl_queue: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_12(typing.Protocol):
    """Structural protocol for objects exposing `symbolic_add`."""

    symbolic_add: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_13(typing.Protocol):
    """Structural protocol for objects exposing `sync_gradients_hook`."""

    sync_gradients_hook: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_14(typing.Protocol):
    """Structural protocol for objects exposing `synchronize`."""

    synchronize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_15(typing.Protocol):
    """Structural protocol for objects exposing `t`."""

    t: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_16(typing.Protocol):
    """Structural protocol for objects exposing `take`."""

    take: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_17(typing.Protocol):
    """Structural protocol for objects exposing `take_along_axis`."""

    take_along_axis: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_18(typing.Protocol):
    """Structural protocol for objects exposing `tan`."""

    tan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_19(typing.Protocol):
    """Structural protocol for objects exposing `tanh`."""

    tanh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_20(typing.Protocol):
    """Structural protocol for objects exposing `target`."""

    target: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_21(typing.Protocol):
    """Structural protocol for objects exposing `target_fn`."""

    target_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_22(typing.Protocol):
    """Structural protocol for objects exposing `target_idx`."""

    target_idx: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_23(typing.Protocol):
    """Structural protocol for objects exposing `target_node`."""

    target_node: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_24(typing.Protocol):
    """Structural protocol for objects exposing `tensor`."""

    tensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_25(typing.Protocol):
    """Structural protocol for objects exposing `tensor_array`."""

    tensor_array: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_26(typing.Protocol):
    """Structural protocol for objects exposing `tensor_array_operand_protocol`."""

    tensor_array_operand_protocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_27(typing.Protocol):
    """Structural protocol for objects exposing `tensor_array_read`."""

    tensor_array_read: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_28(typing.Protocol):
    """Structural protocol for objects exposing `tensor_array_stack`."""

    tensor_array_stack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_29(typing.Protocol):
    """Structural protocol for objects exposing `tensor_array_write`."""

    tensor_array_write: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_30(typing.Protocol):
    """Structural protocol for objects exposing `tensor_diag`."""

    tensor_diag: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_31(typing.Protocol):
    """Structural protocol for objects exposing `tensor_diag_part`."""

    tensor_diag_part: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_32(typing.Protocol):
    """Structural protocol for objects exposing `tensor_fixtures`."""

    tensor_fixtures: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_33(typing.Protocol):
    """Structural protocol for objects exposing `tensor_operand_protocol`."""

    tensor_operand_protocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_34(typing.Protocol):
    """Structural protocol for objects exposing `tensor_scale_fn`."""

    tensor_scale_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_35(typing.Protocol):
    """Structural protocol for objects exposing `tensorarrayoperandprotocol`."""

    tensorarrayoperandprotocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_36(typing.Protocol):
    """Structural protocol for objects exposing `tensorarrayread`."""

    tensorarrayread: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_37(typing.Protocol):
    """Structural protocol for objects exposing `tensorarraystack`."""

    tensorarraystack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_38(typing.Protocol):
    """Structural protocol for objects exposing `tensorarraywrite`."""

    tensorarraywrite: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_39(typing.Protocol):
    """Structural protocol for objects exposing `tensordot`."""

    tensordot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_40(typing.Protocol):
    """Structural protocol for objects exposing `tensorflow`."""

    tensorflow: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_41(typing.Protocol):
    """Structural protocol for objects exposing `tensorinv`."""

    tensorinv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_42(typing.Protocol):
    """Structural protocol for objects exposing `tensoroperandprotocol`."""

    tensoroperandprotocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_43(typing.Protocol):
    """Structural protocol for objects exposing `tensorscattersub`."""

    tensorscattersub: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember48_44(typing.Protocol):
    """Structural protocol for objects exposing `tensorscatterupdate`."""

    tensorscatterupdate: EagerTensorProtocol


SHARD_48_READERS: dict[str, MemberReader] = {
    "subtract": lambda t: (True, t.subtract) if isinstance(t, _HasMember48_00) else (False, None),
    "suggest_closest_symbol": lambda t: (
        (True, t.suggest_closest_symbol) if isinstance(t, _HasMember48_01) else (False, None)
    ),
    "sum": lambda t: (True, t.sum) if isinstance(t, _HasMember48_02) else (False, None),
    "svd": lambda t: (True, t.svd) if isinstance(t, _HasMember48_03) else (False, None),
    "svdvals": lambda t: (True, t.svdvals) if isinstance(t, _HasMember48_04) else (False, None),
    "swapaxes": lambda t: (True, t.swapaxes) if isinstance(t, _HasMember48_05) else (False, None),
    "switch": lambda t: (True, t.switch) if isinstance(t, _HasMember48_06) else (False, None),
    "switch_case": lambda t: (True, t.switch_case) if isinstance(t, _HasMember48_07) else (False, None),
    "switch_op": lambda t: (True, t.switch_op) if isinstance(t, _HasMember48_08) else (False, None),
    "switchop": lambda t: (True, t.switchop) if isinstance(t, _HasMember48_09) else (False, None),
    "sycl_device": lambda t: (True, t.sycl_device) if isinstance(t, _HasMember48_10) else (False, None),
    "sycl_queue": lambda t: (True, t.sycl_queue) if isinstance(t, _HasMember48_11) else (False, None),
    "symbolic_add": lambda t: (True, t.symbolic_add) if isinstance(t, _HasMember48_12) else (False, None),
    "sync_gradients_hook": lambda t: (True, t.sync_gradients_hook) if isinstance(t, _HasMember48_13) else (False, None),
    "synchronize": lambda t: (True, t.synchronize) if isinstance(t, _HasMember48_14) else (False, None),
    "t": lambda t: (True, t.t) if isinstance(t, _HasMember48_15) else (False, None),
    "take": lambda t: (True, t.take) if isinstance(t, _HasMember48_16) else (False, None),
    "take_along_axis": lambda t: (True, t.take_along_axis) if isinstance(t, _HasMember48_17) else (False, None),
    "tan": lambda t: (True, t.tan) if isinstance(t, _HasMember48_18) else (False, None),
    "tanh": lambda t: (True, t.tanh) if isinstance(t, _HasMember48_19) else (False, None),
    "target": lambda t: (True, t.target) if isinstance(t, _HasMember48_20) else (False, None),
    "target_fn": lambda t: (True, t.target_fn) if isinstance(t, _HasMember48_21) else (False, None),
    "target_idx": lambda t: (True, t.target_idx) if isinstance(t, _HasMember48_22) else (False, None),
    "target_node": lambda t: (True, t.target_node) if isinstance(t, _HasMember48_23) else (False, None),
    "tensor": lambda t: (True, t.tensor) if isinstance(t, _HasMember48_24) else (False, None),
    "tensor_array": lambda t: (True, t.tensor_array) if isinstance(t, _HasMember48_25) else (False, None),
    "tensor_array_operand_protocol": lambda t: (
        (True, t.tensor_array_operand_protocol) if isinstance(t, _HasMember48_26) else (False, None)
    ),
    "tensor_array_read": lambda t: (True, t.tensor_array_read) if isinstance(t, _HasMember48_27) else (False, None),
    "tensor_array_stack": lambda t: (True, t.tensor_array_stack) if isinstance(t, _HasMember48_28) else (False, None),
    "tensor_array_write": lambda t: (True, t.tensor_array_write) if isinstance(t, _HasMember48_29) else (False, None),
    "tensor_diag": lambda t: (True, t.tensor_diag) if isinstance(t, _HasMember48_30) else (False, None),
    "tensor_diag_part": lambda t: (True, t.tensor_diag_part) if isinstance(t, _HasMember48_31) else (False, None),
    "tensor_fixtures": lambda t: (True, t.tensor_fixtures) if isinstance(t, _HasMember48_32) else (False, None),
    "tensor_operand_protocol": lambda t: (
        (True, t.tensor_operand_protocol) if isinstance(t, _HasMember48_33) else (False, None)
    ),
    "tensor_scale_fn": lambda t: (True, t.tensor_scale_fn) if isinstance(t, _HasMember48_34) else (False, None),
    "tensorarrayoperandprotocol": lambda t: (
        (True, t.tensorarrayoperandprotocol) if isinstance(t, _HasMember48_35) else (False, None)
    ),
    "tensorarrayread": lambda t: (True, t.tensorarrayread) if isinstance(t, _HasMember48_36) else (False, None),
    "tensorarraystack": lambda t: (True, t.tensorarraystack) if isinstance(t, _HasMember48_37) else (False, None),
    "tensorarraywrite": lambda t: (True, t.tensorarraywrite) if isinstance(t, _HasMember48_38) else (False, None),
    "tensordot": lambda t: (True, t.tensordot) if isinstance(t, _HasMember48_39) else (False, None),
    "tensorflow": lambda t: (True, t.tensorflow) if isinstance(t, _HasMember48_40) else (False, None),
    "tensorinv": lambda t: (True, t.tensorinv) if isinstance(t, _HasMember48_41) else (False, None),
    "tensoroperandprotocol": lambda t: (
        (True, t.tensoroperandprotocol) if isinstance(t, _HasMember48_42) else (False, None)
    ),
    "tensorscattersub": lambda t: (True, t.tensorscattersub) if isinstance(t, _HasMember48_43) else (False, None),
    "tensorscatterupdate": lambda t: (True, t.tensorscatterupdate) if isinstance(t, _HasMember48_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember48_00,
        _HasMember48_01,
        _HasMember48_02,
        _HasMember48_03,
        _HasMember48_04,
        _HasMember48_05,
        _HasMember48_06,
        _HasMember48_07,
        _HasMember48_08,
        _HasMember48_09,
        _HasMember48_10,
        _HasMember48_11,
        _HasMember48_12,
        _HasMember48_13,
        _HasMember48_14,
        _HasMember48_15,
        _HasMember48_16,
        _HasMember48_17,
        _HasMember48_18,
        _HasMember48_19,
        _HasMember48_20,
        _HasMember48_21,
        _HasMember48_22,
        _HasMember48_23,
        _HasMember48_24,
        _HasMember48_25,
        _HasMember48_26,
        _HasMember48_27,
        _HasMember48_28,
        _HasMember48_29,
        _HasMember48_30,
        _HasMember48_31,
        _HasMember48_32,
        _HasMember48_33,
        _HasMember48_34,
        _HasMember48_35,
        _HasMember48_36,
        _HasMember48_37,
        _HasMember48_38,
        _HasMember48_39,
        _HasMember48_40,
        _HasMember48_41,
        _HasMember48_42,
        _HasMember48_43,
        _HasMember48_44,
    )
)
