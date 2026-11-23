"""Static structural protocol bindings shard 31 (logaddexp..mape_loss)."""

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
class _HasMember31_00(typing.Protocol):
    """Structural protocol for objects exposing `logaddexp`."""

    logaddexp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_01(typing.Protocol):
    """Structural protocol for objects exposing `logaddexp2`."""

    logaddexp2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_02(typing.Protocol):
    """Structural protocol for objects exposing `logcosh`."""

    logcosh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_03(typing.Protocol):
    """Structural protocol for objects exposing `logcumsumexp`."""

    logcumsumexp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_04(typing.Protocol):
    """Structural protocol for objects exposing `logdet`."""

    logdet: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_05(typing.Protocol):
    """Structural protocol for objects exposing `logical_and`."""

    logical_and: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_06(typing.Protocol):
    """Structural protocol for objects exposing `logical_not`."""

    logical_not: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_07(typing.Protocol):
    """Structural protocol for objects exposing `logical_or`."""

    logical_or: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_08(typing.Protocol):
    """Structural protocol for objects exposing `logical_xor`."""

    logical_xor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_09(typing.Protocol):
    """Structural protocol for objects exposing `logicaland`."""

    logicaland: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_10(typing.Protocol):
    """Structural protocol for objects exposing `logicalnot`."""

    logicalnot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_11(typing.Protocol):
    """Structural protocol for objects exposing `logicalor`."""

    logicalor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_12(typing.Protocol):
    """Structural protocol for objects exposing `logicalxor`."""

    logicalxor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_13(typing.Protocol):
    """Structural protocol for objects exposing `logit`."""

    logit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_14(typing.Protocol):
    """Structural protocol for objects exposing `logm`."""

    logm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_15(typing.Protocol):
    """Structural protocol for objects exposing `logpoissonloss`."""

    logpoissonloss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_16(typing.Protocol):
    """Structural protocol for objects exposing `logsumexp`."""

    logsumexp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_17(typing.Protocol):
    """Structural protocol for objects exposing `lookup`."""

    lookup: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_18(typing.Protocol):
    """Structural protocol for objects exposing `loss_fn`."""

    loss_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_19(typing.Protocol):
    """Structural protocol for objects exposing `lower`."""

    lower: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_20(typing.Protocol):
    """Structural protocol for objects exposing `lstm_cell`."""

    lstm_cell: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_21(typing.Protocol):
    """Structural protocol for objects exposing `lstsq`."""

    lstsq: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_22(typing.Protocol):
    """Structural protocol for objects exposing `lt`."""

    lt: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_23(typing.Protocol):
    """Structural protocol for objects exposing `lu`."""

    lu: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_24(typing.Protocol):
    """Structural protocol for objects exposing `lu_factor`."""

    lu_factor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_25(typing.Protocol):
    """Structural protocol for objects exposing `lu_matrix_inverse`."""

    lu_matrix_inverse: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_26(typing.Protocol):
    """Structural protocol for objects exposing `lu_reconstruct`."""

    lu_reconstruct: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_27(typing.Protocol):
    """Structural protocol for objects exposing `lu_solve`."""

    lu_solve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_28(typing.Protocol):
    """Structural protocol for objects exposing `ma`."""

    ma: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_29(typing.Protocol):
    """Structural protocol for objects exposing `main`."""

    main: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_30(typing.Protocol):
    """Structural protocol for objects exposing `make_bipipe_pair`."""

    make_bipipe_pair: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_31(typing.Protocol):
    """Structural protocol for objects exposing `make_graph`."""

    make_graph: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_32(typing.Protocol):
    """Structural protocol for objects exposing `make_model`."""

    make_model: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_33(typing.Protocol):
    """Structural protocol for objects exposing `make_node`."""

    make_node: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_34(typing.Protocol):
    """Structural protocol for objects exposing `make_struct`."""

    make_struct: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_35(typing.Protocol):
    """Structural protocol for objects exposing `make_tensor`."""

    make_tensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_36(typing.Protocol):
    """Structural protocol for objects exposing `make_tensor_value_info`."""

    make_tensor_value_info: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_37(typing.Protocol):
    """Structural protocol for objects exposing `manual_seed`."""

    manual_seed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_38(typing.Protocol):
    """Structural protocol for objects exposing `map`."""

    map: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_39(typing.Protocol):
    """Structural protocol for objects exposing `map_blocks`."""

    map_blocks: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_40(typing.Protocol):
    """Structural protocol for objects exposing `map_coordinates`."""

    map_coordinates: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_41(typing.Protocol):
    """Structural protocol for objects exposing `map_flat_values`."""

    map_flat_values: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_42(typing.Protocol):
    """Structural protocol for objects exposing `map_fn`."""

    map_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_43(typing.Protocol):
    """Structural protocol for objects exposing `mape`."""

    mape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember31_44(typing.Protocol):
    """Structural protocol for objects exposing `mape_loss`."""

    mape_loss: EagerTensorProtocol


SHARD_31_READERS: dict[str, MemberReader] = {
    "logaddexp": lambda t: (True, t.logaddexp) if isinstance(t, _HasMember31_00) else (False, None),
    "logaddexp2": lambda t: (True, t.logaddexp2) if isinstance(t, _HasMember31_01) else (False, None),
    "logcosh": lambda t: (True, t.logcosh) if isinstance(t, _HasMember31_02) else (False, None),
    "logcumsumexp": lambda t: (True, t.logcumsumexp) if isinstance(t, _HasMember31_03) else (False, None),
    "logdet": lambda t: (True, t.logdet) if isinstance(t, _HasMember31_04) else (False, None),
    "logical_and": lambda t: (True, t.logical_and) if isinstance(t, _HasMember31_05) else (False, None),
    "logical_not": lambda t: (True, t.logical_not) if isinstance(t, _HasMember31_06) else (False, None),
    "logical_or": lambda t: (True, t.logical_or) if isinstance(t, _HasMember31_07) else (False, None),
    "logical_xor": lambda t: (True, t.logical_xor) if isinstance(t, _HasMember31_08) else (False, None),
    "logicaland": lambda t: (True, t.logicaland) if isinstance(t, _HasMember31_09) else (False, None),
    "logicalnot": lambda t: (True, t.logicalnot) if isinstance(t, _HasMember31_10) else (False, None),
    "logicalor": lambda t: (True, t.logicalor) if isinstance(t, _HasMember31_11) else (False, None),
    "logicalxor": lambda t: (True, t.logicalxor) if isinstance(t, _HasMember31_12) else (False, None),
    "logit": lambda t: (True, t.logit) if isinstance(t, _HasMember31_13) else (False, None),
    "logm": lambda t: (True, t.logm) if isinstance(t, _HasMember31_14) else (False, None),
    "logpoissonloss": lambda t: (True, t.logpoissonloss) if isinstance(t, _HasMember31_15) else (False, None),
    "logsumexp": lambda t: (True, t.logsumexp) if isinstance(t, _HasMember31_16) else (False, None),
    "lookup": lambda t: (True, t.lookup) if isinstance(t, _HasMember31_17) else (False, None),
    "loss_fn": lambda t: (True, t.loss_fn) if isinstance(t, _HasMember31_18) else (False, None),
    "lower": lambda t: (True, t.lower) if isinstance(t, _HasMember31_19) else (False, None),
    "lstm_cell": lambda t: (True, t.lstm_cell) if isinstance(t, _HasMember31_20) else (False, None),
    "lstsq": lambda t: (True, t.lstsq) if isinstance(t, _HasMember31_21) else (False, None),
    "lt": lambda t: (True, t.lt) if isinstance(t, _HasMember31_22) else (False, None),
    "lu": lambda t: (True, t.lu) if isinstance(t, _HasMember31_23) else (False, None),
    "lu_factor": lambda t: (True, t.lu_factor) if isinstance(t, _HasMember31_24) else (False, None),
    "lu_matrix_inverse": lambda t: (True, t.lu_matrix_inverse) if isinstance(t, _HasMember31_25) else (False, None),
    "lu_reconstruct": lambda t: (True, t.lu_reconstruct) if isinstance(t, _HasMember31_26) else (False, None),
    "lu_solve": lambda t: (True, t.lu_solve) if isinstance(t, _HasMember31_27) else (False, None),
    "ma": lambda t: (True, t.ma) if isinstance(t, _HasMember31_28) else (False, None),
    "main": lambda t: (True, t.main) if isinstance(t, _HasMember31_29) else (False, None),
    "make_bipipe_pair": lambda t: (True, t.make_bipipe_pair) if isinstance(t, _HasMember31_30) else (False, None),
    "make_graph": lambda t: (True, t.make_graph) if isinstance(t, _HasMember31_31) else (False, None),
    "make_model": lambda t: (True, t.make_model) if isinstance(t, _HasMember31_32) else (False, None),
    "make_node": lambda t: (True, t.make_node) if isinstance(t, _HasMember31_33) else (False, None),
    "make_struct": lambda t: (True, t.make_struct) if isinstance(t, _HasMember31_34) else (False, None),
    "make_tensor": lambda t: (True, t.make_tensor) if isinstance(t, _HasMember31_35) else (False, None),
    "make_tensor_value_info": lambda t: (
        (True, t.make_tensor_value_info) if isinstance(t, _HasMember31_36) else (False, None)
    ),
    "manual_seed": lambda t: (True, t.manual_seed) if isinstance(t, _HasMember31_37) else (False, None),
    "map": lambda t: (True, t.map) if isinstance(t, _HasMember31_38) else (False, None),
    "map_blocks": lambda t: (True, t.map_blocks) if isinstance(t, _HasMember31_39) else (False, None),
    "map_coordinates": lambda t: (True, t.map_coordinates) if isinstance(t, _HasMember31_40) else (False, None),
    "map_flat_values": lambda t: (True, t.map_flat_values) if isinstance(t, _HasMember31_41) else (False, None),
    "map_fn": lambda t: (True, t.map_fn) if isinstance(t, _HasMember31_42) else (False, None),
    "mape": lambda t: (True, t.mape) if isinstance(t, _HasMember31_43) else (False, None),
    "mape_loss": lambda t: (True, t.mape_loss) if isinstance(t, _HasMember31_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember31_00,
        _HasMember31_01,
        _HasMember31_02,
        _HasMember31_03,
        _HasMember31_04,
        _HasMember31_05,
        _HasMember31_06,
        _HasMember31_07,
        _HasMember31_08,
        _HasMember31_09,
        _HasMember31_10,
        _HasMember31_11,
        _HasMember31_12,
        _HasMember31_13,
        _HasMember31_14,
        _HasMember31_15,
        _HasMember31_16,
        _HasMember31_17,
        _HasMember31_18,
        _HasMember31_19,
        _HasMember31_20,
        _HasMember31_21,
        _HasMember31_22,
        _HasMember31_23,
        _HasMember31_24,
        _HasMember31_25,
        _HasMember31_26,
        _HasMember31_27,
        _HasMember31_28,
        _HasMember31_29,
        _HasMember31_30,
        _HasMember31_31,
        _HasMember31_32,
        _HasMember31_33,
        _HasMember31_34,
        _HasMember31_35,
        _HasMember31_36,
        _HasMember31_37,
        _HasMember31_38,
        _HasMember31_39,
        _HasMember31_40,
        _HasMember31_41,
        _HasMember31_42,
        _HasMember31_43,
        _HasMember31_44,
    )
)
