"""Static structural protocol bindings shard 17 (dispatcher..dummy_vjp)."""

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
class _HasMember17_00(typing.Protocol):
    """Structural protocol for objects exposing `dispatcher`."""

    dispatcher: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_01(typing.Protocol):
    """Structural protocol for objects exposing `distributed`."""

    distributed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_02(typing.Protocol):
    """Structural protocol for objects exposing `distributions`."""

    distributions: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_03(typing.Protocol):
    """Structural protocol for objects exposing `div`."""

    div: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_04(typing.Protocol):
    """Structural protocol for objects exposing `divide`."""

    divide: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_05(typing.Protocol):
    """Structural protocol for objects exposing `divide_no_nan`."""

    divide_no_nan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_06(typing.Protocol):
    """Structural protocol for objects exposing `divmod`."""

    divmod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_07(typing.Protocol):
    """Structural protocol for objects exposing `do_method`."""

    do_method: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_08(typing.Protocol):
    """Structural protocol for objects exposing `dot`."""

    dot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_09(typing.Protocol):
    """Structural protocol for objects exposing `dot_general`."""

    dot_general: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_10(typing.Protocol):
    """Structural protocol for objects exposing `dotgeneral`."""

    dotgeneral: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_11(typing.Protocol):
    """Structural protocol for objects exposing `dotted`."""

    dotted: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_12(typing.Protocol):
    """Structural protocol for objects exposing `double`."""

    double: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_13(typing.Protocol):
    """Structural protocol for objects exposing `double_fn`."""

    double_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_14(typing.Protocol):
    """Structural protocol for objects exposing `drain_later`."""

    drain_later: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_15(typing.Protocol):
    """Structural protocol for objects exposing `draw_bounding_boxes`."""

    draw_bounding_boxes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_16(typing.Protocol):
    """Structural protocol for objects exposing `drop_null`."""

    drop_null: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_17(typing.Protocol):
    """Structural protocol for objects exposing `dropna`."""

    dropna: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_18(typing.Protocol):
    """Structural protocol for objects exposing `dropout`."""

    dropout: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_19(typing.Protocol):
    """Structural protocol for objects exposing `dsplit`."""

    dsplit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_20(typing.Protocol):
    """Structural protocol for objects exposing `dstack`."""

    dstack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_21(typing.Protocol):
    """Structural protocol for objects exposing `dtype`."""

    dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_22(typing.Protocol):
    """Structural protocol for objects exposing `dtypeholderprotocol`."""

    dtypeholderprotocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_23(typing.Protocol):
    """Structural protocol for objects exposing `dummy`."""

    dummy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_24(typing.Protocol):
    """Structural protocol for objects exposing `dummy2`."""

    dummy2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_25(typing.Protocol):
    """Structural protocol for objects exposing `dummy_cell`."""

    dummy_cell: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_26(typing.Protocol):
    """Structural protocol for objects exposing `dummy_dtype`."""

    dummy_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_27(typing.Protocol):
    """Structural protocol for objects exposing `dummy_eager`."""

    dummy_eager: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_28(typing.Protocol):
    """Structural protocol for objects exposing `dummy_emit_creation_node`."""

    dummy_emit_creation_node: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_29(typing.Protocol):
    """Structural protocol for objects exposing `dummy_eval`."""

    dummy_eval: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_30(typing.Protocol):
    """Structural protocol for objects exposing `dummy_export`."""

    dummy_export: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_31(typing.Protocol):
    """Structural protocol for objects exposing `dummy_fn`."""

    dummy_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_32(typing.Protocol):
    """Structural protocol for objects exposing `dummy_fun`."""

    dummy_fun: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_33(typing.Protocol):
    """Structural protocol for objects exposing `dummy_func`."""

    dummy_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_34(typing.Protocol):
    """Structural protocol for objects exposing `dummy_func2`."""

    dummy_func2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_35(typing.Protocol):
    """Structural protocol for objects exposing `dummy_gen`."""

    dummy_gen: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_36(typing.Protocol):
    """Structural protocol for objects exposing `dummy_jvp`."""

    dummy_jvp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_37(typing.Protocol):
    """Structural protocol for objects exposing `dummy_jvp2`."""

    dummy_jvp2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_38(typing.Protocol):
    """Structural protocol for objects exposing `dummy_jvp3`."""

    dummy_jvp3: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_39(typing.Protocol):
    """Structural protocol for objects exposing `dummy_model`."""

    dummy_model: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_40(typing.Protocol):
    """Structural protocol for objects exposing `dummy_model_no_shape`."""

    dummy_model_no_shape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_41(typing.Protocol):
    """Structural protocol for objects exposing `dummy_op`."""

    dummy_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_42(typing.Protocol):
    """Structural protocol for objects exposing `dummy_pass`."""

    dummy_pass: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_43(typing.Protocol):
    """Structural protocol for objects exposing `dummy_update`."""

    dummy_update: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember17_44(typing.Protocol):
    """Structural protocol for objects exposing `dummy_vjp`."""

    dummy_vjp: EagerTensorProtocol


SHARD_17_READERS: dict[str, MemberReader] = {
    "dispatcher": lambda t: (True, t.dispatcher) if isinstance(t, _HasMember17_00) else (False, None),
    "distributed": lambda t: (True, t.distributed) if isinstance(t, _HasMember17_01) else (False, None),
    "distributions": lambda t: (True, t.distributions) if isinstance(t, _HasMember17_02) else (False, None),
    "div": lambda t: (True, t.div) if isinstance(t, _HasMember17_03) else (False, None),
    "divide": lambda t: (True, t.divide) if isinstance(t, _HasMember17_04) else (False, None),
    "divide_no_nan": lambda t: (True, t.divide_no_nan) if isinstance(t, _HasMember17_05) else (False, None),
    "divmod": lambda t: (True, t.divmod) if isinstance(t, _HasMember17_06) else (False, None),
    "do_method": lambda t: (True, t.do_method) if isinstance(t, _HasMember17_07) else (False, None),
    "dot": lambda t: (True, t.dot) if isinstance(t, _HasMember17_08) else (False, None),
    "dot_general": lambda t: (True, t.dot_general) if isinstance(t, _HasMember17_09) else (False, None),
    "dotgeneral": lambda t: (True, t.dotgeneral) if isinstance(t, _HasMember17_10) else (False, None),
    "dotted": lambda t: (True, t.dotted) if isinstance(t, _HasMember17_11) else (False, None),
    "double": lambda t: (True, t.double) if isinstance(t, _HasMember17_12) else (False, None),
    "double_fn": lambda t: (True, t.double_fn) if isinstance(t, _HasMember17_13) else (False, None),
    "drain_later": lambda t: (True, t.drain_later) if isinstance(t, _HasMember17_14) else (False, None),
    "draw_bounding_boxes": lambda t: (True, t.draw_bounding_boxes) if isinstance(t, _HasMember17_15) else (False, None),
    "drop_null": lambda t: (True, t.drop_null) if isinstance(t, _HasMember17_16) else (False, None),
    "dropna": lambda t: (True, t.dropna) if isinstance(t, _HasMember17_17) else (False, None),
    "dropout": lambda t: (True, t.dropout) if isinstance(t, _HasMember17_18) else (False, None),
    "dsplit": lambda t: (True, t.dsplit) if isinstance(t, _HasMember17_19) else (False, None),
    "dstack": lambda t: (True, t.dstack) if isinstance(t, _HasMember17_20) else (False, None),
    "dtype": lambda t: (True, t.dtype) if isinstance(t, _HasMember17_21) else (False, None),
    "dtypeholderprotocol": lambda t: (True, t.dtypeholderprotocol) if isinstance(t, _HasMember17_22) else (False, None),
    "dummy": lambda t: (True, t.dummy) if isinstance(t, _HasMember17_23) else (False, None),
    "dummy2": lambda t: (True, t.dummy2) if isinstance(t, _HasMember17_24) else (False, None),
    "dummy_cell": lambda t: (True, t.dummy_cell) if isinstance(t, _HasMember17_25) else (False, None),
    "dummy_dtype": lambda t: (True, t.dummy_dtype) if isinstance(t, _HasMember17_26) else (False, None),
    "dummy_eager": lambda t: (True, t.dummy_eager) if isinstance(t, _HasMember17_27) else (False, None),
    "dummy_emit_creation_node": lambda t: (
        (True, t.dummy_emit_creation_node) if isinstance(t, _HasMember17_28) else (False, None)
    ),
    "dummy_eval": lambda t: (True, t.dummy_eval) if isinstance(t, _HasMember17_29) else (False, None),
    "dummy_export": lambda t: (True, t.dummy_export) if isinstance(t, _HasMember17_30) else (False, None),
    "dummy_fn": lambda t: (True, t.dummy_fn) if isinstance(t, _HasMember17_31) else (False, None),
    "dummy_fun": lambda t: (True, t.dummy_fun) if isinstance(t, _HasMember17_32) else (False, None),
    "dummy_func": lambda t: (True, t.dummy_func) if isinstance(t, _HasMember17_33) else (False, None),
    "dummy_func2": lambda t: (True, t.dummy_func2) if isinstance(t, _HasMember17_34) else (False, None),
    "dummy_gen": lambda t: (True, t.dummy_gen) if isinstance(t, _HasMember17_35) else (False, None),
    "dummy_jvp": lambda t: (True, t.dummy_jvp) if isinstance(t, _HasMember17_36) else (False, None),
    "dummy_jvp2": lambda t: (True, t.dummy_jvp2) if isinstance(t, _HasMember17_37) else (False, None),
    "dummy_jvp3": lambda t: (True, t.dummy_jvp3) if isinstance(t, _HasMember17_38) else (False, None),
    "dummy_model": lambda t: (True, t.dummy_model) if isinstance(t, _HasMember17_39) else (False, None),
    "dummy_model_no_shape": lambda t: (
        (True, t.dummy_model_no_shape) if isinstance(t, _HasMember17_40) else (False, None)
    ),
    "dummy_op": lambda t: (True, t.dummy_op) if isinstance(t, _HasMember17_41) else (False, None),
    "dummy_pass": lambda t: (True, t.dummy_pass) if isinstance(t, _HasMember17_42) else (False, None),
    "dummy_update": lambda t: (True, t.dummy_update) if isinstance(t, _HasMember17_43) else (False, None),
    "dummy_vjp": lambda t: (True, t.dummy_vjp) if isinstance(t, _HasMember17_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember17_00,
        _HasMember17_01,
        _HasMember17_02,
        _HasMember17_03,
        _HasMember17_04,
        _HasMember17_05,
        _HasMember17_06,
        _HasMember17_07,
        _HasMember17_08,
        _HasMember17_09,
        _HasMember17_10,
        _HasMember17_11,
        _HasMember17_12,
        _HasMember17_13,
        _HasMember17_14,
        _HasMember17_15,
        _HasMember17_16,
        _HasMember17_17,
        _HasMember17_18,
        _HasMember17_19,
        _HasMember17_20,
        _HasMember17_21,
        _HasMember17_22,
        _HasMember17_23,
        _HasMember17_24,
        _HasMember17_25,
        _HasMember17_26,
        _HasMember17_27,
        _HasMember17_28,
        _HasMember17_29,
        _HasMember17_30,
        _HasMember17_31,
        _HasMember17_32,
        _HasMember17_33,
        _HasMember17_34,
        _HasMember17_35,
        _HasMember17_36,
        _HasMember17_37,
        _HasMember17_38,
        _HasMember17_39,
        _HasMember17_40,
        _HasMember17_41,
        _HasMember17_42,
        _HasMember17_43,
        _HasMember17_44,
    )
)
