"""Static structural protocol bindings shard 09 (asin..avgpool2d)."""

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
class _HasMember09_00(typing.Protocol):
    """Structural protocol for objects exposing `asin`."""

    asin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_01(typing.Protocol):
    """Structural protocol for objects exposing `asinh`."""

    asinh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_02(typing.Protocol):
    """Structural protocol for objects exposing `asnumpy`."""

    asnumpy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_03(typing.Protocol):
    """Structural protocol for objects exposing `assert_op`."""

    assert_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_04(typing.Protocol):
    """Structural protocol for objects exposing `assert_value`."""

    assert_value: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_05(typing.Protocol):
    """Structural protocol for objects exposing `assertop`."""

    assertop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_06(typing.Protocol):
    """Structural protocol for objects exposing `assign`."""

    assign: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_07(typing.Protocol):
    """Structural protocol for objects exposing `assign_var_name`."""

    assign_var_name: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_08(typing.Protocol):
    """Structural protocol for objects exposing `assign_variable`."""

    assign_variable: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_09(typing.Protocol):
    """Structural protocol for objects exposing `assign_variable_op`."""

    assign_variable_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_10(typing.Protocol):
    """Structural protocol for objects exposing `assignvariable`."""

    assignvariable: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_11(typing.Protocol):
    """Structural protocol for objects exposing `associative_scan`."""

    associative_scan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_12(typing.Protocol):
    """Structural protocol for objects exposing `associativescan`."""

    associativescan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_13(typing.Protocol):
    """Structural protocol for objects exposing `asstringconfig`."""

    asstringconfig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_14(typing.Protocol):
    """Structural protocol for objects exposing `ast_func`."""

    ast_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_15(typing.Protocol):
    """Structural protocol for objects exposing `ast_int_return`."""

    ast_int_return: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_16(typing.Protocol):
    """Structural protocol for objects exposing `ast_str_return`."""

    ast_str_return: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_17(typing.Protocol):
    """Structural protocol for objects exposing `astype`."""

    astype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_18(typing.Protocol):
    """Structural protocol for objects exposing `async_i_o_future`."""

    async_i_o_future: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_19(typing.Protocol):
    """Structural protocol for objects exposing `asynciofuture`."""

    asynciofuture: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_20(typing.Protocol):
    """Structural protocol for objects exposing `at`."""

    at: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_21(typing.Protocol):
    """Structural protocol for objects exposing `atan`."""

    atan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_22(typing.Protocol):
    """Structural protocol for objects exposing `atan2`."""

    atan2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_23(typing.Protocol):
    """Structural protocol for objects exposing `atanh`."""

    atanh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_24(typing.Protocol):
    """Structural protocol for objects exposing `atleast1d`."""

    atleast1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_25(typing.Protocol):
    """Structural protocol for objects exposing `atleast2d`."""

    atleast2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_26(typing.Protocol):
    """Structural protocol for objects exposing `atleast3d`."""

    atleast3d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_27(typing.Protocol):
    """Structural protocol for objects exposing `atleast_1d`."""

    atleast_1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_28(typing.Protocol):
    """Structural protocol for objects exposing `atleast_2d`."""

    atleast_2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_29(typing.Protocol):
    """Structural protocol for objects exposing `atleast_3d`."""

    atleast_3d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_30(typing.Protocol):
    """Structural protocol for objects exposing `attr`."""

    attr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_31(typing.Protocol):
    """Structural protocol for objects exposing `attributes`."""

    attributes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_32(typing.Protocol):
    """Structural protocol for objects exposing `attrs`."""

    attrs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_33(typing.Protocol):
    """Structural protocol for objects exposing `aug_mix`."""

    aug_mix: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_34(typing.Protocol):
    """Structural protocol for objects exposing `augmix`."""

    augmix: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_35(typing.Protocol):
    """Structural protocol for objects exposing `auto_contrast`."""

    auto_contrast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_36(typing.Protocol):
    """Structural protocol for objects exposing `autocontrast`."""

    autocontrast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_37(typing.Protocol):
    """Structural protocol for objects exposing `aux_fn`."""

    aux_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_38(typing.Protocol):
    """Structural protocol for objects exposing `average`."""

    average: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_39(typing.Protocol):
    """Structural protocol for objects exposing `average_pool`."""

    average_pool: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_40(typing.Protocol):
    """Structural protocol for objects exposing `avg_pool`."""

    avg_pool: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_41(typing.Protocol):
    """Structural protocol for objects exposing `avg_pool1d`."""

    avg_pool1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_42(typing.Protocol):
    """Structural protocol for objects exposing `avg_pool2d`."""

    avg_pool2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_43(typing.Protocol):
    """Structural protocol for objects exposing `avg_pool3d`."""

    avg_pool3d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember09_44(typing.Protocol):
    """Structural protocol for objects exposing `avgpool2d`."""

    avgpool2d: EagerTensorProtocol


SHARD_09_READERS: dict[str, MemberReader] = {
    "asin": lambda t: (True, t.asin) if isinstance(t, _HasMember09_00) else (False, None),
    "asinh": lambda t: (True, t.asinh) if isinstance(t, _HasMember09_01) else (False, None),
    "asnumpy": lambda t: (True, t.asnumpy) if isinstance(t, _HasMember09_02) else (False, None),
    "assert_op": lambda t: (True, t.assert_op) if isinstance(t, _HasMember09_03) else (False, None),
    "assert_value": lambda t: (True, t.assert_value) if isinstance(t, _HasMember09_04) else (False, None),
    "assertop": lambda t: (True, t.assertop) if isinstance(t, _HasMember09_05) else (False, None),
    "assign": lambda t: (True, t.assign) if isinstance(t, _HasMember09_06) else (False, None),
    "assign_var_name": lambda t: (True, t.assign_var_name) if isinstance(t, _HasMember09_07) else (False, None),
    "assign_variable": lambda t: (True, t.assign_variable) if isinstance(t, _HasMember09_08) else (False, None),
    "assign_variable_op": lambda t: (True, t.assign_variable_op) if isinstance(t, _HasMember09_09) else (False, None),
    "assignvariable": lambda t: (True, t.assignvariable) if isinstance(t, _HasMember09_10) else (False, None),
    "associative_scan": lambda t: (True, t.associative_scan) if isinstance(t, _HasMember09_11) else (False, None),
    "associativescan": lambda t: (True, t.associativescan) if isinstance(t, _HasMember09_12) else (False, None),
    "asstringconfig": lambda t: (True, t.asstringconfig) if isinstance(t, _HasMember09_13) else (False, None),
    "ast_func": lambda t: (True, t.ast_func) if isinstance(t, _HasMember09_14) else (False, None),
    "ast_int_return": lambda t: (True, t.ast_int_return) if isinstance(t, _HasMember09_15) else (False, None),
    "ast_str_return": lambda t: (True, t.ast_str_return) if isinstance(t, _HasMember09_16) else (False, None),
    "astype": lambda t: (True, t.astype) if isinstance(t, _HasMember09_17) else (False, None),
    "async_i_o_future": lambda t: (True, t.async_i_o_future) if isinstance(t, _HasMember09_18) else (False, None),
    "asynciofuture": lambda t: (True, t.asynciofuture) if isinstance(t, _HasMember09_19) else (False, None),
    "at": lambda t: (True, t.at) if isinstance(t, _HasMember09_20) else (False, None),
    "atan": lambda t: (True, t.atan) if isinstance(t, _HasMember09_21) else (False, None),
    "atan2": lambda t: (True, t.atan2) if isinstance(t, _HasMember09_22) else (False, None),
    "atanh": lambda t: (True, t.atanh) if isinstance(t, _HasMember09_23) else (False, None),
    "atleast1d": lambda t: (True, t.atleast1d) if isinstance(t, _HasMember09_24) else (False, None),
    "atleast2d": lambda t: (True, t.atleast2d) if isinstance(t, _HasMember09_25) else (False, None),
    "atleast3d": lambda t: (True, t.atleast3d) if isinstance(t, _HasMember09_26) else (False, None),
    "atleast_1d": lambda t: (True, t.atleast_1d) if isinstance(t, _HasMember09_27) else (False, None),
    "atleast_2d": lambda t: (True, t.atleast_2d) if isinstance(t, _HasMember09_28) else (False, None),
    "atleast_3d": lambda t: (True, t.atleast_3d) if isinstance(t, _HasMember09_29) else (False, None),
    "attr": lambda t: (True, t.attr) if isinstance(t, _HasMember09_30) else (False, None),
    "attributes": lambda t: (True, t.attributes) if isinstance(t, _HasMember09_31) else (False, None),
    "attrs": lambda t: (True, t.attrs) if isinstance(t, _HasMember09_32) else (False, None),
    "aug_mix": lambda t: (True, t.aug_mix) if isinstance(t, _HasMember09_33) else (False, None),
    "augmix": lambda t: (True, t.augmix) if isinstance(t, _HasMember09_34) else (False, None),
    "auto_contrast": lambda t: (True, t.auto_contrast) if isinstance(t, _HasMember09_35) else (False, None),
    "autocontrast": lambda t: (True, t.autocontrast) if isinstance(t, _HasMember09_36) else (False, None),
    "aux_fn": lambda t: (True, t.aux_fn) if isinstance(t, _HasMember09_37) else (False, None),
    "average": lambda t: (True, t.average) if isinstance(t, _HasMember09_38) else (False, None),
    "average_pool": lambda t: (True, t.average_pool) if isinstance(t, _HasMember09_39) else (False, None),
    "avg_pool": lambda t: (True, t.avg_pool) if isinstance(t, _HasMember09_40) else (False, None),
    "avg_pool1d": lambda t: (True, t.avg_pool1d) if isinstance(t, _HasMember09_41) else (False, None),
    "avg_pool2d": lambda t: (True, t.avg_pool2d) if isinstance(t, _HasMember09_42) else (False, None),
    "avg_pool3d": lambda t: (True, t.avg_pool3d) if isinstance(t, _HasMember09_43) else (False, None),
    "avgpool2d": lambda t: (True, t.avgpool2d) if isinstance(t, _HasMember09_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember09_00,
        _HasMember09_01,
        _HasMember09_02,
        _HasMember09_03,
        _HasMember09_04,
        _HasMember09_05,
        _HasMember09_06,
        _HasMember09_07,
        _HasMember09_08,
        _HasMember09_09,
        _HasMember09_10,
        _HasMember09_11,
        _HasMember09_12,
        _HasMember09_13,
        _HasMember09_14,
        _HasMember09_15,
        _HasMember09_16,
        _HasMember09_17,
        _HasMember09_18,
        _HasMember09_19,
        _HasMember09_20,
        _HasMember09_21,
        _HasMember09_22,
        _HasMember09_23,
        _HasMember09_24,
        _HasMember09_25,
        _HasMember09_26,
        _HasMember09_27,
        _HasMember09_28,
        _HasMember09_29,
        _HasMember09_30,
        _HasMember09_31,
        _HasMember09_32,
        _HasMember09_33,
        _HasMember09_34,
        _HasMember09_35,
        _HasMember09_36,
        _HasMember09_37,
        _HasMember09_38,
        _HasMember09_39,
        _HasMember09_40,
        _HasMember09_41,
        _HasMember09_42,
        _HasMember09_43,
        _HasMember09_44,
    )
)
