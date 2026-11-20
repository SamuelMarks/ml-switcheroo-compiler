"""Static structural protocol bindings shard 08 (amin..ascii_upper)."""

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
class _HasMember08_00(typing.Protocol):
    """Structural protocol for objects exposing `amin`."""

    amin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_01(typing.Protocol):
    """Structural protocol for objects exposing `and_`."""

    and_: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_02(typing.Protocol):
    """Structural protocol for objects exposing `angle`."""

    angle: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_03(typing.Protocol):
    """Structural protocol for objects exposing `annotated_fn`."""

    annotated_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_04(typing.Protocol):
    """Structural protocol for objects exposing `annotations`."""

    annotations: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_05(typing.Protocol):
    """Structural protocol for objects exposing `another_func`."""

    another_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_06(typing.Protocol):
    """Structural protocol for objects exposing `any`."""

    any: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_07(typing.Protocol):
    """Structural protocol for objects exposing `api`."""

    api: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_08(typing.Protocol):
    """Structural protocol for objects exposing `append`."""

    append: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_09(typing.Protocol):
    """Structural protocol for objects exposing `apply`."""

    apply: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_10(typing.Protocol):
    """Structural protocol for objects exposing `apply_model`."""

    apply_model: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_11(typing.Protocol):
    """Structural protocol for objects exposing `apply_over_axes`."""

    apply_over_axes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_12(typing.Protocol):
    """Structural protocol for objects exposing `applyoveraxes`."""

    applyoveraxes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_13(typing.Protocol):
    """Structural protocol for objects exposing `approx_max_k`."""

    approx_max_k: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_14(typing.Protocol):
    """Structural protocol for objects exposing `approx_min_k`."""

    approx_min_k: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_15(typing.Protocol):
    """Structural protocol for objects exposing `approximate_median`."""

    approximate_median: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_16(typing.Protocol):
    """Structural protocol for objects exposing `approximatemedian`."""

    approximatemedian: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_17(typing.Protocol):
    """Structural protocol for objects exposing `arange`."""

    arange: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_18(typing.Protocol):
    """Structural protocol for objects exposing `arccos`."""

    arccos: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_19(typing.Protocol):
    """Structural protocol for objects exposing `arccosh`."""

    arccosh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_20(typing.Protocol):
    """Structural protocol for objects exposing `arcsin`."""

    arcsin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_21(typing.Protocol):
    """Structural protocol for objects exposing `arcsinh`."""

    arcsinh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_22(typing.Protocol):
    """Structural protocol for objects exposing `arctan`."""

    arctan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_23(typing.Protocol):
    """Structural protocol for objects exposing `arctan2`."""

    arctan2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_24(typing.Protocol):
    """Structural protocol for objects exposing `arctanh`."""

    arctanh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_25(typing.Protocol):
    """Structural protocol for objects exposing `argmax`."""

    argmax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_26(typing.Protocol):
    """Structural protocol for objects exposing `argmin`."""

    argmin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_27(typing.Protocol):
    """Structural protocol for objects exposing `argpartition`."""

    argpartition: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_28(typing.Protocol):
    """Structural protocol for objects exposing `argsort`."""

    argsort: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_29(typing.Protocol):
    """Structural protocol for objects exposing `argwhere`."""

    argwhere: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_30(typing.Protocol):
    """Structural protocol for objects exposing `array`."""

    array: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_31(typing.Protocol):
    """Structural protocol for objects exposing `array_repr`."""

    array_repr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_32(typing.Protocol):
    """Structural protocol for objects exposing `array_split`."""

    array_split: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_33(typing.Protocol):
    """Structural protocol for objects exposing `array_str`."""

    array_str: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_34(typing.Protocol):
    """Structural protocol for objects exposing `arrayrepr`."""

    arrayrepr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_35(typing.Protocol):
    """Structural protocol for objects exposing `arraystr`."""

    arraystr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_36(typing.Protocol):
    """Structural protocol for objects exposing `as_py`."""

    as_py: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_37(typing.Protocol):
    """Structural protocol for objects exposing `as_string`."""

    as_string: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_38(typing.Protocol):
    """Structural protocol for objects exposing `as_string_config`."""

    as_string_config: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_39(typing.Protocol):
    """Structural protocol for objects exposing `asarray`."""

    asarray: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_40(typing.Protocol):
    """Structural protocol for objects exposing `asarray_fn`."""

    asarray_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_41(typing.Protocol):
    """Structural protocol for objects exposing `ascii_capitalize`."""

    ascii_capitalize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_42(typing.Protocol):
    """Structural protocol for objects exposing `ascii_lower`."""

    ascii_lower: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_43(typing.Protocol):
    """Structural protocol for objects exposing `ascii_title`."""

    ascii_title: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember08_44(typing.Protocol):
    """Structural protocol for objects exposing `ascii_upper`."""

    ascii_upper: EagerTensorProtocol


SHARD_08_READERS: dict[str, MemberReader] = {
    "amin": lambda t: (True, t.amin) if isinstance(t, _HasMember08_00) else (False, None),
    "and_": lambda t: (True, t.and_) if isinstance(t, _HasMember08_01) else (False, None),
    "angle": lambda t: (True, t.angle) if isinstance(t, _HasMember08_02) else (False, None),
    "annotated_fn": lambda t: (True, t.annotated_fn) if isinstance(t, _HasMember08_03) else (False, None),
    "annotations": lambda t: (True, t.annotations) if isinstance(t, _HasMember08_04) else (False, None),
    "another_func": lambda t: (True, t.another_func) if isinstance(t, _HasMember08_05) else (False, None),
    "any": lambda t: (True, t.any) if isinstance(t, _HasMember08_06) else (False, None),
    "api": lambda t: (True, t.api) if isinstance(t, _HasMember08_07) else (False, None),
    "append": lambda t: (True, t.append) if isinstance(t, _HasMember08_08) else (False, None),
    "apply": lambda t: (True, t.apply) if isinstance(t, _HasMember08_09) else (False, None),
    "apply_model": lambda t: (True, t.apply_model) if isinstance(t, _HasMember08_10) else (False, None),
    "apply_over_axes": lambda t: (True, t.apply_over_axes) if isinstance(t, _HasMember08_11) else (False, None),
    "applyoveraxes": lambda t: (True, t.applyoveraxes) if isinstance(t, _HasMember08_12) else (False, None),
    "approx_max_k": lambda t: (True, t.approx_max_k) if isinstance(t, _HasMember08_13) else (False, None),
    "approx_min_k": lambda t: (True, t.approx_min_k) if isinstance(t, _HasMember08_14) else (False, None),
    "approximate_median": lambda t: (True, t.approximate_median) if isinstance(t, _HasMember08_15) else (False, None),
    "approximatemedian": lambda t: (True, t.approximatemedian) if isinstance(t, _HasMember08_16) else (False, None),
    "arange": lambda t: (True, t.arange) if isinstance(t, _HasMember08_17) else (False, None),
    "arccos": lambda t: (True, t.arccos) if isinstance(t, _HasMember08_18) else (False, None),
    "arccosh": lambda t: (True, t.arccosh) if isinstance(t, _HasMember08_19) else (False, None),
    "arcsin": lambda t: (True, t.arcsin) if isinstance(t, _HasMember08_20) else (False, None),
    "arcsinh": lambda t: (True, t.arcsinh) if isinstance(t, _HasMember08_21) else (False, None),
    "arctan": lambda t: (True, t.arctan) if isinstance(t, _HasMember08_22) else (False, None),
    "arctan2": lambda t: (True, t.arctan2) if isinstance(t, _HasMember08_23) else (False, None),
    "arctanh": lambda t: (True, t.arctanh) if isinstance(t, _HasMember08_24) else (False, None),
    "argmax": lambda t: (True, t.argmax) if isinstance(t, _HasMember08_25) else (False, None),
    "argmin": lambda t: (True, t.argmin) if isinstance(t, _HasMember08_26) else (False, None),
    "argpartition": lambda t: (True, t.argpartition) if isinstance(t, _HasMember08_27) else (False, None),
    "argsort": lambda t: (True, t.argsort) if isinstance(t, _HasMember08_28) else (False, None),
    "argwhere": lambda t: (True, t.argwhere) if isinstance(t, _HasMember08_29) else (False, None),
    "array": lambda t: (True, t.array) if isinstance(t, _HasMember08_30) else (False, None),
    "array_repr": lambda t: (True, t.array_repr) if isinstance(t, _HasMember08_31) else (False, None),
    "array_split": lambda t: (True, t.array_split) if isinstance(t, _HasMember08_32) else (False, None),
    "array_str": lambda t: (True, t.array_str) if isinstance(t, _HasMember08_33) else (False, None),
    "arrayrepr": lambda t: (True, t.arrayrepr) if isinstance(t, _HasMember08_34) else (False, None),
    "arraystr": lambda t: (True, t.arraystr) if isinstance(t, _HasMember08_35) else (False, None),
    "as_py": lambda t: (True, t.as_py) if isinstance(t, _HasMember08_36) else (False, None),
    "as_string": lambda t: (True, t.as_string) if isinstance(t, _HasMember08_37) else (False, None),
    "as_string_config": lambda t: (True, t.as_string_config) if isinstance(t, _HasMember08_38) else (False, None),
    "asarray": lambda t: (True, t.asarray) if isinstance(t, _HasMember08_39) else (False, None),
    "asarray_fn": lambda t: (True, t.asarray_fn) if isinstance(t, _HasMember08_40) else (False, None),
    "ascii_capitalize": lambda t: (True, t.ascii_capitalize) if isinstance(t, _HasMember08_41) else (False, None),
    "ascii_lower": lambda t: (True, t.ascii_lower) if isinstance(t, _HasMember08_42) else (False, None),
    "ascii_title": lambda t: (True, t.ascii_title) if isinstance(t, _HasMember08_43) else (False, None),
    "ascii_upper": lambda t: (True, t.ascii_upper) if isinstance(t, _HasMember08_44) else (False, None),
}
