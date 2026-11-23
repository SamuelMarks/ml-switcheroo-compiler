"""Static structural protocol bindings shard 41 (rcclReduceScatter..replace_with_mask)."""

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
class _HasMember41_00(typing.Protocol):
    """Structural protocol for objects exposing `rcclReduceScatter`."""

    rcclReduceScatter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_01(typing.Protocol):
    """Structural protocol for objects exposing `read_binary_records`."""

    read_binary_records: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_02(typing.Protocol):
    """Structural protocol for objects exposing `read_file`."""

    read_file: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_03(typing.Protocol):
    """Structural protocol for objects exposing `read_variable`."""

    read_variable: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_04(typing.Protocol):
    """Structural protocol for objects exposing `read_variable_op`."""

    read_variable_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_05(typing.Protocol):
    """Structural protocol for objects exposing `readfile`."""

    readfile: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_06(typing.Protocol):
    """Structural protocol for objects exposing `readvariable`."""

    readvariable: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_07(typing.Protocol):
    """Structural protocol for objects exposing `real`."""

    real: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_08(typing.Protocol):
    """Structural protocol for objects exposing `reciprocal`."""

    reciprocal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_09(typing.Protocol):
    """Structural protocol for objects exposing `reciprocal_no_nan`."""

    reciprocal_no_nan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_10(typing.Protocol):
    """Structural protocol for objects exposing `recv`."""

    recv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_11(typing.Protocol):
    """Structural protocol for objects exposing `reduce`."""

    reduce: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_12(typing.Protocol):
    """Structural protocol for objects exposing `reduce_all`."""

    reduce_all: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_13(typing.Protocol):
    """Structural protocol for objects exposing `reduce_any`."""

    reduce_any: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_14(typing.Protocol):
    """Structural protocol for objects exposing `reduce_euclidean_norm`."""

    reduce_euclidean_norm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_15(typing.Protocol):
    """Structural protocol for objects exposing `reduce_logsumexp`."""

    reduce_logsumexp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_16(typing.Protocol):
    """Structural protocol for objects exposing `reduce_max`."""

    reduce_max: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_17(typing.Protocol):
    """Structural protocol for objects exposing `reduce_mean`."""

    reduce_mean: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_18(typing.Protocol):
    """Structural protocol for objects exposing `reduce_min`."""

    reduce_min: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_19(typing.Protocol):
    """Structural protocol for objects exposing `reduce_prod`."""

    reduce_prod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_20(typing.Protocol):
    """Structural protocol for objects exposing `reduce_scatter`."""

    reduce_scatter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_21(typing.Protocol):
    """Structural protocol for objects exposing `reduce_std`."""

    reduce_std: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_22(typing.Protocol):
    """Structural protocol for objects exposing `reduce_sum`."""

    reduce_sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_23(typing.Protocol):
    """Structural protocol for objects exposing `reduce_variance`."""

    reduce_variance: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_24(typing.Protocol):
    """Structural protocol for objects exposing `reduce_window`."""

    reduce_window: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_25(typing.Protocol):
    """Structural protocol for objects exposing `reducemax`."""

    reducemax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_26(typing.Protocol):
    """Structural protocol for objects exposing `reducemin`."""

    reducemin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_27(typing.Protocol):
    """Structural protocol for objects exposing `reductions`."""

    reductions: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_28(typing.Protocol):
    """Structural protocol for objects exposing `regex_full_match`."""

    regex_full_match: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_29(typing.Protocol):
    """Structural protocol for objects exposing `regex_replace`."""

    regex_replace: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_30(typing.Protocol):
    """Structural protocol for objects exposing `register`."""

    register: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_31(typing.Protocol):
    """Structural protocol for objects exposing `register_op`."""

    register_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_32(typing.Protocol):
    """Structural protocol for objects exposing `register_parameter`."""

    register_parameter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_33(typing.Protocol):
    """Structural protocol for objects exposing `registry`."""

    registry: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_34(typing.Protocol):
    """Structural protocol for objects exposing `relu`."""

    relu: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_35(typing.Protocol):
    """Structural protocol for objects exposing `rem`."""

    rem: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_36(typing.Protocol):
    """Structural protocol for objects exposing `rem2`."""

    rem2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_37(typing.Protocol):
    """Structural protocol for objects exposing `rem3`."""

    rem3: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_38(typing.Protocol):
    """Structural protocol for objects exposing `remainder`."""

    remainder: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_39(typing.Protocol):
    """Structural protocol for objects exposing `remove_and_load`."""

    remove_and_load: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_40(typing.Protocol):
    """Structural protocol for objects exposing `repeat`."""

    repeat: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_41(typing.Protocol):
    """Structural protocol for objects exposing `repeat_interleave`."""

    repeat_interleave: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_42(typing.Protocol):
    """Structural protocol for objects exposing `replace_substring`."""

    replace_substring: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_43(typing.Protocol):
    """Structural protocol for objects exposing `replace_substring_regex`."""

    replace_substring_regex: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember41_44(typing.Protocol):
    """Structural protocol for objects exposing `replace_with_mask`."""

    replace_with_mask: EagerTensorProtocol


SHARD_41_READERS: dict[str, MemberReader] = {
    "rcclReduceScatter": lambda t: (True, t.rcclReduceScatter) if isinstance(t, _HasMember41_00) else (False, None),
    "read_binary_records": lambda t: (True, t.read_binary_records) if isinstance(t, _HasMember41_01) else (False, None),
    "read_file": lambda t: (True, t.read_file) if isinstance(t, _HasMember41_02) else (False, None),
    "read_variable": lambda t: (True, t.read_variable) if isinstance(t, _HasMember41_03) else (False, None),
    "read_variable_op": lambda t: (True, t.read_variable_op) if isinstance(t, _HasMember41_04) else (False, None),
    "readfile": lambda t: (True, t.readfile) if isinstance(t, _HasMember41_05) else (False, None),
    "readvariable": lambda t: (True, t.readvariable) if isinstance(t, _HasMember41_06) else (False, None),
    "real": lambda t: (True, t.real) if isinstance(t, _HasMember41_07) else (False, None),
    "reciprocal": lambda t: (True, t.reciprocal) if isinstance(t, _HasMember41_08) else (False, None),
    "reciprocal_no_nan": lambda t: (True, t.reciprocal_no_nan) if isinstance(t, _HasMember41_09) else (False, None),
    "recv": lambda t: (True, t.recv) if isinstance(t, _HasMember41_10) else (False, None),
    "reduce": lambda t: (True, t.reduce) if isinstance(t, _HasMember41_11) else (False, None),
    "reduce_all": lambda t: (True, t.reduce_all) if isinstance(t, _HasMember41_12) else (False, None),
    "reduce_any": lambda t: (True, t.reduce_any) if isinstance(t, _HasMember41_13) else (False, None),
    "reduce_euclidean_norm": lambda t: (
        (True, t.reduce_euclidean_norm) if isinstance(t, _HasMember41_14) else (False, None)
    ),
    "reduce_logsumexp": lambda t: (True, t.reduce_logsumexp) if isinstance(t, _HasMember41_15) else (False, None),
    "reduce_max": lambda t: (True, t.reduce_max) if isinstance(t, _HasMember41_16) else (False, None),
    "reduce_mean": lambda t: (True, t.reduce_mean) if isinstance(t, _HasMember41_17) else (False, None),
    "reduce_min": lambda t: (True, t.reduce_min) if isinstance(t, _HasMember41_18) else (False, None),
    "reduce_prod": lambda t: (True, t.reduce_prod) if isinstance(t, _HasMember41_19) else (False, None),
    "reduce_scatter": lambda t: (True, t.reduce_scatter) if isinstance(t, _HasMember41_20) else (False, None),
    "reduce_std": lambda t: (True, t.reduce_std) if isinstance(t, _HasMember41_21) else (False, None),
    "reduce_sum": lambda t: (True, t.reduce_sum) if isinstance(t, _HasMember41_22) else (False, None),
    "reduce_variance": lambda t: (True, t.reduce_variance) if isinstance(t, _HasMember41_23) else (False, None),
    "reduce_window": lambda t: (True, t.reduce_window) if isinstance(t, _HasMember41_24) else (False, None),
    "reducemax": lambda t: (True, t.reducemax) if isinstance(t, _HasMember41_25) else (False, None),
    "reducemin": lambda t: (True, t.reducemin) if isinstance(t, _HasMember41_26) else (False, None),
    "reductions": lambda t: (True, t.reductions) if isinstance(t, _HasMember41_27) else (False, None),
    "regex_full_match": lambda t: (True, t.regex_full_match) if isinstance(t, _HasMember41_28) else (False, None),
    "regex_replace": lambda t: (True, t.regex_replace) if isinstance(t, _HasMember41_29) else (False, None),
    "register": lambda t: (True, t.register) if isinstance(t, _HasMember41_30) else (False, None),
    "register_op": lambda t: (True, t.register_op) if isinstance(t, _HasMember41_31) else (False, None),
    "register_parameter": lambda t: (True, t.register_parameter) if isinstance(t, _HasMember41_32) else (False, None),
    "registry": lambda t: (True, t.registry) if isinstance(t, _HasMember41_33) else (False, None),
    "relu": lambda t: (True, t.relu) if isinstance(t, _HasMember41_34) else (False, None),
    "rem": lambda t: (True, t.rem) if isinstance(t, _HasMember41_35) else (False, None),
    "rem2": lambda t: (True, t.rem2) if isinstance(t, _HasMember41_36) else (False, None),
    "rem3": lambda t: (True, t.rem3) if isinstance(t, _HasMember41_37) else (False, None),
    "remainder": lambda t: (True, t.remainder) if isinstance(t, _HasMember41_38) else (False, None),
    "remove_and_load": lambda t: (True, t.remove_and_load) if isinstance(t, _HasMember41_39) else (False, None),
    "repeat": lambda t: (True, t.repeat) if isinstance(t, _HasMember41_40) else (False, None),
    "repeat_interleave": lambda t: (True, t.repeat_interleave) if isinstance(t, _HasMember41_41) else (False, None),
    "replace_substring": lambda t: (True, t.replace_substring) if isinstance(t, _HasMember41_42) else (False, None),
    "replace_substring_regex": lambda t: (
        (True, t.replace_substring_regex) if isinstance(t, _HasMember41_43) else (False, None)
    ),
    "replace_with_mask": lambda t: (True, t.replace_with_mask) if isinstance(t, _HasMember41_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember41_00,
        _HasMember41_01,
        _HasMember41_02,
        _HasMember41_03,
        _HasMember41_04,
        _HasMember41_05,
        _HasMember41_06,
        _HasMember41_07,
        _HasMember41_08,
        _HasMember41_09,
        _HasMember41_10,
        _HasMember41_11,
        _HasMember41_12,
        _HasMember41_13,
        _HasMember41_14,
        _HasMember41_15,
        _HasMember41_16,
        _HasMember41_17,
        _HasMember41_18,
        _HasMember41_19,
        _HasMember41_20,
        _HasMember41_21,
        _HasMember41_22,
        _HasMember41_23,
        _HasMember41_24,
        _HasMember41_25,
        _HasMember41_26,
        _HasMember41_27,
        _HasMember41_28,
        _HasMember41_29,
        _HasMember41_30,
        _HasMember41_31,
        _HasMember41_32,
        _HasMember41_33,
        _HasMember41_34,
        _HasMember41_35,
        _HasMember41_36,
        _HasMember41_37,
        _HasMember41_38,
        _HasMember41_39,
        _HasMember41_40,
        _HasMember41_41,
        _HasMember41_42,
        _HasMember41_43,
        _HasMember41_44,
    )
)
