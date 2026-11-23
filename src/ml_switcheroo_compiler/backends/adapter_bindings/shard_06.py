"""Static structural protocol bindings shard 06 (WindowConfig..absolute)."""

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
class _HasMember06_00(typing.Protocol):
    """Structural protocol for objects exposing `WindowConfig`."""

    WindowConfig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_01(typing.Protocol):
    """Structural protocol for objects exposing `WindowHamming`."""

    WindowHamming: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_02(typing.Protocol):
    """Structural protocol for objects exposing `WindowHann`."""

    WindowHann: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_03(typing.Protocol):
    """Structural protocol for objects exposing `WriteFile`."""

    WriteFile: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_04(typing.Protocol):
    """Structural protocol for objects exposing `_ACTIVE_BACKEND`."""

    _ACTIVE_BACKEND: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_05(typing.Protocol):
    """Structural protocol for objects exposing `_build_onnx_value_infos`."""

    _build_onnx_value_infos: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_06(typing.Protocol):
    """Structural protocol for objects exposing `_conv_nd`."""

    _conv_nd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_07(typing.Protocol):
    """Structural protocol for objects exposing `_emit_block`."""

    _emit_block: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_08(typing.Protocol):
    """Structural protocol for objects exposing `_emit_body_return`."""

    _emit_body_return: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_09(typing.Protocol):
    """Structural protocol for objects exposing `_emit_leaf_or_op`."""

    _emit_leaf_or_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_10(typing.Protocol):
    """Structural protocol for objects exposing `_emit_stmt_or_structure`."""

    _emit_stmt_or_structure: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_11(typing.Protocol):
    """Structural protocol for objects exposing `_execute_accumulate_n`."""

    _execute_accumulate_n: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_12(typing.Protocol):
    """Structural protocol for objects exposing `_execute_adaptive_avg_pool`."""

    _execute_adaptive_avg_pool: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_13(typing.Protocol):
    """Structural protocol for objects exposing `_execute_adaptive_max_pool`."""

    _execute_adaptive_max_pool: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_14(typing.Protocol):
    """Structural protocol for objects exposing `_execute_bessel_jn`."""

    _execute_bessel_jn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_15(typing.Protocol):
    """Structural protocol for objects exposing `_execute_binom_cdf`."""

    _execute_binom_cdf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_16(typing.Protocol):
    """Structural protocol for objects exposing `_execute_binom_pmf`."""

    _execute_binom_pmf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_17(typing.Protocol):
    """Structural protocol for objects exposing `_execute_cast`."""

    _execute_cast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_18(typing.Protocol):
    """Structural protocol for objects exposing `_execute_convolve2d`."""

    _execute_convolve2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_19(typing.Protocol):
    """Structural protocol for objects exposing `_execute_ragged_tensor_to_dense`."""

    _execute_ragged_tensor_to_dense: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_20(typing.Protocol):
    """Structural protocol for objects exposing `_execute_unsorted_segment_max`."""

    _execute_unsorted_segment_max: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_21(typing.Protocol):
    """Structural protocol for objects exposing `_execute_unsorted_segment_min`."""

    _execute_unsorted_segment_min: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_22(typing.Protocol):
    """Structural protocol for objects exposing `_execute_unsorted_segment_prod`."""

    _execute_unsorted_segment_prod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_23(typing.Protocol):
    """Structural protocol for objects exposing `_execute_unsorted_segment_sum`."""

    _execute_unsorted_segment_sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_24(typing.Protocol):
    """Structural protocol for objects exposing `_execute_variance`."""

    _execute_variance: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_25(typing.Protocol):
    """Structural protocol for objects exposing `_execute_welch`."""

    _execute_welch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_26(typing.Protocol):
    """Structural protocol for objects exposing `_hidden`."""

    _hidden: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_27(typing.Protocol):
    """Structural protocol for objects exposing `_mock_name`."""

    _mock_name: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_28(typing.Protocol):
    """Structural protocol for objects exposing `_np_betapdf`."""

    _np_betapdf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_29(typing.Protocol):
    """Structural protocol for objects exposing `_np_deg2rad`."""

    _np_deg2rad: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_30(typing.Protocol):
    """Structural protocol for objects exposing `_np_gamma`."""

    _np_gamma: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_31(typing.Protocol):
    """Structural protocol for objects exposing `_np_modifiedbesseli1`."""

    _np_modifiedbesseli1: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_32(typing.Protocol):
    """Structural protocol for objects exposing `_np_packbits`."""

    _np_packbits: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_33(typing.Protocol):
    """Structural protocol for objects exposing `_np_polyint`."""

    _np_polyint: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_34(typing.Protocol):
    """Structural protocol for objects exposing `_np_takealongaxis`."""

    _np_takealongaxis: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_35(typing.Protocol):
    """Structural protocol for objects exposing `_np_triangular`."""

    _np_triangular: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_36(typing.Protocol):
    """Structural protocol for objects exposing `_np_xlog1py`."""

    _np_xlog1py: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_37(typing.Protocol):
    """Structural protocol for objects exposing `_np_xlogy`."""

    _np_xlogy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_38(typing.Protocol):
    """Structural protocol for objects exposing `_torch_tensordot`."""

    _torch_tensordot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_39(typing.Protocol):
    """Structural protocol for objects exposing `_torch_variance`."""

    _torch_variance: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_40(typing.Protocol):
    """Structural protocol for objects exposing `_validate_delimiters`."""

    _validate_delimiters: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_41(typing.Protocol):
    """Structural protocol for objects exposing `_validate_raw_attributes`."""

    _validate_raw_attributes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_42(typing.Protocol):
    """Structural protocol for objects exposing `_z`."""

    _z: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_43(typing.Protocol):
    """Structural protocol for objects exposing `abs`."""

    abs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember06_44(typing.Protocol):
    """Structural protocol for objects exposing `absolute`."""

    absolute: EagerTensorProtocol


SHARD_06_READERS: dict[str, MemberReader] = {
    "WindowConfig": lambda t: (True, t.WindowConfig) if isinstance(t, _HasMember06_00) else (False, None),
    "WindowHamming": lambda t: (True, t.WindowHamming) if isinstance(t, _HasMember06_01) else (False, None),
    "WindowHann": lambda t: (True, t.WindowHann) if isinstance(t, _HasMember06_02) else (False, None),
    "WriteFile": lambda t: (True, t.WriteFile) if isinstance(t, _HasMember06_03) else (False, None),
    "_ACTIVE_BACKEND": lambda t: (True, t._ACTIVE_BACKEND) if isinstance(t, _HasMember06_04) else (False, None),
    "_build_onnx_value_infos": lambda t: (
        (True, t._build_onnx_value_infos) if isinstance(t, _HasMember06_05) else (False, None)
    ),
    "_conv_nd": lambda t: (True, t._conv_nd) if isinstance(t, _HasMember06_06) else (False, None),
    "_emit_block": lambda t: (True, t._emit_block) if isinstance(t, _HasMember06_07) else (False, None),
    "_emit_body_return": lambda t: (True, t._emit_body_return) if isinstance(t, _HasMember06_08) else (False, None),
    "_emit_leaf_or_op": lambda t: (True, t._emit_leaf_or_op) if isinstance(t, _HasMember06_09) else (False, None),
    "_emit_stmt_or_structure": lambda t: (
        (True, t._emit_stmt_or_structure) if isinstance(t, _HasMember06_10) else (False, None)
    ),
    "_execute_accumulate_n": lambda t: (
        (True, t._execute_accumulate_n) if isinstance(t, _HasMember06_11) else (False, None)
    ),
    "_execute_adaptive_avg_pool": lambda t: (
        (True, t._execute_adaptive_avg_pool) if isinstance(t, _HasMember06_12) else (False, None)
    ),
    "_execute_adaptive_max_pool": lambda t: (
        (True, t._execute_adaptive_max_pool) if isinstance(t, _HasMember06_13) else (False, None)
    ),
    "_execute_bessel_jn": lambda t: (True, t._execute_bessel_jn) if isinstance(t, _HasMember06_14) else (False, None),
    "_execute_binom_cdf": lambda t: (True, t._execute_binom_cdf) if isinstance(t, _HasMember06_15) else (False, None),
    "_execute_binom_pmf": lambda t: (True, t._execute_binom_pmf) if isinstance(t, _HasMember06_16) else (False, None),
    "_execute_cast": lambda t: (True, t._execute_cast) if isinstance(t, _HasMember06_17) else (False, None),
    "_execute_convolve2d": lambda t: (True, t._execute_convolve2d) if isinstance(t, _HasMember06_18) else (False, None),
    "_execute_ragged_tensor_to_dense": lambda t: (
        (True, t._execute_ragged_tensor_to_dense) if isinstance(t, _HasMember06_19) else (False, None)
    ),
    "_execute_unsorted_segment_max": lambda t: (
        (True, t._execute_unsorted_segment_max) if isinstance(t, _HasMember06_20) else (False, None)
    ),
    "_execute_unsorted_segment_min": lambda t: (
        (True, t._execute_unsorted_segment_min) if isinstance(t, _HasMember06_21) else (False, None)
    ),
    "_execute_unsorted_segment_prod": lambda t: (
        (True, t._execute_unsorted_segment_prod) if isinstance(t, _HasMember06_22) else (False, None)
    ),
    "_execute_unsorted_segment_sum": lambda t: (
        (True, t._execute_unsorted_segment_sum) if isinstance(t, _HasMember06_23) else (False, None)
    ),
    "_execute_variance": lambda t: (True, t._execute_variance) if isinstance(t, _HasMember06_24) else (False, None),
    "_execute_welch": lambda t: (True, t._execute_welch) if isinstance(t, _HasMember06_25) else (False, None),
    "_hidden": lambda t: (True, t._hidden) if isinstance(t, _HasMember06_26) else (False, None),
    "_mock_name": lambda t: (True, t._mock_name) if isinstance(t, _HasMember06_27) else (False, None),
    "_np_betapdf": lambda t: (True, t._np_betapdf) if isinstance(t, _HasMember06_28) else (False, None),
    "_np_deg2rad": lambda t: (True, t._np_deg2rad) if isinstance(t, _HasMember06_29) else (False, None),
    "_np_gamma": lambda t: (True, t._np_gamma) if isinstance(t, _HasMember06_30) else (False, None),
    "_np_modifiedbesseli1": lambda t: (
        (True, t._np_modifiedbesseli1) if isinstance(t, _HasMember06_31) else (False, None)
    ),
    "_np_packbits": lambda t: (True, t._np_packbits) if isinstance(t, _HasMember06_32) else (False, None),
    "_np_polyint": lambda t: (True, t._np_polyint) if isinstance(t, _HasMember06_33) else (False, None),
    "_np_takealongaxis": lambda t: (True, t._np_takealongaxis) if isinstance(t, _HasMember06_34) else (False, None),
    "_np_triangular": lambda t: (True, t._np_triangular) if isinstance(t, _HasMember06_35) else (False, None),
    "_np_xlog1py": lambda t: (True, t._np_xlog1py) if isinstance(t, _HasMember06_36) else (False, None),
    "_np_xlogy": lambda t: (True, t._np_xlogy) if isinstance(t, _HasMember06_37) else (False, None),
    "_torch_tensordot": lambda t: (True, t._torch_tensordot) if isinstance(t, _HasMember06_38) else (False, None),
    "_torch_variance": lambda t: (True, t._torch_variance) if isinstance(t, _HasMember06_39) else (False, None),
    "_validate_delimiters": lambda t: (
        (True, t._validate_delimiters) if isinstance(t, _HasMember06_40) else (False, None)
    ),
    "_validate_raw_attributes": lambda t: (
        (True, t._validate_raw_attributes) if isinstance(t, _HasMember06_41) else (False, None)
    ),
    "_z": lambda t: (True, t._z) if isinstance(t, _HasMember06_42) else (False, None),
    "abs": lambda t: (True, t.abs) if isinstance(t, _HasMember06_43) else (False, None),
    "absolute": lambda t: (True, t.absolute) if isinstance(t, _HasMember06_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember06_00,
        _HasMember06_01,
        _HasMember06_02,
        _HasMember06_03,
        _HasMember06_04,
        _HasMember06_05,
        _HasMember06_06,
        _HasMember06_07,
        _HasMember06_08,
        _HasMember06_09,
        _HasMember06_10,
        _HasMember06_11,
        _HasMember06_12,
        _HasMember06_13,
        _HasMember06_14,
        _HasMember06_15,
        _HasMember06_16,
        _HasMember06_17,
        _HasMember06_18,
        _HasMember06_19,
        _HasMember06_20,
        _HasMember06_21,
        _HasMember06_22,
        _HasMember06_23,
        _HasMember06_24,
        _HasMember06_25,
        _HasMember06_26,
        _HasMember06_27,
        _HasMember06_28,
        _HasMember06_29,
        _HasMember06_30,
        _HasMember06_31,
        _HasMember06_32,
        _HasMember06_33,
        _HasMember06_34,
        _HasMember06_35,
        _HasMember06_36,
        _HasMember06_37,
        _HasMember06_38,
        _HasMember06_39,
        _HasMember06_40,
        _HasMember06_41,
        _HasMember06_42,
        _HasMember06_43,
        _HasMember06_44,
    )
)
