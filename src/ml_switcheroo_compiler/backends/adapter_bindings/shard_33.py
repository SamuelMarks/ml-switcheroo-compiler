"""Static structural protocol bindings shard 33 (mlx..mock_getsource)."""

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
class _HasMember33_00(typing.Protocol):
    """Structural protocol for objects exposing `mlx`."""

    mlx: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_01(typing.Protocol):
    """Structural protocol for objects exposing `mock_add`."""

    mock_add: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_02(typing.Protocol):
    """Structural protocol for objects exposing `mock_ak_op`."""

    mock_ak_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_03(typing.Protocol):
    """Structural protocol for objects exposing `mock_all_gather`."""

    mock_all_gather: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_04(typing.Protocol):
    """Structural protocol for objects exposing `mock_backend`."""

    mock_backend: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_05(typing.Protocol):
    """Structural protocol for objects exposing `mock_bwd`."""

    mock_bwd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_06(typing.Protocol):
    """Structural protocol for objects exposing `mock_calls`."""

    mock_calls: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_07(typing.Protocol):
    """Structural protocol for objects exposing `mock_cdll`."""

    mock_cdll: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_08(typing.Protocol):
    """Structural protocol for objects exposing `mock_cell`."""

    mock_cell: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_09(typing.Protocol):
    """Structural protocol for objects exposing `mock_cf_scan`."""

    mock_cf_scan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_10(typing.Protocol):
    """Structural protocol for objects exposing `mock_client`."""

    mock_client: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_11(typing.Protocol):
    """Structural protocol for objects exposing `mock_client_err`."""

    mock_client_err: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_12(typing.Protocol):
    """Structural protocol for objects exposing `mock_conv2d`."""

    mock_conv2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_13(typing.Protocol):
    """Structural protocol for objects exposing `mock_conv2d_transpose`."""

    mock_conv2d_transpose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_14(typing.Protocol):
    """Structural protocol for objects exposing `mock_conv_nd`."""

    mock_conv_nd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_15(typing.Protocol):
    """Structural protocol for objects exposing `mock_custom_op`."""

    mock_custom_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_16(typing.Protocol):
    """Structural protocol for objects exposing `mock_default_dir_exists`."""

    mock_default_dir_exists: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_17(typing.Protocol):
    """Structural protocol for objects exposing `mock_default_dir_listdir`."""

    mock_default_dir_listdir: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_18(typing.Protocol):
    """Structural protocol for objects exposing `mock_einsum`."""

    mock_einsum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_19(typing.Protocol):
    """Structural protocol for objects exposing `mock_elapsed`."""

    mock_elapsed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_20(typing.Protocol):
    """Structural protocol for objects exposing `mock_emit`."""

    mock_emit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_21(typing.Protocol):
    """Structural protocol for objects exposing `mock_eval`."""

    mock_eval: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_22(typing.Protocol):
    """Structural protocol for objects exposing `mock_eval_node`."""

    mock_eval_node: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_23(typing.Protocol):
    """Structural protocol for objects exposing `mock_exec`."""

    mock_exec: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_24(typing.Protocol):
    """Structural protocol for objects exposing `mock_exists`."""

    mock_exists: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_25(typing.Protocol):
    """Structural protocol for objects exposing `mock_exists_fallback`."""

    mock_exists_fallback: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_26(typing.Protocol):
    """Structural protocol for objects exposing `mock_exists_no_bp`."""

    mock_exists_no_bp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_27(typing.Protocol):
    """Structural protocol for objects exposing `mock_exists_no_col`."""

    mock_exists_no_col: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_28(typing.Protocol):
    """Structural protocol for objects exposing `mock_exists_no_standard`."""

    mock_exists_no_standard: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_29(typing.Protocol):
    """Structural protocol for objects exposing `mock_exists_no_top`."""

    mock_exists_no_top: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_30(typing.Protocol):
    """Structural protocol for objects exposing `mock_exists_none`."""

    mock_exists_none: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_31(typing.Protocol):
    """Structural protocol for objects exposing `mock_exists_primary`."""

    mock_exists_primary: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_32(typing.Protocol):
    """Structural protocol for objects exposing `mock_expanduser`."""

    mock_expanduser: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_33(typing.Protocol):
    """Structural protocol for objects exposing `mock_find_spec`."""

    mock_find_spec: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_34(typing.Protocol):
    """Structural protocol for objects exposing `mock_func`."""

    mock_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_35(typing.Protocol):
    """Structural protocol for objects exposing `mock_fwd`."""

    mock_fwd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_36(typing.Protocol):
    """Structural protocol for objects exposing `mock_generate_body`."""

    mock_generate_body: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_37(typing.Protocol):
    """Structural protocol for objects exposing `mock_get`."""

    mock_get: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_38(typing.Protocol):
    """Structural protocol for objects exposing `mock_get_backend`."""

    mock_get_backend: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_39(typing.Protocol):
    """Structural protocol for objects exposing `mock_get_data`."""

    mock_get_data: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_40(typing.Protocol):
    """Structural protocol for objects exposing `mock_get_func`."""

    mock_get_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_41(typing.Protocol):
    """Structural protocol for objects exposing `mock_get_op`."""

    mock_get_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_42(typing.Protocol):
    """Structural protocol for objects exposing `mock_get_template`."""

    mock_get_template: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_43(typing.Protocol):
    """Structural protocol for objects exposing `mock_getframeinfo`."""

    mock_getframeinfo: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember33_44(typing.Protocol):
    """Structural protocol for objects exposing `mock_getsource`."""

    mock_getsource: EagerTensorProtocol


SHARD_33_READERS: dict[str, MemberReader] = {
    "mlx": lambda t: (True, t.mlx) if isinstance(t, _HasMember33_00) else (False, None),
    "mock_add": lambda t: (True, t.mock_add) if isinstance(t, _HasMember33_01) else (False, None),
    "mock_ak_op": lambda t: (True, t.mock_ak_op) if isinstance(t, _HasMember33_02) else (False, None),
    "mock_all_gather": lambda t: (True, t.mock_all_gather) if isinstance(t, _HasMember33_03) else (False, None),
    "mock_backend": lambda t: (True, t.mock_backend) if isinstance(t, _HasMember33_04) else (False, None),
    "mock_bwd": lambda t: (True, t.mock_bwd) if isinstance(t, _HasMember33_05) else (False, None),
    "mock_calls": lambda t: (True, t.mock_calls) if isinstance(t, _HasMember33_06) else (False, None),
    "mock_cdll": lambda t: (True, t.mock_cdll) if isinstance(t, _HasMember33_07) else (False, None),
    "mock_cell": lambda t: (True, t.mock_cell) if isinstance(t, _HasMember33_08) else (False, None),
    "mock_cf_scan": lambda t: (True, t.mock_cf_scan) if isinstance(t, _HasMember33_09) else (False, None),
    "mock_client": lambda t: (True, t.mock_client) if isinstance(t, _HasMember33_10) else (False, None),
    "mock_client_err": lambda t: (True, t.mock_client_err) if isinstance(t, _HasMember33_11) else (False, None),
    "mock_conv2d": lambda t: (True, t.mock_conv2d) if isinstance(t, _HasMember33_12) else (False, None),
    "mock_conv2d_transpose": lambda t: (
        (True, t.mock_conv2d_transpose) if isinstance(t, _HasMember33_13) else (False, None)
    ),
    "mock_conv_nd": lambda t: (True, t.mock_conv_nd) if isinstance(t, _HasMember33_14) else (False, None),
    "mock_custom_op": lambda t: (True, t.mock_custom_op) if isinstance(t, _HasMember33_15) else (False, None),
    "mock_default_dir_exists": lambda t: (
        (True, t.mock_default_dir_exists) if isinstance(t, _HasMember33_16) else (False, None)
    ),
    "mock_default_dir_listdir": lambda t: (
        (True, t.mock_default_dir_listdir) if isinstance(t, _HasMember33_17) else (False, None)
    ),
    "mock_einsum": lambda t: (True, t.mock_einsum) if isinstance(t, _HasMember33_18) else (False, None),
    "mock_elapsed": lambda t: (True, t.mock_elapsed) if isinstance(t, _HasMember33_19) else (False, None),
    "mock_emit": lambda t: (True, t.mock_emit) if isinstance(t, _HasMember33_20) else (False, None),
    "mock_eval": lambda t: (True, t.mock_eval) if isinstance(t, _HasMember33_21) else (False, None),
    "mock_eval_node": lambda t: (True, t.mock_eval_node) if isinstance(t, _HasMember33_22) else (False, None),
    "mock_exec": lambda t: (True, t.mock_exec) if isinstance(t, _HasMember33_23) else (False, None),
    "mock_exists": lambda t: (True, t.mock_exists) if isinstance(t, _HasMember33_24) else (False, None),
    "mock_exists_fallback": lambda t: (
        (True, t.mock_exists_fallback) if isinstance(t, _HasMember33_25) else (False, None)
    ),
    "mock_exists_no_bp": lambda t: (True, t.mock_exists_no_bp) if isinstance(t, _HasMember33_26) else (False, None),
    "mock_exists_no_col": lambda t: (True, t.mock_exists_no_col) if isinstance(t, _HasMember33_27) else (False, None),
    "mock_exists_no_standard": lambda t: (
        (True, t.mock_exists_no_standard) if isinstance(t, _HasMember33_28) else (False, None)
    ),
    "mock_exists_no_top": lambda t: (True, t.mock_exists_no_top) if isinstance(t, _HasMember33_29) else (False, None),
    "mock_exists_none": lambda t: (True, t.mock_exists_none) if isinstance(t, _HasMember33_30) else (False, None),
    "mock_exists_primary": lambda t: (True, t.mock_exists_primary) if isinstance(t, _HasMember33_31) else (False, None),
    "mock_expanduser": lambda t: (True, t.mock_expanduser) if isinstance(t, _HasMember33_32) else (False, None),
    "mock_find_spec": lambda t: (True, t.mock_find_spec) if isinstance(t, _HasMember33_33) else (False, None),
    "mock_func": lambda t: (True, t.mock_func) if isinstance(t, _HasMember33_34) else (False, None),
    "mock_fwd": lambda t: (True, t.mock_fwd) if isinstance(t, _HasMember33_35) else (False, None),
    "mock_generate_body": lambda t: (True, t.mock_generate_body) if isinstance(t, _HasMember33_36) else (False, None),
    "mock_get": lambda t: (True, t.mock_get) if isinstance(t, _HasMember33_37) else (False, None),
    "mock_get_backend": lambda t: (True, t.mock_get_backend) if isinstance(t, _HasMember33_38) else (False, None),
    "mock_get_data": lambda t: (True, t.mock_get_data) if isinstance(t, _HasMember33_39) else (False, None),
    "mock_get_func": lambda t: (True, t.mock_get_func) if isinstance(t, _HasMember33_40) else (False, None),
    "mock_get_op": lambda t: (True, t.mock_get_op) if isinstance(t, _HasMember33_41) else (False, None),
    "mock_get_template": lambda t: (True, t.mock_get_template) if isinstance(t, _HasMember33_42) else (False, None),
    "mock_getframeinfo": lambda t: (True, t.mock_getframeinfo) if isinstance(t, _HasMember33_43) else (False, None),
    "mock_getsource": lambda t: (True, t.mock_getsource) if isinstance(t, _HasMember33_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember33_00,
        _HasMember33_01,
        _HasMember33_02,
        _HasMember33_03,
        _HasMember33_04,
        _HasMember33_05,
        _HasMember33_06,
        _HasMember33_07,
        _HasMember33_08,
        _HasMember33_09,
        _HasMember33_10,
        _HasMember33_11,
        _HasMember33_12,
        _HasMember33_13,
        _HasMember33_14,
        _HasMember33_15,
        _HasMember33_16,
        _HasMember33_17,
        _HasMember33_18,
        _HasMember33_19,
        _HasMember33_20,
        _HasMember33_21,
        _HasMember33_22,
        _HasMember33_23,
        _HasMember33_24,
        _HasMember33_25,
        _HasMember33_26,
        _HasMember33_27,
        _HasMember33_28,
        _HasMember33_29,
        _HasMember33_30,
        _HasMember33_31,
        _HasMember33_32,
        _HasMember33_33,
        _HasMember33_34,
        _HasMember33_35,
        _HasMember33_36,
        _HasMember33_37,
        _HasMember33_38,
        _HasMember33_39,
        _HasMember33_40,
        _HasMember33_41,
        _HasMember33_42,
        _HasMember33_43,
        _HasMember33_44,
    )
)
