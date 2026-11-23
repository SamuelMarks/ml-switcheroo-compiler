"""Static structural protocol bindings shard 34 (mock_glob..model_dump)."""

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
class _HasMember34_00(typing.Protocol):
    """Structural protocol for objects exposing `mock_glob`."""

    mock_glob: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_01(typing.Protocol):
    """Structural protocol for objects exposing `mock_hasattr`."""

    mock_hasattr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_02(typing.Protocol):
    """Structural protocol for objects exposing `mock_hist`."""

    mock_hist: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_03(typing.Protocol):
    """Structural protocol for objects exposing `mock_hist2`."""

    mock_hist2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_04(typing.Protocol):
    """Structural protocol for objects exposing `mock_import`."""

    mock_import: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_05(typing.Protocol):
    """Structural protocol for objects exposing `mock_import_ge_fail`."""

    mock_import_ge_fail: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_06(typing.Protocol):
    """Structural protocol for objects exposing `mock_infer_op_shape`."""

    mock_infer_op_shape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_07(typing.Protocol):
    """Structural protocol for objects exposing `mock_iou`."""

    mock_iou: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_08(typing.Protocol):
    """Structural protocol for objects exposing `mock_isdir`."""

    mock_isdir: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_09(typing.Protocol):
    """Structural protocol for objects exposing `mock_isdir_false`."""

    mock_isdir_false: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_10(typing.Protocol):
    """Structural protocol for objects exposing `mock_join`."""

    mock_join: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_11(typing.Protocol):
    """Structural protocol for objects exposing `mock_jvp`."""

    mock_jvp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_12(typing.Protocol):
    """Structural protocol for objects exposing `mock_jvp_bad`."""

    mock_jvp_bad: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_13(typing.Protocol):
    """Structural protocol for objects exposing `mock_listdir`."""

    mock_listdir: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_14(typing.Protocol):
    """Structural protocol for objects exposing `mock_listdir_cache`."""

    mock_listdir_cache: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_15(typing.Protocol):
    """Structural protocol for objects exposing `mock_matmul`."""

    mock_matmul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_16(typing.Protocol):
    """Structural protocol for objects exposing `mock_nb_custom`."""

    mock_nb_custom: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_17(typing.Protocol):
    """Structural protocol for objects exposing `mock_nms`."""

    mock_nms: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_18(typing.Protocol):
    """Structural protocol for objects exposing `mock_open`."""

    mock_open: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_19(typing.Protocol):
    """Structural protocol for objects exposing `mock_open_error`."""

    mock_open_error: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_20(typing.Protocol):
    """Structural protocol for objects exposing `mock_open_file`."""

    mock_open_file: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_21(typing.Protocol):
    """Structural protocol for objects exposing `mock_open_impl`."""

    mock_open_impl: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_22(typing.Protocol):
    """Structural protocol for objects exposing `mock_perf_counter`."""

    mock_perf_counter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_23(typing.Protocol):
    """Structural protocol for objects exposing `mock_prepare`."""

    mock_prepare: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_24(typing.Protocol):
    """Structural protocol for objects exposing `mock_recv`."""

    mock_recv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_25(typing.Protocol):
    """Structural protocol for objects exposing `mock_recv_ping`."""

    mock_recv_ping: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_26(typing.Protocol):
    """Structural protocol for objects exposing `mock_relpath`."""

    mock_relpath: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_27(typing.Protocol):
    """Structural protocol for objects exposing `mock_replace_error`."""

    mock_replace_error: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_28(typing.Protocol):
    """Structural protocol for objects exposing `mock_rnn`."""

    mock_rnn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_29(typing.Protocol):
    """Structural protocol for objects exposing `mock_scan`."""

    mock_scan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_30(typing.Protocol):
    """Structural protocol for objects exposing `mock_shard_tensor`."""

    mock_shard_tensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_31(typing.Protocol):
    """Structural protocol for objects exposing `mock_sleep`."""

    mock_sleep: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_32(typing.Protocol):
    """Structural protocol for objects exposing `mock_subprocess_run`."""

    mock_subprocess_run: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_33(typing.Protocol):
    """Structural protocol for objects exposing `mock_svd`."""

    mock_svd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_34(typing.Protocol):
    """Structural protocol for objects exposing `mock_sync_drain`."""

    mock_sync_drain: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_35(typing.Protocol):
    """Structural protocol for objects exposing `mock_target`."""

    mock_target: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_36(typing.Protocol):
    """Structural protocol for objects exposing `mock_template_resolver`."""

    mock_template_resolver: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_37(typing.Protocol):
    """Structural protocol for objects exposing `mock_time`."""

    mock_time: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_38(typing.Protocol):
    """Structural protocol for objects exposing `mock_uuid4`."""

    mock_uuid4: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_39(typing.Protocol):
    """Structural protocol for objects exposing `mock_wait_drain`."""

    mock_wait_drain: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_40(typing.Protocol):
    """Structural protocol for objects exposing `mock_walk`."""

    mock_walk: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_41(typing.Protocol):
    """Structural protocol for objects exposing `mock_warn`."""

    mock_warn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_42(typing.Protocol):
    """Structural protocol for objects exposing `mod`."""

    mod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_43(typing.Protocol):
    """Structural protocol for objects exposing `mode`."""

    mode: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember34_44(typing.Protocol):
    """Structural protocol for objects exposing `model_dump`."""

    model_dump: EagerTensorProtocol


SHARD_34_READERS: dict[str, MemberReader] = {
    "mock_glob": lambda t: (True, t.mock_glob) if isinstance(t, _HasMember34_00) else (False, None),
    "mock_hasattr": lambda t: (True, t.mock_hasattr) if isinstance(t, _HasMember34_01) else (False, None),
    "mock_hist": lambda t: (True, t.mock_hist) if isinstance(t, _HasMember34_02) else (False, None),
    "mock_hist2": lambda t: (True, t.mock_hist2) if isinstance(t, _HasMember34_03) else (False, None),
    "mock_import": lambda t: (True, t.mock_import) if isinstance(t, _HasMember34_04) else (False, None),
    "mock_import_ge_fail": lambda t: (True, t.mock_import_ge_fail) if isinstance(t, _HasMember34_05) else (False, None),
    "mock_infer_op_shape": lambda t: (True, t.mock_infer_op_shape) if isinstance(t, _HasMember34_06) else (False, None),
    "mock_iou": lambda t: (True, t.mock_iou) if isinstance(t, _HasMember34_07) else (False, None),
    "mock_isdir": lambda t: (True, t.mock_isdir) if isinstance(t, _HasMember34_08) else (False, None),
    "mock_isdir_false": lambda t: (True, t.mock_isdir_false) if isinstance(t, _HasMember34_09) else (False, None),
    "mock_join": lambda t: (True, t.mock_join) if isinstance(t, _HasMember34_10) else (False, None),
    "mock_jvp": lambda t: (True, t.mock_jvp) if isinstance(t, _HasMember34_11) else (False, None),
    "mock_jvp_bad": lambda t: (True, t.mock_jvp_bad) if isinstance(t, _HasMember34_12) else (False, None),
    "mock_listdir": lambda t: (True, t.mock_listdir) if isinstance(t, _HasMember34_13) else (False, None),
    "mock_listdir_cache": lambda t: (True, t.mock_listdir_cache) if isinstance(t, _HasMember34_14) else (False, None),
    "mock_matmul": lambda t: (True, t.mock_matmul) if isinstance(t, _HasMember34_15) else (False, None),
    "mock_nb_custom": lambda t: (True, t.mock_nb_custom) if isinstance(t, _HasMember34_16) else (False, None),
    "mock_nms": lambda t: (True, t.mock_nms) if isinstance(t, _HasMember34_17) else (False, None),
    "mock_open": lambda t: (True, t.mock_open) if isinstance(t, _HasMember34_18) else (False, None),
    "mock_open_error": lambda t: (True, t.mock_open_error) if isinstance(t, _HasMember34_19) else (False, None),
    "mock_open_file": lambda t: (True, t.mock_open_file) if isinstance(t, _HasMember34_20) else (False, None),
    "mock_open_impl": lambda t: (True, t.mock_open_impl) if isinstance(t, _HasMember34_21) else (False, None),
    "mock_perf_counter": lambda t: (True, t.mock_perf_counter) if isinstance(t, _HasMember34_22) else (False, None),
    "mock_prepare": lambda t: (True, t.mock_prepare) if isinstance(t, _HasMember34_23) else (False, None),
    "mock_recv": lambda t: (True, t.mock_recv) if isinstance(t, _HasMember34_24) else (False, None),
    "mock_recv_ping": lambda t: (True, t.mock_recv_ping) if isinstance(t, _HasMember34_25) else (False, None),
    "mock_relpath": lambda t: (True, t.mock_relpath) if isinstance(t, _HasMember34_26) else (False, None),
    "mock_replace_error": lambda t: (True, t.mock_replace_error) if isinstance(t, _HasMember34_27) else (False, None),
    "mock_rnn": lambda t: (True, t.mock_rnn) if isinstance(t, _HasMember34_28) else (False, None),
    "mock_scan": lambda t: (True, t.mock_scan) if isinstance(t, _HasMember34_29) else (False, None),
    "mock_shard_tensor": lambda t: (True, t.mock_shard_tensor) if isinstance(t, _HasMember34_30) else (False, None),
    "mock_sleep": lambda t: (True, t.mock_sleep) if isinstance(t, _HasMember34_31) else (False, None),
    "mock_subprocess_run": lambda t: (True, t.mock_subprocess_run) if isinstance(t, _HasMember34_32) else (False, None),
    "mock_svd": lambda t: (True, t.mock_svd) if isinstance(t, _HasMember34_33) else (False, None),
    "mock_sync_drain": lambda t: (True, t.mock_sync_drain) if isinstance(t, _HasMember34_34) else (False, None),
    "mock_target": lambda t: (True, t.mock_target) if isinstance(t, _HasMember34_35) else (False, None),
    "mock_template_resolver": lambda t: (
        (True, t.mock_template_resolver) if isinstance(t, _HasMember34_36) else (False, None)
    ),
    "mock_time": lambda t: (True, t.mock_time) if isinstance(t, _HasMember34_37) else (False, None),
    "mock_uuid4": lambda t: (True, t.mock_uuid4) if isinstance(t, _HasMember34_38) else (False, None),
    "mock_wait_drain": lambda t: (True, t.mock_wait_drain) if isinstance(t, _HasMember34_39) else (False, None),
    "mock_walk": lambda t: (True, t.mock_walk) if isinstance(t, _HasMember34_40) else (False, None),
    "mock_warn": lambda t: (True, t.mock_warn) if isinstance(t, _HasMember34_41) else (False, None),
    "mod": lambda t: (True, t.mod) if isinstance(t, _HasMember34_42) else (False, None),
    "mode": lambda t: (True, t.mode) if isinstance(t, _HasMember34_43) else (False, None),
    "model_dump": lambda t: (True, t.model_dump) if isinstance(t, _HasMember34_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember34_00,
        _HasMember34_01,
        _HasMember34_02,
        _HasMember34_03,
        _HasMember34_04,
        _HasMember34_05,
        _HasMember34_06,
        _HasMember34_07,
        _HasMember34_08,
        _HasMember34_09,
        _HasMember34_10,
        _HasMember34_11,
        _HasMember34_12,
        _HasMember34_13,
        _HasMember34_14,
        _HasMember34_15,
        _HasMember34_16,
        _HasMember34_17,
        _HasMember34_18,
        _HasMember34_19,
        _HasMember34_20,
        _HasMember34_21,
        _HasMember34_22,
        _HasMember34_23,
        _HasMember34_24,
        _HasMember34_25,
        _HasMember34_26,
        _HasMember34_27,
        _HasMember34_28,
        _HasMember34_29,
        _HasMember34_30,
        _HasMember34_31,
        _HasMember34_32,
        _HasMember34_33,
        _HasMember34_34,
        _HasMember34_35,
        _HasMember34_36,
        _HasMember34_37,
        _HasMember34_38,
        _HasMember34_39,
        _HasMember34_40,
        _HasMember34_41,
        _HasMember34_42,
        _HasMember34_43,
        _HasMember34_44,
    )
)
