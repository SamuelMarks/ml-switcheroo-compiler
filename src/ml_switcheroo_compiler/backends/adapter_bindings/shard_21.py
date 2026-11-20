"""Static structural protocol bindings shard 21 (fabs..fix)."""

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
class _HasMember21_00(typing.Protocol):
    """Structural protocol for objects exposing `fabs`."""

    fabs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_01(typing.Protocol):
    """Structural protocol for objects exposing `fail_sig`."""

    fail_sig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_02(typing.Protocol):
    """Structural protocol for objects exposing `failing_import`."""

    failing_import: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_03(typing.Protocol):
    """Structural protocol for objects exposing `failing_model`."""

    failing_model: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_04(typing.Protocol):
    """Structural protocol for objects exposing `failing_prop`."""

    failing_prop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_05(typing.Protocol):
    """Structural protocol for objects exposing `failing_sig`."""

    failing_sig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_06(typing.Protocol):
    """Structural protocol for objects exposing `failing_typing_hints`."""

    failing_typing_hints: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_07(typing.Protocol):
    """Structural protocol for objects exposing `fake_abs`."""

    fake_abs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_08(typing.Protocol):
    """Structural protocol for objects exposing `fake_abspath`."""

    fake_abspath: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_09(typing.Protocol):
    """Structural protocol for objects exposing `fake_exists`."""

    fake_exists: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_10(typing.Protocol):
    """Structural protocol for objects exposing `fake_hip_malloc_success`."""

    fake_hip_malloc_success: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_11(typing.Protocol):
    """Structural protocol for objects exposing `fake_import`."""

    fake_import: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_12(typing.Protocol):
    """Structural protocol for objects exposing `fake_jvp`."""

    fake_jvp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_13(typing.Protocol):
    """Structural protocol for objects exposing `fake_jvp2`."""

    fake_jvp2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_14(typing.Protocol):
    """Structural protocol for objects exposing `fake_malloc_success`."""

    fake_malloc_success: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_15(typing.Protocol):
    """Structural protocol for objects exposing `fake_nb_import`."""

    fake_nb_import: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_16(typing.Protocol):
    """Structural protocol for objects exposing `fake_run_success`."""

    fake_run_success: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_17(typing.Protocol):
    """Structural protocol for objects exposing `fake_sparse_import`."""

    fake_sparse_import: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_18(typing.Protocol):
    """Structural protocol for objects exposing `fake_vjp`."""

    fake_vjp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_19(typing.Protocol):
    """Structural protocol for objects exposing `fake_vjp2`."""

    fake_vjp2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_20(typing.Protocol):
    """Structural protocol for objects exposing `fallback_op`."""

    fallback_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_21(typing.Protocol):
    """Structural protocol for objects exposing `false_fn`."""

    false_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_22(typing.Protocol):
    """Structural protocol for objects exposing `fast`."""

    fast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_23(typing.Protocol):
    """Structural protocol for objects exposing `fast_run_single`."""

    fast_run_single: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_24(typing.Protocol):
    """Structural protocol for objects exposing `fft`."""

    fft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_25(typing.Protocol):
    """Structural protocol for objects exposing `fft2`."""

    fft2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_26(typing.Protocol):
    """Structural protocol for objects exposing `fftconvolve`."""

    fftconvolve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_27(typing.Protocol):
    """Structural protocol for objects exposing `fftfreq`."""

    fftfreq: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_28(typing.Protocol):
    """Structural protocol for objects exposing `fftn`."""

    fftn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_29(typing.Protocol):
    """Structural protocol for objects exposing `fftnd`."""

    fftnd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_30(typing.Protocol):
    """Structural protocol for objects exposing `fftshift`."""

    fftshift: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_31(typing.Protocol):
    """Structural protocol for objects exposing `field`."""

    field: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_32(typing.Protocol):
    """Structural protocol for objects exposing `fill_diagonal`."""

    fill_diagonal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_33(typing.Protocol):
    """Structural protocol for objects exposing `fill_null`."""

    fill_null: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_34(typing.Protocol):
    """Structural protocol for objects exposing `fill_null_backward`."""

    fill_null_backward: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_35(typing.Protocol):
    """Structural protocol for objects exposing `fill_null_forward`."""

    fill_null_forward: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_36(typing.Protocol):
    """Structural protocol for objects exposing `fill_value`."""

    fill_value: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_37(typing.Protocol):
    """Structural protocol for objects exposing `filldiagonal`."""

    filldiagonal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_38(typing.Protocol):
    """Structural protocol for objects exposing `filled`."""

    filled: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_39(typing.Protocol):
    """Structural protocol for objects exposing `fillna`."""

    fillna: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_40(typing.Protocol):
    """Structural protocol for objects exposing `filter`."""

    filter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_41(typing.Protocol):
    """Structural protocol for objects exposing `filter_state`."""

    filter_state: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_42(typing.Protocol):
    """Structural protocol for objects exposing `filterstate`."""

    filterstate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_43(typing.Protocol):
    """Structural protocol for objects exposing `final_combine`."""

    final_combine: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember21_44(typing.Protocol):
    """Structural protocol for objects exposing `fix`."""

    fix: EagerTensorProtocol


SHARD_21_READERS: dict[str, MemberReader] = {
    "fabs": lambda t: (True, t.fabs) if isinstance(t, _HasMember21_00) else (False, None),
    "fail_sig": lambda t: (True, t.fail_sig) if isinstance(t, _HasMember21_01) else (False, None),
    "failing_import": lambda t: (True, t.failing_import) if isinstance(t, _HasMember21_02) else (False, None),
    "failing_model": lambda t: (True, t.failing_model) if isinstance(t, _HasMember21_03) else (False, None),
    "failing_prop": lambda t: (True, t.failing_prop) if isinstance(t, _HasMember21_04) else (False, None),
    "failing_sig": lambda t: (True, t.failing_sig) if isinstance(t, _HasMember21_05) else (False, None),
    "failing_typing_hints": lambda t: (
        (True, t.failing_typing_hints) if isinstance(t, _HasMember21_06) else (False, None)
    ),
    "fake_abs": lambda t: (True, t.fake_abs) if isinstance(t, _HasMember21_07) else (False, None),
    "fake_abspath": lambda t: (True, t.fake_abspath) if isinstance(t, _HasMember21_08) else (False, None),
    "fake_exists": lambda t: (True, t.fake_exists) if isinstance(t, _HasMember21_09) else (False, None),
    "fake_hip_malloc_success": lambda t: (
        (True, t.fake_hip_malloc_success) if isinstance(t, _HasMember21_10) else (False, None)
    ),
    "fake_import": lambda t: (True, t.fake_import) if isinstance(t, _HasMember21_11) else (False, None),
    "fake_jvp": lambda t: (True, t.fake_jvp) if isinstance(t, _HasMember21_12) else (False, None),
    "fake_jvp2": lambda t: (True, t.fake_jvp2) if isinstance(t, _HasMember21_13) else (False, None),
    "fake_malloc_success": lambda t: (True, t.fake_malloc_success) if isinstance(t, _HasMember21_14) else (False, None),
    "fake_nb_import": lambda t: (True, t.fake_nb_import) if isinstance(t, _HasMember21_15) else (False, None),
    "fake_run_success": lambda t: (True, t.fake_run_success) if isinstance(t, _HasMember21_16) else (False, None),
    "fake_sparse_import": lambda t: (True, t.fake_sparse_import) if isinstance(t, _HasMember21_17) else (False, None),
    "fake_vjp": lambda t: (True, t.fake_vjp) if isinstance(t, _HasMember21_18) else (False, None),
    "fake_vjp2": lambda t: (True, t.fake_vjp2) if isinstance(t, _HasMember21_19) else (False, None),
    "fallback_op": lambda t: (True, t.fallback_op) if isinstance(t, _HasMember21_20) else (False, None),
    "false_fn": lambda t: (True, t.false_fn) if isinstance(t, _HasMember21_21) else (False, None),
    "fast": lambda t: (True, t.fast) if isinstance(t, _HasMember21_22) else (False, None),
    "fast_run_single": lambda t: (True, t.fast_run_single) if isinstance(t, _HasMember21_23) else (False, None),
    "fft": lambda t: (True, t.fft) if isinstance(t, _HasMember21_24) else (False, None),
    "fft2": lambda t: (True, t.fft2) if isinstance(t, _HasMember21_25) else (False, None),
    "fftconvolve": lambda t: (True, t.fftconvolve) if isinstance(t, _HasMember21_26) else (False, None),
    "fftfreq": lambda t: (True, t.fftfreq) if isinstance(t, _HasMember21_27) else (False, None),
    "fftn": lambda t: (True, t.fftn) if isinstance(t, _HasMember21_28) else (False, None),
    "fftnd": lambda t: (True, t.fftnd) if isinstance(t, _HasMember21_29) else (False, None),
    "fftshift": lambda t: (True, t.fftshift) if isinstance(t, _HasMember21_30) else (False, None),
    "field": lambda t: (True, t.field) if isinstance(t, _HasMember21_31) else (False, None),
    "fill_diagonal": lambda t: (True, t.fill_diagonal) if isinstance(t, _HasMember21_32) else (False, None),
    "fill_null": lambda t: (True, t.fill_null) if isinstance(t, _HasMember21_33) else (False, None),
    "fill_null_backward": lambda t: (True, t.fill_null_backward) if isinstance(t, _HasMember21_34) else (False, None),
    "fill_null_forward": lambda t: (True, t.fill_null_forward) if isinstance(t, _HasMember21_35) else (False, None),
    "fill_value": lambda t: (True, t.fill_value) if isinstance(t, _HasMember21_36) else (False, None),
    "filldiagonal": lambda t: (True, t.filldiagonal) if isinstance(t, _HasMember21_37) else (False, None),
    "filled": lambda t: (True, t.filled) if isinstance(t, _HasMember21_38) else (False, None),
    "fillna": lambda t: (True, t.fillna) if isinstance(t, _HasMember21_39) else (False, None),
    "filter": lambda t: (True, t.filter) if isinstance(t, _HasMember21_40) else (False, None),
    "filter_state": lambda t: (True, t.filter_state) if isinstance(t, _HasMember21_41) else (False, None),
    "filterstate": lambda t: (True, t.filterstate) if isinstance(t, _HasMember21_42) else (False, None),
    "final_combine": lambda t: (True, t.final_combine) if isinstance(t, _HasMember21_43) else (False, None),
    "fix": lambda t: (True, t.fix) if isinstance(t, _HasMember21_44) else (False, None),
}
