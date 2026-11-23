"""Static structural protocol bindings shard 35 (model_raising..my_target)."""

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
class _HasMember35_00(typing.Protocol):
    """Structural protocol for objects exposing `model_raising`."""

    model_raising: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_01(typing.Protocol):
    """Structural protocol for objects exposing `model_scalar_out`."""

    model_scalar_out: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_02(typing.Protocol):
    """Structural protocol for objects exposing `model_with_shape`."""

    model_with_shape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_03(typing.Protocol):
    """Structural protocol for objects exposing `modf`."""

    modf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_04(typing.Protocol):
    """Structural protocol for objects exposing `modified_bessel_i0`."""

    modified_bessel_i0: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_05(typing.Protocol):
    """Structural protocol for objects exposing `modified_bessel_i1`."""

    modified_bessel_i1: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_06(typing.Protocol):
    """Structural protocol for objects exposing `modified_bessel_k0`."""

    modified_bessel_k0: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_07(typing.Protocol):
    """Structural protocol for objects exposing `modified_bessel_k1`."""

    modified_bessel_k1: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_08(typing.Protocol):
    """Structural protocol for objects exposing `module`."""

    module: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_09(typing.Protocol):
    """Structural protocol for objects exposing `moveaxis`."""

    moveaxis: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_10(typing.Protocol):
    """Structural protocol for objects exposing `mps`."""

    mps: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_11(typing.Protocol):
    """Structural protocol for objects exposing `mse_loss`."""

    mse_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_12(typing.Protocol):
    """Structural protocol for objects exposing `msle`."""

    msle: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_13(typing.Protocol):
    """Structural protocol for objects exposing `msle_loss`."""

    msle_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_14(typing.Protocol):
    """Structural protocol for objects exposing `mul`."""

    mul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_15(typing.Protocol):
    """Structural protocol for objects exposing `multi_dot`."""

    multi_dot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_16(typing.Protocol):
    """Structural protocol for objects exposing `multi_head_attention_forward`."""

    multi_head_attention_forward: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_17(typing.Protocol):
    """Structural protocol for objects exposing `multiply`."""

    multiply: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_18(typing.Protocol):
    """Structural protocol for objects exposing `multiply_no_nan`."""

    multiply_no_nan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_19(typing.Protocol):
    """Structural protocol for objects exposing `mvlgamma`."""

    mvlgamma: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_20(typing.Protocol):
    """Structural protocol for objects exposing `my_backend_op`."""

    my_backend_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_21(typing.Protocol):
    """Structural protocol for objects exposing `my_bwd`."""

    my_bwd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_22(typing.Protocol):
    """Structural protocol for objects exposing `my_caller`."""

    my_caller: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_23(typing.Protocol):
    """Structural protocol for objects exposing `my_cell`."""

    my_cell: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_24(typing.Protocol):
    """Structural protocol for objects exposing `my_cell_res`."""

    my_cell_res: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_25(typing.Protocol):
    """Structural protocol for objects exposing `my_cell_trace`."""

    my_cell_trace: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_26(typing.Protocol):
    """Structural protocol for objects exposing `my_cm`."""

    my_cm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_27(typing.Protocol):
    """Structural protocol for objects exposing `my_complex_func`."""

    my_complex_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_28(typing.Protocol):
    """Structural protocol for objects exposing `my_desc`."""

    my_desc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_29(typing.Protocol):
    """Structural protocol for objects exposing `my_dist`."""

    my_dist: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_30(typing.Protocol):
    """Structural protocol for objects exposing `my_f`."""

    my_f: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_31(typing.Protocol):
    """Structural protocol for objects exposing `my_fn`."""

    my_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_32(typing.Protocol):
    """Structural protocol for objects exposing `my_frontend_fn`."""

    my_frontend_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_33(typing.Protocol):
    """Structural protocol for objects exposing `my_fun`."""

    my_fun: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_34(typing.Protocol):
    """Structural protocol for objects exposing `my_func`."""

    my_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_35(typing.Protocol):
    """Structural protocol for objects exposing `my_hook`."""

    my_hook: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_36(typing.Protocol):
    """Structural protocol for objects exposing `my_infer_shape`."""

    my_infer_shape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_37(typing.Protocol):
    """Structural protocol for objects exposing `my_jvp`."""

    my_jvp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_38(typing.Protocol):
    """Structural protocol for objects exposing `my_linalg_func`."""

    my_linalg_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_39(typing.Protocol):
    """Structural protocol for objects exposing `my_method`."""

    my_method: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_40(typing.Protocol):
    """Structural protocol for objects exposing `my_op`."""

    my_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_41(typing.Protocol):
    """Structural protocol for objects exposing `my_rem`."""

    my_rem: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_42(typing.Protocol):
    """Structural protocol for objects exposing `my_res_fn`."""

    my_res_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_43(typing.Protocol):
    """Structural protocol for objects exposing `my_sum`."""

    my_sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember35_44(typing.Protocol):
    """Structural protocol for objects exposing `my_target`."""

    my_target: EagerTensorProtocol


SHARD_35_READERS: dict[str, MemberReader] = {
    "model_raising": lambda t: (True, t.model_raising) if isinstance(t, _HasMember35_00) else (False, None),
    "model_scalar_out": lambda t: (True, t.model_scalar_out) if isinstance(t, _HasMember35_01) else (False, None),
    "model_with_shape": lambda t: (True, t.model_with_shape) if isinstance(t, _HasMember35_02) else (False, None),
    "modf": lambda t: (True, t.modf) if isinstance(t, _HasMember35_03) else (False, None),
    "modified_bessel_i0": lambda t: (True, t.modified_bessel_i0) if isinstance(t, _HasMember35_04) else (False, None),
    "modified_bessel_i1": lambda t: (True, t.modified_bessel_i1) if isinstance(t, _HasMember35_05) else (False, None),
    "modified_bessel_k0": lambda t: (True, t.modified_bessel_k0) if isinstance(t, _HasMember35_06) else (False, None),
    "modified_bessel_k1": lambda t: (True, t.modified_bessel_k1) if isinstance(t, _HasMember35_07) else (False, None),
    "module": lambda t: (True, t.module) if isinstance(t, _HasMember35_08) else (False, None),
    "moveaxis": lambda t: (True, t.moveaxis) if isinstance(t, _HasMember35_09) else (False, None),
    "mps": lambda t: (True, t.mps) if isinstance(t, _HasMember35_10) else (False, None),
    "mse_loss": lambda t: (True, t.mse_loss) if isinstance(t, _HasMember35_11) else (False, None),
    "msle": lambda t: (True, t.msle) if isinstance(t, _HasMember35_12) else (False, None),
    "msle_loss": lambda t: (True, t.msle_loss) if isinstance(t, _HasMember35_13) else (False, None),
    "mul": lambda t: (True, t.mul) if isinstance(t, _HasMember35_14) else (False, None),
    "multi_dot": lambda t: (True, t.multi_dot) if isinstance(t, _HasMember35_15) else (False, None),
    "multi_head_attention_forward": lambda t: (
        (True, t.multi_head_attention_forward) if isinstance(t, _HasMember35_16) else (False, None)
    ),
    "multiply": lambda t: (True, t.multiply) if isinstance(t, _HasMember35_17) else (False, None),
    "multiply_no_nan": lambda t: (True, t.multiply_no_nan) if isinstance(t, _HasMember35_18) else (False, None),
    "mvlgamma": lambda t: (True, t.mvlgamma) if isinstance(t, _HasMember35_19) else (False, None),
    "my_backend_op": lambda t: (True, t.my_backend_op) if isinstance(t, _HasMember35_20) else (False, None),
    "my_bwd": lambda t: (True, t.my_bwd) if isinstance(t, _HasMember35_21) else (False, None),
    "my_caller": lambda t: (True, t.my_caller) if isinstance(t, _HasMember35_22) else (False, None),
    "my_cell": lambda t: (True, t.my_cell) if isinstance(t, _HasMember35_23) else (False, None),
    "my_cell_res": lambda t: (True, t.my_cell_res) if isinstance(t, _HasMember35_24) else (False, None),
    "my_cell_trace": lambda t: (True, t.my_cell_trace) if isinstance(t, _HasMember35_25) else (False, None),
    "my_cm": lambda t: (True, t.my_cm) if isinstance(t, _HasMember35_26) else (False, None),
    "my_complex_func": lambda t: (True, t.my_complex_func) if isinstance(t, _HasMember35_27) else (False, None),
    "my_desc": lambda t: (True, t.my_desc) if isinstance(t, _HasMember35_28) else (False, None),
    "my_dist": lambda t: (True, t.my_dist) if isinstance(t, _HasMember35_29) else (False, None),
    "my_f": lambda t: (True, t.my_f) if isinstance(t, _HasMember35_30) else (False, None),
    "my_fn": lambda t: (True, t.my_fn) if isinstance(t, _HasMember35_31) else (False, None),
    "my_frontend_fn": lambda t: (True, t.my_frontend_fn) if isinstance(t, _HasMember35_32) else (False, None),
    "my_fun": lambda t: (True, t.my_fun) if isinstance(t, _HasMember35_33) else (False, None),
    "my_func": lambda t: (True, t.my_func) if isinstance(t, _HasMember35_34) else (False, None),
    "my_hook": lambda t: (True, t.my_hook) if isinstance(t, _HasMember35_35) else (False, None),
    "my_infer_shape": lambda t: (True, t.my_infer_shape) if isinstance(t, _HasMember35_36) else (False, None),
    "my_jvp": lambda t: (True, t.my_jvp) if isinstance(t, _HasMember35_37) else (False, None),
    "my_linalg_func": lambda t: (True, t.my_linalg_func) if isinstance(t, _HasMember35_38) else (False, None),
    "my_method": lambda t: (True, t.my_method) if isinstance(t, _HasMember35_39) else (False, None),
    "my_op": lambda t: (True, t.my_op) if isinstance(t, _HasMember35_40) else (False, None),
    "my_rem": lambda t: (True, t.my_rem) if isinstance(t, _HasMember35_41) else (False, None),
    "my_res_fn": lambda t: (True, t.my_res_fn) if isinstance(t, _HasMember35_42) else (False, None),
    "my_sum": lambda t: (True, t.my_sum) if isinstance(t, _HasMember35_43) else (False, None),
    "my_target": lambda t: (True, t.my_target) if isinstance(t, _HasMember35_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember35_00,
        _HasMember35_01,
        _HasMember35_02,
        _HasMember35_03,
        _HasMember35_04,
        _HasMember35_05,
        _HasMember35_06,
        _HasMember35_07,
        _HasMember35_08,
        _HasMember35_09,
        _HasMember35_10,
        _HasMember35_11,
        _HasMember35_12,
        _HasMember35_13,
        _HasMember35_14,
        _HasMember35_15,
        _HasMember35_16,
        _HasMember35_17,
        _HasMember35_18,
        _HasMember35_19,
        _HasMember35_20,
        _HasMember35_21,
        _HasMember35_22,
        _HasMember35_23,
        _HasMember35_24,
        _HasMember35_25,
        _HasMember35_26,
        _HasMember35_27,
        _HasMember35_28,
        _HasMember35_29,
        _HasMember35_30,
        _HasMember35_31,
        _HasMember35_32,
        _HasMember35_33,
        _HasMember35_34,
        _HasMember35_35,
        _HasMember35_36,
        _HasMember35_37,
        _HasMember35_38,
        _HasMember35_39,
        _HasMember35_40,
        _HasMember35_41,
        _HasMember35_42,
        _HasMember35_43,
        _HasMember35_44,
    )
)
