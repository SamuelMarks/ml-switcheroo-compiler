"""Static structural protocol bindings shard 20 (extract_data..f_varpos_kw)."""

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
class _HasMember20_00(typing.Protocol):
    """Structural protocol for objects exposing `extract_data`."""

    extract_data: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_01(typing.Protocol):
    """Structural protocol for objects exposing `extract_device`."""

    extract_device: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_02(typing.Protocol):
    """Structural protocol for objects exposing `extract_dtype`."""

    extract_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_03(typing.Protocol):
    """Structural protocol for objects exposing `extract_patches`."""

    extract_patches: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_04(typing.Protocol):
    """Structural protocol for objects exposing `extract_patches_options`."""

    extract_patches_options: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_05(typing.Protocol):
    """Structural protocol for objects exposing `extract_proxy_inputs`."""

    extract_proxy_inputs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_06(typing.Protocol):
    """Structural protocol for objects exposing `extract_shape`."""

    extract_shape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_07(typing.Protocol):
    """Structural protocol for objects exposing `extract_volume_patches`."""

    extract_volume_patches: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_08(typing.Protocol):
    """Structural protocol for objects exposing `extractboundingboxes`."""

    extractboundingboxes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_09(typing.Protocol):
    """Structural protocol for objects exposing `extractpatchesoptions`."""

    extractpatchesoptions: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_10(typing.Protocol):
    """Structural protocol for objects exposing `eye`."""

    eye: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_11(typing.Protocol):
    """Structural protocol for objects exposing `f`."""

    f: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_12(typing.Protocol):
    """Structural protocol for objects exposing `f2`."""

    f2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_13(typing.Protocol):
    """Structural protocol for objects exposing `f_act`."""

    f_act: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_14(typing.Protocol):
    """Structural protocol for objects exposing `f_aux`."""

    f_aux: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_15(typing.Protocol):
    """Structural protocol for objects exposing `f_aux_fwd`."""

    f_aux_fwd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_16(typing.Protocol):
    """Structural protocol for objects exposing `f_aux_rev`."""

    f_aux_rev: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_17(typing.Protocol):
    """Structural protocol for objects exposing `f_bool`."""

    f_bool: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_18(typing.Protocol):
    """Structural protocol for objects exposing `f_cube`."""

    f_cube: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_19(typing.Protocol):
    """Structural protocol for objects exposing `f_custom`."""

    f_custom: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_20(typing.Protocol):
    """Structural protocol for objects exposing `f_custom_jvp`."""

    f_custom_jvp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_21(typing.Protocol):
    """Structural protocol for objects exposing `f_dot`."""

    f_dot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_22(typing.Protocol):
    """Structural protocol for objects exposing `f_erf`."""

    f_erf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_23(typing.Protocol):
    """Structural protocol for objects exposing `f_expand`."""

    f_expand: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_24(typing.Protocol):
    """Structural protocol for objects exposing `f_jac_aux`."""

    f_jac_aux: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_25(typing.Protocol):
    """Structural protocol for objects exposing `f_kwonly`."""

    f_kwonly: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_26(typing.Protocol):
    """Structural protocol for objects exposing `f_linear`."""

    f_linear: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_27(typing.Protocol):
    """Structural protocol for objects exposing `f_ln`."""

    f_ln: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_28(typing.Protocol):
    """Structural protocol for objects exposing `f_loss`."""

    f_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_29(typing.Protocol):
    """Structural protocol for objects exposing `f_matmul`."""

    f_matmul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_30(typing.Protocol):
    """Structural protocol for objects exposing `f_mean`."""

    f_mean: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_31(typing.Protocol):
    """Structural protocol for objects exposing `f_multi`."""

    f_multi: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_32(typing.Protocol):
    """Structural protocol for objects exposing `f_normal`."""

    f_normal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_33(typing.Protocol):
    """Structural protocol for objects exposing `f_outer`."""

    f_outer: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_34(typing.Protocol):
    """Structural protocol for objects exposing `f_relu`."""

    f_relu: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_35(typing.Protocol):
    """Structural protocol for objects exposing `f_reshape`."""

    f_reshape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_36(typing.Protocol):
    """Structural protocol for objects exposing `f_scalar`."""

    f_scalar: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_37(typing.Protocol):
    """Structural protocol for objects exposing `f_sig`."""

    f_sig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_38(typing.Protocol):
    """Structural protocol for objects exposing `f_sigmoid`."""

    f_sigmoid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_39(typing.Protocol):
    """Structural protocol for objects exposing `f_softmax`."""

    f_softmax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_40(typing.Protocol):
    """Structural protocol for objects exposing `f_squeeze`."""

    f_squeeze: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_41(typing.Protocol):
    """Structural protocol for objects exposing `f_sum`."""

    f_sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_42(typing.Protocol):
    """Structural protocol for objects exposing `f_transpose`."""

    f_transpose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_43(typing.Protocol):
    """Structural protocol for objects exposing `f_varargs`."""

    f_varargs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember20_44(typing.Protocol):
    """Structural protocol for objects exposing `f_varpos_kw`."""

    f_varpos_kw: EagerTensorProtocol


SHARD_20_READERS: dict[str, MemberReader] = {
    "extract_data": lambda t: (True, t.extract_data) if isinstance(t, _HasMember20_00) else (False, None),
    "extract_device": lambda t: (True, t.extract_device) if isinstance(t, _HasMember20_01) else (False, None),
    "extract_dtype": lambda t: (True, t.extract_dtype) if isinstance(t, _HasMember20_02) else (False, None),
    "extract_patches": lambda t: (True, t.extract_patches) if isinstance(t, _HasMember20_03) else (False, None),
    "extract_patches_options": lambda t: (
        (True, t.extract_patches_options) if isinstance(t, _HasMember20_04) else (False, None)
    ),
    "extract_proxy_inputs": lambda t: (
        (True, t.extract_proxy_inputs) if isinstance(t, _HasMember20_05) else (False, None)
    ),
    "extract_shape": lambda t: (True, t.extract_shape) if isinstance(t, _HasMember20_06) else (False, None),
    "extract_volume_patches": lambda t: (
        (True, t.extract_volume_patches) if isinstance(t, _HasMember20_07) else (False, None)
    ),
    "extractboundingboxes": lambda t: (
        (True, t.extractboundingboxes) if isinstance(t, _HasMember20_08) else (False, None)
    ),
    "extractpatchesoptions": lambda t: (
        (True, t.extractpatchesoptions) if isinstance(t, _HasMember20_09) else (False, None)
    ),
    "eye": lambda t: (True, t.eye) if isinstance(t, _HasMember20_10) else (False, None),
    "f": lambda t: (True, t.f) if isinstance(t, _HasMember20_11) else (False, None),
    "f2": lambda t: (True, t.f2) if isinstance(t, _HasMember20_12) else (False, None),
    "f_act": lambda t: (True, t.f_act) if isinstance(t, _HasMember20_13) else (False, None),
    "f_aux": lambda t: (True, t.f_aux) if isinstance(t, _HasMember20_14) else (False, None),
    "f_aux_fwd": lambda t: (True, t.f_aux_fwd) if isinstance(t, _HasMember20_15) else (False, None),
    "f_aux_rev": lambda t: (True, t.f_aux_rev) if isinstance(t, _HasMember20_16) else (False, None),
    "f_bool": lambda t: (True, t.f_bool) if isinstance(t, _HasMember20_17) else (False, None),
    "f_cube": lambda t: (True, t.f_cube) if isinstance(t, _HasMember20_18) else (False, None),
    "f_custom": lambda t: (True, t.f_custom) if isinstance(t, _HasMember20_19) else (False, None),
    "f_custom_jvp": lambda t: (True, t.f_custom_jvp) if isinstance(t, _HasMember20_20) else (False, None),
    "f_dot": lambda t: (True, t.f_dot) if isinstance(t, _HasMember20_21) else (False, None),
    "f_erf": lambda t: (True, t.f_erf) if isinstance(t, _HasMember20_22) else (False, None),
    "f_expand": lambda t: (True, t.f_expand) if isinstance(t, _HasMember20_23) else (False, None),
    "f_jac_aux": lambda t: (True, t.f_jac_aux) if isinstance(t, _HasMember20_24) else (False, None),
    "f_kwonly": lambda t: (True, t.f_kwonly) if isinstance(t, _HasMember20_25) else (False, None),
    "f_linear": lambda t: (True, t.f_linear) if isinstance(t, _HasMember20_26) else (False, None),
    "f_ln": lambda t: (True, t.f_ln) if isinstance(t, _HasMember20_27) else (False, None),
    "f_loss": lambda t: (True, t.f_loss) if isinstance(t, _HasMember20_28) else (False, None),
    "f_matmul": lambda t: (True, t.f_matmul) if isinstance(t, _HasMember20_29) else (False, None),
    "f_mean": lambda t: (True, t.f_mean) if isinstance(t, _HasMember20_30) else (False, None),
    "f_multi": lambda t: (True, t.f_multi) if isinstance(t, _HasMember20_31) else (False, None),
    "f_normal": lambda t: (True, t.f_normal) if isinstance(t, _HasMember20_32) else (False, None),
    "f_outer": lambda t: (True, t.f_outer) if isinstance(t, _HasMember20_33) else (False, None),
    "f_relu": lambda t: (True, t.f_relu) if isinstance(t, _HasMember20_34) else (False, None),
    "f_reshape": lambda t: (True, t.f_reshape) if isinstance(t, _HasMember20_35) else (False, None),
    "f_scalar": lambda t: (True, t.f_scalar) if isinstance(t, _HasMember20_36) else (False, None),
    "f_sig": lambda t: (True, t.f_sig) if isinstance(t, _HasMember20_37) else (False, None),
    "f_sigmoid": lambda t: (True, t.f_sigmoid) if isinstance(t, _HasMember20_38) else (False, None),
    "f_softmax": lambda t: (True, t.f_softmax) if isinstance(t, _HasMember20_39) else (False, None),
    "f_squeeze": lambda t: (True, t.f_squeeze) if isinstance(t, _HasMember20_40) else (False, None),
    "f_sum": lambda t: (True, t.f_sum) if isinstance(t, _HasMember20_41) else (False, None),
    "f_transpose": lambda t: (True, t.f_transpose) if isinstance(t, _HasMember20_42) else (False, None),
    "f_varargs": lambda t: (True, t.f_varargs) if isinstance(t, _HasMember20_43) else (False, None),
    "f_varpos_kw": lambda t: (True, t.f_varpos_kw) if isinstance(t, _HasMember20_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember20_00,
        _HasMember20_01,
        _HasMember20_02,
        _HasMember20_03,
        _HasMember20_04,
        _HasMember20_05,
        _HasMember20_06,
        _HasMember20_07,
        _HasMember20_08,
        _HasMember20_09,
        _HasMember20_10,
        _HasMember20_11,
        _HasMember20_12,
        _HasMember20_13,
        _HasMember20_14,
        _HasMember20_15,
        _HasMember20_16,
        _HasMember20_17,
        _HasMember20_18,
        _HasMember20_19,
        _HasMember20_20,
        _HasMember20_21,
        _HasMember20_22,
        _HasMember20_23,
        _HasMember20_24,
        _HasMember20_25,
        _HasMember20_26,
        _HasMember20_27,
        _HasMember20_28,
        _HasMember20_29,
        _HasMember20_30,
        _HasMember20_31,
        _HasMember20_32,
        _HasMember20_33,
        _HasMember20_34,
        _HasMember20_35,
        _HasMember20_36,
        _HasMember20_37,
        _HasMember20_38,
        _HasMember20_39,
        _HasMember20_40,
        _HasMember20_41,
        _HasMember20_42,
        _HasMember20_43,
        _HasMember20_44,
    )
)
