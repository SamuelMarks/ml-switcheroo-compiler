"""Static structural protocol bindings shard 13 (conjugate_gradient..coo_sub)."""

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
class _HasMember13_00(typing.Protocol):
    """Structural protocol for objects exposing `conjugate_gradient`."""

    conjugate_gradient: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_01(typing.Protocol):
    """Structural protocol for objects exposing `constant_folding_pass`."""

    constant_folding_pass: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_02(typing.Protocol):
    """Structural protocol for objects exposing `contents`."""

    contents: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_03(typing.Protocol):
    """Structural protocol for objects exposing `control_flow`."""

    control_flow: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_04(typing.Protocol):
    """Structural protocol for objects exposing `control_flow_utils`."""

    control_flow_utils: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_05(typing.Protocol):
    """Structural protocol for objects exposing `conv`."""

    conv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_06(typing.Protocol):
    """Structural protocol for objects exposing `conv1d`."""

    conv1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_07(typing.Protocol):
    """Structural protocol for objects exposing `conv1d_lstm_cell`."""

    conv1d_lstm_cell: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_08(typing.Protocol):
    """Structural protocol for objects exposing `conv1d_transpose`."""

    conv1d_transpose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_09(typing.Protocol):
    """Structural protocol for objects exposing `conv2d`."""

    conv2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_10(typing.Protocol):
    """Structural protocol for objects exposing `conv2d_lstm_cell`."""

    conv2d_lstm_cell: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_11(typing.Protocol):
    """Structural protocol for objects exposing `conv2d_transpose`."""

    conv2d_transpose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_12(typing.Protocol):
    """Structural protocol for objects exposing `conv3d`."""

    conv3d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_13(typing.Protocol):
    """Structural protocol for objects exposing `conv3d_lstm_cell`."""

    conv3d_lstm_cell: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_14(typing.Protocol):
    """Structural protocol for objects exposing `conv3d_transpose`."""

    conv3d_transpose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_15(typing.Protocol):
    """Structural protocol for objects exposing `conv_general`."""

    conv_general: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_16(typing.Protocol):
    """Structural protocol for objects exposing `conv_general_dilated`."""

    conv_general_dilated: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_17(typing.Protocol):
    """Structural protocol for objects exposing `conv_general_dilated_local`."""

    conv_general_dilated_local: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_18(typing.Protocol):
    """Structural protocol for objects exposing `conv_general_dilated_patches`."""

    conv_general_dilated_patches: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_19(typing.Protocol):
    """Structural protocol for objects exposing `conv_transpose`."""

    conv_transpose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_20(typing.Protocol):
    """Structural protocol for objects exposing `conv_transpose1d`."""

    conv_transpose1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_21(typing.Protocol):
    """Structural protocol for objects exposing `conv_transpose2d`."""

    conv_transpose2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_22(typing.Protocol):
    """Structural protocol for objects exposing `conv_transpose3d`."""

    conv_transpose3d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_23(typing.Protocol):
    """Structural protocol for objects exposing `conv_with_general_padding`."""

    conv_with_general_padding: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_24(typing.Protocol):
    """Structural protocol for objects exposing `convert_to_numpy`."""

    convert_to_numpy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_25(typing.Protocol):
    """Structural protocol for objects exposing `convert_to_tensor`."""

    convert_to_tensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_26(typing.Protocol):
    """Structural protocol for objects exposing `convgeneraldilated`."""

    convgeneraldilated: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_27(typing.Protocol):
    """Structural protocol for objects exposing `convgeneraldilatedlocal`."""

    convgeneraldilatedlocal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_28(typing.Protocol):
    """Structural protocol for objects exposing `convgeneraldilatedpatches`."""

    convgeneraldilatedpatches: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_29(typing.Protocol):
    """Structural protocol for objects exposing `convolve`."""

    convolve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_30(typing.Protocol):
    """Structural protocol for objects exposing `convolve2d`."""

    convolve2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_31(typing.Protocol):
    """Structural protocol for objects exposing `convtranspose`."""

    convtranspose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_32(typing.Protocol):
    """Structural protocol for objects exposing `convwithgeneralpadding`."""

    convwithgeneralpadding: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_33(typing.Protocol):
    """Structural protocol for objects exposing `coo_abs`."""

    coo_abs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_34(typing.Protocol):
    """Structural protocol for objects exposing `coo_add`."""

    coo_add: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_35(typing.Protocol):
    """Structural protocol for objects exposing `coo_dot`."""

    coo_dot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_36(typing.Protocol):
    """Structural protocol for objects exposing `coo_fromdense`."""

    coo_fromdense: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_37(typing.Protocol):
    """Structural protocol for objects exposing `coo_matmat`."""

    coo_matmat: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_38(typing.Protocol):
    """Structural protocol for objects exposing `coo_matvec`."""

    coo_matvec: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_39(typing.Protocol):
    """Structural protocol for objects exposing `coo_mean`."""

    coo_mean: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_40(typing.Protocol):
    """Structural protocol for objects exposing `coo_mul`."""

    coo_mul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_41(typing.Protocol):
    """Structural protocol for objects exposing `coo_neg`."""

    coo_neg: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_42(typing.Protocol):
    """Structural protocol for objects exposing `coo_relu`."""

    coo_relu: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_43(typing.Protocol):
    """Structural protocol for objects exposing `coo_reshape`."""

    coo_reshape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember13_44(typing.Protocol):
    """Structural protocol for objects exposing `coo_sub`."""

    coo_sub: EagerTensorProtocol


SHARD_13_READERS: dict[str, MemberReader] = {
    "conjugate_gradient": lambda t: (True, t.conjugate_gradient) if isinstance(t, _HasMember13_00) else (False, None),
    "constant_folding_pass": lambda t: (
        (True, t.constant_folding_pass) if isinstance(t, _HasMember13_01) else (False, None)
    ),
    "contents": lambda t: (True, t.contents) if isinstance(t, _HasMember13_02) else (False, None),
    "control_flow": lambda t: (True, t.control_flow) if isinstance(t, _HasMember13_03) else (False, None),
    "control_flow_utils": lambda t: (True, t.control_flow_utils) if isinstance(t, _HasMember13_04) else (False, None),
    "conv": lambda t: (True, t.conv) if isinstance(t, _HasMember13_05) else (False, None),
    "conv1d": lambda t: (True, t.conv1d) if isinstance(t, _HasMember13_06) else (False, None),
    "conv1d_lstm_cell": lambda t: (True, t.conv1d_lstm_cell) if isinstance(t, _HasMember13_07) else (False, None),
    "conv1d_transpose": lambda t: (True, t.conv1d_transpose) if isinstance(t, _HasMember13_08) else (False, None),
    "conv2d": lambda t: (True, t.conv2d) if isinstance(t, _HasMember13_09) else (False, None),
    "conv2d_lstm_cell": lambda t: (True, t.conv2d_lstm_cell) if isinstance(t, _HasMember13_10) else (False, None),
    "conv2d_transpose": lambda t: (True, t.conv2d_transpose) if isinstance(t, _HasMember13_11) else (False, None),
    "conv3d": lambda t: (True, t.conv3d) if isinstance(t, _HasMember13_12) else (False, None),
    "conv3d_lstm_cell": lambda t: (True, t.conv3d_lstm_cell) if isinstance(t, _HasMember13_13) else (False, None),
    "conv3d_transpose": lambda t: (True, t.conv3d_transpose) if isinstance(t, _HasMember13_14) else (False, None),
    "conv_general": lambda t: (True, t.conv_general) if isinstance(t, _HasMember13_15) else (False, None),
    "conv_general_dilated": lambda t: (
        (True, t.conv_general_dilated) if isinstance(t, _HasMember13_16) else (False, None)
    ),
    "conv_general_dilated_local": lambda t: (
        (True, t.conv_general_dilated_local) if isinstance(t, _HasMember13_17) else (False, None)
    ),
    "conv_general_dilated_patches": lambda t: (
        (True, t.conv_general_dilated_patches) if isinstance(t, _HasMember13_18) else (False, None)
    ),
    "conv_transpose": lambda t: (True, t.conv_transpose) if isinstance(t, _HasMember13_19) else (False, None),
    "conv_transpose1d": lambda t: (True, t.conv_transpose1d) if isinstance(t, _HasMember13_20) else (False, None),
    "conv_transpose2d": lambda t: (True, t.conv_transpose2d) if isinstance(t, _HasMember13_21) else (False, None),
    "conv_transpose3d": lambda t: (True, t.conv_transpose3d) if isinstance(t, _HasMember13_22) else (False, None),
    "conv_with_general_padding": lambda t: (
        (True, t.conv_with_general_padding) if isinstance(t, _HasMember13_23) else (False, None)
    ),
    "convert_to_numpy": lambda t: (True, t.convert_to_numpy) if isinstance(t, _HasMember13_24) else (False, None),
    "convert_to_tensor": lambda t: (True, t.convert_to_tensor) if isinstance(t, _HasMember13_25) else (False, None),
    "convgeneraldilated": lambda t: (True, t.convgeneraldilated) if isinstance(t, _HasMember13_26) else (False, None),
    "convgeneraldilatedlocal": lambda t: (
        (True, t.convgeneraldilatedlocal) if isinstance(t, _HasMember13_27) else (False, None)
    ),
    "convgeneraldilatedpatches": lambda t: (
        (True, t.convgeneraldilatedpatches) if isinstance(t, _HasMember13_28) else (False, None)
    ),
    "convolve": lambda t: (True, t.convolve) if isinstance(t, _HasMember13_29) else (False, None),
    "convolve2d": lambda t: (True, t.convolve2d) if isinstance(t, _HasMember13_30) else (False, None),
    "convtranspose": lambda t: (True, t.convtranspose) if isinstance(t, _HasMember13_31) else (False, None),
    "convwithgeneralpadding": lambda t: (
        (True, t.convwithgeneralpadding) if isinstance(t, _HasMember13_32) else (False, None)
    ),
    "coo_abs": lambda t: (True, t.coo_abs) if isinstance(t, _HasMember13_33) else (False, None),
    "coo_add": lambda t: (True, t.coo_add) if isinstance(t, _HasMember13_34) else (False, None),
    "coo_dot": lambda t: (True, t.coo_dot) if isinstance(t, _HasMember13_35) else (False, None),
    "coo_fromdense": lambda t: (True, t.coo_fromdense) if isinstance(t, _HasMember13_36) else (False, None),
    "coo_matmat": lambda t: (True, t.coo_matmat) if isinstance(t, _HasMember13_37) else (False, None),
    "coo_matvec": lambda t: (True, t.coo_matvec) if isinstance(t, _HasMember13_38) else (False, None),
    "coo_mean": lambda t: (True, t.coo_mean) if isinstance(t, _HasMember13_39) else (False, None),
    "coo_mul": lambda t: (True, t.coo_mul) if isinstance(t, _HasMember13_40) else (False, None),
    "coo_neg": lambda t: (True, t.coo_neg) if isinstance(t, _HasMember13_41) else (False, None),
    "coo_relu": lambda t: (True, t.coo_relu) if isinstance(t, _HasMember13_42) else (False, None),
    "coo_reshape": lambda t: (True, t.coo_reshape) if isinstance(t, _HasMember13_43) else (False, None),
    "coo_sub": lambda t: (True, t.coo_sub) if isinstance(t, _HasMember13_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember13_00,
        _HasMember13_01,
        _HasMember13_02,
        _HasMember13_03,
        _HasMember13_04,
        _HasMember13_05,
        _HasMember13_06,
        _HasMember13_07,
        _HasMember13_08,
        _HasMember13_09,
        _HasMember13_10,
        _HasMember13_11,
        _HasMember13_12,
        _HasMember13_13,
        _HasMember13_14,
        _HasMember13_15,
        _HasMember13_16,
        _HasMember13_17,
        _HasMember13_18,
        _HasMember13_19,
        _HasMember13_20,
        _HasMember13_21,
        _HasMember13_22,
        _HasMember13_23,
        _HasMember13_24,
        _HasMember13_25,
        _HasMember13_26,
        _HasMember13_27,
        _HasMember13_28,
        _HasMember13_29,
        _HasMember13_30,
        _HasMember13_31,
        _HasMember13_32,
        _HasMember13_33,
        _HasMember13_34,
        _HasMember13_35,
        _HasMember13_36,
        _HasMember13_37,
        _HasMember13_38,
        _HasMember13_39,
        _HasMember13_40,
        _HasMember13_41,
        _HasMember13_42,
        _HasMember13_43,
        _HasMember13_44,
    )
)
