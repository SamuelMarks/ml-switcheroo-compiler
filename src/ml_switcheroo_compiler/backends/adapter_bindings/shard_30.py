"""Static structural protocol bindings shard 30 (linear_operator_toeplitz..log_softmax)."""

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
class _HasMember30_00(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_toeplitz`."""

    linear_operator_toeplitz: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_01(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_tridiag`."""

    linear_operator_tridiag: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_02(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_zeros`."""

    linear_operator_zeros: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_03(typing.Protocol):
    """Structural protocol for objects exposing `linearoperator`."""

    linearoperator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_04(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatoradjoint`."""

    linearoperatoradjoint: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_05(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorblockdiag`."""

    linearoperatorblockdiag: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_06(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorblocklowertriangular`."""

    linearoperatorblocklowertriangular: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_07(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorcirculant`."""

    linearoperatorcirculant: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_08(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorcirculant2d`."""

    linearoperatorcirculant2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_09(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorcirculant3d`."""

    linearoperatorcirculant3d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_10(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorcomposition`."""

    linearoperatorcomposition: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_11(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatordiag`."""

    linearoperatordiag: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_12(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorfullmatrix`."""

    linearoperatorfullmatrix: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_13(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorhouseholder`."""

    linearoperatorhouseholder: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_14(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatoridentity`."""

    linearoperatoridentity: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_15(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorinversion`."""

    linearoperatorinversion: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_16(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorkronecker`."""

    linearoperatorkronecker: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_17(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorlowertriangular`."""

    linearoperatorlowertriangular: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_18(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorlowrankupdate`."""

    linearoperatorlowrankupdate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_19(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatoroperandprotocol`."""

    linearoperatoroperandprotocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_20(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorpermutation`."""

    linearoperatorpermutation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_21(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorscaledidentity`."""

    linearoperatorscaledidentity: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_22(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatortoeplitz`."""

    linearoperatortoeplitz: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_23(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatortridiag`."""

    linearoperatortridiag: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_24(typing.Protocol):
    """Structural protocol for objects exposing `linearoperatorzeros`."""

    linearoperatorzeros: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_25(typing.Protocol):
    """Structural protocol for objects exposing `linspace`."""

    linspace: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_26(typing.Protocol):
    """Structural protocol for objects exposing `list_element`."""

    list_element: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_27(typing.Protocol):
    """Structural protocol for objects exposing `list_flatten`."""

    list_flatten: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_28(typing.Protocol):
    """Structural protocol for objects exposing `list_parent_indices`."""

    list_parent_indices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_29(typing.Protocol):
    """Structural protocol for objects exposing `list_slice`."""

    list_slice: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_30(typing.Protocol):
    """Structural protocol for objects exposing `list_value_length`."""

    list_value_length: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_31(typing.Protocol):
    """Structural protocol for objects exposing `ln`."""

    ln: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_32(typing.Protocol):
    """Structural protocol for objects exposing `load`."""

    load: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_33(typing.Protocol):
    """Structural protocol for objects exposing `load_h5`."""

    load_h5: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_34(typing.Protocol):
    """Structural protocol for objects exposing `load_npz`."""

    load_npz: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_35(typing.Protocol):
    """Structural protocol for objects exposing `load_state_dict`."""

    load_state_dict: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_36(typing.Protocol):
    """Structural protocol for objects exposing `load_target`."""

    load_target: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_37(typing.Protocol):
    """Structural protocol for objects exposing `log`."""

    log: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_38(typing.Protocol):
    """Structural protocol for objects exposing `log10`."""

    log10: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_39(typing.Protocol):
    """Structural protocol for objects exposing `log1p`."""

    log1p: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_40(typing.Protocol):
    """Structural protocol for objects exposing `log2`."""

    log2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_41(typing.Protocol):
    """Structural protocol for objects exposing `log_cosh_loss`."""

    log_cosh_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_42(typing.Protocol):
    """Structural protocol for objects exposing `log_poisson_loss`."""

    log_poisson_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_43(typing.Protocol):
    """Structural protocol for objects exposing `log_sigmoid`."""

    log_sigmoid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember30_44(typing.Protocol):
    """Structural protocol for objects exposing `log_softmax`."""

    log_softmax: EagerTensorProtocol


SHARD_30_READERS: dict[str, MemberReader] = {
    "linear_operator_toeplitz": lambda t: (
        (True, t.linear_operator_toeplitz) if isinstance(t, _HasMember30_00) else (False, None)
    ),
    "linear_operator_tridiag": lambda t: (
        (True, t.linear_operator_tridiag) if isinstance(t, _HasMember30_01) else (False, None)
    ),
    "linear_operator_zeros": lambda t: (
        (True, t.linear_operator_zeros) if isinstance(t, _HasMember30_02) else (False, None)
    ),
    "linearoperator": lambda t: (True, t.linearoperator) if isinstance(t, _HasMember30_03) else (False, None),
    "linearoperatoradjoint": lambda t: (
        (True, t.linearoperatoradjoint) if isinstance(t, _HasMember30_04) else (False, None)
    ),
    "linearoperatorblockdiag": lambda t: (
        (True, t.linearoperatorblockdiag) if isinstance(t, _HasMember30_05) else (False, None)
    ),
    "linearoperatorblocklowertriangular": lambda t: (
        (True, t.linearoperatorblocklowertriangular) if isinstance(t, _HasMember30_06) else (False, None)
    ),
    "linearoperatorcirculant": lambda t: (
        (True, t.linearoperatorcirculant) if isinstance(t, _HasMember30_07) else (False, None)
    ),
    "linearoperatorcirculant2d": lambda t: (
        (True, t.linearoperatorcirculant2d) if isinstance(t, _HasMember30_08) else (False, None)
    ),
    "linearoperatorcirculant3d": lambda t: (
        (True, t.linearoperatorcirculant3d) if isinstance(t, _HasMember30_09) else (False, None)
    ),
    "linearoperatorcomposition": lambda t: (
        (True, t.linearoperatorcomposition) if isinstance(t, _HasMember30_10) else (False, None)
    ),
    "linearoperatordiag": lambda t: (True, t.linearoperatordiag) if isinstance(t, _HasMember30_11) else (False, None),
    "linearoperatorfullmatrix": lambda t: (
        (True, t.linearoperatorfullmatrix) if isinstance(t, _HasMember30_12) else (False, None)
    ),
    "linearoperatorhouseholder": lambda t: (
        (True, t.linearoperatorhouseholder) if isinstance(t, _HasMember30_13) else (False, None)
    ),
    "linearoperatoridentity": lambda t: (
        (True, t.linearoperatoridentity) if isinstance(t, _HasMember30_14) else (False, None)
    ),
    "linearoperatorinversion": lambda t: (
        (True, t.linearoperatorinversion) if isinstance(t, _HasMember30_15) else (False, None)
    ),
    "linearoperatorkronecker": lambda t: (
        (True, t.linearoperatorkronecker) if isinstance(t, _HasMember30_16) else (False, None)
    ),
    "linearoperatorlowertriangular": lambda t: (
        (True, t.linearoperatorlowertriangular) if isinstance(t, _HasMember30_17) else (False, None)
    ),
    "linearoperatorlowrankupdate": lambda t: (
        (True, t.linearoperatorlowrankupdate) if isinstance(t, _HasMember30_18) else (False, None)
    ),
    "linearoperatoroperandprotocol": lambda t: (
        (True, t.linearoperatoroperandprotocol) if isinstance(t, _HasMember30_19) else (False, None)
    ),
    "linearoperatorpermutation": lambda t: (
        (True, t.linearoperatorpermutation) if isinstance(t, _HasMember30_20) else (False, None)
    ),
    "linearoperatorscaledidentity": lambda t: (
        (True, t.linearoperatorscaledidentity) if isinstance(t, _HasMember30_21) else (False, None)
    ),
    "linearoperatortoeplitz": lambda t: (
        (True, t.linearoperatortoeplitz) if isinstance(t, _HasMember30_22) else (False, None)
    ),
    "linearoperatortridiag": lambda t: (
        (True, t.linearoperatortridiag) if isinstance(t, _HasMember30_23) else (False, None)
    ),
    "linearoperatorzeros": lambda t: (True, t.linearoperatorzeros) if isinstance(t, _HasMember30_24) else (False, None),
    "linspace": lambda t: (True, t.linspace) if isinstance(t, _HasMember30_25) else (False, None),
    "list_element": lambda t: (True, t.list_element) if isinstance(t, _HasMember30_26) else (False, None),
    "list_flatten": lambda t: (True, t.list_flatten) if isinstance(t, _HasMember30_27) else (False, None),
    "list_parent_indices": lambda t: (True, t.list_parent_indices) if isinstance(t, _HasMember30_28) else (False, None),
    "list_slice": lambda t: (True, t.list_slice) if isinstance(t, _HasMember30_29) else (False, None),
    "list_value_length": lambda t: (True, t.list_value_length) if isinstance(t, _HasMember30_30) else (False, None),
    "ln": lambda t: (True, t.ln) if isinstance(t, _HasMember30_31) else (False, None),
    "load": lambda t: (True, t.load) if isinstance(t, _HasMember30_32) else (False, None),
    "load_h5": lambda t: (True, t.load_h5) if isinstance(t, _HasMember30_33) else (False, None),
    "load_npz": lambda t: (True, t.load_npz) if isinstance(t, _HasMember30_34) else (False, None),
    "load_state_dict": lambda t: (True, t.load_state_dict) if isinstance(t, _HasMember30_35) else (False, None),
    "load_target": lambda t: (True, t.load_target) if isinstance(t, _HasMember30_36) else (False, None),
    "log": lambda t: (True, t.log) if isinstance(t, _HasMember30_37) else (False, None),
    "log10": lambda t: (True, t.log10) if isinstance(t, _HasMember30_38) else (False, None),
    "log1p": lambda t: (True, t.log1p) if isinstance(t, _HasMember30_39) else (False, None),
    "log2": lambda t: (True, t.log2) if isinstance(t, _HasMember30_40) else (False, None),
    "log_cosh_loss": lambda t: (True, t.log_cosh_loss) if isinstance(t, _HasMember30_41) else (False, None),
    "log_poisson_loss": lambda t: (True, t.log_poisson_loss) if isinstance(t, _HasMember30_42) else (False, None),
    "log_sigmoid": lambda t: (True, t.log_sigmoid) if isinstance(t, _HasMember30_43) else (False, None),
    "log_softmax": lambda t: (True, t.log_softmax) if isinstance(t, _HasMember30_44) else (False, None),
}
