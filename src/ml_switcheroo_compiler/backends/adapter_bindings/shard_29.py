"""Static structural protocol bindings shard 29 (kwarg_translations..linear_operator_scaled_identity)."""

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
class _HasMember29_00(typing.Protocol):
    """Structural protocol for objects exposing `kwarg_translations`."""

    kwarg_translations: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_01(typing.Protocol):
    """Structural protocol for objects exposing `l1_loss`."""

    l1_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_02(typing.Protocol):
    """Structural protocol for objects exposing `l2_loss`."""

    l2_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_03(typing.Protocol):
    """Structural protocol for objects exposing `l2_normalize`."""

    l2_normalize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_04(typing.Protocol):
    """Structural protocol for objects exposing `laguerre_polynomial_l`."""

    laguerre_polynomial_l: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_05(typing.Protocol):
    """Structural protocol for objects exposing `lax`."""

    lax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_06(typing.Protocol):
    """Structural protocol for objects exposing `layout`."""

    layout: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_07(typing.Protocol):
    """Structural protocol for objects exposing `lbeta`."""

    lbeta: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_08(typing.Protocol):
    """Structural protocol for objects exposing `lcm`."""

    lcm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_09(typing.Protocol):
    """Structural protocol for objects exposing `ldexp`."""

    ldexp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_10(typing.Protocol):
    """Structural protocol for objects exposing `le`."""

    le: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_11(typing.Protocol):
    """Structural protocol for objects exposing `leaky_relu`."""

    leaky_relu: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_12(typing.Protocol):
    """Structural protocol for objects exposing `least_common_multiple`."""

    least_common_multiple: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_13(typing.Protocol):
    """Structural protocol for objects exposing `left`."""

    left: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_14(typing.Protocol):
    """Structural protocol for objects exposing `left_shift`."""

    left_shift: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_15(typing.Protocol):
    """Structural protocol for objects exposing `leftshift`."""

    leftshift: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_16(typing.Protocol):
    """Structural protocol for objects exposing `legendre_polynomial_p`."""

    legendre_polynomial_p: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_17(typing.Protocol):
    """Structural protocol for objects exposing `length`."""

    length: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_18(typing.Protocol):
    """Structural protocol for objects exposing `less`."""

    less: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_19(typing.Protocol):
    """Structural protocol for objects exposing `less_equal`."""

    less_equal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_20(typing.Protocol):
    """Structural protocol for objects exposing `lessequal`."""

    lessequal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_21(typing.Protocol):
    """Structural protocol for objects exposing `lexsort`."""

    lexsort: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_22(typing.Protocol):
    """Structural protocol for objects exposing `lgamma`."""

    lgamma: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_23(typing.Protocol):
    """Structural protocol for objects exposing `lhs_spec`."""

    lhs_spec: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_24(typing.Protocol):
    """Structural protocol for objects exposing `linalg`."""

    linalg: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_25(typing.Protocol):
    """Structural protocol for objects exposing `linear_comb_fn`."""

    linear_comb_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_26(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator`."""

    linear_operator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_27(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_adjoint`."""

    linear_operator_adjoint: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_28(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_block_diag`."""

    linear_operator_block_diag: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_29(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_block_lower_triangular`."""

    linear_operator_block_lower_triangular: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_30(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_circulant`."""

    linear_operator_circulant: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_31(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_circulant2_d`."""

    linear_operator_circulant2_d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_32(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_circulant3_d`."""

    linear_operator_circulant3_d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_33(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_composition`."""

    linear_operator_composition: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_34(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_diag`."""

    linear_operator_diag: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_35(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_full_matrix`."""

    linear_operator_full_matrix: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_36(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_householder`."""

    linear_operator_householder: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_37(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_identity`."""

    linear_operator_identity: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_38(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_inversion`."""

    linear_operator_inversion: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_39(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_kronecker`."""

    linear_operator_kronecker: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_40(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_low_rank_update`."""

    linear_operator_low_rank_update: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_41(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_lower_triangular`."""

    linear_operator_lower_triangular: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_42(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_operand_protocol`."""

    linear_operator_operand_protocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_43(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_permutation`."""

    linear_operator_permutation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember29_44(typing.Protocol):
    """Structural protocol for objects exposing `linear_operator_scaled_identity`."""

    linear_operator_scaled_identity: EagerTensorProtocol


SHARD_29_READERS: dict[str, MemberReader] = {
    "kwarg_translations": lambda t: (True, t.kwarg_translations) if isinstance(t, _HasMember29_00) else (False, None),
    "l1_loss": lambda t: (True, t.l1_loss) if isinstance(t, _HasMember29_01) else (False, None),
    "l2_loss": lambda t: (True, t.l2_loss) if isinstance(t, _HasMember29_02) else (False, None),
    "l2_normalize": lambda t: (True, t.l2_normalize) if isinstance(t, _HasMember29_03) else (False, None),
    "laguerre_polynomial_l": lambda t: (
        (True, t.laguerre_polynomial_l) if isinstance(t, _HasMember29_04) else (False, None)
    ),
    "lax": lambda t: (True, t.lax) if isinstance(t, _HasMember29_05) else (False, None),
    "layout": lambda t: (True, t.layout) if isinstance(t, _HasMember29_06) else (False, None),
    "lbeta": lambda t: (True, t.lbeta) if isinstance(t, _HasMember29_07) else (False, None),
    "lcm": lambda t: (True, t.lcm) if isinstance(t, _HasMember29_08) else (False, None),
    "ldexp": lambda t: (True, t.ldexp) if isinstance(t, _HasMember29_09) else (False, None),
    "le": lambda t: (True, t.le) if isinstance(t, _HasMember29_10) else (False, None),
    "leaky_relu": lambda t: (True, t.leaky_relu) if isinstance(t, _HasMember29_11) else (False, None),
    "least_common_multiple": lambda t: (
        (True, t.least_common_multiple) if isinstance(t, _HasMember29_12) else (False, None)
    ),
    "left": lambda t: (True, t.left) if isinstance(t, _HasMember29_13) else (False, None),
    "left_shift": lambda t: (True, t.left_shift) if isinstance(t, _HasMember29_14) else (False, None),
    "leftshift": lambda t: (True, t.leftshift) if isinstance(t, _HasMember29_15) else (False, None),
    "legendre_polynomial_p": lambda t: (
        (True, t.legendre_polynomial_p) if isinstance(t, _HasMember29_16) else (False, None)
    ),
    "length": lambda t: (True, t.length) if isinstance(t, _HasMember29_17) else (False, None),
    "less": lambda t: (True, t.less) if isinstance(t, _HasMember29_18) else (False, None),
    "less_equal": lambda t: (True, t.less_equal) if isinstance(t, _HasMember29_19) else (False, None),
    "lessequal": lambda t: (True, t.lessequal) if isinstance(t, _HasMember29_20) else (False, None),
    "lexsort": lambda t: (True, t.lexsort) if isinstance(t, _HasMember29_21) else (False, None),
    "lgamma": lambda t: (True, t.lgamma) if isinstance(t, _HasMember29_22) else (False, None),
    "lhs_spec": lambda t: (True, t.lhs_spec) if isinstance(t, _HasMember29_23) else (False, None),
    "linalg": lambda t: (True, t.linalg) if isinstance(t, _HasMember29_24) else (False, None),
    "linear_comb_fn": lambda t: (True, t.linear_comb_fn) if isinstance(t, _HasMember29_25) else (False, None),
    "linear_operator": lambda t: (True, t.linear_operator) if isinstance(t, _HasMember29_26) else (False, None),
    "linear_operator_adjoint": lambda t: (
        (True, t.linear_operator_adjoint) if isinstance(t, _HasMember29_27) else (False, None)
    ),
    "linear_operator_block_diag": lambda t: (
        (True, t.linear_operator_block_diag) if isinstance(t, _HasMember29_28) else (False, None)
    ),
    "linear_operator_block_lower_triangular": lambda t: (
        (True, t.linear_operator_block_lower_triangular) if isinstance(t, _HasMember29_29) else (False, None)
    ),
    "linear_operator_circulant": lambda t: (
        (True, t.linear_operator_circulant) if isinstance(t, _HasMember29_30) else (False, None)
    ),
    "linear_operator_circulant2_d": lambda t: (
        (True, t.linear_operator_circulant2_d) if isinstance(t, _HasMember29_31) else (False, None)
    ),
    "linear_operator_circulant3_d": lambda t: (
        (True, t.linear_operator_circulant3_d) if isinstance(t, _HasMember29_32) else (False, None)
    ),
    "linear_operator_composition": lambda t: (
        (True, t.linear_operator_composition) if isinstance(t, _HasMember29_33) else (False, None)
    ),
    "linear_operator_diag": lambda t: (
        (True, t.linear_operator_diag) if isinstance(t, _HasMember29_34) else (False, None)
    ),
    "linear_operator_full_matrix": lambda t: (
        (True, t.linear_operator_full_matrix) if isinstance(t, _HasMember29_35) else (False, None)
    ),
    "linear_operator_householder": lambda t: (
        (True, t.linear_operator_householder) if isinstance(t, _HasMember29_36) else (False, None)
    ),
    "linear_operator_identity": lambda t: (
        (True, t.linear_operator_identity) if isinstance(t, _HasMember29_37) else (False, None)
    ),
    "linear_operator_inversion": lambda t: (
        (True, t.linear_operator_inversion) if isinstance(t, _HasMember29_38) else (False, None)
    ),
    "linear_operator_kronecker": lambda t: (
        (True, t.linear_operator_kronecker) if isinstance(t, _HasMember29_39) else (False, None)
    ),
    "linear_operator_low_rank_update": lambda t: (
        (True, t.linear_operator_low_rank_update) if isinstance(t, _HasMember29_40) else (False, None)
    ),
    "linear_operator_lower_triangular": lambda t: (
        (True, t.linear_operator_lower_triangular) if isinstance(t, _HasMember29_41) else (False, None)
    ),
    "linear_operator_operand_protocol": lambda t: (
        (True, t.linear_operator_operand_protocol) if isinstance(t, _HasMember29_42) else (False, None)
    ),
    "linear_operator_permutation": lambda t: (
        (True, t.linear_operator_permutation) if isinstance(t, _HasMember29_43) else (False, None)
    ),
    "linear_operator_scaled_identity": lambda t: (
        (True, t.linear_operator_scaled_identity) if isinstance(t, _HasMember29_44) else (False, None)
    ),
}

register_dynamic_protocols(
    (
        _HasMember29_00,
        _HasMember29_01,
        _HasMember29_02,
        _HasMember29_03,
        _HasMember29_04,
        _HasMember29_05,
        _HasMember29_06,
        _HasMember29_07,
        _HasMember29_08,
        _HasMember29_09,
        _HasMember29_10,
        _HasMember29_11,
        _HasMember29_12,
        _HasMember29_13,
        _HasMember29_14,
        _HasMember29_15,
        _HasMember29_16,
        _HasMember29_17,
        _HasMember29_18,
        _HasMember29_19,
        _HasMember29_20,
        _HasMember29_21,
        _HasMember29_22,
        _HasMember29_23,
        _HasMember29_24,
        _HasMember29_25,
        _HasMember29_26,
        _HasMember29_27,
        _HasMember29_28,
        _HasMember29_29,
        _HasMember29_30,
        _HasMember29_31,
        _HasMember29_32,
        _HasMember29_33,
        _HasMember29_34,
        _HasMember29_35,
        _HasMember29_36,
        _HasMember29_37,
        _HasMember29_38,
        _HasMember29_39,
        _HasMember29_40,
        _HasMember29_41,
        _HasMember29_42,
        _HasMember29_43,
        _HasMember29_44,
    )
)
