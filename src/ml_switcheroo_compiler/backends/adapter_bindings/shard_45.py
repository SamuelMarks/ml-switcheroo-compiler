"""Static structural protocol bindings shard 45 (shape_metadata..sort_complex)."""

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
class _HasMember45_00(typing.Protocol):
    """Structural protocol for objects exposing `shape_metadata`."""

    shape_metadata: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_01(typing.Protocol):
    """Structural protocol for objects exposing `shape_metadata_protocol`."""

    shape_metadata_protocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_02(typing.Protocol):
    """Structural protocol for objects exposing `shape_protocol`."""

    shape_protocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_03(typing.Protocol):
    """Structural protocol for objects exposing `shapemetadataprotocol`."""

    shapemetadataprotocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_04(typing.Protocol):
    """Structural protocol for objects exposing `shapeprotocol`."""

    shapeprotocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_05(typing.Protocol):
    """Structural protocol for objects exposing `sharpen`."""

    sharpen: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_06(typing.Protocol):
    """Structural protocol for objects exposing `shift_left`."""

    shift_left: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_07(typing.Protocol):
    """Structural protocol for objects exposing `shift_right`."""

    shift_right: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_08(typing.Protocol):
    """Structural protocol for objects exposing `shifted_chebyshev_polynomial_t`."""

    shifted_chebyshev_polynomial_t: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_09(typing.Protocol):
    """Structural protocol for objects exposing `shifted_chebyshev_polynomial_u`."""

    shifted_chebyshev_polynomial_u: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_10(typing.Protocol):
    """Structural protocol for objects exposing `shifted_chebyshev_polynomial_v`."""

    shifted_chebyshev_polynomial_v: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_11(typing.Protocol):
    """Structural protocol for objects exposing `shifted_chebyshev_polynomial_w`."""

    shifted_chebyshev_polynomial_w: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_12(typing.Protocol):
    """Structural protocol for objects exposing `shuffle`."""

    shuffle: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_13(typing.Protocol):
    """Structural protocol for objects exposing `sigmoid`."""

    sigmoid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_14(typing.Protocol):
    """Structural protocol for objects exposing `sign`."""

    sign: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_15(typing.Protocol):
    """Structural protocol for objects exposing `signal`."""

    signal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_16(typing.Protocol):
    """Structural protocol for objects exposing `signbit`."""

    signbit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_17(typing.Protocol):
    """Structural protocol for objects exposing `silu`."""

    silu: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_18(typing.Protocol):
    """Structural protocol for objects exposing `simple_f`."""

    simple_f: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_19(typing.Protocol):
    """Structural protocol for objects exposing `simple_fn`."""

    simple_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_20(typing.Protocol):
    """Structural protocol for objects exposing `simple_identity`."""

    simple_identity: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_21(typing.Protocol):
    """Structural protocol for objects exposing `simplify`."""

    simplify: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_22(typing.Protocol):
    """Structural protocol for objects exposing `sin`."""

    sin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_23(typing.Protocol):
    """Structural protocol for objects exposing `sinc`."""

    sinc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_24(typing.Protocol):
    """Structural protocol for objects exposing `single_child`."""

    single_child: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_25(typing.Protocol):
    """Structural protocol for objects exposing `sinh`."""

    sinh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_26(typing.Protocol):
    """Structural protocol for objects exposing `size`."""

    size: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_27(typing.Protocol):
    """Structural protocol for objects exposing `slice`."""

    slice: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_28(typing.Protocol):
    """Structural protocol for objects exposing `slicing`."""

    slicing: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_29(typing.Protocol):
    """Structural protocol for objects exposing `slogdet`."""

    slogdet: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_30(typing.Protocol):
    """Structural protocol for objects exposing `smart_sig`."""

    smart_sig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_31(typing.Protocol):
    """Structural protocol for objects exposing `smm`."""

    smm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_32(typing.Protocol):
    """Structural protocol for objects exposing `smooth_l1_loss`."""

    smooth_l1_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_33(typing.Protocol):
    """Structural protocol for objects exposing `sobol_sample`."""

    sobol_sample: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_34(typing.Protocol):
    """Structural protocol for objects exposing `sobolsample`."""

    sobolsample: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_35(typing.Protocol):
    """Structural protocol for objects exposing `softmax`."""

    softmax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_36(typing.Protocol):
    """Structural protocol for objects exposing `softplus`."""

    softplus: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_37(typing.Protocol):
    """Structural protocol for objects exposing `softsign`."""

    softsign: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_38(typing.Protocol):
    """Structural protocol for objects exposing `solarize`."""

    solarize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_39(typing.Protocol):
    """Structural protocol for objects exposing `solve`."""

    solve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_40(typing.Protocol):
    """Structural protocol for objects exposing `solve_banded`."""

    solve_banded: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_41(typing.Protocol):
    """Structural protocol for objects exposing `solve_ex`."""

    solve_ex: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_42(typing.Protocol):
    """Structural protocol for objects exposing `solve_triangular`."""

    solve_triangular: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_43(typing.Protocol):
    """Structural protocol for objects exposing `sort`."""

    sort: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember45_44(typing.Protocol):
    """Structural protocol for objects exposing `sort_complex`."""

    sort_complex: EagerTensorProtocol


SHARD_45_READERS: dict[str, MemberReader] = {
    "shape_metadata": lambda t: (True, t.shape_metadata) if isinstance(t, _HasMember45_00) else (False, None),
    "shape_metadata_protocol": lambda t: (
        (True, t.shape_metadata_protocol) if isinstance(t, _HasMember45_01) else (False, None)
    ),
    "shape_protocol": lambda t: (True, t.shape_protocol) if isinstance(t, _HasMember45_02) else (False, None),
    "shapemetadataprotocol": lambda t: (
        (True, t.shapemetadataprotocol) if isinstance(t, _HasMember45_03) else (False, None)
    ),
    "shapeprotocol": lambda t: (True, t.shapeprotocol) if isinstance(t, _HasMember45_04) else (False, None),
    "sharpen": lambda t: (True, t.sharpen) if isinstance(t, _HasMember45_05) else (False, None),
    "shift_left": lambda t: (True, t.shift_left) if isinstance(t, _HasMember45_06) else (False, None),
    "shift_right": lambda t: (True, t.shift_right) if isinstance(t, _HasMember45_07) else (False, None),
    "shifted_chebyshev_polynomial_t": lambda t: (
        (True, t.shifted_chebyshev_polynomial_t) if isinstance(t, _HasMember45_08) else (False, None)
    ),
    "shifted_chebyshev_polynomial_u": lambda t: (
        (True, t.shifted_chebyshev_polynomial_u) if isinstance(t, _HasMember45_09) else (False, None)
    ),
    "shifted_chebyshev_polynomial_v": lambda t: (
        (True, t.shifted_chebyshev_polynomial_v) if isinstance(t, _HasMember45_10) else (False, None)
    ),
    "shifted_chebyshev_polynomial_w": lambda t: (
        (True, t.shifted_chebyshev_polynomial_w) if isinstance(t, _HasMember45_11) else (False, None)
    ),
    "shuffle": lambda t: (True, t.shuffle) if isinstance(t, _HasMember45_12) else (False, None),
    "sigmoid": lambda t: (True, t.sigmoid) if isinstance(t, _HasMember45_13) else (False, None),
    "sign": lambda t: (True, t.sign) if isinstance(t, _HasMember45_14) else (False, None),
    "signal": lambda t: (True, t.signal) if isinstance(t, _HasMember45_15) else (False, None),
    "signbit": lambda t: (True, t.signbit) if isinstance(t, _HasMember45_16) else (False, None),
    "silu": lambda t: (True, t.silu) if isinstance(t, _HasMember45_17) else (False, None),
    "simple_f": lambda t: (True, t.simple_f) if isinstance(t, _HasMember45_18) else (False, None),
    "simple_fn": lambda t: (True, t.simple_fn) if isinstance(t, _HasMember45_19) else (False, None),
    "simple_identity": lambda t: (True, t.simple_identity) if isinstance(t, _HasMember45_20) else (False, None),
    "simplify": lambda t: (True, t.simplify) if isinstance(t, _HasMember45_21) else (False, None),
    "sin": lambda t: (True, t.sin) if isinstance(t, _HasMember45_22) else (False, None),
    "sinc": lambda t: (True, t.sinc) if isinstance(t, _HasMember45_23) else (False, None),
    "single_child": lambda t: (True, t.single_child) if isinstance(t, _HasMember45_24) else (False, None),
    "sinh": lambda t: (True, t.sinh) if isinstance(t, _HasMember45_25) else (False, None),
    "size": lambda t: (True, t.size) if isinstance(t, _HasMember45_26) else (False, None),
    "slice": lambda t: (True, t.slice) if isinstance(t, _HasMember45_27) else (False, None),
    "slicing": lambda t: (True, t.slicing) if isinstance(t, _HasMember45_28) else (False, None),
    "slogdet": lambda t: (True, t.slogdet) if isinstance(t, _HasMember45_29) else (False, None),
    "smart_sig": lambda t: (True, t.smart_sig) if isinstance(t, _HasMember45_30) else (False, None),
    "smm": lambda t: (True, t.smm) if isinstance(t, _HasMember45_31) else (False, None),
    "smooth_l1_loss": lambda t: (True, t.smooth_l1_loss) if isinstance(t, _HasMember45_32) else (False, None),
    "sobol_sample": lambda t: (True, t.sobol_sample) if isinstance(t, _HasMember45_33) else (False, None),
    "sobolsample": lambda t: (True, t.sobolsample) if isinstance(t, _HasMember45_34) else (False, None),
    "softmax": lambda t: (True, t.softmax) if isinstance(t, _HasMember45_35) else (False, None),
    "softplus": lambda t: (True, t.softplus) if isinstance(t, _HasMember45_36) else (False, None),
    "softsign": lambda t: (True, t.softsign) if isinstance(t, _HasMember45_37) else (False, None),
    "solarize": lambda t: (True, t.solarize) if isinstance(t, _HasMember45_38) else (False, None),
    "solve": lambda t: (True, t.solve) if isinstance(t, _HasMember45_39) else (False, None),
    "solve_banded": lambda t: (True, t.solve_banded) if isinstance(t, _HasMember45_40) else (False, None),
    "solve_ex": lambda t: (True, t.solve_ex) if isinstance(t, _HasMember45_41) else (False, None),
    "solve_triangular": lambda t: (True, t.solve_triangular) if isinstance(t, _HasMember45_42) else (False, None),
    "sort": lambda t: (True, t.sort) if isinstance(t, _HasMember45_43) else (False, None),
    "sort_complex": lambda t: (True, t.sort_complex) if isinstance(t, _HasMember45_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember45_00,
        _HasMember45_01,
        _HasMember45_02,
        _HasMember45_03,
        _HasMember45_04,
        _HasMember45_05,
        _HasMember45_06,
        _HasMember45_07,
        _HasMember45_08,
        _HasMember45_09,
        _HasMember45_10,
        _HasMember45_11,
        _HasMember45_12,
        _HasMember45_13,
        _HasMember45_14,
        _HasMember45_15,
        _HasMember45_16,
        _HasMember45_17,
        _HasMember45_18,
        _HasMember45_19,
        _HasMember45_20,
        _HasMember45_21,
        _HasMember45_22,
        _HasMember45_23,
        _HasMember45_24,
        _HasMember45_25,
        _HasMember45_26,
        _HasMember45_27,
        _HasMember45_28,
        _HasMember45_29,
        _HasMember45_30,
        _HasMember45_31,
        _HasMember45_32,
        _HasMember45_33,
        _HasMember45_34,
        _HasMember45_35,
        _HasMember45_36,
        _HasMember45_37,
        _HasMember45_38,
        _HasMember45_39,
        _HasMember45_40,
        _HasMember45_41,
        _HasMember45_42,
        _HasMember45_43,
        _HasMember45_44,
    )
)
