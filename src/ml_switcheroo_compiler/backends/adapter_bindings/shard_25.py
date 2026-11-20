"""Static structural protocol bindings shard 25 (gfile_glob..hinge_loss)."""

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
class _HasMember25_00(typing.Protocol):
    """Structural protocol for objects exposing `gfile_glob`."""

    gfile_glob: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_01(typing.Protocol):
    """Structural protocol for objects exposing `gfile_makedirs`."""

    gfile_makedirs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_02(typing.Protocol):
    """Structural protocol for objects exposing `gfile_stat`."""

    gfile_stat: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_03(typing.Protocol):
    """Structural protocol for objects exposing `gfoo`."""

    gfoo: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_04(typing.Protocol):
    """Structural protocol for objects exposing `global_norm`."""

    global_norm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_05(typing.Protocol):
    """Structural protocol for objects exposing `global_wasm_mock`."""

    global_wasm_mock: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_06(typing.Protocol):
    """Structural protocol for objects exposing `glorot_uniform`."""

    glorot_uniform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_07(typing.Protocol):
    """Structural protocol for objects exposing `grad`."""

    grad: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_08(typing.Protocol):
    """Structural protocol for objects exposing `gradient`."""

    gradient: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_09(typing.Protocol):
    """Structural protocol for objects exposing `greater`."""

    greater: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_10(typing.Protocol):
    """Structural protocol for objects exposing `greater_equal`."""

    greater_equal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_11(typing.Protocol):
    """Structural protocol for objects exposing `greaterequal`."""

    greaterequal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_12(typing.Protocol):
    """Structural protocol for objects exposing `greatest_common_divisor`."""

    greatest_common_divisor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_13(typing.Protocol):
    """Structural protocol for objects exposing `grid_sample`."""

    grid_sample: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_14(typing.Protocol):
    """Structural protocol for objects exposing `group`."""

    group: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_15(typing.Protocol):
    """Structural protocol for objects exposing `group_by`."""

    group_by: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_16(typing.Protocol):
    """Structural protocol for objects exposing `group_variance`."""

    group_variance: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_17(typing.Protocol):
    """Structural protocol for objects exposing `gru`."""

    gru: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_18(typing.Protocol):
    """Structural protocol for objects exposing `gru_cell`."""

    gru_cell: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_19(typing.Protocol):
    """Structural protocol for objects exposing `gt`."""

    gt: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_20(typing.Protocol):
    """Structural protocol for objects exposing `gumbel`."""

    gumbel: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_21(typing.Protocol):
    """Structural protocol for objects exposing `hadamard_transform`."""

    hadamard_transform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_22(typing.Protocol):
    """Structural protocol for objects exposing `half_pixel_centers`."""

    half_pixel_centers: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_23(typing.Protocol):
    """Structural protocol for objects exposing `hard_silu`."""

    hard_silu: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_24(typing.Protocol):
    """Structural protocol for objects exposing `hard_swish`."""

    hard_swish: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_25(typing.Protocol):
    """Structural protocol for objects exposing `hardswish`."""

    hardswish: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_26(typing.Protocol):
    """Structural protocol for objects exposing `hash_all`."""

    hash_all: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_27(typing.Protocol):
    """Structural protocol for objects exposing `hash_any`."""

    hash_any: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_28(typing.Protocol):
    """Structural protocol for objects exposing `hash_count`."""

    hash_count: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_29(typing.Protocol):
    """Structural protocol for objects exposing `hash_count_distinct`."""

    hash_count_distinct: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_30(typing.Protocol):
    """Structural protocol for objects exposing `hash_first`."""

    hash_first: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_31(typing.Protocol):
    """Structural protocol for objects exposing `hash_last`."""

    hash_last: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_32(typing.Protocol):
    """Structural protocol for objects exposing `hash_list`."""

    hash_list: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_33(typing.Protocol):
    """Structural protocol for objects exposing `hash_max`."""

    hash_max: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_34(typing.Protocol):
    """Structural protocol for objects exposing `hash_mean`."""

    hash_mean: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_35(typing.Protocol):
    """Structural protocol for objects exposing `hash_min`."""

    hash_min: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_36(typing.Protocol):
    """Structural protocol for objects exposing `hash_product`."""

    hash_product: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_37(typing.Protocol):
    """Structural protocol for objects exposing `hash_stddev`."""

    hash_stddev: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_38(typing.Protocol):
    """Structural protocol for objects exposing `hash_sum`."""

    hash_sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_39(typing.Protocol):
    """Structural protocol for objects exposing `hash_variance`."""

    hash_variance: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_40(typing.Protocol):
    """Structural protocol for objects exposing `heaviside`."""

    heaviside: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_41(typing.Protocol):
    """Structural protocol for objects exposing `hermite_polynomial_h`."""

    hermite_polynomial_h: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_42(typing.Protocol):
    """Structural protocol for objects exposing `hermite_polynomial_he`."""

    hermite_polynomial_he: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_43(typing.Protocol):
    """Structural protocol for objects exposing `hfft`."""

    hfft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember25_44(typing.Protocol):
    """Structural protocol for objects exposing `hinge_loss`."""

    hinge_loss: EagerTensorProtocol


SHARD_25_READERS: dict[str, MemberReader] = {
    "gfile_glob": lambda t: (True, t.gfile_glob) if isinstance(t, _HasMember25_00) else (False, None),
    "gfile_makedirs": lambda t: (True, t.gfile_makedirs) if isinstance(t, _HasMember25_01) else (False, None),
    "gfile_stat": lambda t: (True, t.gfile_stat) if isinstance(t, _HasMember25_02) else (False, None),
    "gfoo": lambda t: (True, t.gfoo) if isinstance(t, _HasMember25_03) else (False, None),
    "global_norm": lambda t: (True, t.global_norm) if isinstance(t, _HasMember25_04) else (False, None),
    "global_wasm_mock": lambda t: (True, t.global_wasm_mock) if isinstance(t, _HasMember25_05) else (False, None),
    "glorot_uniform": lambda t: (True, t.glorot_uniform) if isinstance(t, _HasMember25_06) else (False, None),
    "grad": lambda t: (True, t.grad) if isinstance(t, _HasMember25_07) else (False, None),
    "gradient": lambda t: (True, t.gradient) if isinstance(t, _HasMember25_08) else (False, None),
    "greater": lambda t: (True, t.greater) if isinstance(t, _HasMember25_09) else (False, None),
    "greater_equal": lambda t: (True, t.greater_equal) if isinstance(t, _HasMember25_10) else (False, None),
    "greaterequal": lambda t: (True, t.greaterequal) if isinstance(t, _HasMember25_11) else (False, None),
    "greatest_common_divisor": lambda t: (
        (True, t.greatest_common_divisor) if isinstance(t, _HasMember25_12) else (False, None)
    ),
    "grid_sample": lambda t: (True, t.grid_sample) if isinstance(t, _HasMember25_13) else (False, None),
    "group": lambda t: (True, t.group) if isinstance(t, _HasMember25_14) else (False, None),
    "group_by": lambda t: (True, t.group_by) if isinstance(t, _HasMember25_15) else (False, None),
    "group_variance": lambda t: (True, t.group_variance) if isinstance(t, _HasMember25_16) else (False, None),
    "gru": lambda t: (True, t.gru) if isinstance(t, _HasMember25_17) else (False, None),
    "gru_cell": lambda t: (True, t.gru_cell) if isinstance(t, _HasMember25_18) else (False, None),
    "gt": lambda t: (True, t.gt) if isinstance(t, _HasMember25_19) else (False, None),
    "gumbel": lambda t: (True, t.gumbel) if isinstance(t, _HasMember25_20) else (False, None),
    "hadamard_transform": lambda t: (True, t.hadamard_transform) if isinstance(t, _HasMember25_21) else (False, None),
    "half_pixel_centers": lambda t: (True, t.half_pixel_centers) if isinstance(t, _HasMember25_22) else (False, None),
    "hard_silu": lambda t: (True, t.hard_silu) if isinstance(t, _HasMember25_23) else (False, None),
    "hard_swish": lambda t: (True, t.hard_swish) if isinstance(t, _HasMember25_24) else (False, None),
    "hardswish": lambda t: (True, t.hardswish) if isinstance(t, _HasMember25_25) else (False, None),
    "hash_all": lambda t: (True, t.hash_all) if isinstance(t, _HasMember25_26) else (False, None),
    "hash_any": lambda t: (True, t.hash_any) if isinstance(t, _HasMember25_27) else (False, None),
    "hash_count": lambda t: (True, t.hash_count) if isinstance(t, _HasMember25_28) else (False, None),
    "hash_count_distinct": lambda t: (True, t.hash_count_distinct) if isinstance(t, _HasMember25_29) else (False, None),
    "hash_first": lambda t: (True, t.hash_first) if isinstance(t, _HasMember25_30) else (False, None),
    "hash_last": lambda t: (True, t.hash_last) if isinstance(t, _HasMember25_31) else (False, None),
    "hash_list": lambda t: (True, t.hash_list) if isinstance(t, _HasMember25_32) else (False, None),
    "hash_max": lambda t: (True, t.hash_max) if isinstance(t, _HasMember25_33) else (False, None),
    "hash_mean": lambda t: (True, t.hash_mean) if isinstance(t, _HasMember25_34) else (False, None),
    "hash_min": lambda t: (True, t.hash_min) if isinstance(t, _HasMember25_35) else (False, None),
    "hash_product": lambda t: (True, t.hash_product) if isinstance(t, _HasMember25_36) else (False, None),
    "hash_stddev": lambda t: (True, t.hash_stddev) if isinstance(t, _HasMember25_37) else (False, None),
    "hash_sum": lambda t: (True, t.hash_sum) if isinstance(t, _HasMember25_38) else (False, None),
    "hash_variance": lambda t: (True, t.hash_variance) if isinstance(t, _HasMember25_39) else (False, None),
    "heaviside": lambda t: (True, t.heaviside) if isinstance(t, _HasMember25_40) else (False, None),
    "hermite_polynomial_h": lambda t: (
        (True, t.hermite_polynomial_h) if isinstance(t, _HasMember25_41) else (False, None)
    ),
    "hermite_polynomial_he": lambda t: (
        (True, t.hermite_polynomial_he) if isinstance(t, _HasMember25_42) else (False, None)
    ),
    "hfft": lambda t: (True, t.hfft) if isinstance(t, _HasMember25_43) else (False, None),
    "hinge_loss": lambda t: (True, t.hinge_loss) if isinstance(t, _HasMember25_44) else (False, None),
}
