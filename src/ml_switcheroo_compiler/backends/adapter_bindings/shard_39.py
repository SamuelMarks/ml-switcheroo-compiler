"""Static structural protocol bindings shard 39 (profile_graph..raggedsegmentidstorowsplits)."""

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
class _HasMember39_00(typing.Protocol):
    """Structural protocol for objects exposing `profile_graph`."""

    profile_graph: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_01(typing.Protocol):
    """Structural protocol for objects exposing `promote_types`."""

    promote_types: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_02(typing.Protocol):
    """Structural protocol for objects exposing `ps_pull_weights`."""

    ps_pull_weights: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_03(typing.Protocol):
    """Structural protocol for objects exposing `ps_push_gradients`."""

    ps_push_gradients: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_04(typing.Protocol):
    """Structural protocol for objects exposing `pshuffle`."""

    pshuffle: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_05(typing.Protocol):
    """Structural protocol for objects exposing `psum`."""

    psum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_06(typing.Protocol):
    """Structural protocol for objects exposing `psum_scatter`."""

    psum_scatter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_07(typing.Protocol):
    """Structural protocol for objects exposing `pswapaxes`."""

    pswapaxes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_08(typing.Protocol):
    """Structural protocol for objects exposing `ptr`."""

    ptr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_09(typing.Protocol):
    """Structural protocol for objects exposing `put`."""

    put: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_10(typing.Protocol):
    """Structural protocol for objects exposing `put_along_axis`."""

    put_along_axis: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_11(typing.Protocol):
    """Structural protocol for objects exposing `put_msg`."""

    put_msg: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_12(typing.Protocol):
    """Structural protocol for objects exposing `qr`."""

    qr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_13(typing.Protocol):
    """Structural protocol for objects exposing `quantile`."""

    quantile: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_14(typing.Protocol):
    """Structural protocol for objects exposing `r`."""

    r: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_15(typing.Protocol):
    """Structural protocol for objects exposing `r_n_n_config`."""

    r_n_n_config: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_16(typing.Protocol):
    """Structural protocol for objects exposing `rad2deg`."""

    rad2deg: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_17(typing.Protocol):
    """Structural protocol for objects exposing `rademacher`."""

    rademacher: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_18(typing.Protocol):
    """Structural protocol for objects exposing `radians`."""

    radians: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_19(typing.Protocol):
    """Structural protocol for objects exposing `ragged`."""

    ragged: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_20(typing.Protocol):
    """Structural protocol for objects exposing `ragged_add`."""

    ragged_add: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_21(typing.Protocol):
    """Structural protocol for objects exposing `ragged_constant`."""

    ragged_constant: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_22(typing.Protocol):
    """Structural protocol for objects exposing `ragged_cross_hashed`."""

    ragged_cross_hashed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_23(typing.Protocol):
    """Structural protocol for objects exposing `ragged_dot`."""

    ragged_dot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_24(typing.Protocol):
    """Structural protocol for objects exposing `ragged_dynamic_broadcast`."""

    ragged_dynamic_broadcast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_25(typing.Protocol):
    """Structural protocol for objects exposing `ragged_gather`."""

    ragged_gather: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_26(typing.Protocol):
    """Structural protocol for objects exposing `ragged_mat_mul`."""

    ragged_mat_mul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_27(typing.Protocol):
    """Structural protocol for objects exposing `ragged_operand_protocol`."""

    ragged_operand_protocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_28(typing.Protocol):
    """Structural protocol for objects exposing `ragged_range`."""

    ragged_range: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_29(typing.Protocol):
    """Structural protocol for objects exposing `ragged_row_splits_to_segment_ids`."""

    ragged_row_splits_to_segment_ids: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_30(typing.Protocol):
    """Structural protocol for objects exposing `ragged_segment_ids_to_row_splits`."""

    ragged_segment_ids_to_row_splits: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_31(typing.Protocol):
    """Structural protocol for objects exposing `ragged_stack`."""

    ragged_stack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_32(typing.Protocol):
    """Structural protocol for objects exposing `ragged_stack_dynamic_partitions`."""

    ragged_stack_dynamic_partitions: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_33(typing.Protocol):
    """Structural protocol for objects exposing `ragged_tensor_to_dense`."""

    ragged_tensor_to_dense: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_34(typing.Protocol):
    """Structural protocol for objects exposing `raggedadd`."""

    raggedadd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_35(typing.Protocol):
    """Structural protocol for objects exposing `raggedconstant`."""

    raggedconstant: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_36(typing.Protocol):
    """Structural protocol for objects exposing `raggedcrosshashed`."""

    raggedcrosshashed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_37(typing.Protocol):
    """Structural protocol for objects exposing `raggeddot`."""

    raggeddot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_38(typing.Protocol):
    """Structural protocol for objects exposing `raggeddynamicbroadcast`."""

    raggeddynamicbroadcast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_39(typing.Protocol):
    """Structural protocol for objects exposing `raggedgather`."""

    raggedgather: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_40(typing.Protocol):
    """Structural protocol for objects exposing `raggedmatmul`."""

    raggedmatmul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_41(typing.Protocol):
    """Structural protocol for objects exposing `raggedoperandprotocol`."""

    raggedoperandprotocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_42(typing.Protocol):
    """Structural protocol for objects exposing `raggedrange`."""

    raggedrange: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_43(typing.Protocol):
    """Structural protocol for objects exposing `raggedrowsplitstosegmentids`."""

    raggedrowsplitstosegmentids: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember39_44(typing.Protocol):
    """Structural protocol for objects exposing `raggedsegmentidstorowsplits`."""

    raggedsegmentidstorowsplits: EagerTensorProtocol


SHARD_39_READERS: dict[str, MemberReader] = {
    "profile_graph": lambda t: (True, t.profile_graph) if isinstance(t, _HasMember39_00) else (False, None),
    "promote_types": lambda t: (True, t.promote_types) if isinstance(t, _HasMember39_01) else (False, None),
    "ps_pull_weights": lambda t: (True, t.ps_pull_weights) if isinstance(t, _HasMember39_02) else (False, None),
    "ps_push_gradients": lambda t: (True, t.ps_push_gradients) if isinstance(t, _HasMember39_03) else (False, None),
    "pshuffle": lambda t: (True, t.pshuffle) if isinstance(t, _HasMember39_04) else (False, None),
    "psum": lambda t: (True, t.psum) if isinstance(t, _HasMember39_05) else (False, None),
    "psum_scatter": lambda t: (True, t.psum_scatter) if isinstance(t, _HasMember39_06) else (False, None),
    "pswapaxes": lambda t: (True, t.pswapaxes) if isinstance(t, _HasMember39_07) else (False, None),
    "ptr": lambda t: (True, t.ptr) if isinstance(t, _HasMember39_08) else (False, None),
    "put": lambda t: (True, t.put) if isinstance(t, _HasMember39_09) else (False, None),
    "put_along_axis": lambda t: (True, t.put_along_axis) if isinstance(t, _HasMember39_10) else (False, None),
    "put_msg": lambda t: (True, t.put_msg) if isinstance(t, _HasMember39_11) else (False, None),
    "qr": lambda t: (True, t.qr) if isinstance(t, _HasMember39_12) else (False, None),
    "quantile": lambda t: (True, t.quantile) if isinstance(t, _HasMember39_13) else (False, None),
    "r": lambda t: (True, t.r) if isinstance(t, _HasMember39_14) else (False, None),
    "r_n_n_config": lambda t: (True, t.r_n_n_config) if isinstance(t, _HasMember39_15) else (False, None),
    "rad2deg": lambda t: (True, t.rad2deg) if isinstance(t, _HasMember39_16) else (False, None),
    "rademacher": lambda t: (True, t.rademacher) if isinstance(t, _HasMember39_17) else (False, None),
    "radians": lambda t: (True, t.radians) if isinstance(t, _HasMember39_18) else (False, None),
    "ragged": lambda t: (True, t.ragged) if isinstance(t, _HasMember39_19) else (False, None),
    "ragged_add": lambda t: (True, t.ragged_add) if isinstance(t, _HasMember39_20) else (False, None),
    "ragged_constant": lambda t: (True, t.ragged_constant) if isinstance(t, _HasMember39_21) else (False, None),
    "ragged_cross_hashed": lambda t: (True, t.ragged_cross_hashed) if isinstance(t, _HasMember39_22) else (False, None),
    "ragged_dot": lambda t: (True, t.ragged_dot) if isinstance(t, _HasMember39_23) else (False, None),
    "ragged_dynamic_broadcast": lambda t: (
        (True, t.ragged_dynamic_broadcast) if isinstance(t, _HasMember39_24) else (False, None)
    ),
    "ragged_gather": lambda t: (True, t.ragged_gather) if isinstance(t, _HasMember39_25) else (False, None),
    "ragged_mat_mul": lambda t: (True, t.ragged_mat_mul) if isinstance(t, _HasMember39_26) else (False, None),
    "ragged_operand_protocol": lambda t: (
        (True, t.ragged_operand_protocol) if isinstance(t, _HasMember39_27) else (False, None)
    ),
    "ragged_range": lambda t: (True, t.ragged_range) if isinstance(t, _HasMember39_28) else (False, None),
    "ragged_row_splits_to_segment_ids": lambda t: (
        (True, t.ragged_row_splits_to_segment_ids) if isinstance(t, _HasMember39_29) else (False, None)
    ),
    "ragged_segment_ids_to_row_splits": lambda t: (
        (True, t.ragged_segment_ids_to_row_splits) if isinstance(t, _HasMember39_30) else (False, None)
    ),
    "ragged_stack": lambda t: (True, t.ragged_stack) if isinstance(t, _HasMember39_31) else (False, None),
    "ragged_stack_dynamic_partitions": lambda t: (
        (True, t.ragged_stack_dynamic_partitions) if isinstance(t, _HasMember39_32) else (False, None)
    ),
    "ragged_tensor_to_dense": lambda t: (
        (True, t.ragged_tensor_to_dense) if isinstance(t, _HasMember39_33) else (False, None)
    ),
    "raggedadd": lambda t: (True, t.raggedadd) if isinstance(t, _HasMember39_34) else (False, None),
    "raggedconstant": lambda t: (True, t.raggedconstant) if isinstance(t, _HasMember39_35) else (False, None),
    "raggedcrosshashed": lambda t: (True, t.raggedcrosshashed) if isinstance(t, _HasMember39_36) else (False, None),
    "raggeddot": lambda t: (True, t.raggeddot) if isinstance(t, _HasMember39_37) else (False, None),
    "raggeddynamicbroadcast": lambda t: (
        (True, t.raggeddynamicbroadcast) if isinstance(t, _HasMember39_38) else (False, None)
    ),
    "raggedgather": lambda t: (True, t.raggedgather) if isinstance(t, _HasMember39_39) else (False, None),
    "raggedmatmul": lambda t: (True, t.raggedmatmul) if isinstance(t, _HasMember39_40) else (False, None),
    "raggedoperandprotocol": lambda t: (
        (True, t.raggedoperandprotocol) if isinstance(t, _HasMember39_41) else (False, None)
    ),
    "raggedrange": lambda t: (True, t.raggedrange) if isinstance(t, _HasMember39_42) else (False, None),
    "raggedrowsplitstosegmentids": lambda t: (
        (True, t.raggedrowsplitstosegmentids) if isinstance(t, _HasMember39_43) else (False, None)
    ),
    "raggedsegmentidstorowsplits": lambda t: (
        (True, t.raggedsegmentidstorowsplits) if isinstance(t, _HasMember39_44) else (False, None)
    ),
}

register_dynamic_protocols(
    (
        _HasMember39_00,
        _HasMember39_01,
        _HasMember39_02,
        _HasMember39_03,
        _HasMember39_04,
        _HasMember39_05,
        _HasMember39_06,
        _HasMember39_07,
        _HasMember39_08,
        _HasMember39_09,
        _HasMember39_10,
        _HasMember39_11,
        _HasMember39_12,
        _HasMember39_13,
        _HasMember39_14,
        _HasMember39_15,
        _HasMember39_16,
        _HasMember39_17,
        _HasMember39_18,
        _HasMember39_19,
        _HasMember39_20,
        _HasMember39_21,
        _HasMember39_22,
        _HasMember39_23,
        _HasMember39_24,
        _HasMember39_25,
        _HasMember39_26,
        _HasMember39_27,
        _HasMember39_28,
        _HasMember39_29,
        _HasMember39_30,
        _HasMember39_31,
        _HasMember39_32,
        _HasMember39_33,
        _HasMember39_34,
        _HasMember39_35,
        _HasMember39_36,
        _HasMember39_37,
        _HasMember39_38,
        _HasMember39_39,
        _HasMember39_40,
        _HasMember39_41,
        _HasMember39_42,
        _HasMember39_43,
        _HasMember39_44,
    )
)
