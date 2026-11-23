"""Static structural protocol bindings shard 46 (sort_indices..split_pattern_regex)."""

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
class _HasMember46_00(typing.Protocol):
    """Structural protocol for objects exposing `sort_indices`."""

    sort_indices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_01(typing.Protocol):
    """Structural protocol for objects exposing `source`."""

    source: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_02(typing.Protocol):
    """Structural protocol for objects exposing `source_idx`."""

    source_idx: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_03(typing.Protocol):
    """Structural protocol for objects exposing `source_node`."""

    source_node: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_04(typing.Protocol):
    """Structural protocol for objects exposing `space_batch`."""

    space_batch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_05(typing.Protocol):
    """Structural protocol for objects exposing `space_to_batch`."""

    space_to_batch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_06(typing.Protocol):
    """Structural protocol for objects exposing `space_to_batch_nd`."""

    space_to_batch_nd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_07(typing.Protocol):
    """Structural protocol for objects exposing `spacing`."""

    spacing: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_08(typing.Protocol):
    """Structural protocol for objects exposing `sparse`."""

    sparse: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_09(typing.Protocol):
    """Structural protocol for objects exposing `sparse_categorical_crossentropy`."""

    sparse_categorical_crossentropy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_10(typing.Protocol):
    """Structural protocol for objects exposing `sparse_concat`."""

    sparse_concat: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_11(typing.Protocol):
    """Structural protocol for objects exposing `sparse_dense_mat_mul`."""

    sparse_dense_mat_mul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_12(typing.Protocol):
    """Structural protocol for objects exposing `sparse_eye`."""

    sparse_eye: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_13(typing.Protocol):
    """Structural protocol for objects exposing `sparse_map_values`."""

    sparse_map_values: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_14(typing.Protocol):
    """Structural protocol for objects exposing `sparse_maximum`."""

    sparse_maximum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_15(typing.Protocol):
    """Structural protocol for objects exposing `sparse_operand_protocol`."""

    sparse_operand_protocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_16(typing.Protocol):
    """Structural protocol for objects exposing `sparse_plus`."""

    sparse_plus: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_17(typing.Protocol):
    """Structural protocol for objects exposing `sparse_reduce_max`."""

    sparse_reduce_max: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_18(typing.Protocol):
    """Structural protocol for objects exposing `sparse_reorder`."""

    sparse_reorder: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_19(typing.Protocol):
    """Structural protocol for objects exposing `sparse_reshape`."""

    sparse_reshape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_20(typing.Protocol):
    """Structural protocol for objects exposing `sparse_sampled_add`."""

    sparse_sampled_add: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_21(typing.Protocol):
    """Structural protocol for objects exposing `sparse_segment_mean`."""

    sparse_segment_mean: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_22(typing.Protocol):
    """Structural protocol for objects exposing `sparse_segment_sum`."""

    sparse_segment_sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_23(typing.Protocol):
    """Structural protocol for objects exposing `sparse_sigmoid`."""

    sparse_sigmoid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_24(typing.Protocol):
    """Structural protocol for objects exposing `sparse_softmax`."""

    sparse_softmax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_25(typing.Protocol):
    """Structural protocol for objects exposing `sparse_split`."""

    sparse_split: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_26(typing.Protocol):
    """Structural protocol for objects exposing `sparse_to_dense`."""

    sparse_to_dense: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_27(typing.Protocol):
    """Structural protocol for objects exposing `sparse_transpose`."""

    sparse_transpose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_28(typing.Protocol):
    """Structural protocol for objects exposing `sparseconcat`."""

    sparseconcat: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_29(typing.Protocol):
    """Structural protocol for objects exposing `sparsedensematmul`."""

    sparsedensematmul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_30(typing.Protocol):
    """Structural protocol for objects exposing `sparsemapvalues`."""

    sparsemapvalues: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_31(typing.Protocol):
    """Structural protocol for objects exposing `sparseoperandprotocol`."""

    sparseoperandprotocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_32(typing.Protocol):
    """Structural protocol for objects exposing `sparseplus`."""

    sparseplus: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_33(typing.Protocol):
    """Structural protocol for objects exposing `sparsereducemax`."""

    sparsereducemax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_34(typing.Protocol):
    """Structural protocol for objects exposing `sparsereshape`."""

    sparsereshape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_35(typing.Protocol):
    """Structural protocol for objects exposing `sparsesampledadd`."""

    sparsesampledadd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_36(typing.Protocol):
    """Structural protocol for objects exposing `sparsesegmentsum`."""

    sparsesegmentsum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_37(typing.Protocol):
    """Structural protocol for objects exposing `sparsesigmoid`."""

    sparsesigmoid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_38(typing.Protocol):
    """Structural protocol for objects exposing `sparsesplit`."""

    sparsesplit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_39(typing.Protocol):
    """Structural protocol for objects exposing `sparsetodense`."""

    sparsetodense: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_40(typing.Protocol):
    """Structural protocol for objects exposing `sparsetranspose`."""

    sparsetranspose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_41(typing.Protocol):
    """Structural protocol for objects exposing `special`."""

    special: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_42(typing.Protocol):
    """Structural protocol for objects exposing `split`."""

    split: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_43(typing.Protocol):
    """Structural protocol for objects exposing `split_pattern`."""

    split_pattern: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember46_44(typing.Protocol):
    """Structural protocol for objects exposing `split_pattern_regex`."""

    split_pattern_regex: EagerTensorProtocol


SHARD_46_READERS: dict[str, MemberReader] = {
    "sort_indices": lambda t: (True, t.sort_indices) if isinstance(t, _HasMember46_00) else (False, None),
    "source": lambda t: (True, t.source) if isinstance(t, _HasMember46_01) else (False, None),
    "source_idx": lambda t: (True, t.source_idx) if isinstance(t, _HasMember46_02) else (False, None),
    "source_node": lambda t: (True, t.source_node) if isinstance(t, _HasMember46_03) else (False, None),
    "space_batch": lambda t: (True, t.space_batch) if isinstance(t, _HasMember46_04) else (False, None),
    "space_to_batch": lambda t: (True, t.space_to_batch) if isinstance(t, _HasMember46_05) else (False, None),
    "space_to_batch_nd": lambda t: (True, t.space_to_batch_nd) if isinstance(t, _HasMember46_06) else (False, None),
    "spacing": lambda t: (True, t.spacing) if isinstance(t, _HasMember46_07) else (False, None),
    "sparse": lambda t: (True, t.sparse) if isinstance(t, _HasMember46_08) else (False, None),
    "sparse_categorical_crossentropy": lambda t: (
        (True, t.sparse_categorical_crossentropy) if isinstance(t, _HasMember46_09) else (False, None)
    ),
    "sparse_concat": lambda t: (True, t.sparse_concat) if isinstance(t, _HasMember46_10) else (False, None),
    "sparse_dense_mat_mul": lambda t: (
        (True, t.sparse_dense_mat_mul) if isinstance(t, _HasMember46_11) else (False, None)
    ),
    "sparse_eye": lambda t: (True, t.sparse_eye) if isinstance(t, _HasMember46_12) else (False, None),
    "sparse_map_values": lambda t: (True, t.sparse_map_values) if isinstance(t, _HasMember46_13) else (False, None),
    "sparse_maximum": lambda t: (True, t.sparse_maximum) if isinstance(t, _HasMember46_14) else (False, None),
    "sparse_operand_protocol": lambda t: (
        (True, t.sparse_operand_protocol) if isinstance(t, _HasMember46_15) else (False, None)
    ),
    "sparse_plus": lambda t: (True, t.sparse_plus) if isinstance(t, _HasMember46_16) else (False, None),
    "sparse_reduce_max": lambda t: (True, t.sparse_reduce_max) if isinstance(t, _HasMember46_17) else (False, None),
    "sparse_reorder": lambda t: (True, t.sparse_reorder) if isinstance(t, _HasMember46_18) else (False, None),
    "sparse_reshape": lambda t: (True, t.sparse_reshape) if isinstance(t, _HasMember46_19) else (False, None),
    "sparse_sampled_add": lambda t: (True, t.sparse_sampled_add) if isinstance(t, _HasMember46_20) else (False, None),
    "sparse_segment_mean": lambda t: (True, t.sparse_segment_mean) if isinstance(t, _HasMember46_21) else (False, None),
    "sparse_segment_sum": lambda t: (True, t.sparse_segment_sum) if isinstance(t, _HasMember46_22) else (False, None),
    "sparse_sigmoid": lambda t: (True, t.sparse_sigmoid) if isinstance(t, _HasMember46_23) else (False, None),
    "sparse_softmax": lambda t: (True, t.sparse_softmax) if isinstance(t, _HasMember46_24) else (False, None),
    "sparse_split": lambda t: (True, t.sparse_split) if isinstance(t, _HasMember46_25) else (False, None),
    "sparse_to_dense": lambda t: (True, t.sparse_to_dense) if isinstance(t, _HasMember46_26) else (False, None),
    "sparse_transpose": lambda t: (True, t.sparse_transpose) if isinstance(t, _HasMember46_27) else (False, None),
    "sparseconcat": lambda t: (True, t.sparseconcat) if isinstance(t, _HasMember46_28) else (False, None),
    "sparsedensematmul": lambda t: (True, t.sparsedensematmul) if isinstance(t, _HasMember46_29) else (False, None),
    "sparsemapvalues": lambda t: (True, t.sparsemapvalues) if isinstance(t, _HasMember46_30) else (False, None),
    "sparseoperandprotocol": lambda t: (
        (True, t.sparseoperandprotocol) if isinstance(t, _HasMember46_31) else (False, None)
    ),
    "sparseplus": lambda t: (True, t.sparseplus) if isinstance(t, _HasMember46_32) else (False, None),
    "sparsereducemax": lambda t: (True, t.sparsereducemax) if isinstance(t, _HasMember46_33) else (False, None),
    "sparsereshape": lambda t: (True, t.sparsereshape) if isinstance(t, _HasMember46_34) else (False, None),
    "sparsesampledadd": lambda t: (True, t.sparsesampledadd) if isinstance(t, _HasMember46_35) else (False, None),
    "sparsesegmentsum": lambda t: (True, t.sparsesegmentsum) if isinstance(t, _HasMember46_36) else (False, None),
    "sparsesigmoid": lambda t: (True, t.sparsesigmoid) if isinstance(t, _HasMember46_37) else (False, None),
    "sparsesplit": lambda t: (True, t.sparsesplit) if isinstance(t, _HasMember46_38) else (False, None),
    "sparsetodense": lambda t: (True, t.sparsetodense) if isinstance(t, _HasMember46_39) else (False, None),
    "sparsetranspose": lambda t: (True, t.sparsetranspose) if isinstance(t, _HasMember46_40) else (False, None),
    "special": lambda t: (True, t.special) if isinstance(t, _HasMember46_41) else (False, None),
    "split": lambda t: (True, t.split) if isinstance(t, _HasMember46_42) else (False, None),
    "split_pattern": lambda t: (True, t.split_pattern) if isinstance(t, _HasMember46_43) else (False, None),
    "split_pattern_regex": lambda t: (True, t.split_pattern_regex) if isinstance(t, _HasMember46_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember46_00,
        _HasMember46_01,
        _HasMember46_02,
        _HasMember46_03,
        _HasMember46_04,
        _HasMember46_05,
        _HasMember46_06,
        _HasMember46_07,
        _HasMember46_08,
        _HasMember46_09,
        _HasMember46_10,
        _HasMember46_11,
        _HasMember46_12,
        _HasMember46_13,
        _HasMember46_14,
        _HasMember46_15,
        _HasMember46_16,
        _HasMember46_17,
        _HasMember46_18,
        _HasMember46_19,
        _HasMember46_20,
        _HasMember46_21,
        _HasMember46_22,
        _HasMember46_23,
        _HasMember46_24,
        _HasMember46_25,
        _HasMember46_26,
        _HasMember46_27,
        _HasMember46_28,
        _HasMember46_29,
        _HasMember46_30,
        _HasMember46_31,
        _HasMember46_32,
        _HasMember46_33,
        _HasMember46_34,
        _HasMember46_35,
        _HasMember46_36,
        _HasMember46_37,
        _HasMember46_38,
        _HasMember46_39,
        _HasMember46_40,
        _HasMember46_41,
        _HasMember46_42,
        _HasMember46_43,
        _HasMember46_44,
    )
)
