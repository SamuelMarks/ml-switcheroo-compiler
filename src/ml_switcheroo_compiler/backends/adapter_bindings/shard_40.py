"""Static structural protocol bindings shard 40 (raggedstack..rcclGetUniqueId)."""

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
class _HasMember40_00(typing.Protocol):
    """Structural protocol for objects exposing `raggedstack`."""

    raggedstack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_01(typing.Protocol):
    """Structural protocol for objects exposing `raggedstackdynamicpartitions`."""

    raggedstackdynamicpartitions: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_02(typing.Protocol):
    """Structural protocol for objects exposing `raggedtensortodense`."""

    raggedtensortodense: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_03(typing.Protocol):
    """Structural protocol for objects exposing `raise_custom`."""

    raise_custom: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_04(typing.Protocol):
    """Structural protocol for objects exposing `raise_err`."""

    raise_err: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_05(typing.Protocol):
    """Structural protocol for objects exposing `rand`."""

    rand: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_06(typing.Protocol):
    """Structural protocol for objects exposing `rand_augment`."""

    rand_augment: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_07(typing.Protocol):
    """Structural protocol for objects exposing `randint`."""

    randint: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_08(typing.Protocol):
    """Structural protocol for objects exposing `randn`."""

    randn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_09(typing.Protocol):
    """Structural protocol for objects exposing `random`."""

    random: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_10(typing.Protocol):
    """Structural protocol for objects exposing `random_color_jitter`."""

    random_color_jitter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_11(typing.Protocol):
    """Structural protocol for objects exposing `random_crop`."""

    random_crop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_12(typing.Protocol):
    """Structural protocol for objects exposing `random_elastic_transform`."""

    random_elastic_transform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_13(typing.Protocol):
    """Structural protocol for objects exposing `random_erasing`."""

    random_erasing: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_14(typing.Protocol):
    """Structural protocol for objects exposing `random_flip`."""

    random_flip: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_15(typing.Protocol):
    """Structural protocol for objects exposing `random_gaussian_blur`."""

    random_gaussian_blur: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_16(typing.Protocol):
    """Structural protocol for objects exposing `random_ops`."""

    random_ops: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_17(typing.Protocol):
    """Structural protocol for objects exposing `random_perspective`."""

    random_perspective: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_18(typing.Protocol):
    """Structural protocol for objects exposing `random_rotation`."""

    random_rotation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_19(typing.Protocol):
    """Structural protocol for objects exposing `random_sharpness`."""

    random_sharpness: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_20(typing.Protocol):
    """Structural protocol for objects exposing `random_shear`."""

    random_shear: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_21(typing.Protocol):
    """Structural protocol for objects exposing `random_translation`."""

    random_translation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_22(typing.Protocol):
    """Structural protocol for objects exposing `random_zoom`."""

    random_zoom: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_23(typing.Protocol):
    """Structural protocol for objects exposing `rank`."""

    rank: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_24(typing.Protocol):
    """Structural protocol for objects exposing `ravel`."""

    ravel: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_25(typing.Protocol):
    """Structural protocol for objects exposing `ravel_multi_index`."""

    ravel_multi_index: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_26(typing.Protocol):
    """Structural protocol for objects exposing `ravelmultiindex`."""

    ravelmultiindex: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_27(typing.Protocol):
    """Structural protocol for objects exposing `raw_conv2_d`."""

    raw_conv2_d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_28(typing.Protocol):
    """Structural protocol for objects exposing `raw_mat_mul`."""

    raw_mat_mul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_29(typing.Protocol):
    """Structural protocol for objects exposing `raw_merge`."""

    raw_merge: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_30(typing.Protocol):
    """Structural protocol for objects exposing `raw_op`."""

    raw_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_31(typing.Protocol):
    """Structural protocol for objects exposing `raw_ops`."""

    raw_ops: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_32(typing.Protocol):
    """Structural protocol for objects exposing `raw_switch`."""

    raw_switch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_33(typing.Protocol):
    """Structural protocol for objects exposing `rawconv2d`."""

    rawconv2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_34(typing.Protocol):
    """Structural protocol for objects exposing `rawmatmul`."""

    rawmatmul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_35(typing.Protocol):
    """Structural protocol for objects exposing `rawmerge`."""

    rawmerge: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_36(typing.Protocol):
    """Structural protocol for objects exposing `rawop`."""

    rawop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_37(typing.Protocol):
    """Structural protocol for objects exposing `rawswitch`."""

    rawswitch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_38(typing.Protocol):
    """Structural protocol for objects exposing `rcclAllGather`."""

    rcclAllGather: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_39(typing.Protocol):
    """Structural protocol for objects exposing `rcclAllReduce`."""

    rcclAllReduce: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_40(typing.Protocol):
    """Structural protocol for objects exposing `rcclAllToAll`."""

    rcclAllToAll: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_41(typing.Protocol):
    """Structural protocol for objects exposing `rcclBroadcast`."""

    rcclBroadcast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_42(typing.Protocol):
    """Structural protocol for objects exposing `rcclCommDestroy`."""

    rcclCommDestroy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_43(typing.Protocol):
    """Structural protocol for objects exposing `rcclCommInitRank`."""

    rcclCommInitRank: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember40_44(typing.Protocol):
    """Structural protocol for objects exposing `rcclGetUniqueId`."""

    rcclGetUniqueId: EagerTensorProtocol


SHARD_40_READERS: dict[str, MemberReader] = {
    "raggedstack": lambda t: (True, t.raggedstack) if isinstance(t, _HasMember40_00) else (False, None),
    "raggedstackdynamicpartitions": lambda t: (
        (True, t.raggedstackdynamicpartitions) if isinstance(t, _HasMember40_01) else (False, None)
    ),
    "raggedtensortodense": lambda t: (True, t.raggedtensortodense) if isinstance(t, _HasMember40_02) else (False, None),
    "raise_custom": lambda t: (True, t.raise_custom) if isinstance(t, _HasMember40_03) else (False, None),
    "raise_err": lambda t: (True, t.raise_err) if isinstance(t, _HasMember40_04) else (False, None),
    "rand": lambda t: (True, t.rand) if isinstance(t, _HasMember40_05) else (False, None),
    "rand_augment": lambda t: (True, t.rand_augment) if isinstance(t, _HasMember40_06) else (False, None),
    "randint": lambda t: (True, t.randint) if isinstance(t, _HasMember40_07) else (False, None),
    "randn": lambda t: (True, t.randn) if isinstance(t, _HasMember40_08) else (False, None),
    "random": lambda t: (True, t.random) if isinstance(t, _HasMember40_09) else (False, None),
    "random_color_jitter": lambda t: (True, t.random_color_jitter) if isinstance(t, _HasMember40_10) else (False, None),
    "random_crop": lambda t: (True, t.random_crop) if isinstance(t, _HasMember40_11) else (False, None),
    "random_elastic_transform": lambda t: (
        (True, t.random_elastic_transform) if isinstance(t, _HasMember40_12) else (False, None)
    ),
    "random_erasing": lambda t: (True, t.random_erasing) if isinstance(t, _HasMember40_13) else (False, None),
    "random_flip": lambda t: (True, t.random_flip) if isinstance(t, _HasMember40_14) else (False, None),
    "random_gaussian_blur": lambda t: (
        (True, t.random_gaussian_blur) if isinstance(t, _HasMember40_15) else (False, None)
    ),
    "random_ops": lambda t: (True, t.random_ops) if isinstance(t, _HasMember40_16) else (False, None),
    "random_perspective": lambda t: (True, t.random_perspective) if isinstance(t, _HasMember40_17) else (False, None),
    "random_rotation": lambda t: (True, t.random_rotation) if isinstance(t, _HasMember40_18) else (False, None),
    "random_sharpness": lambda t: (True, t.random_sharpness) if isinstance(t, _HasMember40_19) else (False, None),
    "random_shear": lambda t: (True, t.random_shear) if isinstance(t, _HasMember40_20) else (False, None),
    "random_translation": lambda t: (True, t.random_translation) if isinstance(t, _HasMember40_21) else (False, None),
    "random_zoom": lambda t: (True, t.random_zoom) if isinstance(t, _HasMember40_22) else (False, None),
    "rank": lambda t: (True, t.rank) if isinstance(t, _HasMember40_23) else (False, None),
    "ravel": lambda t: (True, t.ravel) if isinstance(t, _HasMember40_24) else (False, None),
    "ravel_multi_index": lambda t: (True, t.ravel_multi_index) if isinstance(t, _HasMember40_25) else (False, None),
    "ravelmultiindex": lambda t: (True, t.ravelmultiindex) if isinstance(t, _HasMember40_26) else (False, None),
    "raw_conv2_d": lambda t: (True, t.raw_conv2_d) if isinstance(t, _HasMember40_27) else (False, None),
    "raw_mat_mul": lambda t: (True, t.raw_mat_mul) if isinstance(t, _HasMember40_28) else (False, None),
    "raw_merge": lambda t: (True, t.raw_merge) if isinstance(t, _HasMember40_29) else (False, None),
    "raw_op": lambda t: (True, t.raw_op) if isinstance(t, _HasMember40_30) else (False, None),
    "raw_ops": lambda t: (True, t.raw_ops) if isinstance(t, _HasMember40_31) else (False, None),
    "raw_switch": lambda t: (True, t.raw_switch) if isinstance(t, _HasMember40_32) else (False, None),
    "rawconv2d": lambda t: (True, t.rawconv2d) if isinstance(t, _HasMember40_33) else (False, None),
    "rawmatmul": lambda t: (True, t.rawmatmul) if isinstance(t, _HasMember40_34) else (False, None),
    "rawmerge": lambda t: (True, t.rawmerge) if isinstance(t, _HasMember40_35) else (False, None),
    "rawop": lambda t: (True, t.rawop) if isinstance(t, _HasMember40_36) else (False, None),
    "rawswitch": lambda t: (True, t.rawswitch) if isinstance(t, _HasMember40_37) else (False, None),
    "rcclAllGather": lambda t: (True, t.rcclAllGather) if isinstance(t, _HasMember40_38) else (False, None),
    "rcclAllReduce": lambda t: (True, t.rcclAllReduce) if isinstance(t, _HasMember40_39) else (False, None),
    "rcclAllToAll": lambda t: (True, t.rcclAllToAll) if isinstance(t, _HasMember40_40) else (False, None),
    "rcclBroadcast": lambda t: (True, t.rcclBroadcast) if isinstance(t, _HasMember40_41) else (False, None),
    "rcclCommDestroy": lambda t: (True, t.rcclCommDestroy) if isinstance(t, _HasMember40_42) else (False, None),
    "rcclCommInitRank": lambda t: (True, t.rcclCommInitRank) if isinstance(t, _HasMember40_43) else (False, None),
    "rcclGetUniqueId": lambda t: (True, t.rcclGetUniqueId) if isinstance(t, _HasMember40_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember40_00,
        _HasMember40_01,
        _HasMember40_02,
        _HasMember40_03,
        _HasMember40_04,
        _HasMember40_05,
        _HasMember40_06,
        _HasMember40_07,
        _HasMember40_08,
        _HasMember40_09,
        _HasMember40_10,
        _HasMember40_11,
        _HasMember40_12,
        _HasMember40_13,
        _HasMember40_14,
        _HasMember40_15,
        _HasMember40_16,
        _HasMember40_17,
        _HasMember40_18,
        _HasMember40_19,
        _HasMember40_20,
        _HasMember40_21,
        _HasMember40_22,
        _HasMember40_23,
        _HasMember40_24,
        _HasMember40_25,
        _HasMember40_26,
        _HasMember40_27,
        _HasMember40_28,
        _HasMember40_29,
        _HasMember40_30,
        _HasMember40_31,
        _HasMember40_32,
        _HasMember40_33,
        _HasMember40_34,
        _HasMember40_35,
        _HasMember40_36,
        _HasMember40_37,
        _HasMember40_38,
        _HasMember40_39,
        _HasMember40_40,
        _HasMember40_41,
        _HasMember40_42,
        _HasMember40_43,
        _HasMember40_44,
    )
)
