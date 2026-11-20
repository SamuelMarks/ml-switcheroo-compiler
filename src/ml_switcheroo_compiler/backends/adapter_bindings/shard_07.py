"""Static structural protocol bindings shard 07 (accept..amax)."""

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
class _HasMember07_00(typing.Protocol):
    """Structural protocol for objects exposing `accept`."""

    accept: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_01(typing.Protocol):
    """Structural protocol for objects exposing `access_mode`."""

    access_mode: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_02(typing.Protocol):
    """Structural protocol for objects exposing `accumulate`."""

    accumulate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_03(typing.Protocol):
    """Structural protocol for objects exposing `accumulate_n`."""

    accumulate_n: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_04(typing.Protocol):
    """Structural protocol for objects exposing `acos`."""

    acos: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_05(typing.Protocol):
    """Structural protocol for objects exposing `acosh`."""

    acosh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_06(typing.Protocol):
    """Structural protocol for objects exposing `adaptive_avg_pool2d`."""

    adaptive_avg_pool2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_07(typing.Protocol):
    """Structural protocol for objects exposing `adaptive_avg_pool3d`."""

    adaptive_avg_pool3d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_08(typing.Protocol):
    """Structural protocol for objects exposing `adaptive_log_softmax_with_loss`."""

    adaptive_log_softmax_with_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_09(typing.Protocol):
    """Structural protocol for objects exposing `adaptive_max_pool2d`."""

    adaptive_max_pool2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_10(typing.Protocol):
    """Structural protocol for objects exposing `adaptive_max_pool3d`."""

    adaptive_max_pool3d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_11(typing.Protocol):
    """Structural protocol for objects exposing `adaptivelogsoftmaxwithloss`."""

    adaptivelogsoftmaxwithloss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_12(typing.Protocol):
    """Structural protocol for objects exposing `add`."""

    add: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_13(typing.Protocol):
    """Structural protocol for objects exposing `add_line`."""

    add_line: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_14(typing.Protocol):
    """Structural protocol for objects exposing `add_n`."""

    add_n: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_15(typing.Protocol):
    """Structural protocol for objects exposing `add_node`."""

    add_node: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_16(typing.Protocol):
    """Structural protocol for objects exposing `addmm`."""

    addmm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_17(typing.Protocol):
    """Structural protocol for objects exposing `address`."""

    address: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_18(typing.Protocol):
    """Structural protocol for objects exposing `address_space`."""

    address_space: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_19(typing.Protocol):
    """Structural protocol for objects exposing `adjoint`."""

    adjoint: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_20(typing.Protocol):
    """Structural protocol for objects exposing `adjust_brightness`."""

    adjust_brightness: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_21(typing.Protocol):
    """Structural protocol for objects exposing `adjust_contrast`."""

    adjust_contrast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_22(typing.Protocol):
    """Structural protocol for objects exposing `adjust_hue`."""

    adjust_hue: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_23(typing.Protocol):
    """Structural protocol for objects exposing `adjust_saturation`."""

    adjust_saturation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_24(typing.Protocol):
    """Structural protocol for objects exposing `adjustbrightness`."""

    adjustbrightness: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_25(typing.Protocol):
    """Structural protocol for objects exposing `adjustcontrast`."""

    adjustcontrast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_26(typing.Protocol):
    """Structural protocol for objects exposing `adjusthue`."""

    adjusthue: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_27(typing.Protocol):
    """Structural protocol for objects exposing `adjustsaturation`."""

    adjustsaturation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_28(typing.Protocol):
    """Structural protocol for objects exposing `affine_config`."""

    affine_config: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_29(typing.Protocol):
    """Structural protocol for objects exposing `affine_generator`."""

    affine_generator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_30(typing.Protocol):
    """Structural protocol for objects exposing `affine_grid`."""

    affine_grid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_31(typing.Protocol):
    """Structural protocol for objects exposing `affine_transform`."""

    affine_transform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_32(typing.Protocol):
    """Structural protocol for objects exposing `affineconfig`."""

    affineconfig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_33(typing.Protocol):
    """Structural protocol for objects exposing `affinegenerator`."""

    affinegenerator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_34(typing.Protocol):
    """Structural protocol for objects exposing `affinegrid`."""

    affinegrid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_35(typing.Protocol):
    """Structural protocol for objects exposing `affinetransform`."""

    affinetransform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_36(typing.Protocol):
    """Structural protocol for objects exposing `aggregate`."""

    aggregate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_37(typing.Protocol):
    """Structural protocol for objects exposing `align_corners`."""

    align_corners: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_38(typing.Protocol):
    """Structural protocol for objects exposing `all`."""

    all: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_39(typing.Protocol):
    """Structural protocol for objects exposing `all_gather`."""

    all_gather: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_40(typing.Protocol):
    """Structural protocol for objects exposing `all_reduce_ring`."""

    all_reduce_ring: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_41(typing.Protocol):
    """Structural protocol for objects exposing `all_sum`."""

    all_sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_42(typing.Protocol):
    """Structural protocol for objects exposing `all_to_all`."""

    all_to_all: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_43(typing.Protocol):
    """Structural protocol for objects exposing `allclose`."""

    allclose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember07_44(typing.Protocol):
    """Structural protocol for objects exposing `amax`."""

    amax: EagerTensorProtocol


SHARD_07_READERS: dict[str, MemberReader] = {
    "accept": lambda t: (True, t.accept) if isinstance(t, _HasMember07_00) else (False, None),
    "access_mode": lambda t: (True, t.access_mode) if isinstance(t, _HasMember07_01) else (False, None),
    "accumulate": lambda t: (True, t.accumulate) if isinstance(t, _HasMember07_02) else (False, None),
    "accumulate_n": lambda t: (True, t.accumulate_n) if isinstance(t, _HasMember07_03) else (False, None),
    "acos": lambda t: (True, t.acos) if isinstance(t, _HasMember07_04) else (False, None),
    "acosh": lambda t: (True, t.acosh) if isinstance(t, _HasMember07_05) else (False, None),
    "adaptive_avg_pool2d": lambda t: (True, t.adaptive_avg_pool2d) if isinstance(t, _HasMember07_06) else (False, None),
    "adaptive_avg_pool3d": lambda t: (True, t.adaptive_avg_pool3d) if isinstance(t, _HasMember07_07) else (False, None),
    "adaptive_log_softmax_with_loss": lambda t: (
        (True, t.adaptive_log_softmax_with_loss) if isinstance(t, _HasMember07_08) else (False, None)
    ),
    "adaptive_max_pool2d": lambda t: (True, t.adaptive_max_pool2d) if isinstance(t, _HasMember07_09) else (False, None),
    "adaptive_max_pool3d": lambda t: (True, t.adaptive_max_pool3d) if isinstance(t, _HasMember07_10) else (False, None),
    "adaptivelogsoftmaxwithloss": lambda t: (
        (True, t.adaptivelogsoftmaxwithloss) if isinstance(t, _HasMember07_11) else (False, None)
    ),
    "add": lambda t: (True, t.add) if isinstance(t, _HasMember07_12) else (False, None),
    "add_line": lambda t: (True, t.add_line) if isinstance(t, _HasMember07_13) else (False, None),
    "add_n": lambda t: (True, t.add_n) if isinstance(t, _HasMember07_14) else (False, None),
    "add_node": lambda t: (True, t.add_node) if isinstance(t, _HasMember07_15) else (False, None),
    "addmm": lambda t: (True, t.addmm) if isinstance(t, _HasMember07_16) else (False, None),
    "address": lambda t: (True, t.address) if isinstance(t, _HasMember07_17) else (False, None),
    "address_space": lambda t: (True, t.address_space) if isinstance(t, _HasMember07_18) else (False, None),
    "adjoint": lambda t: (True, t.adjoint) if isinstance(t, _HasMember07_19) else (False, None),
    "adjust_brightness": lambda t: (True, t.adjust_brightness) if isinstance(t, _HasMember07_20) else (False, None),
    "adjust_contrast": lambda t: (True, t.adjust_contrast) if isinstance(t, _HasMember07_21) else (False, None),
    "adjust_hue": lambda t: (True, t.adjust_hue) if isinstance(t, _HasMember07_22) else (False, None),
    "adjust_saturation": lambda t: (True, t.adjust_saturation) if isinstance(t, _HasMember07_23) else (False, None),
    "adjustbrightness": lambda t: (True, t.adjustbrightness) if isinstance(t, _HasMember07_24) else (False, None),
    "adjustcontrast": lambda t: (True, t.adjustcontrast) if isinstance(t, _HasMember07_25) else (False, None),
    "adjusthue": lambda t: (True, t.adjusthue) if isinstance(t, _HasMember07_26) else (False, None),
    "adjustsaturation": lambda t: (True, t.adjustsaturation) if isinstance(t, _HasMember07_27) else (False, None),
    "affine_config": lambda t: (True, t.affine_config) if isinstance(t, _HasMember07_28) else (False, None),
    "affine_generator": lambda t: (True, t.affine_generator) if isinstance(t, _HasMember07_29) else (False, None),
    "affine_grid": lambda t: (True, t.affine_grid) if isinstance(t, _HasMember07_30) else (False, None),
    "affine_transform": lambda t: (True, t.affine_transform) if isinstance(t, _HasMember07_31) else (False, None),
    "affineconfig": lambda t: (True, t.affineconfig) if isinstance(t, _HasMember07_32) else (False, None),
    "affinegenerator": lambda t: (True, t.affinegenerator) if isinstance(t, _HasMember07_33) else (False, None),
    "affinegrid": lambda t: (True, t.affinegrid) if isinstance(t, _HasMember07_34) else (False, None),
    "affinetransform": lambda t: (True, t.affinetransform) if isinstance(t, _HasMember07_35) else (False, None),
    "aggregate": lambda t: (True, t.aggregate) if isinstance(t, _HasMember07_36) else (False, None),
    "align_corners": lambda t: (True, t.align_corners) if isinstance(t, _HasMember07_37) else (False, None),
    "all": lambda t: (True, t.all) if isinstance(t, _HasMember07_38) else (False, None),
    "all_gather": lambda t: (True, t.all_gather) if isinstance(t, _HasMember07_39) else (False, None),
    "all_reduce_ring": lambda t: (True, t.all_reduce_ring) if isinstance(t, _HasMember07_40) else (False, None),
    "all_sum": lambda t: (True, t.all_sum) if isinstance(t, _HasMember07_41) else (False, None),
    "all_to_all": lambda t: (True, t.all_to_all) if isinstance(t, _HasMember07_42) else (False, None),
    "allclose": lambda t: (True, t.allclose) if isinstance(t, _HasMember07_43) else (False, None),
    "amax": lambda t: (True, t.amax) if isinstance(t, _HasMember07_44) else (False, None),
}
