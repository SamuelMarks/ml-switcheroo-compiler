"""Static structural protocol bindings shard 32 (mapflatvalues..mlir)."""

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
class _HasMember32_00(typing.Protocol):
    """Structural protocol for objects exposing `mapflatvalues`."""

    mapflatvalues: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_01(typing.Protocol):
    """Structural protocol for objects exposing `margin_ranking_loss`."""

    margin_ranking_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_02(typing.Protocol):
    """Structural protocol for objects exposing `mask_indices`."""

    mask_indices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_03(typing.Protocol):
    """Structural protocol for objects exposing `masked_equal`."""

    masked_equal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_04(typing.Protocol):
    """Structural protocol for objects exposing `match_substring`."""

    match_substring: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_05(typing.Protocol):
    """Structural protocol for objects exposing `match_substring_regex`."""

    match_substring_regex: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_06(typing.Protocol):
    """Structural protocol for objects exposing `math`."""

    math: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_07(typing.Protocol):
    """Structural protocol for objects exposing `matmul`."""

    matmul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_08(typing.Protocol):
    """Structural protocol for objects exposing `matmul_fn`."""

    matmul_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_09(typing.Protocol):
    """Structural protocol for objects exposing `matrix_norm`."""

    matrix_norm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_10(typing.Protocol):
    """Structural protocol for objects exposing `matrix_power`."""

    matrix_power: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_11(typing.Protocol):
    """Structural protocol for objects exposing `matrix_rank`."""

    matrix_rank: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_12(typing.Protocol):
    """Structural protocol for objects exposing `matrix_transpose`."""

    matrix_transpose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_13(typing.Protocol):
    """Structural protocol for objects exposing `matrixrank`."""

    matrixrank: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_14(typing.Protocol):
    """Structural protocol for objects exposing `matrixtranspose`."""

    matrixtranspose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_15(typing.Protocol):
    """Structural protocol for objects exposing `matvec`."""

    matvec: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_16(typing.Protocol):
    """Structural protocol for objects exposing `max`."""

    max: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_17(typing.Protocol):
    """Structural protocol for objects exposing `max2`."""

    max2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_18(typing.Protocol):
    """Structural protocol for objects exposing `max_element_wise`."""

    max_element_wise: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_19(typing.Protocol):
    """Structural protocol for objects exposing `max_pool`."""

    max_pool: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_20(typing.Protocol):
    """Structural protocol for objects exposing `max_pool1d`."""

    max_pool1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_21(typing.Protocol):
    """Structural protocol for objects exposing `max_pool2d`."""

    max_pool2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_22(typing.Protocol):
    """Structural protocol for objects exposing `max_pool3d`."""

    max_pool3d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_23(typing.Protocol):
    """Structural protocol for objects exposing `maximum`."""

    maximum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_24(typing.Protocol):
    """Structural protocol for objects exposing `maxpool2d`."""

    maxpool2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_25(typing.Protocol):
    """Structural protocol for objects exposing `mean`."""

    mean: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_26(typing.Protocol):
    """Structural protocol for objects exposing `median`."""

    median: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_27(typing.Protocol):
    """Structural protocol for objects exposing `median_filter`."""

    median_filter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_28(typing.Protocol):
    """Structural protocol for objects exposing `memmove`."""

    memmove: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_29(typing.Protocol):
    """Structural protocol for objects exposing `mesh_mapping`."""

    mesh_mapping: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_30(typing.Protocol):
    """Structural protocol for objects exposing `meshgrid`."""

    meshgrid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_31(typing.Protocol):
    """Structural protocol for objects exposing `metal`."""

    metal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_32(typing.Protocol):
    """Structural protocol for objects exposing `mgrid`."""

    mgrid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_33(typing.Protocol):
    """Structural protocol for objects exposing `min`."""

    min: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_34(typing.Protocol):
    """Structural protocol for objects exposing `min2`."""

    min2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_35(typing.Protocol):
    """Structural protocol for objects exposing `min_element_wise`."""

    min_element_wise: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_36(typing.Protocol):
    """Structural protocol for objects exposing `minimum`."""

    minimum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_37(typing.Protocol):
    """Structural protocol for objects exposing `misc`."""

    misc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_38(typing.Protocol):
    """Structural protocol for objects exposing `mish`."""

    mish: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_39(typing.Protocol):
    """Structural protocol for objects exposing `missing`."""

    missing: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_40(typing.Protocol):
    """Structural protocol for objects exposing `missing_cupy_attr`."""

    missing_cupy_attr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_41(typing.Protocol):
    """Structural protocol for objects exposing `missing_method`."""

    missing_method: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_42(typing.Protocol):
    """Structural protocol for objects exposing `mixup`."""

    mixup: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_43(typing.Protocol):
    """Structural protocol for objects exposing `ml_switcheroo_compiler`."""

    ml_switcheroo_compiler: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember32_44(typing.Protocol):
    """Structural protocol for objects exposing `mlir`."""

    mlir: EagerTensorProtocol


SHARD_32_READERS: dict[str, MemberReader] = {
    "mapflatvalues": lambda t: (True, t.mapflatvalues) if isinstance(t, _HasMember32_00) else (False, None),
    "margin_ranking_loss": lambda t: (True, t.margin_ranking_loss) if isinstance(t, _HasMember32_01) else (False, None),
    "mask_indices": lambda t: (True, t.mask_indices) if isinstance(t, _HasMember32_02) else (False, None),
    "masked_equal": lambda t: (True, t.masked_equal) if isinstance(t, _HasMember32_03) else (False, None),
    "match_substring": lambda t: (True, t.match_substring) if isinstance(t, _HasMember32_04) else (False, None),
    "match_substring_regex": lambda t: (
        (True, t.match_substring_regex) if isinstance(t, _HasMember32_05) else (False, None)
    ),
    "math": lambda t: (True, t.math) if isinstance(t, _HasMember32_06) else (False, None),
    "matmul": lambda t: (True, t.matmul) if isinstance(t, _HasMember32_07) else (False, None),
    "matmul_fn": lambda t: (True, t.matmul_fn) if isinstance(t, _HasMember32_08) else (False, None),
    "matrix_norm": lambda t: (True, t.matrix_norm) if isinstance(t, _HasMember32_09) else (False, None),
    "matrix_power": lambda t: (True, t.matrix_power) if isinstance(t, _HasMember32_10) else (False, None),
    "matrix_rank": lambda t: (True, t.matrix_rank) if isinstance(t, _HasMember32_11) else (False, None),
    "matrix_transpose": lambda t: (True, t.matrix_transpose) if isinstance(t, _HasMember32_12) else (False, None),
    "matrixrank": lambda t: (True, t.matrixrank) if isinstance(t, _HasMember32_13) else (False, None),
    "matrixtranspose": lambda t: (True, t.matrixtranspose) if isinstance(t, _HasMember32_14) else (False, None),
    "matvec": lambda t: (True, t.matvec) if isinstance(t, _HasMember32_15) else (False, None),
    "max": lambda t: (True, t.max) if isinstance(t, _HasMember32_16) else (False, None),
    "max2": lambda t: (True, t.max2) if isinstance(t, _HasMember32_17) else (False, None),
    "max_element_wise": lambda t: (True, t.max_element_wise) if isinstance(t, _HasMember32_18) else (False, None),
    "max_pool": lambda t: (True, t.max_pool) if isinstance(t, _HasMember32_19) else (False, None),
    "max_pool1d": lambda t: (True, t.max_pool1d) if isinstance(t, _HasMember32_20) else (False, None),
    "max_pool2d": lambda t: (True, t.max_pool2d) if isinstance(t, _HasMember32_21) else (False, None),
    "max_pool3d": lambda t: (True, t.max_pool3d) if isinstance(t, _HasMember32_22) else (False, None),
    "maximum": lambda t: (True, t.maximum) if isinstance(t, _HasMember32_23) else (False, None),
    "maxpool2d": lambda t: (True, t.maxpool2d) if isinstance(t, _HasMember32_24) else (False, None),
    "mean": lambda t: (True, t.mean) if isinstance(t, _HasMember32_25) else (False, None),
    "median": lambda t: (True, t.median) if isinstance(t, _HasMember32_26) else (False, None),
    "median_filter": lambda t: (True, t.median_filter) if isinstance(t, _HasMember32_27) else (False, None),
    "memmove": lambda t: (True, t.memmove) if isinstance(t, _HasMember32_28) else (False, None),
    "mesh_mapping": lambda t: (True, t.mesh_mapping) if isinstance(t, _HasMember32_29) else (False, None),
    "meshgrid": lambda t: (True, t.meshgrid) if isinstance(t, _HasMember32_30) else (False, None),
    "metal": lambda t: (True, t.metal) if isinstance(t, _HasMember32_31) else (False, None),
    "mgrid": lambda t: (True, t.mgrid) if isinstance(t, _HasMember32_32) else (False, None),
    "min": lambda t: (True, t.min) if isinstance(t, _HasMember32_33) else (False, None),
    "min2": lambda t: (True, t.min2) if isinstance(t, _HasMember32_34) else (False, None),
    "min_element_wise": lambda t: (True, t.min_element_wise) if isinstance(t, _HasMember32_35) else (False, None),
    "minimum": lambda t: (True, t.minimum) if isinstance(t, _HasMember32_36) else (False, None),
    "misc": lambda t: (True, t.misc) if isinstance(t, _HasMember32_37) else (False, None),
    "mish": lambda t: (True, t.mish) if isinstance(t, _HasMember32_38) else (False, None),
    "missing": lambda t: (True, t.missing) if isinstance(t, _HasMember32_39) else (False, None),
    "missing_cupy_attr": lambda t: (True, t.missing_cupy_attr) if isinstance(t, _HasMember32_40) else (False, None),
    "missing_method": lambda t: (True, t.missing_method) if isinstance(t, _HasMember32_41) else (False, None),
    "mixup": lambda t: (True, t.mixup) if isinstance(t, _HasMember32_42) else (False, None),
    "ml_switcheroo_compiler": lambda t: (
        (True, t.ml_switcheroo_compiler) if isinstance(t, _HasMember32_43) else (False, None)
    ),
    "mlir": lambda t: (True, t.mlir) if isinstance(t, _HasMember32_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember32_00,
        _HasMember32_01,
        _HasMember32_02,
        _HasMember32_03,
        _HasMember32_04,
        _HasMember32_05,
        _HasMember32_06,
        _HasMember32_07,
        _HasMember32_08,
        _HasMember32_09,
        _HasMember32_10,
        _HasMember32_11,
        _HasMember32_12,
        _HasMember32_13,
        _HasMember32_14,
        _HasMember32_15,
        _HasMember32_16,
        _HasMember32_17,
        _HasMember32_18,
        _HasMember32_19,
        _HasMember32_20,
        _HasMember32_21,
        _HasMember32_22,
        _HasMember32_23,
        _HasMember32_24,
        _HasMember32_25,
        _HasMember32_26,
        _HasMember32_27,
        _HasMember32_28,
        _HasMember32_29,
        _HasMember32_30,
        _HasMember32_31,
        _HasMember32_32,
        _HasMember32_33,
        _HasMember32_34,
        _HasMember32_35,
        _HasMember32_36,
        _HasMember32_37,
        _HasMember32_38,
        _HasMember32_39,
        _HasMember32_40,
        _HasMember32_41,
        _HasMember32_42,
        _HasMember32_43,
        _HasMember32_44,
    )
)
