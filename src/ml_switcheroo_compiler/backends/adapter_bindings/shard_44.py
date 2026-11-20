"""Static structural protocol bindings shard 44 (scan_op..shape_inference)."""

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
class _HasMember44_00(typing.Protocol):
    """Structural protocol for objects exposing `scan_op`."""

    scan_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_01(typing.Protocol):
    """Structural protocol for objects exposing `scan_tracing`."""

    scan_tracing: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_02(typing.Protocol):
    """Structural protocol for objects exposing `scanop`."""

    scanop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_03(typing.Protocol):
    """Structural protocol for objects exposing `scatter`."""

    scatter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_04(typing.Protocol):
    """Structural protocol for objects exposing `scatter_nd`."""

    scatter_nd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_05(typing.Protocol):
    """Structural protocol for objects exposing `scatterapply`."""

    scatterapply: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_06(typing.Protocol):
    """Structural protocol for objects exposing `scattermax`."""

    scattermax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_07(typing.Protocol):
    """Structural protocol for objects exposing `scattermin`."""

    scattermin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_08(typing.Protocol):
    """Structural protocol for objects exposing `scattermul`."""

    scattermul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_09(typing.Protocol):
    """Structural protocol for objects exposing `scatternd`."""

    scatternd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_10(typing.Protocol):
    """Structural protocol for objects exposing `schema`."""

    schema: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_11(typing.Protocol):
    """Structural protocol for objects exposing `schur`."""

    schur: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_12(typing.Protocol):
    """Structural protocol for objects exposing `scipy`."""

    scipy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_13(typing.Protocol):
    """Structural protocol for objects exposing `searchsorted`."""

    searchsorted: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_14(typing.Protocol):
    """Structural protocol for objects exposing `seed`."""

    seed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_15(typing.Protocol):
    """Structural protocol for objects exposing `segment_max`."""

    segment_max: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_16(typing.Protocol):
    """Structural protocol for objects exposing `segment_mean`."""

    segment_mean: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_17(typing.Protocol):
    """Structural protocol for objects exposing `segment_min`."""

    segment_min: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_18(typing.Protocol):
    """Structural protocol for objects exposing `segment_prod`."""

    segment_prod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_19(typing.Protocol):
    """Structural protocol for objects exposing `segment_sum`."""

    segment_sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_20(typing.Protocol):
    """Structural protocol for objects exposing `segmented_mm`."""

    segmented_mm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_21(typing.Protocol):
    """Structural protocol for objects exposing `select`."""

    select: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_22(typing.Protocol):
    """Structural protocol for objects exposing `send`."""

    send: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_23(typing.Protocol):
    """Structural protocol for objects exposing `sendMessage`."""

    sendMessage: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_24(typing.Protocol):
    """Structural protocol for objects exposing `separable_conv`."""

    separable_conv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_25(typing.Protocol):
    """Structural protocol for objects exposing `separable_conv1d`."""

    separable_conv1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_26(typing.Protocol):
    """Structural protocol for objects exposing `separable_conv2d`."""

    separable_conv2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_27(typing.Protocol):
    """Structural protocol for objects exposing `serialize`."""

    serialize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_28(typing.Protocol):
    """Structural protocol for objects exposing `serialize_tensor`."""

    serialize_tensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_29(typing.Protocol):
    """Structural protocol for objects exposing `serializetensor`."""

    serializetensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_30(typing.Protocol):
    """Structural protocol for objects exposing `serializetensorcamel`."""

    serializetensorcamel: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_31(typing.Protocol):
    """Structural protocol for objects exposing `setBuffer_offset_atIndex_`."""

    setBuffer_offset_atIndex_: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_32(typing.Protocol):
    """Structural protocol for objects exposing `setComputePipelineState_`."""

    setComputePipelineState_: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_33(typing.Protocol):
    """Structural protocol for objects exposing `setUp`."""

    setUp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_34(typing.Protocol):
    """Structural protocol for objects exposing `set_count_positive`."""

    set_count_positive: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_35(typing.Protocol):
    """Structural protocol for objects exposing `set_count_zero`."""

    set_count_zero: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_36(typing.Protocol):
    """Structural protocol for objects exposing `set_default_stream`."""

    set_default_stream: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_37(typing.Protocol):
    """Structural protocol for objects exposing `set_diag`."""

    set_diag: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_38(typing.Protocol):
    """Structural protocol for objects exposing `set_memory_limit`."""

    set_memory_limit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_39(typing.Protocol):
    """Structural protocol for objects exposing `set_not_running`."""

    set_not_running: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_40(typing.Protocol):
    """Structural protocol for objects exposing `set_wired_limit`."""

    set_wired_limit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_41(typing.Protocol):
    """Structural protocol for objects exposing `setblocking`."""

    setblocking: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_42(typing.Protocol):
    """Structural protocol for objects exposing `setup_mock_src`."""

    setup_mock_src: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_43(typing.Protocol):
    """Structural protocol for objects exposing `shape`."""

    shape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember44_44(typing.Protocol):
    """Structural protocol for objects exposing `shape_inference`."""

    shape_inference: EagerTensorProtocol


SHARD_44_READERS: dict[str, MemberReader] = {
    "scan_op": lambda t: (True, t.scan_op) if isinstance(t, _HasMember44_00) else (False, None),
    "scan_tracing": lambda t: (True, t.scan_tracing) if isinstance(t, _HasMember44_01) else (False, None),
    "scanop": lambda t: (True, t.scanop) if isinstance(t, _HasMember44_02) else (False, None),
    "scatter": lambda t: (True, t.scatter) if isinstance(t, _HasMember44_03) else (False, None),
    "scatter_nd": lambda t: (True, t.scatter_nd) if isinstance(t, _HasMember44_04) else (False, None),
    "scatterapply": lambda t: (True, t.scatterapply) if isinstance(t, _HasMember44_05) else (False, None),
    "scattermax": lambda t: (True, t.scattermax) if isinstance(t, _HasMember44_06) else (False, None),
    "scattermin": lambda t: (True, t.scattermin) if isinstance(t, _HasMember44_07) else (False, None),
    "scattermul": lambda t: (True, t.scattermul) if isinstance(t, _HasMember44_08) else (False, None),
    "scatternd": lambda t: (True, t.scatternd) if isinstance(t, _HasMember44_09) else (False, None),
    "schema": lambda t: (True, t.schema) if isinstance(t, _HasMember44_10) else (False, None),
    "schur": lambda t: (True, t.schur) if isinstance(t, _HasMember44_11) else (False, None),
    "scipy": lambda t: (True, t.scipy) if isinstance(t, _HasMember44_12) else (False, None),
    "searchsorted": lambda t: (True, t.searchsorted) if isinstance(t, _HasMember44_13) else (False, None),
    "seed": lambda t: (True, t.seed) if isinstance(t, _HasMember44_14) else (False, None),
    "segment_max": lambda t: (True, t.segment_max) if isinstance(t, _HasMember44_15) else (False, None),
    "segment_mean": lambda t: (True, t.segment_mean) if isinstance(t, _HasMember44_16) else (False, None),
    "segment_min": lambda t: (True, t.segment_min) if isinstance(t, _HasMember44_17) else (False, None),
    "segment_prod": lambda t: (True, t.segment_prod) if isinstance(t, _HasMember44_18) else (False, None),
    "segment_sum": lambda t: (True, t.segment_sum) if isinstance(t, _HasMember44_19) else (False, None),
    "segmented_mm": lambda t: (True, t.segmented_mm) if isinstance(t, _HasMember44_20) else (False, None),
    "select": lambda t: (True, t.select) if isinstance(t, _HasMember44_21) else (False, None),
    "send": lambda t: (True, t.send) if isinstance(t, _HasMember44_22) else (False, None),
    "sendMessage": lambda t: (True, t.sendMessage) if isinstance(t, _HasMember44_23) else (False, None),
    "separable_conv": lambda t: (True, t.separable_conv) if isinstance(t, _HasMember44_24) else (False, None),
    "separable_conv1d": lambda t: (True, t.separable_conv1d) if isinstance(t, _HasMember44_25) else (False, None),
    "separable_conv2d": lambda t: (True, t.separable_conv2d) if isinstance(t, _HasMember44_26) else (False, None),
    "serialize": lambda t: (True, t.serialize) if isinstance(t, _HasMember44_27) else (False, None),
    "serialize_tensor": lambda t: (True, t.serialize_tensor) if isinstance(t, _HasMember44_28) else (False, None),
    "serializetensor": lambda t: (True, t.serializetensor) if isinstance(t, _HasMember44_29) else (False, None),
    "serializetensorcamel": lambda t: (
        (True, t.serializetensorcamel) if isinstance(t, _HasMember44_30) else (False, None)
    ),
    "setBuffer_offset_atIndex_": lambda t: (
        (True, t.setBuffer_offset_atIndex_) if isinstance(t, _HasMember44_31) else (False, None)
    ),
    "setComputePipelineState_": lambda t: (
        (True, t.setComputePipelineState_) if isinstance(t, _HasMember44_32) else (False, None)
    ),
    "setUp": lambda t: (True, t.setUp) if isinstance(t, _HasMember44_33) else (False, None),
    "set_count_positive": lambda t: (True, t.set_count_positive) if isinstance(t, _HasMember44_34) else (False, None),
    "set_count_zero": lambda t: (True, t.set_count_zero) if isinstance(t, _HasMember44_35) else (False, None),
    "set_default_stream": lambda t: (True, t.set_default_stream) if isinstance(t, _HasMember44_36) else (False, None),
    "set_diag": lambda t: (True, t.set_diag) if isinstance(t, _HasMember44_37) else (False, None),
    "set_memory_limit": lambda t: (True, t.set_memory_limit) if isinstance(t, _HasMember44_38) else (False, None),
    "set_not_running": lambda t: (True, t.set_not_running) if isinstance(t, _HasMember44_39) else (False, None),
    "set_wired_limit": lambda t: (True, t.set_wired_limit) if isinstance(t, _HasMember44_40) else (False, None),
    "setblocking": lambda t: (True, t.setblocking) if isinstance(t, _HasMember44_41) else (False, None),
    "setup_mock_src": lambda t: (True, t.setup_mock_src) if isinstance(t, _HasMember44_42) else (False, None),
    "shape": lambda t: (True, t.shape) if isinstance(t, _HasMember44_43) else (False, None),
    "shape_inference": lambda t: (True, t.shape_inference) if isinstance(t, _HasMember44_44) else (False, None),
}
