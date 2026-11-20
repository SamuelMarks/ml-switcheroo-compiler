"""Static structural protocol bindings shard 15 (cummax..decode_image_bytes)."""

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
class _HasMember15_00(typing.Protocol):
    """Structural protocol for objects exposing `cummax`."""

    cummax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_01(typing.Protocol):
    """Structural protocol for objects exposing `cummin`."""

    cummin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_02(typing.Protocol):
    """Structural protocol for objects exposing `cumprod`."""

    cumprod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_03(typing.Protocol):
    """Structural protocol for objects exposing `cumsum`."""

    cumsum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_04(typing.Protocol):
    """Structural protocol for objects exposing `cumulative_logsumexp`."""

    cumulative_logsumexp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_05(typing.Protocol):
    """Structural protocol for objects exposing `cumulative_max`."""

    cumulative_max: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_06(typing.Protocol):
    """Structural protocol for objects exposing `cumulative_min`."""

    cumulative_min: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_07(typing.Protocol):
    """Structural protocol for objects exposing `cumulative_prod`."""

    cumulative_prod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_08(typing.Protocol):
    """Structural protocol for objects exposing `cumulative_sum`."""

    cumulative_sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_09(typing.Protocol):
    """Structural protocol for objects exposing `cumulativemax`."""

    cumulativemax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_10(typing.Protocol):
    """Structural protocol for objects exposing `cumulativemin`."""

    cumulativemin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_11(typing.Protocol):
    """Structural protocol for objects exposing `cumulativeprod`."""

    cumulativeprod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_12(typing.Protocol):
    """Structural protocol for objects exposing `cumulativesum`."""

    cumulativesum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_13(typing.Protocol):
    """Structural protocol for objects exposing `cupy`."""

    cupy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_14(typing.Protocol):
    """Structural protocol for objects exposing `current_allocated_memory`."""

    current_allocated_memory: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_15(typing.Protocol):
    """Structural protocol for objects exposing `custom_disp`."""

    custom_disp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_16(typing.Protocol):
    """Structural protocol for objects exposing `custom_dispatcher`."""

    custom_dispatcher: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_17(typing.Protocol):
    """Structural protocol for objects exposing `custom_eager`."""

    custom_eager: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_18(typing.Protocol):
    """Structural protocol for objects exposing `custom_gradient`."""

    custom_gradient: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_19(typing.Protocol):
    """Structural protocol for objects exposing `custom_hasattr`."""

    custom_hasattr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_20(typing.Protocol):
    """Structural protocol for objects exposing `custom_linalg_op`."""

    custom_linalg_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_21(typing.Protocol):
    """Structural protocol for objects exposing `custom_linear_solve`."""

    custom_linear_solve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_22(typing.Protocol):
    """Structural protocol for objects exposing `custom_op`."""

    custom_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_23(typing.Protocol):
    """Structural protocol for objects exposing `custom_root`."""

    custom_root: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_24(typing.Protocol):
    """Structural protocol for objects exposing `custom_rules`."""

    custom_rules: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_25(typing.Protocol):
    """Structural protocol for objects exposing `customlinearsolve`."""

    customlinearsolve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_26(typing.Protocol):
    """Structural protocol for objects exposing `customroot`."""

    customroot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_27(typing.Protocol):
    """Structural protocol for objects exposing `cutmix`."""

    cutmix: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_28(typing.Protocol):
    """Structural protocol for objects exposing `d_type_holder_protocol`."""

    d_type_holder_protocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_29(typing.Protocol):
    """Structural protocol for objects exposing `da`."""

    da: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_30(typing.Protocol):
    """Structural protocol for objects exposing `dask`."""

    dask: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_31(typing.Protocol):
    """Structural protocol for objects exposing `data`."""

    data: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_32(typing.Protocol):
    """Structural protocol for objects exposing `data_format`."""

    data_format: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_33(typing.Protocol):
    """Structural protocol for objects exposing `data_holder_protocol`."""

    data_holder_protocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_34(typing.Protocol):
    """Structural protocol for objects exposing `dataholderprotocol`."""

    dataholderprotocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_35(typing.Protocol):
    """Structural protocol for objects exposing `debug_infs`."""

    debug_infs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_36(typing.Protocol):
    """Structural protocol for objects exposing `debug_nans`."""

    debug_nans: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_37(typing.Protocol):
    """Structural protocol for objects exposing `debuginfs`."""

    debuginfs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_38(typing.Protocol):
    """Structural protocol for objects exposing `debugnans`."""

    debugnans: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_39(typing.Protocol):
    """Structural protocol for objects exposing `decode_base64`."""

    decode_base64: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_40(typing.Protocol):
    """Structural protocol for objects exposing `decode_bmp`."""

    decode_bmp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_41(typing.Protocol):
    """Structural protocol for objects exposing `decode_csv`."""

    decode_csv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_42(typing.Protocol):
    """Structural protocol for objects exposing `decode_gif`."""

    decode_gif: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_43(typing.Protocol):
    """Structural protocol for objects exposing `decode_image`."""

    decode_image: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember15_44(typing.Protocol):
    """Structural protocol for objects exposing `decode_image_bytes`."""

    decode_image_bytes: EagerTensorProtocol


SHARD_15_READERS: dict[str, MemberReader] = {
    "cummax": lambda t: (True, t.cummax) if isinstance(t, _HasMember15_00) else (False, None),
    "cummin": lambda t: (True, t.cummin) if isinstance(t, _HasMember15_01) else (False, None),
    "cumprod": lambda t: (True, t.cumprod) if isinstance(t, _HasMember15_02) else (False, None),
    "cumsum": lambda t: (True, t.cumsum) if isinstance(t, _HasMember15_03) else (False, None),
    "cumulative_logsumexp": lambda t: (
        (True, t.cumulative_logsumexp) if isinstance(t, _HasMember15_04) else (False, None)
    ),
    "cumulative_max": lambda t: (True, t.cumulative_max) if isinstance(t, _HasMember15_05) else (False, None),
    "cumulative_min": lambda t: (True, t.cumulative_min) if isinstance(t, _HasMember15_06) else (False, None),
    "cumulative_prod": lambda t: (True, t.cumulative_prod) if isinstance(t, _HasMember15_07) else (False, None),
    "cumulative_sum": lambda t: (True, t.cumulative_sum) if isinstance(t, _HasMember15_08) else (False, None),
    "cumulativemax": lambda t: (True, t.cumulativemax) if isinstance(t, _HasMember15_09) else (False, None),
    "cumulativemin": lambda t: (True, t.cumulativemin) if isinstance(t, _HasMember15_10) else (False, None),
    "cumulativeprod": lambda t: (True, t.cumulativeprod) if isinstance(t, _HasMember15_11) else (False, None),
    "cumulativesum": lambda t: (True, t.cumulativesum) if isinstance(t, _HasMember15_12) else (False, None),
    "cupy": lambda t: (True, t.cupy) if isinstance(t, _HasMember15_13) else (False, None),
    "current_allocated_memory": lambda t: (
        (True, t.current_allocated_memory) if isinstance(t, _HasMember15_14) else (False, None)
    ),
    "custom_disp": lambda t: (True, t.custom_disp) if isinstance(t, _HasMember15_15) else (False, None),
    "custom_dispatcher": lambda t: (True, t.custom_dispatcher) if isinstance(t, _HasMember15_16) else (False, None),
    "custom_eager": lambda t: (True, t.custom_eager) if isinstance(t, _HasMember15_17) else (False, None),
    "custom_gradient": lambda t: (True, t.custom_gradient) if isinstance(t, _HasMember15_18) else (False, None),
    "custom_hasattr": lambda t: (True, t.custom_hasattr) if isinstance(t, _HasMember15_19) else (False, None),
    "custom_linalg_op": lambda t: (True, t.custom_linalg_op) if isinstance(t, _HasMember15_20) else (False, None),
    "custom_linear_solve": lambda t: (True, t.custom_linear_solve) if isinstance(t, _HasMember15_21) else (False, None),
    "custom_op": lambda t: (True, t.custom_op) if isinstance(t, _HasMember15_22) else (False, None),
    "custom_root": lambda t: (True, t.custom_root) if isinstance(t, _HasMember15_23) else (False, None),
    "custom_rules": lambda t: (True, t.custom_rules) if isinstance(t, _HasMember15_24) else (False, None),
    "customlinearsolve": lambda t: (True, t.customlinearsolve) if isinstance(t, _HasMember15_25) else (False, None),
    "customroot": lambda t: (True, t.customroot) if isinstance(t, _HasMember15_26) else (False, None),
    "cutmix": lambda t: (True, t.cutmix) if isinstance(t, _HasMember15_27) else (False, None),
    "d_type_holder_protocol": lambda t: (
        (True, t.d_type_holder_protocol) if isinstance(t, _HasMember15_28) else (False, None)
    ),
    "da": lambda t: (True, t.da) if isinstance(t, _HasMember15_29) else (False, None),
    "dask": lambda t: (True, t.dask) if isinstance(t, _HasMember15_30) else (False, None),
    "data": lambda t: (True, t.data) if isinstance(t, _HasMember15_31) else (False, None),
    "data_format": lambda t: (True, t.data_format) if isinstance(t, _HasMember15_32) else (False, None),
    "data_holder_protocol": lambda t: (
        (True, t.data_holder_protocol) if isinstance(t, _HasMember15_33) else (False, None)
    ),
    "dataholderprotocol": lambda t: (True, t.dataholderprotocol) if isinstance(t, _HasMember15_34) else (False, None),
    "debug_infs": lambda t: (True, t.debug_infs) if isinstance(t, _HasMember15_35) else (False, None),
    "debug_nans": lambda t: (True, t.debug_nans) if isinstance(t, _HasMember15_36) else (False, None),
    "debuginfs": lambda t: (True, t.debuginfs) if isinstance(t, _HasMember15_37) else (False, None),
    "debugnans": lambda t: (True, t.debugnans) if isinstance(t, _HasMember15_38) else (False, None),
    "decode_base64": lambda t: (True, t.decode_base64) if isinstance(t, _HasMember15_39) else (False, None),
    "decode_bmp": lambda t: (True, t.decode_bmp) if isinstance(t, _HasMember15_40) else (False, None),
    "decode_csv": lambda t: (True, t.decode_csv) if isinstance(t, _HasMember15_41) else (False, None),
    "decode_gif": lambda t: (True, t.decode_gif) if isinstance(t, _HasMember15_42) else (False, None),
    "decode_image": lambda t: (True, t.decode_image) if isinstance(t, _HasMember15_43) else (False, None),
    "decode_image_bytes": lambda t: (True, t.decode_image_bytes) if isinstance(t, _HasMember15_44) else (False, None),
}
