"""Static structural protocol bindings shard 26 (hipDeviceSynchronize..infer_shape)."""

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
class _HasMember26_00(typing.Protocol):
    """Structural protocol for objects exposing `hipDeviceSynchronize`."""

    hipDeviceSynchronize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_01(typing.Protocol):
    """Structural protocol for objects exposing `hipFree`."""

    hipFree: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_02(typing.Protocol):
    """Structural protocol for objects exposing `hipGetDeviceCount`."""

    hipGetDeviceCount: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_03(typing.Protocol):
    """Structural protocol for objects exposing `hipInit`."""

    hipInit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_04(typing.Protocol):
    """Structural protocol for objects exposing `hipMalloc`."""

    hipMalloc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_05(typing.Protocol):
    """Structural protocol for objects exposing `hipMemcpy`."""

    hipMemcpy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_06(typing.Protocol):
    """Structural protocol for objects exposing `hipStreamSynchronize`."""

    hipStreamSynchronize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_07(typing.Protocol):
    """Structural protocol for objects exposing `hiprtcDestroyProgram`."""

    hiprtcDestroyProgram: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_08(typing.Protocol):
    """Structural protocol for objects exposing `histogram`."""

    histogram: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_09(typing.Protocol):
    """Structural protocol for objects exposing `hook`."""

    hook: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_10(typing.Protocol):
    """Structural protocol for objects exposing `householder_product`."""

    householder_product: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_11(typing.Protocol):
    """Structural protocol for objects exposing `hsplit`."""

    hsplit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_12(typing.Protocol):
    """Structural protocol for objects exposing `hstack`."""

    hstack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_13(typing.Protocol):
    """Structural protocol for objects exposing `hsv_to_rgb`."""

    hsv_to_rgb: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_14(typing.Protocol):
    """Structural protocol for objects exposing `huber_loss`."""

    huber_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_15(typing.Protocol):
    """Structural protocol for objects exposing `hvp_test_func`."""

    hvp_test_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_16(typing.Protocol):
    """Structural protocol for objects exposing `hypot`."""

    hypot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_17(typing.Protocol):
    """Structural protocol for objects exposing `i0`."""

    i0: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_18(typing.Protocol):
    """Structural protocol for objects exposing `i0e`."""

    i0e: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_19(typing.Protocol):
    """Structural protocol for objects exposing `i1`."""

    i1: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_20(typing.Protocol):
    """Structural protocol for objects exposing `i1e`."""

    i1e: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_21(typing.Protocol):
    """Structural protocol for objects exposing `id`."""

    id: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_22(typing.Protocol):
    """Structural protocol for objects exposing `identity`."""

    identity: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_23(typing.Protocol):
    """Structural protocol for objects exposing `if_else`."""

    if_else: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_24(typing.Protocol):
    """Structural protocol for objects exposing `ifelse`."""

    ifelse: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_25(typing.Protocol):
    """Structural protocol for objects exposing `ifft`."""

    ifft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_26(typing.Protocol):
    """Structural protocol for objects exposing `ifft2`."""

    ifft2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_27(typing.Protocol):
    """Structural protocol for objects exposing `ifftn`."""

    ifftn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_28(typing.Protocol):
    """Structural protocol for objects exposing `ifftnd`."""

    ifftnd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_29(typing.Protocol):
    """Structural protocol for objects exposing `ifftshift`."""

    ifftshift: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_30(typing.Protocol):
    """Structural protocol for objects exposing `igamma`."""

    igamma: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_31(typing.Protocol):
    """Structural protocol for objects exposing `igammac`."""

    igammac: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_32(typing.Protocol):
    """Structural protocol for objects exposing `ihfft`."""

    ihfft: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_33(typing.Protocol):
    """Structural protocol for objects exposing `imag`."""

    imag: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_34(typing.Protocol):
    """Structural protocol for objects exposing `image`."""

    image: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_35(typing.Protocol):
    """Structural protocol for objects exposing `in_top_k`."""

    in_top_k: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_36(typing.Protocol):
    """Structural protocol for objects exposing `indent`."""

    indent: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_37(typing.Protocol):
    """Structural protocol for objects exposing `index`."""

    index: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_38(typing.Protocol):
    """Structural protocol for objects exposing `indexing`."""

    indexing: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_39(typing.Protocol):
    """Structural protocol for objects exposing `indexing_advanced`."""

    indexing_advanced: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_40(typing.Protocol):
    """Structural protocol for objects exposing `indices`."""

    indices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_41(typing.Protocol):
    """Structural protocol for objects exposing `indices_nonzero`."""

    indices_nonzero: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_42(typing.Protocol):
    """Structural protocol for objects exposing `inf`."""

    inf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_43(typing.Protocol):
    """Structural protocol for objects exposing `infeed`."""

    infeed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember26_44(typing.Protocol):
    """Structural protocol for objects exposing `infer_shape`."""

    infer_shape: EagerTensorProtocol


SHARD_26_READERS: dict[str, MemberReader] = {
    "hipDeviceSynchronize": lambda t: (
        (True, t.hipDeviceSynchronize) if isinstance(t, _HasMember26_00) else (False, None)
    ),
    "hipFree": lambda t: (True, t.hipFree) if isinstance(t, _HasMember26_01) else (False, None),
    "hipGetDeviceCount": lambda t: (True, t.hipGetDeviceCount) if isinstance(t, _HasMember26_02) else (False, None),
    "hipInit": lambda t: (True, t.hipInit) if isinstance(t, _HasMember26_03) else (False, None),
    "hipMalloc": lambda t: (True, t.hipMalloc) if isinstance(t, _HasMember26_04) else (False, None),
    "hipMemcpy": lambda t: (True, t.hipMemcpy) if isinstance(t, _HasMember26_05) else (False, None),
    "hipStreamSynchronize": lambda t: (
        (True, t.hipStreamSynchronize) if isinstance(t, _HasMember26_06) else (False, None)
    ),
    "hiprtcDestroyProgram": lambda t: (
        (True, t.hiprtcDestroyProgram) if isinstance(t, _HasMember26_07) else (False, None)
    ),
    "histogram": lambda t: (True, t.histogram) if isinstance(t, _HasMember26_08) else (False, None),
    "hook": lambda t: (True, t.hook) if isinstance(t, _HasMember26_09) else (False, None),
    "householder_product": lambda t: (True, t.householder_product) if isinstance(t, _HasMember26_10) else (False, None),
    "hsplit": lambda t: (True, t.hsplit) if isinstance(t, _HasMember26_11) else (False, None),
    "hstack": lambda t: (True, t.hstack) if isinstance(t, _HasMember26_12) else (False, None),
    "hsv_to_rgb": lambda t: (True, t.hsv_to_rgb) if isinstance(t, _HasMember26_13) else (False, None),
    "huber_loss": lambda t: (True, t.huber_loss) if isinstance(t, _HasMember26_14) else (False, None),
    "hvp_test_func": lambda t: (True, t.hvp_test_func) if isinstance(t, _HasMember26_15) else (False, None),
    "hypot": lambda t: (True, t.hypot) if isinstance(t, _HasMember26_16) else (False, None),
    "i0": lambda t: (True, t.i0) if isinstance(t, _HasMember26_17) else (False, None),
    "i0e": lambda t: (True, t.i0e) if isinstance(t, _HasMember26_18) else (False, None),
    "i1": lambda t: (True, t.i1) if isinstance(t, _HasMember26_19) else (False, None),
    "i1e": lambda t: (True, t.i1e) if isinstance(t, _HasMember26_20) else (False, None),
    "id": lambda t: (True, t.id) if isinstance(t, _HasMember26_21) else (False, None),
    "identity": lambda t: (True, t.identity) if isinstance(t, _HasMember26_22) else (False, None),
    "if_else": lambda t: (True, t.if_else) if isinstance(t, _HasMember26_23) else (False, None),
    "ifelse": lambda t: (True, t.ifelse) if isinstance(t, _HasMember26_24) else (False, None),
    "ifft": lambda t: (True, t.ifft) if isinstance(t, _HasMember26_25) else (False, None),
    "ifft2": lambda t: (True, t.ifft2) if isinstance(t, _HasMember26_26) else (False, None),
    "ifftn": lambda t: (True, t.ifftn) if isinstance(t, _HasMember26_27) else (False, None),
    "ifftnd": lambda t: (True, t.ifftnd) if isinstance(t, _HasMember26_28) else (False, None),
    "ifftshift": lambda t: (True, t.ifftshift) if isinstance(t, _HasMember26_29) else (False, None),
    "igamma": lambda t: (True, t.igamma) if isinstance(t, _HasMember26_30) else (False, None),
    "igammac": lambda t: (True, t.igammac) if isinstance(t, _HasMember26_31) else (False, None),
    "ihfft": lambda t: (True, t.ihfft) if isinstance(t, _HasMember26_32) else (False, None),
    "imag": lambda t: (True, t.imag) if isinstance(t, _HasMember26_33) else (False, None),
    "image": lambda t: (True, t.image) if isinstance(t, _HasMember26_34) else (False, None),
    "in_top_k": lambda t: (True, t.in_top_k) if isinstance(t, _HasMember26_35) else (False, None),
    "indent": lambda t: (True, t.indent) if isinstance(t, _HasMember26_36) else (False, None),
    "index": lambda t: (True, t.index) if isinstance(t, _HasMember26_37) else (False, None),
    "indexing": lambda t: (True, t.indexing) if isinstance(t, _HasMember26_38) else (False, None),
    "indexing_advanced": lambda t: (True, t.indexing_advanced) if isinstance(t, _HasMember26_39) else (False, None),
    "indices": lambda t: (True, t.indices) if isinstance(t, _HasMember26_40) else (False, None),
    "indices_nonzero": lambda t: (True, t.indices_nonzero) if isinstance(t, _HasMember26_41) else (False, None),
    "inf": lambda t: (True, t.inf) if isinstance(t, _HasMember26_42) else (False, None),
    "infeed": lambda t: (True, t.infeed) if isinstance(t, _HasMember26_43) else (False, None),
    "infer_shape": lambda t: (True, t.infer_shape) if isinstance(t, _HasMember26_44) else (False, None),
}
