"""Static structural protocol bindings shard 14 (coo_sum..cumlogsumexp)."""

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
class _HasMember14_00(typing.Protocol):
    """Structural protocol for objects exposing `coo_sum`."""

    coo_sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_01(typing.Protocol):
    """Structural protocol for objects exposing `coo_todense`."""

    coo_todense: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_02(typing.Protocol):
    """Structural protocol for objects exposing `coo_transpose`."""

    coo_transpose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_03(typing.Protocol):
    """Structural protocol for objects exposing `cookiejar`."""

    cookiejar: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_04(typing.Protocol):
    """Structural protocol for objects exposing `coords`."""

    coords: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_05(typing.Protocol):
    """Structural protocol for objects exposing `copy`."""

    copy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_06(typing.Protocol):
    """Structural protocol for objects exposing `copysign`."""

    copysign: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_07(typing.Protocol):
    """Structural protocol for objects exposing `core`."""

    core: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_08(typing.Protocol):
    """Structural protocol for objects exposing `corrcoef`."""

    corrcoef: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_09(typing.Protocol):
    """Structural protocol for objects exposing `correlate`."""

    correlate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_10(typing.Protocol):
    """Structural protocol for objects exposing `cos`."""

    cos: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_11(typing.Protocol):
    """Structural protocol for objects exposing `cosh`."""

    cosh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_12(typing.Protocol):
    """Structural protocol for objects exposing `cosine_similarity_loss`."""

    cosine_similarity_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_13(typing.Protocol):
    """Structural protocol for objects exposing `count`."""

    count: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_14(typing.Protocol):
    """Structural protocol for objects exposing `count_nonzero`."""

    count_nonzero: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_15(typing.Protocol):
    """Structural protocol for objects exposing `cov`."""

    cov: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_16(typing.Protocol):
    """Structural protocol for objects exposing `cp`."""

    cp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_17(typing.Protocol):
    """Structural protocol for objects exposing `cpu`."""

    cpu: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_18(typing.Protocol):
    """Structural protocol for objects exposing `create_eager_tensor`."""

    create_eager_tensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_19(typing.Protocol):
    """Structural protocol for objects exposing `create_interconnected_mesh`."""

    create_interconnected_mesh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_20(typing.Protocol):
    """Structural protocol for objects exposing `create_program`."""

    create_program: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_21(typing.Protocol):
    """Structural protocol for objects exposing `create_token`."""

    create_token: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_22(typing.Protocol):
    """Structural protocol for objects exposing `createtoken`."""

    createtoken: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_23(typing.Protocol):
    """Structural protocol for objects exposing `creation`."""

    creation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_24(typing.Protocol):
    """Structural protocol for objects exposing `crop`."""

    crop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_25(typing.Protocol):
    """Structural protocol for objects exposing `crop_and_resize`."""

    crop_and_resize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_26(typing.Protocol):
    """Structural protocol for objects exposing `crop_images`."""

    crop_images: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_27(typing.Protocol):
    """Structural protocol for objects exposing `cross`."""

    cross: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_28(typing.Protocol):
    """Structural protocol for objects exposing `ctc_decode`."""

    ctc_decode: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_29(typing.Protocol):
    """Structural protocol for objects exposing `ctc_loss`."""

    ctc_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_30(typing.Protocol):
    """Structural protocol for objects exposing `ctcloss`."""

    ctcloss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_31(typing.Protocol):
    """Structural protocol for objects exposing `ctypes`."""

    ctypes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_32(typing.Protocol):
    """Structural protocol for objects exposing `cuCtxSynchronize`."""

    cuCtxSynchronize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_33(typing.Protocol):
    """Structural protocol for objects exposing `cuDeviceGetCount`."""

    cuDeviceGetCount: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_34(typing.Protocol):
    """Structural protocol for objects exposing `cuInit`."""

    cuInit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_35(typing.Protocol):
    """Structural protocol for objects exposing `cube_custom`."""

    cube_custom: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_36(typing.Protocol):
    """Structural protocol for objects exposing `cube_custom_jvp`."""

    cube_custom_jvp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_37(typing.Protocol):
    """Structural protocol for objects exposing `cubic`."""

    cubic: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_38(typing.Protocol):
    """Structural protocol for objects exposing `cuda`."""

    cuda: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_39(typing.Protocol):
    """Structural protocol for objects exposing `cudaDeviceSynchronize`."""

    cudaDeviceSynchronize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_40(typing.Protocol):
    """Structural protocol for objects exposing `cudaFree`."""

    cudaFree: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_41(typing.Protocol):
    """Structural protocol for objects exposing `cudaMalloc`."""

    cudaMalloc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_42(typing.Protocol):
    """Structural protocol for objects exposing `cudaMemcpy`."""

    cudaMemcpy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_43(typing.Protocol):
    """Structural protocol for objects exposing `cudaStreamSynchronize`."""

    cudaStreamSynchronize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember14_44(typing.Protocol):
    """Structural protocol for objects exposing `cumlogsumexp`."""

    cumlogsumexp: EagerTensorProtocol


SHARD_14_READERS: dict[str, MemberReader] = {
    "coo_sum": lambda t: (True, t.coo_sum) if isinstance(t, _HasMember14_00) else (False, None),
    "coo_todense": lambda t: (True, t.coo_todense) if isinstance(t, _HasMember14_01) else (False, None),
    "coo_transpose": lambda t: (True, t.coo_transpose) if isinstance(t, _HasMember14_02) else (False, None),
    "cookiejar": lambda t: (True, t.cookiejar) if isinstance(t, _HasMember14_03) else (False, None),
    "coords": lambda t: (True, t.coords) if isinstance(t, _HasMember14_04) else (False, None),
    "copy": lambda t: (True, t.copy) if isinstance(t, _HasMember14_05) else (False, None),
    "copysign": lambda t: (True, t.copysign) if isinstance(t, _HasMember14_06) else (False, None),
    "core": lambda t: (True, t.core) if isinstance(t, _HasMember14_07) else (False, None),
    "corrcoef": lambda t: (True, t.corrcoef) if isinstance(t, _HasMember14_08) else (False, None),
    "correlate": lambda t: (True, t.correlate) if isinstance(t, _HasMember14_09) else (False, None),
    "cos": lambda t: (True, t.cos) if isinstance(t, _HasMember14_10) else (False, None),
    "cosh": lambda t: (True, t.cosh) if isinstance(t, _HasMember14_11) else (False, None),
    "cosine_similarity_loss": lambda t: (
        (True, t.cosine_similarity_loss) if isinstance(t, _HasMember14_12) else (False, None)
    ),
    "count": lambda t: (True, t.count) if isinstance(t, _HasMember14_13) else (False, None),
    "count_nonzero": lambda t: (True, t.count_nonzero) if isinstance(t, _HasMember14_14) else (False, None),
    "cov": lambda t: (True, t.cov) if isinstance(t, _HasMember14_15) else (False, None),
    "cp": lambda t: (True, t.cp) if isinstance(t, _HasMember14_16) else (False, None),
    "cpu": lambda t: (True, t.cpu) if isinstance(t, _HasMember14_17) else (False, None),
    "create_eager_tensor": lambda t: (True, t.create_eager_tensor) if isinstance(t, _HasMember14_18) else (False, None),
    "create_interconnected_mesh": lambda t: (
        (True, t.create_interconnected_mesh) if isinstance(t, _HasMember14_19) else (False, None)
    ),
    "create_program": lambda t: (True, t.create_program) if isinstance(t, _HasMember14_20) else (False, None),
    "create_token": lambda t: (True, t.create_token) if isinstance(t, _HasMember14_21) else (False, None),
    "createtoken": lambda t: (True, t.createtoken) if isinstance(t, _HasMember14_22) else (False, None),
    "creation": lambda t: (True, t.creation) if isinstance(t, _HasMember14_23) else (False, None),
    "crop": lambda t: (True, t.crop) if isinstance(t, _HasMember14_24) else (False, None),
    "crop_and_resize": lambda t: (True, t.crop_and_resize) if isinstance(t, _HasMember14_25) else (False, None),
    "crop_images": lambda t: (True, t.crop_images) if isinstance(t, _HasMember14_26) else (False, None),
    "cross": lambda t: (True, t.cross) if isinstance(t, _HasMember14_27) else (False, None),
    "ctc_decode": lambda t: (True, t.ctc_decode) if isinstance(t, _HasMember14_28) else (False, None),
    "ctc_loss": lambda t: (True, t.ctc_loss) if isinstance(t, _HasMember14_29) else (False, None),
    "ctcloss": lambda t: (True, t.ctcloss) if isinstance(t, _HasMember14_30) else (False, None),
    "ctypes": lambda t: (True, t.ctypes) if isinstance(t, _HasMember14_31) else (False, None),
    "cuCtxSynchronize": lambda t: (True, t.cuCtxSynchronize) if isinstance(t, _HasMember14_32) else (False, None),
    "cuDeviceGetCount": lambda t: (True, t.cuDeviceGetCount) if isinstance(t, _HasMember14_33) else (False, None),
    "cuInit": lambda t: (True, t.cuInit) if isinstance(t, _HasMember14_34) else (False, None),
    "cube_custom": lambda t: (True, t.cube_custom) if isinstance(t, _HasMember14_35) else (False, None),
    "cube_custom_jvp": lambda t: (True, t.cube_custom_jvp) if isinstance(t, _HasMember14_36) else (False, None),
    "cubic": lambda t: (True, t.cubic) if isinstance(t, _HasMember14_37) else (False, None),
    "cuda": lambda t: (True, t.cuda) if isinstance(t, _HasMember14_38) else (False, None),
    "cudaDeviceSynchronize": lambda t: (
        (True, t.cudaDeviceSynchronize) if isinstance(t, _HasMember14_39) else (False, None)
    ),
    "cudaFree": lambda t: (True, t.cudaFree) if isinstance(t, _HasMember14_40) else (False, None),
    "cudaMalloc": lambda t: (True, t.cudaMalloc) if isinstance(t, _HasMember14_41) else (False, None),
    "cudaMemcpy": lambda t: (True, t.cudaMemcpy) if isinstance(t, _HasMember14_42) else (False, None),
    "cudaStreamSynchronize": lambda t: (
        (True, t.cudaStreamSynchronize) if isinstance(t, _HasMember14_43) else (False, None)
    ),
    "cumlogsumexp": lambda t: (True, t.cumlogsumexp) if isinstance(t, _HasMember14_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember14_00,
        _HasMember14_01,
        _HasMember14_02,
        _HasMember14_03,
        _HasMember14_04,
        _HasMember14_05,
        _HasMember14_06,
        _HasMember14_07,
        _HasMember14_08,
        _HasMember14_09,
        _HasMember14_10,
        _HasMember14_11,
        _HasMember14_12,
        _HasMember14_13,
        _HasMember14_14,
        _HasMember14_15,
        _HasMember14_16,
        _HasMember14_17,
        _HasMember14_18,
        _HasMember14_19,
        _HasMember14_20,
        _HasMember14_21,
        _HasMember14_22,
        _HasMember14_23,
        _HasMember14_24,
        _HasMember14_25,
        _HasMember14_26,
        _HasMember14_27,
        _HasMember14_28,
        _HasMember14_29,
        _HasMember14_30,
        _HasMember14_31,
        _HasMember14_32,
        _HasMember14_33,
        _HasMember14_34,
        _HasMember14_35,
        _HasMember14_36,
        _HasMember14_37,
        _HasMember14_38,
        _HasMember14_39,
        _HasMember14_40,
        _HasMember14_41,
        _HasMember14_42,
        _HasMember14_43,
        _HasMember14_44,
    )
)
