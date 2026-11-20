"""Static structural protocol bindings shard 43 (run_bad_op..scan_fn)."""

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
class _HasMember43_00(typing.Protocol):
    """Structural protocol for objects exposing `run_bad_op`."""

    run_bad_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_01(typing.Protocol):
    """Structural protocol for objects exposing `run_barrier`."""

    run_barrier: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_02(typing.Protocol):
    """Structural protocol for objects exposing `run_dir_rs`."""

    run_dir_rs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_03(typing.Protocol):
    """Structural protocol for objects exposing `run_direct`."""

    run_direct: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_04(typing.Protocol):
    """Structural protocol for objects exposing `run_ftrl`."""

    run_ftrl: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_05(typing.Protocol):
    """Structural protocol for objects exposing `run_kernel`."""

    run_kernel: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_06(typing.Protocol):
    """Structural protocol for objects exposing `run_method`."""

    run_method: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_07(typing.Protocol):
    """Structural protocol for objects exposing `run_rank`."""

    run_rank: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_08(typing.Protocol):
    """Structural protocol for objects exposing `run_rank2`."""

    run_rank2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_09(typing.Protocol):
    """Structural protocol for objects exposing `run_rank_3`."""

    run_rank_3: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_10(typing.Protocol):
    """Structural protocol for objects exposing `run_rank_max`."""

    run_rank_max: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_11(typing.Protocol):
    """Structural protocol for objects exposing `run_rank_min`."""

    run_rank_min: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_12(typing.Protocol):
    """Structural protocol for objects exposing `run_rank_prod`."""

    run_rank_prod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_13(typing.Protocol):
    """Structural protocol for objects exposing `run_rank_ring`."""

    run_rank_ring: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_14(typing.Protocol):
    """Structural protocol for objects exposing `run_rank_sum`."""

    run_rank_sum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_15(typing.Protocol):
    """Structural protocol for objects exposing `run_rd`."""

    run_rd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_16(typing.Protocol):
    """Structural protocol for objects exposing `run_ring`."""

    run_ring: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_17(typing.Protocol):
    """Structural protocol for objects exposing `run_ring_disp`."""

    run_ring_disp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_18(typing.Protocol):
    """Structural protocol for objects exposing `run_ring_op`."""

    run_ring_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_19(typing.Protocol):
    """Structural protocol for objects exposing `run_rmsprop`."""

    run_rmsprop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_20(typing.Protocol):
    """Structural protocol for objects exposing `runtime`."""

    runtime: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_21(typing.Protocol):
    """Structural protocol for objects exposing `safe_call`."""

    safe_call: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_22(typing.Protocol):
    """Structural protocol for objects exposing `sample_fn`."""

    sample_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_23(typing.Protocol):
    """Structural protocol for objects exposing `sample_op_func`."""

    sample_op_func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_24(typing.Protocol):
    """Structural protocol for objects exposing `save`."""

    save: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_25(typing.Protocol):
    """Structural protocol for objects exposing `save_gguf`."""

    save_gguf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_26(typing.Protocol):
    """Structural protocol for objects exposing `save_h5`."""

    save_h5: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_27(typing.Protocol):
    """Structural protocol for objects exposing `save_npz`."""

    save_npz: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_28(typing.Protocol):
    """Structural protocol for objects exposing `save_safetensors`."""

    save_safetensors: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_29(typing.Protocol):
    """Structural protocol for objects exposing `savegguf`."""

    savegguf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_30(typing.Protocol):
    """Structural protocol for objects exposing `savez`."""

    savez: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_31(typing.Protocol):
    """Structural protocol for objects exposing `savez_compressed`."""

    savez_compressed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_32(typing.Protocol):
    """Structural protocol for objects exposing `savezcompressed`."""

    savezcompressed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_33(typing.Protocol):
    """Structural protocol for objects exposing `scalar_mul`."""

    scalar_mul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_34(typing.Protocol):
    """Structural protocol for objects exposing `scalar_quad`."""

    scalar_quad: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_35(typing.Protocol):
    """Structural protocol for objects exposing `scale_fn`."""

    scale_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_36(typing.Protocol):
    """Structural protocol for objects exposing `scale_regularization_loss`."""

    scale_regularization_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_37(typing.Protocol):
    """Structural protocol for objects exposing `scaled_dot_product_attention`."""

    scaled_dot_product_attention: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_38(typing.Protocol):
    """Structural protocol for objects exposing `scaled_modified_bessel_k0`."""

    scaled_modified_bessel_k0: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_39(typing.Protocol):
    """Structural protocol for objects exposing `scaled_modified_bessel_k1`."""

    scaled_modified_bessel_k1: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_40(typing.Protocol):
    """Structural protocol for objects exposing `scan`."""

    scan: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_41(typing.Protocol):
    """Structural protocol for objects exposing `scan_bind`."""

    scan_bind: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_42(typing.Protocol):
    """Structural protocol for objects exposing `scan_dummy`."""

    scan_dummy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_43(typing.Protocol):
    """Structural protocol for objects exposing `scan_eager`."""

    scan_eager: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember43_44(typing.Protocol):
    """Structural protocol for objects exposing `scan_fn`."""

    scan_fn: EagerTensorProtocol


SHARD_43_READERS: dict[str, MemberReader] = {
    "run_bad_op": lambda t: (True, t.run_bad_op) if isinstance(t, _HasMember43_00) else (False, None),
    "run_barrier": lambda t: (True, t.run_barrier) if isinstance(t, _HasMember43_01) else (False, None),
    "run_dir_rs": lambda t: (True, t.run_dir_rs) if isinstance(t, _HasMember43_02) else (False, None),
    "run_direct": lambda t: (True, t.run_direct) if isinstance(t, _HasMember43_03) else (False, None),
    "run_ftrl": lambda t: (True, t.run_ftrl) if isinstance(t, _HasMember43_04) else (False, None),
    "run_kernel": lambda t: (True, t.run_kernel) if isinstance(t, _HasMember43_05) else (False, None),
    "run_method": lambda t: (True, t.run_method) if isinstance(t, _HasMember43_06) else (False, None),
    "run_rank": lambda t: (True, t.run_rank) if isinstance(t, _HasMember43_07) else (False, None),
    "run_rank2": lambda t: (True, t.run_rank2) if isinstance(t, _HasMember43_08) else (False, None),
    "run_rank_3": lambda t: (True, t.run_rank_3) if isinstance(t, _HasMember43_09) else (False, None),
    "run_rank_max": lambda t: (True, t.run_rank_max) if isinstance(t, _HasMember43_10) else (False, None),
    "run_rank_min": lambda t: (True, t.run_rank_min) if isinstance(t, _HasMember43_11) else (False, None),
    "run_rank_prod": lambda t: (True, t.run_rank_prod) if isinstance(t, _HasMember43_12) else (False, None),
    "run_rank_ring": lambda t: (True, t.run_rank_ring) if isinstance(t, _HasMember43_13) else (False, None),
    "run_rank_sum": lambda t: (True, t.run_rank_sum) if isinstance(t, _HasMember43_14) else (False, None),
    "run_rd": lambda t: (True, t.run_rd) if isinstance(t, _HasMember43_15) else (False, None),
    "run_ring": lambda t: (True, t.run_ring) if isinstance(t, _HasMember43_16) else (False, None),
    "run_ring_disp": lambda t: (True, t.run_ring_disp) if isinstance(t, _HasMember43_17) else (False, None),
    "run_ring_op": lambda t: (True, t.run_ring_op) if isinstance(t, _HasMember43_18) else (False, None),
    "run_rmsprop": lambda t: (True, t.run_rmsprop) if isinstance(t, _HasMember43_19) else (False, None),
    "runtime": lambda t: (True, t.runtime) if isinstance(t, _HasMember43_20) else (False, None),
    "safe_call": lambda t: (True, t.safe_call) if isinstance(t, _HasMember43_21) else (False, None),
    "sample_fn": lambda t: (True, t.sample_fn) if isinstance(t, _HasMember43_22) else (False, None),
    "sample_op_func": lambda t: (True, t.sample_op_func) if isinstance(t, _HasMember43_23) else (False, None),
    "save": lambda t: (True, t.save) if isinstance(t, _HasMember43_24) else (False, None),
    "save_gguf": lambda t: (True, t.save_gguf) if isinstance(t, _HasMember43_25) else (False, None),
    "save_h5": lambda t: (True, t.save_h5) if isinstance(t, _HasMember43_26) else (False, None),
    "save_npz": lambda t: (True, t.save_npz) if isinstance(t, _HasMember43_27) else (False, None),
    "save_safetensors": lambda t: (True, t.save_safetensors) if isinstance(t, _HasMember43_28) else (False, None),
    "savegguf": lambda t: (True, t.savegguf) if isinstance(t, _HasMember43_29) else (False, None),
    "savez": lambda t: (True, t.savez) if isinstance(t, _HasMember43_30) else (False, None),
    "savez_compressed": lambda t: (True, t.savez_compressed) if isinstance(t, _HasMember43_31) else (False, None),
    "savezcompressed": lambda t: (True, t.savezcompressed) if isinstance(t, _HasMember43_32) else (False, None),
    "scalar_mul": lambda t: (True, t.scalar_mul) if isinstance(t, _HasMember43_33) else (False, None),
    "scalar_quad": lambda t: (True, t.scalar_quad) if isinstance(t, _HasMember43_34) else (False, None),
    "scale_fn": lambda t: (True, t.scale_fn) if isinstance(t, _HasMember43_35) else (False, None),
    "scale_regularization_loss": lambda t: (
        (True, t.scale_regularization_loss) if isinstance(t, _HasMember43_36) else (False, None)
    ),
    "scaled_dot_product_attention": lambda t: (
        (True, t.scaled_dot_product_attention) if isinstance(t, _HasMember43_37) else (False, None)
    ),
    "scaled_modified_bessel_k0": lambda t: (
        (True, t.scaled_modified_bessel_k0) if isinstance(t, _HasMember43_38) else (False, None)
    ),
    "scaled_modified_bessel_k1": lambda t: (
        (True, t.scaled_modified_bessel_k1) if isinstance(t, _HasMember43_39) else (False, None)
    ),
    "scan": lambda t: (True, t.scan) if isinstance(t, _HasMember43_40) else (False, None),
    "scan_bind": lambda t: (True, t.scan_bind) if isinstance(t, _HasMember43_41) else (False, None),
    "scan_dummy": lambda t: (True, t.scan_dummy) if isinstance(t, _HasMember43_42) else (False, None),
    "scan_eager": lambda t: (True, t.scan_eager) if isinstance(t, _HasMember43_43) else (False, None),
    "scan_fn": lambda t: (True, t.scan_fn) if isinstance(t, _HasMember43_44) else (False, None),
}
