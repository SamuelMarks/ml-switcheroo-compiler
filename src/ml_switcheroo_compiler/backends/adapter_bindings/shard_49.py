"""Static structural protocol bindings shard 49 (tensorsolve..tree_leaves)."""

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
class _HasMember49_00(typing.Protocol):
    """Structural protocol for objects exposing `tensorsolve`."""

    tensorsolve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_01(typing.Protocol):
    """Structural protocol for objects exposing `terminate`."""

    terminate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_02(typing.Protocol):
    """Structural protocol for objects exposing `testglobal_tracing_state_tape`."""

    testglobal_tracing_state_tape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_03(typing.Protocol):
    """Structural protocol for objects exposing `text`."""

    text: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_04(typing.Protocol):
    """Structural protocol for objects exposing `text_vectorization`."""

    text_vectorization: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_05(typing.Protocol):
    """Structural protocol for objects exposing `textvectorization`."""

    textvectorization: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_06(typing.Protocol):
    """Structural protocol for objects exposing `tf`."""

    tf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_07(typing.Protocol):
    """Structural protocol for objects exposing `tile`."""

    tile: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_08(typing.Protocol):
    """Structural protocol for objects exposing `title`."""

    title: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_09(typing.Protocol):
    """Structural protocol for objects exposing `to`."""

    to: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_10(typing.Protocol):
    """Structural protocol for objects exposing `to_batches`."""

    to_batches: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_11(typing.Protocol):
    """Structural protocol for objects exposing `to_cpp_dtype`."""

    to_cpp_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_12(typing.Protocol):
    """Structural protocol for objects exposing `to_cupy_dtype`."""

    to_cupy_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_13(typing.Protocol):
    """Structural protocol for objects exposing `to_dask_dtype`."""

    to_dask_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_14(typing.Protocol):
    """Structural protocol for objects exposing `to_dense`."""

    to_dense: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_15(typing.Protocol):
    """Structural protocol for objects exposing `to_jax_dtype`."""

    to_jax_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_16(typing.Protocol):
    """Structural protocol for objects exposing `to_json`."""

    to_json: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_17(typing.Protocol):
    """Structural protocol for objects exposing `to_keras_dtype`."""

    to_keras_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_18(typing.Protocol):
    """Structural protocol for objects exposing `to_list`."""

    to_list: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_19(typing.Protocol):
    """Structural protocol for objects exposing `to_mlx_dtype`."""

    to_mlx_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_20(typing.Protocol):
    """Structural protocol for objects exposing `to_numpy`."""

    to_numpy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_21(typing.Protocol):
    """Structural protocol for objects exposing `to_numpy_dtype`."""

    to_numpy_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_22(typing.Protocol):
    """Structural protocol for objects exposing `to_padded_tensor`."""

    to_padded_tensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_23(typing.Protocol):
    """Structural protocol for objects exposing `to_polynomial`."""

    to_polynomial: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_24(typing.Protocol):
    """Structural protocol for objects exposing `to_pylist`."""

    to_pylist: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_25(typing.Protocol):
    """Structural protocol for objects exposing `to_regular`."""

    to_regular: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_26(typing.Protocol):
    """Structural protocol for objects exposing `to_text`."""

    to_text: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_27(typing.Protocol):
    """Structural protocol for objects exposing `to_tf_dtype`."""

    to_tf_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_28(typing.Protocol):
    """Structural protocol for objects exposing `to_torch_dtype`."""

    to_torch_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_29(typing.Protocol):
    """Structural protocol for objects exposing `tobytes`."""

    tobytes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_30(typing.Protocol):
    """Structural protocol for objects exposing `todense`."""

    todense: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_31(typing.Protocol):
    """Structural protocol for objects exposing `toggle_not_ready`."""

    toggle_not_ready: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_32(typing.Protocol):
    """Structural protocol for objects exposing `tolist`."""

    tolist: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_33(typing.Protocol):
    """Structural protocol for objects exposing `top_k`."""

    top_k: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_34(typing.Protocol):
    """Structural protocol for objects exposing `topk`."""

    topk: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_35(typing.Protocol):
    """Structural protocol for objects exposing `torch`."""

    torch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_36(typing.Protocol):
    """Structural protocol for objects exposing `total_loss`."""

    total_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_37(typing.Protocol):
    """Structural protocol for objects exposing `trace`."""

    trace: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_38(typing.Protocol):
    """Structural protocol for objects exposing `traced_fn`."""

    traced_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_39(typing.Protocol):
    """Structural protocol for objects exposing `trainable`."""

    trainable: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_40(typing.Protocol):
    """Structural protocol for objects exposing `transpose`."""

    transpose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_41(typing.Protocol):
    """Structural protocol for objects exposing `trapezoid`."""

    trapezoid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_42(typing.Protocol):
    """Structural protocol for objects exposing `trapezoidalintegral`."""

    trapezoidalintegral: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_43(typing.Protocol):
    """Structural protocol for objects exposing `trapz`."""

    trapz: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember49_44(typing.Protocol):
    """Structural protocol for objects exposing `tree_leaves`."""

    tree_leaves: EagerTensorProtocol


SHARD_49_READERS: dict[str, MemberReader] = {
    "tensorsolve": lambda t: (True, t.tensorsolve) if isinstance(t, _HasMember49_00) else (False, None),
    "terminate": lambda t: (True, t.terminate) if isinstance(t, _HasMember49_01) else (False, None),
    "testglobal_tracing_state_tape": lambda t: (
        (True, t.testglobal_tracing_state_tape) if isinstance(t, _HasMember49_02) else (False, None)
    ),
    "text": lambda t: (True, t.text) if isinstance(t, _HasMember49_03) else (False, None),
    "text_vectorization": lambda t: (True, t.text_vectorization) if isinstance(t, _HasMember49_04) else (False, None),
    "textvectorization": lambda t: (True, t.textvectorization) if isinstance(t, _HasMember49_05) else (False, None),
    "tf": lambda t: (True, t.tf) if isinstance(t, _HasMember49_06) else (False, None),
    "tile": lambda t: (True, t.tile) if isinstance(t, _HasMember49_07) else (False, None),
    "title": lambda t: (True, t.title) if isinstance(t, _HasMember49_08) else (False, None),
    "to": lambda t: (True, t.to) if isinstance(t, _HasMember49_09) else (False, None),
    "to_batches": lambda t: (True, t.to_batches) if isinstance(t, _HasMember49_10) else (False, None),
    "to_cpp_dtype": lambda t: (True, t.to_cpp_dtype) if isinstance(t, _HasMember49_11) else (False, None),
    "to_cupy_dtype": lambda t: (True, t.to_cupy_dtype) if isinstance(t, _HasMember49_12) else (False, None),
    "to_dask_dtype": lambda t: (True, t.to_dask_dtype) if isinstance(t, _HasMember49_13) else (False, None),
    "to_dense": lambda t: (True, t.to_dense) if isinstance(t, _HasMember49_14) else (False, None),
    "to_jax_dtype": lambda t: (True, t.to_jax_dtype) if isinstance(t, _HasMember49_15) else (False, None),
    "to_json": lambda t: (True, t.to_json) if isinstance(t, _HasMember49_16) else (False, None),
    "to_keras_dtype": lambda t: (True, t.to_keras_dtype) if isinstance(t, _HasMember49_17) else (False, None),
    "to_list": lambda t: (True, t.to_list) if isinstance(t, _HasMember49_18) else (False, None),
    "to_mlx_dtype": lambda t: (True, t.to_mlx_dtype) if isinstance(t, _HasMember49_19) else (False, None),
    "to_numpy": lambda t: (True, t.to_numpy) if isinstance(t, _HasMember49_20) else (False, None),
    "to_numpy_dtype": lambda t: (True, t.to_numpy_dtype) if isinstance(t, _HasMember49_21) else (False, None),
    "to_padded_tensor": lambda t: (True, t.to_padded_tensor) if isinstance(t, _HasMember49_22) else (False, None),
    "to_polynomial": lambda t: (True, t.to_polynomial) if isinstance(t, _HasMember49_23) else (False, None),
    "to_pylist": lambda t: (True, t.to_pylist) if isinstance(t, _HasMember49_24) else (False, None),
    "to_regular": lambda t: (True, t.to_regular) if isinstance(t, _HasMember49_25) else (False, None),
    "to_text": lambda t: (True, t.to_text) if isinstance(t, _HasMember49_26) else (False, None),
    "to_tf_dtype": lambda t: (True, t.to_tf_dtype) if isinstance(t, _HasMember49_27) else (False, None),
    "to_torch_dtype": lambda t: (True, t.to_torch_dtype) if isinstance(t, _HasMember49_28) else (False, None),
    "tobytes": lambda t: (True, t.tobytes) if isinstance(t, _HasMember49_29) else (False, None),
    "todense": lambda t: (True, t.todense) if isinstance(t, _HasMember49_30) else (False, None),
    "toggle_not_ready": lambda t: (True, t.toggle_not_ready) if isinstance(t, _HasMember49_31) else (False, None),
    "tolist": lambda t: (True, t.tolist) if isinstance(t, _HasMember49_32) else (False, None),
    "top_k": lambda t: (True, t.top_k) if isinstance(t, _HasMember49_33) else (False, None),
    "topk": lambda t: (True, t.topk) if isinstance(t, _HasMember49_34) else (False, None),
    "torch": lambda t: (True, t.torch) if isinstance(t, _HasMember49_35) else (False, None),
    "total_loss": lambda t: (True, t.total_loss) if isinstance(t, _HasMember49_36) else (False, None),
    "trace": lambda t: (True, t.trace) if isinstance(t, _HasMember49_37) else (False, None),
    "traced_fn": lambda t: (True, t.traced_fn) if isinstance(t, _HasMember49_38) else (False, None),
    "trainable": lambda t: (True, t.trainable) if isinstance(t, _HasMember49_39) else (False, None),
    "transpose": lambda t: (True, t.transpose) if isinstance(t, _HasMember49_40) else (False, None),
    "trapezoid": lambda t: (True, t.trapezoid) if isinstance(t, _HasMember49_41) else (False, None),
    "trapezoidalintegral": lambda t: (True, t.trapezoidalintegral) if isinstance(t, _HasMember49_42) else (False, None),
    "trapz": lambda t: (True, t.trapz) if isinstance(t, _HasMember49_43) else (False, None),
    "tree_leaves": lambda t: (True, t.tree_leaves) if isinstance(t, _HasMember49_44) else (False, None),
}
