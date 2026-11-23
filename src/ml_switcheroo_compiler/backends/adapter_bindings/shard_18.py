"""Static structural protocol bindings shard 18 (dynamic_partition..ensure_module_type)."""

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
class _HasMember18_00(typing.Protocol):
    """Structural protocol for objects exposing `dynamic_partition`."""

    dynamic_partition: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_01(typing.Protocol):
    """Structural protocol for objects exposing `dynamic_slicing`."""

    dynamic_slicing: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_02(typing.Protocol):
    """Structural protocol for objects exposing `dynamic_stitch`."""

    dynamic_stitch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_03(typing.Protocol):
    """Structural protocol for objects exposing `eager_eval`."""

    eager_eval: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_04(typing.Protocol):
    """Structural protocol for objects exposing `eager_evaluator`."""

    eager_evaluator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_05(typing.Protocol):
    """Structural protocol for objects exposing `eager_mode`."""

    eager_mode: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_06(typing.Protocol):
    """Structural protocol for objects exposing `eager_registry`."""

    eager_registry: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_07(typing.Protocol):
    """Structural protocol for objects exposing `edges`."""

    edges: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_08(typing.Protocol):
    """Structural protocol for objects exposing `ediff1d`."""

    ediff1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_09(typing.Protocol):
    """Structural protocol for objects exposing `edit_distance`."""

    edit_distance: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_10(typing.Protocol):
    """Structural protocol for objects exposing `eig`."""

    eig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_11(typing.Protocol):
    """Structural protocol for objects exposing `eigh`."""

    eigh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_12(typing.Protocol):
    """Structural protocol for objects exposing `eigh_tridiagonal`."""

    eigh_tridiagonal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_13(typing.Protocol):
    """Structural protocol for objects exposing `eigvals`."""

    eigvals: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_14(typing.Protocol):
    """Structural protocol for objects exposing `eigvalsh`."""

    eigvalsh: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_15(typing.Protocol):
    """Structural protocol for objects exposing `einsum`."""

    einsum: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_16(typing.Protocol):
    """Structural protocol for objects exposing `einsum_path`."""

    einsum_path: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_17(typing.Protocol):
    """Structural protocol for objects exposing `einsumpath`."""

    einsumpath: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_18(typing.Protocol):
    """Structural protocol for objects exposing `elastic_transform`."""

    elastic_transform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_19(typing.Protocol):
    """Structural protocol for objects exposing `elastictransform`."""

    elastictransform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_20(typing.Protocol):
    """Structural protocol for objects exposing `elu`."""

    elu: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_21(typing.Protocol):
    """Structural protocol for objects exposing `emit`."""

    emit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_22(typing.Protocol):
    """Structural protocol for objects exposing `emit_constant`."""

    emit_constant: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_23(typing.Protocol):
    """Structural protocol for objects exposing `emit_ir_node`."""

    emit_ir_node: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_24(typing.Protocol):
    """Structural protocol for objects exposing `emit_jax`."""

    emit_jax: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_25(typing.Protocol):
    """Structural protocol for objects exposing `emit_keras`."""

    emit_keras: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_26(typing.Protocol):
    """Structural protocol for objects exposing `emit_mlx`."""

    emit_mlx: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_27(typing.Protocol):
    """Structural protocol for objects exposing `emit_pytorch`."""

    emit_pytorch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_28(typing.Protocol):
    """Structural protocol for objects exposing `emit_tensorflow`."""

    emit_tensorflow: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_29(typing.Protocol):
    """Structural protocol for objects exposing `empty`."""

    empty: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_30(typing.Protocol):
    """Structural protocol for objects exposing `empty_callable`."""

    empty_callable: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_31(typing.Protocol):
    """Structural protocol for objects exposing `empty_like`."""

    empty_like: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_32(typing.Protocol):
    """Structural protocol for objects exposing `enable_eager_mode`."""

    enable_eager_mode: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_33(typing.Protocol):
    """Structural protocol for objects exposing `enable_tracing_mode`."""

    enable_tracing_mode: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_34(typing.Protocol):
    """Structural protocol for objects exposing `encode_base64`."""

    encode_base64: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_35(typing.Protocol):
    """Structural protocol for objects exposing `encodebase64`."""

    encodebase64: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_36(typing.Protocol):
    """Structural protocol for objects exposing `end_lineno`."""

    end_lineno: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_37(typing.Protocol):
    """Structural protocol for objects exposing `endpoint`."""

    endpoint: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_38(typing.Protocol):
    """Structural protocol for objects exposing `ensure_attr_val`."""

    ensure_attr_val: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_39(typing.Protocol):
    """Structural protocol for objects exposing `ensure_attrs`."""

    ensure_attrs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember18_42(typing.Protocol):
    """Structural protocol for objects exposing `ensure_graph`."""

    ensure_graph: EagerTensorProtocol


SHARD_18_READERS: dict[str, MemberReader] = {
    "dynamic_partition": lambda t: (True, t.dynamic_partition) if isinstance(t, _HasMember18_00) else (False, None),
    "dynamic_slicing": lambda t: (True, t.dynamic_slicing) if isinstance(t, _HasMember18_01) else (False, None),
    "dynamic_stitch": lambda t: (True, t.dynamic_stitch) if isinstance(t, _HasMember18_02) else (False, None),
    "eager_eval": lambda t: (True, t.eager_eval) if isinstance(t, _HasMember18_03) else (False, None),
    "eager_evaluator": lambda t: (True, t.eager_evaluator) if isinstance(t, _HasMember18_04) else (False, None),
    "eager_mode": lambda t: (True, t.eager_mode) if isinstance(t, _HasMember18_05) else (False, None),
    "eager_registry": lambda t: (True, t.eager_registry) if isinstance(t, _HasMember18_06) else (False, None),
    "edges": lambda t: (True, t.edges) if isinstance(t, _HasMember18_07) else (False, None),
    "ediff1d": lambda t: (True, t.ediff1d) if isinstance(t, _HasMember18_08) else (False, None),
    "edit_distance": lambda t: (True, t.edit_distance) if isinstance(t, _HasMember18_09) else (False, None),
    "eig": lambda t: (True, t.eig) if isinstance(t, _HasMember18_10) else (False, None),
    "eigh": lambda t: (True, t.eigh) if isinstance(t, _HasMember18_11) else (False, None),
    "eigh_tridiagonal": lambda t: (True, t.eigh_tridiagonal) if isinstance(t, _HasMember18_12) else (False, None),
    "eigvals": lambda t: (True, t.eigvals) if isinstance(t, _HasMember18_13) else (False, None),
    "eigvalsh": lambda t: (True, t.eigvalsh) if isinstance(t, _HasMember18_14) else (False, None),
    "einsum": lambda t: (True, t.einsum) if isinstance(t, _HasMember18_15) else (False, None),
    "einsum_path": lambda t: (True, t.einsum_path) if isinstance(t, _HasMember18_16) else (False, None),
    "einsumpath": lambda t: (True, t.einsumpath) if isinstance(t, _HasMember18_17) else (False, None),
    "elastic_transform": lambda t: (True, t.elastic_transform) if isinstance(t, _HasMember18_18) else (False, None),
    "elastictransform": lambda t: (True, t.elastictransform) if isinstance(t, _HasMember18_19) else (False, None),
    "elu": lambda t: (True, t.elu) if isinstance(t, _HasMember18_20) else (False, None),
    "emit": lambda t: (True, t.emit) if isinstance(t, _HasMember18_21) else (False, None),
    "emit_constant": lambda t: (True, t.emit_constant) if isinstance(t, _HasMember18_22) else (False, None),
    "emit_ir_node": lambda t: (True, t.emit_ir_node) if isinstance(t, _HasMember18_23) else (False, None),
    "emit_jax": lambda t: (True, t.emit_jax) if isinstance(t, _HasMember18_24) else (False, None),
    "emit_keras": lambda t: (True, t.emit_keras) if isinstance(t, _HasMember18_25) else (False, None),
    "emit_mlx": lambda t: (True, t.emit_mlx) if isinstance(t, _HasMember18_26) else (False, None),
    "emit_pytorch": lambda t: (True, t.emit_pytorch) if isinstance(t, _HasMember18_27) else (False, None),
    "emit_tensorflow": lambda t: (True, t.emit_tensorflow) if isinstance(t, _HasMember18_28) else (False, None),
    "empty": lambda t: (True, t.empty) if isinstance(t, _HasMember18_29) else (False, None),
    "empty_callable": lambda t: (True, t.empty_callable) if isinstance(t, _HasMember18_30) else (False, None),
    "empty_like": lambda t: (True, t.empty_like) if isinstance(t, _HasMember18_31) else (False, None),
    "enable_eager_mode": lambda t: (True, t.enable_eager_mode) if isinstance(t, _HasMember18_32) else (False, None),
    "enable_tracing_mode": lambda t: (True, t.enable_tracing_mode) if isinstance(t, _HasMember18_33) else (False, None),
    "encode_base64": lambda t: (True, t.encode_base64) if isinstance(t, _HasMember18_34) else (False, None),
    "encodebase64": lambda t: (True, t.encodebase64) if isinstance(t, _HasMember18_35) else (False, None),
    "end_lineno": lambda t: (True, t.end_lineno) if isinstance(t, _HasMember18_36) else (False, None),
    "endpoint": lambda t: (True, t.endpoint) if isinstance(t, _HasMember18_37) else (False, None),
    "ensure_attr_val": lambda t: (True, t.ensure_attr_val) if isinstance(t, _HasMember18_38) else (False, None),
    "ensure_attrs": lambda t: (True, t.ensure_attrs) if isinstance(t, _HasMember18_39) else (False, None),
    "ensure_graph": lambda t: (True, t.ensure_graph) if isinstance(t, _HasMember18_42) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember18_00,
        _HasMember18_01,
        _HasMember18_02,
        _HasMember18_03,
        _HasMember18_04,
        _HasMember18_05,
        _HasMember18_06,
        _HasMember18_07,
        _HasMember18_08,
        _HasMember18_09,
        _HasMember18_10,
        _HasMember18_11,
        _HasMember18_12,
        _HasMember18_13,
        _HasMember18_14,
        _HasMember18_15,
        _HasMember18_16,
        _HasMember18_17,
        _HasMember18_18,
        _HasMember18_19,
        _HasMember18_20,
        _HasMember18_21,
        _HasMember18_22,
        _HasMember18_23,
        _HasMember18_24,
        _HasMember18_25,
        _HasMember18_26,
        _HasMember18_27,
        _HasMember18_28,
        _HasMember18_29,
        _HasMember18_30,
        _HasMember18_31,
        _HasMember18_32,
        _HasMember18_33,
        _HasMember18_34,
        _HasMember18_35,
        _HasMember18_36,
        _HasMember18_37,
        _HasMember18_38,
        _HasMember18_39,
        _HasMember18_42,
    )
)
