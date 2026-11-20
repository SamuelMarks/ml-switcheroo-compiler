"""Static structural protocol bindings shard 19 (ensure_ndarray..extract_bounding_boxes)."""

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
class _HasMember19_00(typing.Protocol):
    """Structural protocol for objects exposing `ensure_ndarray`."""

    ensure_ndarray: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_07(typing.Protocol):
    """Structural protocol for objects exposing `entry`."""

    entry: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_08(typing.Protocol):
    """Structural protocol for objects exposing `eq`."""

    eq: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_09(typing.Protocol):
    """Structural protocol for objects exposing `equal`."""

    equal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_10(typing.Protocol):
    """Structural protocol for objects exposing `equalization`."""

    equalization: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_11(typing.Protocol):
    """Structural protocol for objects exposing `erf`."""

    erf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_12(typing.Protocol):
    """Structural protocol for objects exposing `erf_inv`."""

    erf_inv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_13(typing.Protocol):
    """Structural protocol for objects exposing `erfc`."""

    erfc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_14(typing.Protocol):
    """Structural protocol for objects exposing `erfcinv`."""

    erfcinv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_15(typing.Protocol):
    """Structural protocol for objects exposing `erfinv`."""

    erfinv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_16(typing.Protocol):
    """Structural protocol for objects exposing `err`."""

    err: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_17(typing.Protocol):
    """Structural protocol for objects exposing `error_model`."""

    error_model: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_18(typing.Protocol):
    """Structural protocol for objects exposing `errstate`."""

    errstate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_19(typing.Protocol):
    """Structural protocol for objects exposing `eval`."""

    eval: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_20(typing.Protocol):
    """Structural protocol for objects exposing `eval_sh_chebyt`."""

    eval_sh_chebyt: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_21(typing.Protocol):
    """Structural protocol for objects exposing `eval_sh_chebyu`."""

    eval_sh_chebyu: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_22(typing.Protocol):
    """Structural protocol for objects exposing `eval_shape`."""

    eval_shape: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_23(typing.Protocol):
    """Structural protocol for objects exposing `evaluate`."""

    evaluate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_24(typing.Protocol):
    """Structural protocol for objects exposing `evaluate_graph`."""

    evaluate_graph: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_25(typing.Protocol):
    """Structural protocol for objects exposing `event_later`."""

    event_later: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_26(typing.Protocol):
    """Structural protocol for objects exposing `execute`."""

    execute: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_27(typing.Protocol):
    """Structural protocol for objects exposing `execute_graph`."""

    execute_graph: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_28(typing.Protocol):
    """Structural protocol for objects exposing `execute_op`."""

    execute_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_29(typing.Protocol):
    """Structural protocol for objects exposing `exists`."""

    exists: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_30(typing.Protocol):
    """Structural protocol for objects exposing `exists_rule_file`."""

    exists_rule_file: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_31(typing.Protocol):
    """Structural protocol for objects exposing `exists_rules_op`."""

    exists_rules_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_32(typing.Protocol):
    """Structural protocol for objects exposing `exp`."""

    exp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_33(typing.Protocol):
    """Structural protocol for objects exposing `exp2`."""

    exp2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_34(typing.Protocol):
    """Structural protocol for objects exposing `expand`."""

    expand: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_35(typing.Protocol):
    """Structural protocol for objects exposing `expand_dims`."""

    expand_dims: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_36(typing.Protocol):
    """Structural protocol for objects exposing `expanddims`."""

    expanddims: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_37(typing.Protocol):
    """Structural protocol for objects exposing `exploding_fn`."""

    exploding_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_38(typing.Protocol):
    """Structural protocol for objects exposing `exploding_sparse_fn`."""

    exploding_sparse_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_39(typing.Protocol):
    """Structural protocol for objects exposing `expm`."""

    expm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_40(typing.Protocol):
    """Structural protocol for objects exposing `expm1`."""

    expm1: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_41(typing.Protocol):
    """Structural protocol for objects exposing `export_function`."""

    export_function: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_42(typing.Protocol):
    """Structural protocol for objects exposing `expr`."""

    expr: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_43(typing.Protocol):
    """Structural protocol for objects exposing `extract`."""

    extract: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember19_44(typing.Protocol):
    """Structural protocol for objects exposing `extract_bounding_boxes`."""

    extract_bounding_boxes: EagerTensorProtocol


SHARD_19_READERS: dict[str, MemberReader] = {
    "ensure_ndarray": lambda t: (True, t.ensure_ndarray) if isinstance(t, _HasMember19_00) else (False, None),
    "entry": lambda t: (True, t.entry) if isinstance(t, _HasMember19_07) else (False, None),
    "eq": lambda t: (True, t.eq) if isinstance(t, _HasMember19_08) else (False, None),
    "equal": lambda t: (True, t.equal) if isinstance(t, _HasMember19_09) else (False, None),
    "equalization": lambda t: (True, t.equalization) if isinstance(t, _HasMember19_10) else (False, None),
    "erf": lambda t: (True, t.erf) if isinstance(t, _HasMember19_11) else (False, None),
    "erf_inv": lambda t: (True, t.erf_inv) if isinstance(t, _HasMember19_12) else (False, None),
    "erfc": lambda t: (True, t.erfc) if isinstance(t, _HasMember19_13) else (False, None),
    "erfcinv": lambda t: (True, t.erfcinv) if isinstance(t, _HasMember19_14) else (False, None),
    "erfinv": lambda t: (True, t.erfinv) if isinstance(t, _HasMember19_15) else (False, None),
    "err": lambda t: (True, t.err) if isinstance(t, _HasMember19_16) else (False, None),
    "error_model": lambda t: (True, t.error_model) if isinstance(t, _HasMember19_17) else (False, None),
    "errstate": lambda t: (True, t.errstate) if isinstance(t, _HasMember19_18) else (False, None),
    "eval": lambda t: (True, t.eval) if isinstance(t, _HasMember19_19) else (False, None),
    "eval_sh_chebyt": lambda t: (True, t.eval_sh_chebyt) if isinstance(t, _HasMember19_20) else (False, None),
    "eval_sh_chebyu": lambda t: (True, t.eval_sh_chebyu) if isinstance(t, _HasMember19_21) else (False, None),
    "eval_shape": lambda t: (True, t.eval_shape) if isinstance(t, _HasMember19_22) else (False, None),
    "evaluate": lambda t: (True, t.evaluate) if isinstance(t, _HasMember19_23) else (False, None),
    "evaluate_graph": lambda t: (True, t.evaluate_graph) if isinstance(t, _HasMember19_24) else (False, None),
    "event_later": lambda t: (True, t.event_later) if isinstance(t, _HasMember19_25) else (False, None),
    "execute": lambda t: (True, t.execute) if isinstance(t, _HasMember19_26) else (False, None),
    "execute_graph": lambda t: (True, t.execute_graph) if isinstance(t, _HasMember19_27) else (False, None),
    "execute_op": lambda t: (True, t.execute_op) if isinstance(t, _HasMember19_28) else (False, None),
    "exists": lambda t: (True, t.exists) if isinstance(t, _HasMember19_29) else (False, None),
    "exists_rule_file": lambda t: (True, t.exists_rule_file) if isinstance(t, _HasMember19_30) else (False, None),
    "exists_rules_op": lambda t: (True, t.exists_rules_op) if isinstance(t, _HasMember19_31) else (False, None),
    "exp": lambda t: (True, t.exp) if isinstance(t, _HasMember19_32) else (False, None),
    "exp2": lambda t: (True, t.exp2) if isinstance(t, _HasMember19_33) else (False, None),
    "expand": lambda t: (True, t.expand) if isinstance(t, _HasMember19_34) else (False, None),
    "expand_dims": lambda t: (True, t.expand_dims) if isinstance(t, _HasMember19_35) else (False, None),
    "expanddims": lambda t: (True, t.expanddims) if isinstance(t, _HasMember19_36) else (False, None),
    "exploding_fn": lambda t: (True, t.exploding_fn) if isinstance(t, _HasMember19_37) else (False, None),
    "exploding_sparse_fn": lambda t: (True, t.exploding_sparse_fn) if isinstance(t, _HasMember19_38) else (False, None),
    "expm": lambda t: (True, t.expm) if isinstance(t, _HasMember19_39) else (False, None),
    "expm1": lambda t: (True, t.expm1) if isinstance(t, _HasMember19_40) else (False, None),
    "export_function": lambda t: (True, t.export_function) if isinstance(t, _HasMember19_41) else (False, None),
    "expr": lambda t: (True, t.expr) if isinstance(t, _HasMember19_42) else (False, None),
    "extract": lambda t: (True, t.extract) if isinstance(t, _HasMember19_43) else (False, None),
    "extract_bounding_boxes": lambda t: (
        (True, t.extract_bounding_boxes) if isinstance(t, _HasMember19_44) else (False, None)
    ),
}
