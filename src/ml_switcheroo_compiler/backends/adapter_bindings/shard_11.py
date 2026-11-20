"""Static structural protocol bindings shard 11 (bitwisenot..chebyshev_polynomial_t)."""

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
class _HasMember11_00(typing.Protocol):
    """Structural protocol for objects exposing `bitwisenot`."""

    bitwisenot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_01(typing.Protocol):
    """Structural protocol for objects exposing `bitwiseor`."""

    bitwiseor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_02(typing.Protocol):
    """Structural protocol for objects exposing `bitwisexor`."""

    bitwisexor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_03(typing.Protocol):
    """Structural protocol for objects exposing `block`."""

    block: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_04(typing.Protocol):
    """Structural protocol for objects exposing `block_masked_mm`."""

    block_masked_mm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_05(typing.Protocol):
    """Structural protocol for objects exposing `block_until_ready`."""

    block_until_ready: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_06(typing.Protocol):
    """Structural protocol for objects exposing `body`."""

    body: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_07(typing.Protocol):
    """Structural protocol for objects exposing `body_fn`."""

    body_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_08(typing.Protocol):
    """Structural protocol for objects exposing `body_fn_trace`."""

    body_fn_trace: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_09(typing.Protocol):
    """Structural protocol for objects exposing `body_fun`."""

    body_fun: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_10(typing.Protocol):
    """Structural protocol for objects exposing `bool_`."""

    bool_: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_11(typing.Protocol):
    """Structural protocol for objects exposing `boolean_mask`."""

    boolean_mask: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_12(typing.Protocol):
    """Structural protocol for objects exposing `booleanmask`."""

    booleanmask: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_13(typing.Protocol):
    """Structural protocol for objects exposing `broadcast_arrays`."""

    broadcast_arrays: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_14(typing.Protocol):
    """Structural protocol for objects exposing `broadcast_to`."""

    broadcast_to: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_15(typing.Protocol):
    """Structural protocol for objects exposing `broadcastto`."""

    broadcastto: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_16(typing.Protocol):
    """Structural protocol for objects exposing `buffer`."""

    buffer: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_17(typing.Protocol):
    """Structural protocol for objects exposing `bufferedAmount`."""

    bufferedAmount: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_18(typing.Protocol):
    """Structural protocol for objects exposing `buffered_amount`."""

    buffered_amount: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_19(typing.Protocol):
    """Structural protocol for objects exposing `buffers`."""

    buffers: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_20(typing.Protocol):
    """Structural protocol for objects exposing `buggy_desc`."""

    buggy_desc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_21(typing.Protocol):
    """Structural protocol for objects exposing `build_call`."""

    build_call: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_22(typing.Protocol):
    """Structural protocol for objects exposing `build_test_graph`."""

    build_test_graph: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_23(typing.Protocol):
    """Structural protocol for objects exposing `build_test_node`."""

    build_test_node: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_24(typing.Protocol):
    """Structural protocol for objects exposing `bwd`."""

    bwd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_25(typing.Protocol):
    """Structural protocol for objects exposing `bytecode`."""

    bytecode: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_26(typing.Protocol):
    """Structural protocol for objects exposing `byteorder`."""

    byteorder: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_27(typing.Protocol):
    """Structural protocol for objects exposing `c`."""

    c: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_28(typing.Protocol):
    """Structural protocol for objects exposing `callable`."""

    callable: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_29(typing.Protocol):
    """Structural protocol for objects exposing `canonical`."""

    canonical: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_30(typing.Protocol):
    """Structural protocol for objects exposing `capitalize`."""

    capitalize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_31(typing.Protocol):
    """Structural protocol for objects exposing `cartesian_prod`."""

    cartesian_prod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_32(typing.Protocol):
    """Structural protocol for objects exposing `case`."""

    case: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_33(typing.Protocol):
    """Structural protocol for objects exposing `cast`."""

    cast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_34(typing.Protocol):
    """Structural protocol for objects exposing `cat`."""

    cat: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_35(typing.Protocol):
    """Structural protocol for objects exposing `categorical_crossentropy`."""

    categorical_crossentropy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_36(typing.Protocol):
    """Structural protocol for objects exposing `categorical_generalized_cross_entropy`."""

    categorical_generalized_cross_entropy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_37(typing.Protocol):
    """Structural protocol for objects exposing `categoricalgeneralizedcrossentropy`."""

    categoricalgeneralizedcrossentropy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_38(typing.Protocol):
    """Structural protocol for objects exposing `cbrt`."""

    cbrt: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_39(typing.Protocol):
    """Structural protocol for objects exposing `cdf`."""

    cdf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_40(typing.Protocol):
    """Structural protocol for objects exposing `ceil`."""

    ceil: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_41(typing.Protocol):
    """Structural protocol for objects exposing `ceiling`."""

    ceiling: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_42(typing.Protocol):
    """Structural protocol for objects exposing `cg`."""

    cg: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_43(typing.Protocol):
    """Structural protocol for objects exposing `cg2`."""

    cg2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember11_44(typing.Protocol):
    """Structural protocol for objects exposing `chebyshev_polynomial_t`."""

    chebyshev_polynomial_t: EagerTensorProtocol


SHARD_11_READERS: dict[str, MemberReader] = {
    "bitwisenot": lambda t: (True, t.bitwisenot) if isinstance(t, _HasMember11_00) else (False, None),
    "bitwiseor": lambda t: (True, t.bitwiseor) if isinstance(t, _HasMember11_01) else (False, None),
    "bitwisexor": lambda t: (True, t.bitwisexor) if isinstance(t, _HasMember11_02) else (False, None),
    "block": lambda t: (True, t.block) if isinstance(t, _HasMember11_03) else (False, None),
    "block_masked_mm": lambda t: (True, t.block_masked_mm) if isinstance(t, _HasMember11_04) else (False, None),
    "block_until_ready": lambda t: (True, t.block_until_ready) if isinstance(t, _HasMember11_05) else (False, None),
    "body": lambda t: (True, t.body) if isinstance(t, _HasMember11_06) else (False, None),
    "body_fn": lambda t: (True, t.body_fn) if isinstance(t, _HasMember11_07) else (False, None),
    "body_fn_trace": lambda t: (True, t.body_fn_trace) if isinstance(t, _HasMember11_08) else (False, None),
    "body_fun": lambda t: (True, t.body_fun) if isinstance(t, _HasMember11_09) else (False, None),
    "bool_": lambda t: (True, t.bool_) if isinstance(t, _HasMember11_10) else (False, None),
    "boolean_mask": lambda t: (True, t.boolean_mask) if isinstance(t, _HasMember11_11) else (False, None),
    "booleanmask": lambda t: (True, t.booleanmask) if isinstance(t, _HasMember11_12) else (False, None),
    "broadcast_arrays": lambda t: (True, t.broadcast_arrays) if isinstance(t, _HasMember11_13) else (False, None),
    "broadcast_to": lambda t: (True, t.broadcast_to) if isinstance(t, _HasMember11_14) else (False, None),
    "broadcastto": lambda t: (True, t.broadcastto) if isinstance(t, _HasMember11_15) else (False, None),
    "buffer": lambda t: (True, t.buffer) if isinstance(t, _HasMember11_16) else (False, None),
    "bufferedAmount": lambda t: (True, t.bufferedAmount) if isinstance(t, _HasMember11_17) else (False, None),
    "buffered_amount": lambda t: (True, t.buffered_amount) if isinstance(t, _HasMember11_18) else (False, None),
    "buffers": lambda t: (True, t.buffers) if isinstance(t, _HasMember11_19) else (False, None),
    "buggy_desc": lambda t: (True, t.buggy_desc) if isinstance(t, _HasMember11_20) else (False, None),
    "build_call": lambda t: (True, t.build_call) if isinstance(t, _HasMember11_21) else (False, None),
    "build_test_graph": lambda t: (True, t.build_test_graph) if isinstance(t, _HasMember11_22) else (False, None),
    "build_test_node": lambda t: (True, t.build_test_node) if isinstance(t, _HasMember11_23) else (False, None),
    "bwd": lambda t: (True, t.bwd) if isinstance(t, _HasMember11_24) else (False, None),
    "bytecode": lambda t: (True, t.bytecode) if isinstance(t, _HasMember11_25) else (False, None),
    "byteorder": lambda t: (True, t.byteorder) if isinstance(t, _HasMember11_26) else (False, None),
    "c": lambda t: (True, t.c) if isinstance(t, _HasMember11_27) else (False, None),
    "callable": lambda t: (True, t.callable) if isinstance(t, _HasMember11_28) else (False, None),
    "canonical": lambda t: (True, t.canonical) if isinstance(t, _HasMember11_29) else (False, None),
    "capitalize": lambda t: (True, t.capitalize) if isinstance(t, _HasMember11_30) else (False, None),
    "cartesian_prod": lambda t: (True, t.cartesian_prod) if isinstance(t, _HasMember11_31) else (False, None),
    "case": lambda t: (True, t.case) if isinstance(t, _HasMember11_32) else (False, None),
    "cast": lambda t: (True, t.cast) if isinstance(t, _HasMember11_33) else (False, None),
    "cat": lambda t: (True, t.cat) if isinstance(t, _HasMember11_34) else (False, None),
    "categorical_crossentropy": lambda t: (
        (True, t.categorical_crossentropy) if isinstance(t, _HasMember11_35) else (False, None)
    ),
    "categorical_generalized_cross_entropy": lambda t: (
        (True, t.categorical_generalized_cross_entropy) if isinstance(t, _HasMember11_36) else (False, None)
    ),
    "categoricalgeneralizedcrossentropy": lambda t: (
        (True, t.categoricalgeneralizedcrossentropy) if isinstance(t, _HasMember11_37) else (False, None)
    ),
    "cbrt": lambda t: (True, t.cbrt) if isinstance(t, _HasMember11_38) else (False, None),
    "cdf": lambda t: (True, t.cdf) if isinstance(t, _HasMember11_39) else (False, None),
    "ceil": lambda t: (True, t.ceil) if isinstance(t, _HasMember11_40) else (False, None),
    "ceiling": lambda t: (True, t.ceiling) if isinstance(t, _HasMember11_41) else (False, None),
    "cg": lambda t: (True, t.cg) if isinstance(t, _HasMember11_42) else (False, None),
    "cg2": lambda t: (True, t.cg2) if isinstance(t, _HasMember11_43) else (False, None),
    "chebyshev_polynomial_t": lambda t: (
        (True, t.chebyshev_polynomial_t) if isinstance(t, _HasMember11_44) else (False, None)
    ),
}
