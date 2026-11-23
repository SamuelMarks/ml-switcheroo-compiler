"""Static structural protocol bindings shard 12 (chebyshev_polynomial_u..conjugate)."""

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
class _HasMember12_00(typing.Protocol):
    """Structural protocol for objects exposing `chebyshev_polynomial_u`."""

    chebyshev_polynomial_u: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_01(typing.Protocol):
    """Structural protocol for objects exposing `check_model`."""

    check_model: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_02(typing.Protocol):
    """Structural protocol for objects exposing `check_wgsl_shader_compliance`."""

    check_wgsl_shader_compliance: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_03(typing.Protocol):
    """Structural protocol for objects exposing `children`."""

    children: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_04(typing.Protocol):
    """Structural protocol for objects exposing `cho_solve`."""

    cho_solve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_05(typing.Protocol):
    """Structural protocol for objects exposing `choice`."""

    choice: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_06(typing.Protocol):
    """Structural protocol for objects exposing `cholesky`."""

    cholesky: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_07(typing.Protocol):
    """Structural protocol for objects exposing `cholesky_solve`."""

    cholesky_solve: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_08(typing.Protocol):
    """Structural protocol for objects exposing `choose`."""

    choose: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_09(typing.Protocol):
    """Structural protocol for objects exposing `chunked_array`."""

    chunked_array: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_10(typing.Protocol):
    """Structural protocol for objects exposing `circle_loss`."""

    circle_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_11(typing.Protocol):
    """Structural protocol for objects exposing `circleloss`."""

    circleloss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_12(typing.Protocol):
    """Structural protocol for objects exposing `cleanup`."""

    cleanup: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_13(typing.Protocol):
    """Structural protocol for objects exposing `clip`."""

    clip: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_14(typing.Protocol):
    """Structural protocol for objects exposing `clone`."""

    clone: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_15(typing.Protocol):
    """Structural protocol for objects exposing `close`."""

    close: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_16(typing.Protocol):
    """Structural protocol for objects exposing `code`."""

    code: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_17(typing.Protocol):
    """Structural protocol for objects exposing `column`."""

    column: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_18(typing.Protocol):
    """Structural protocol for objects exposing `column_stack`."""

    column_stack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_19(typing.Protocol):
    """Structural protocol for objects exposing `columnstack`."""

    columnstack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_20(typing.Protocol):
    """Structural protocol for objects exposing `combine_chunks`."""

    combine_chunks: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_21(typing.Protocol):
    """Structural protocol for objects exposing `compile`."""

    compile: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_22(typing.Protocol):
    """Structural protocol for objects exposing `compile_aot`."""

    compile_aot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_23(typing.Protocol):
    """Structural protocol for objects exposing `complex128`."""

    complex128: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_24(typing.Protocol):
    """Structural protocol for objects exposing `complex64`."""

    complex64: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_25(typing.Protocol):
    """Structural protocol for objects exposing `complex_forward`."""

    complex_forward: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_26(typing.Protocol):
    """Structural protocol for objects exposing `compress`."""

    compress: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_27(typing.Protocol):
    """Structural protocol for objects exposing `compute`."""

    compute: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_28(typing.Protocol):
    """Structural protocol for objects exposing `compute_graph`."""

    compute_graph: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_29(typing.Protocol):
    """Structural protocol for objects exposing `concat`."""

    concat: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_30(typing.Protocol):
    """Structural protocol for objects exposing `concatenate`."""

    concatenate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_31(typing.Protocol):
    """Structural protocol for objects exposing `cond`."""

    cond: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_32(typing.Protocol):
    """Structural protocol for objects exposing `cond_1`."""

    cond_1: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_33(typing.Protocol):
    """Structural protocol for objects exposing `cond_2`."""

    cond_2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_34(typing.Protocol):
    """Structural protocol for objects exposing `cond_eager`."""

    cond_eager: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_35(typing.Protocol):
    """Structural protocol for objects exposing `cond_fn`."""

    cond_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_36(typing.Protocol):
    """Structural protocol for objects exposing `cond_fn_trace`."""

    cond_fn_trace: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_37(typing.Protocol):
    """Structural protocol for objects exposing `cond_tracing`."""

    cond_tracing: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_38(typing.Protocol):
    """Structural protocol for objects exposing `condition`."""

    condition: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_39(typing.Protocol):
    """Structural protocol for objects exposing `config`."""

    config: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_40(typing.Protocol):
    """Structural protocol for objects exposing `config_models`."""

    config_models: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_41(typing.Protocol):
    """Structural protocol for objects exposing `configs`."""

    configs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_42(typing.Protocol):
    """Structural protocol for objects exposing `confusion_matrix`."""

    confusion_matrix: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_43(typing.Protocol):
    """Structural protocol for objects exposing `conj`."""

    conj: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember12_44(typing.Protocol):
    """Structural protocol for objects exposing `conjugate`."""

    conjugate: EagerTensorProtocol


SHARD_12_READERS: dict[str, MemberReader] = {
    "chebyshev_polynomial_u": lambda t: (
        (True, t.chebyshev_polynomial_u) if isinstance(t, _HasMember12_00) else (False, None)
    ),
    "check_model": lambda t: (True, t.check_model) if isinstance(t, _HasMember12_01) else (False, None),
    "check_wgsl_shader_compliance": lambda t: (
        (True, t.check_wgsl_shader_compliance) if isinstance(t, _HasMember12_02) else (False, None)
    ),
    "children": lambda t: (True, t.children) if isinstance(t, _HasMember12_03) else (False, None),
    "cho_solve": lambda t: (True, t.cho_solve) if isinstance(t, _HasMember12_04) else (False, None),
    "choice": lambda t: (True, t.choice) if isinstance(t, _HasMember12_05) else (False, None),
    "cholesky": lambda t: (True, t.cholesky) if isinstance(t, _HasMember12_06) else (False, None),
    "cholesky_solve": lambda t: (True, t.cholesky_solve) if isinstance(t, _HasMember12_07) else (False, None),
    "choose": lambda t: (True, t.choose) if isinstance(t, _HasMember12_08) else (False, None),
    "chunked_array": lambda t: (True, t.chunked_array) if isinstance(t, _HasMember12_09) else (False, None),
    "circle_loss": lambda t: (True, t.circle_loss) if isinstance(t, _HasMember12_10) else (False, None),
    "circleloss": lambda t: (True, t.circleloss) if isinstance(t, _HasMember12_11) else (False, None),
    "cleanup": lambda t: (True, t.cleanup) if isinstance(t, _HasMember12_12) else (False, None),
    "clip": lambda t: (True, t.clip) if isinstance(t, _HasMember12_13) else (False, None),
    "clone": lambda t: (True, t.clone) if isinstance(t, _HasMember12_14) else (False, None),
    "close": lambda t: (True, t.close) if isinstance(t, _HasMember12_15) else (False, None),
    "code": lambda t: (True, t.code) if isinstance(t, _HasMember12_16) else (False, None),
    "column": lambda t: (True, t.column) if isinstance(t, _HasMember12_17) else (False, None),
    "column_stack": lambda t: (True, t.column_stack) if isinstance(t, _HasMember12_18) else (False, None),
    "columnstack": lambda t: (True, t.columnstack) if isinstance(t, _HasMember12_19) else (False, None),
    "combine_chunks": lambda t: (True, t.combine_chunks) if isinstance(t, _HasMember12_20) else (False, None),
    "compile": lambda t: (True, t.compile) if isinstance(t, _HasMember12_21) else (False, None),
    "compile_aot": lambda t: (True, t.compile_aot) if isinstance(t, _HasMember12_22) else (False, None),
    "complex128": lambda t: (True, t.complex128) if isinstance(t, _HasMember12_23) else (False, None),
    "complex64": lambda t: (True, t.complex64) if isinstance(t, _HasMember12_24) else (False, None),
    "complex_forward": lambda t: (True, t.complex_forward) if isinstance(t, _HasMember12_25) else (False, None),
    "compress": lambda t: (True, t.compress) if isinstance(t, _HasMember12_26) else (False, None),
    "compute": lambda t: (True, t.compute) if isinstance(t, _HasMember12_27) else (False, None),
    "compute_graph": lambda t: (True, t.compute_graph) if isinstance(t, _HasMember12_28) else (False, None),
    "concat": lambda t: (True, t.concat) if isinstance(t, _HasMember12_29) else (False, None),
    "concatenate": lambda t: (True, t.concatenate) if isinstance(t, _HasMember12_30) else (False, None),
    "cond": lambda t: (True, t.cond) if isinstance(t, _HasMember12_31) else (False, None),
    "cond_1": lambda t: (True, t.cond_1) if isinstance(t, _HasMember12_32) else (False, None),
    "cond_2": lambda t: (True, t.cond_2) if isinstance(t, _HasMember12_33) else (False, None),
    "cond_eager": lambda t: (True, t.cond_eager) if isinstance(t, _HasMember12_34) else (False, None),
    "cond_fn": lambda t: (True, t.cond_fn) if isinstance(t, _HasMember12_35) else (False, None),
    "cond_fn_trace": lambda t: (True, t.cond_fn_trace) if isinstance(t, _HasMember12_36) else (False, None),
    "cond_tracing": lambda t: (True, t.cond_tracing) if isinstance(t, _HasMember12_37) else (False, None),
    "condition": lambda t: (True, t.condition) if isinstance(t, _HasMember12_38) else (False, None),
    "config": lambda t: (True, t.config) if isinstance(t, _HasMember12_39) else (False, None),
    "config_models": lambda t: (True, t.config_models) if isinstance(t, _HasMember12_40) else (False, None),
    "configs": lambda t: (True, t.configs) if isinstance(t, _HasMember12_41) else (False, None),
    "confusion_matrix": lambda t: (True, t.confusion_matrix) if isinstance(t, _HasMember12_42) else (False, None),
    "conj": lambda t: (True, t.conj) if isinstance(t, _HasMember12_43) else (False, None),
    "conjugate": lambda t: (True, t.conjugate) if isinstance(t, _HasMember12_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember12_00,
        _HasMember12_01,
        _HasMember12_02,
        _HasMember12_03,
        _HasMember12_04,
        _HasMember12_05,
        _HasMember12_06,
        _HasMember12_07,
        _HasMember12_08,
        _HasMember12_09,
        _HasMember12_10,
        _HasMember12_11,
        _HasMember12_12,
        _HasMember12_13,
        _HasMember12_14,
        _HasMember12_15,
        _HasMember12_16,
        _HasMember12_17,
        _HasMember12_18,
        _HasMember12_19,
        _HasMember12_20,
        _HasMember12_21,
        _HasMember12_22,
        _HasMember12_23,
        _HasMember12_24,
        _HasMember12_25,
        _HasMember12_26,
        _HasMember12_27,
        _HasMember12_28,
        _HasMember12_29,
        _HasMember12_30,
        _HasMember12_31,
        _HasMember12_32,
        _HasMember12_33,
        _HasMember12_34,
        _HasMember12_35,
        _HasMember12_36,
        _HasMember12_37,
        _HasMember12_38,
        _HasMember12_39,
        _HasMember12_40,
        _HasMember12_41,
        _HasMember12_42,
        _HasMember12_43,
        _HasMember12_44,
    )
)
