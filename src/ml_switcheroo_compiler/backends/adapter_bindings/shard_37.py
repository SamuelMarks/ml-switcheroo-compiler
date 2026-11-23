"""Static structural protocol bindings shard 37 (nextafter..out_spec)."""

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
class _HasMember37_00(typing.Protocol):
    """Structural protocol for objects exposing `nextafter`."""

    nextafter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_01(typing.Protocol):
    """Structural protocol for objects exposing `nll_loss`."""

    nll_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_02(typing.Protocol):
    """Structural protocol for objects exposing `nms`."""

    nms: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_03(typing.Protocol):
    """Structural protocol for objects exposing `nn`."""

    nn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_04(typing.Protocol):
    """Structural protocol for objects exposing `node`."""

    node: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_05(typing.Protocol):
    """Structural protocol for objects exposing `nodes`."""

    nodes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_06(typing.Protocol):
    """Structural protocol for objects exposing `non_int_tuple`."""

    non_int_tuple: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_07(typing.Protocol):
    """Structural protocol for objects exposing `non_max_suppression`."""

    non_max_suppression: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_08(typing.Protocol):
    """Structural protocol for objects exposing `nonblocking_load`."""

    nonblocking_load: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_09(typing.Protocol):
    """Structural protocol for objects exposing `nonblocking_save`."""

    nonblocking_save: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_10(typing.Protocol):
    """Structural protocol for objects exposing `nonblockingload`."""

    nonblockingload: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_11(typing.Protocol):
    """Structural protocol for objects exposing `nonblockingsave`."""

    nonblockingsave: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_12(typing.Protocol):
    """Structural protocol for objects exposing `nonexistent_attr_xyz`."""

    nonexistent_attr_xyz: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_13(typing.Protocol):
    """Structural protocol for objects exposing `nonzero`."""

    nonzero: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_14(typing.Protocol):
    """Structural protocol for objects exposing `norm`."""

    norm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_15(typing.Protocol):
    """Structural protocol for objects exposing `norm_config`."""

    norm_config: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_16(typing.Protocol):
    """Structural protocol for objects exposing `normal`."""

    normal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_17(typing.Protocol):
    """Structural protocol for objects exposing `normalization`."""

    normalization: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_18(typing.Protocol):
    """Structural protocol for objects exposing `normalize`."""

    normalize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_19(typing.Protocol):
    """Structural protocol for objects exposing `normconfig`."""

    normconfig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_20(typing.Protocol):
    """Structural protocol for objects exposing `not_equal`."""

    not_equal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_21(typing.Protocol):
    """Structural protocol for objects exposing `notequal`."""

    notequal: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_22(typing.Protocol):
    """Structural protocol for objects exposing `np`."""

    np: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_23(typing.Protocol):
    """Structural protocol for objects exposing `null`."""

    null: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_24(typing.Protocol):
    """Structural protocol for objects exposing `nulls`."""

    nulls: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_25(typing.Protocol):
    """Structural protocol for objects exposing `num`."""

    num: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_26(typing.Protocol):
    """Structural protocol for objects exposing `num_columns`."""

    num_columns: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_27(typing.Protocol):
    """Structural protocol for objects exposing `number`."""

    number: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_28(typing.Protocol):
    """Structural protocol for objects exposing `numel`."""

    numel: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_29(typing.Protocol):
    """Structural protocol for objects exposing `numpy`."""

    numpy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_30(typing.Protocol):
    """Structural protocol for objects exposing `nvrtcDestroyProgram`."""

    nvrtcDestroyProgram: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_31(typing.Protocol):
    """Structural protocol for objects exposing `ogrid`."""

    ogrid: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_32(typing.Protocol):
    """Structural protocol for objects exposing `one_hot`."""

    one_hot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_33(typing.Protocol):
    """Structural protocol for objects exposing `onehot`."""

    onehot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_34(typing.Protocol):
    """Structural protocol for objects exposing `ones`."""

    ones: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_35(typing.Protocol):
    """Structural protocol for objects exposing `ones_like`."""

    ones_like: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_36(typing.Protocol):
    """Structural protocol for objects exposing `op`."""

    op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_37(typing.Protocol):
    """Structural protocol for objects exposing `op_def`."""

    op_def: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_38(typing.Protocol):
    """Structural protocol for objects exposing `op_name`."""

    op_name: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_39(typing.Protocol):
    """Structural protocol for objects exposing `op_type`."""

    op_type: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_40(typing.Protocol):
    """Structural protocol for objects exposing `opdef`."""

    opdef: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_41(typing.Protocol):
    """Structural protocol for objects exposing `ops`."""

    ops: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_42(typing.Protocol):
    """Structural protocol for objects exposing `or_`."""

    or_: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_43(typing.Protocol):
    """Structural protocol for objects exposing `oscillating_pass`."""

    oscillating_pass: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember37_44(typing.Protocol):
    """Structural protocol for objects exposing `out_spec`."""

    out_spec: EagerTensorProtocol


SHARD_37_READERS: dict[str, MemberReader] = {
    "nextafter": lambda t: (True, t.nextafter) if isinstance(t, _HasMember37_00) else (False, None),
    "nll_loss": lambda t: (True, t.nll_loss) if isinstance(t, _HasMember37_01) else (False, None),
    "nms": lambda t: (True, t.nms) if isinstance(t, _HasMember37_02) else (False, None),
    "nn": lambda t: (True, t.nn) if isinstance(t, _HasMember37_03) else (False, None),
    "node": lambda t: (True, t.node) if isinstance(t, _HasMember37_04) else (False, None),
    "nodes": lambda t: (True, t.nodes) if isinstance(t, _HasMember37_05) else (False, None),
    "non_int_tuple": lambda t: (True, t.non_int_tuple) if isinstance(t, _HasMember37_06) else (False, None),
    "non_max_suppression": lambda t: (True, t.non_max_suppression) if isinstance(t, _HasMember37_07) else (False, None),
    "nonblocking_load": lambda t: (True, t.nonblocking_load) if isinstance(t, _HasMember37_08) else (False, None),
    "nonblocking_save": lambda t: (True, t.nonblocking_save) if isinstance(t, _HasMember37_09) else (False, None),
    "nonblockingload": lambda t: (True, t.nonblockingload) if isinstance(t, _HasMember37_10) else (False, None),
    "nonblockingsave": lambda t: (True, t.nonblockingsave) if isinstance(t, _HasMember37_11) else (False, None),
    "nonexistent_attr_xyz": lambda t: (
        (True, t.nonexistent_attr_xyz) if isinstance(t, _HasMember37_12) else (False, None)
    ),
    "nonzero": lambda t: (True, t.nonzero) if isinstance(t, _HasMember37_13) else (False, None),
    "norm": lambda t: (True, t.norm) if isinstance(t, _HasMember37_14) else (False, None),
    "norm_config": lambda t: (True, t.norm_config) if isinstance(t, _HasMember37_15) else (False, None),
    "normal": lambda t: (True, t.normal) if isinstance(t, _HasMember37_16) else (False, None),
    "normalization": lambda t: (True, t.normalization) if isinstance(t, _HasMember37_17) else (False, None),
    "normalize": lambda t: (True, t.normalize) if isinstance(t, _HasMember37_18) else (False, None),
    "normconfig": lambda t: (True, t.normconfig) if isinstance(t, _HasMember37_19) else (False, None),
    "not_equal": lambda t: (True, t.not_equal) if isinstance(t, _HasMember37_20) else (False, None),
    "notequal": lambda t: (True, t.notequal) if isinstance(t, _HasMember37_21) else (False, None),
    "np": lambda t: (True, t.np) if isinstance(t, _HasMember37_22) else (False, None),
    "null": lambda t: (True, t.null) if isinstance(t, _HasMember37_23) else (False, None),
    "nulls": lambda t: (True, t.nulls) if isinstance(t, _HasMember37_24) else (False, None),
    "num": lambda t: (True, t.num) if isinstance(t, _HasMember37_25) else (False, None),
    "num_columns": lambda t: (True, t.num_columns) if isinstance(t, _HasMember37_26) else (False, None),
    "number": lambda t: (True, t.number) if isinstance(t, _HasMember37_27) else (False, None),
    "numel": lambda t: (True, t.numel) if isinstance(t, _HasMember37_28) else (False, None),
    "numpy": lambda t: (True, t.numpy) if isinstance(t, _HasMember37_29) else (False, None),
    "nvrtcDestroyProgram": lambda t: (True, t.nvrtcDestroyProgram) if isinstance(t, _HasMember37_30) else (False, None),
    "ogrid": lambda t: (True, t.ogrid) if isinstance(t, _HasMember37_31) else (False, None),
    "one_hot": lambda t: (True, t.one_hot) if isinstance(t, _HasMember37_32) else (False, None),
    "onehot": lambda t: (True, t.onehot) if isinstance(t, _HasMember37_33) else (False, None),
    "ones": lambda t: (True, t.ones) if isinstance(t, _HasMember37_34) else (False, None),
    "ones_like": lambda t: (True, t.ones_like) if isinstance(t, _HasMember37_35) else (False, None),
    "op": lambda t: (True, t.op) if isinstance(t, _HasMember37_36) else (False, None),
    "op_def": lambda t: (True, t.op_def) if isinstance(t, _HasMember37_37) else (False, None),
    "op_name": lambda t: (True, t.op_name) if isinstance(t, _HasMember37_38) else (False, None),
    "op_type": lambda t: (True, t.op_type) if isinstance(t, _HasMember37_39) else (False, None),
    "opdef": lambda t: (True, t.opdef) if isinstance(t, _HasMember37_40) else (False, None),
    "ops": lambda t: (True, t.ops) if isinstance(t, _HasMember37_41) else (False, None),
    "or_": lambda t: (True, t.or_) if isinstance(t, _HasMember37_42) else (False, None),
    "oscillating_pass": lambda t: (True, t.oscillating_pass) if isinstance(t, _HasMember37_43) else (False, None),
    "out_spec": lambda t: (True, t.out_spec) if isinstance(t, _HasMember37_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember37_00,
        _HasMember37_01,
        _HasMember37_02,
        _HasMember37_03,
        _HasMember37_04,
        _HasMember37_05,
        _HasMember37_06,
        _HasMember37_07,
        _HasMember37_08,
        _HasMember37_09,
        _HasMember37_10,
        _HasMember37_11,
        _HasMember37_12,
        _HasMember37_13,
        _HasMember37_14,
        _HasMember37_15,
        _HasMember37_16,
        _HasMember37_17,
        _HasMember37_18,
        _HasMember37_19,
        _HasMember37_20,
        _HasMember37_21,
        _HasMember37_22,
        _HasMember37_23,
        _HasMember37_24,
        _HasMember37_25,
        _HasMember37_26,
        _HasMember37_27,
        _HasMember37_28,
        _HasMember37_29,
        _HasMember37_30,
        _HasMember37_31,
        _HasMember37_32,
        _HasMember37_33,
        _HasMember37_34,
        _HasMember37_35,
        _HasMember37_36,
        _HasMember37_37,
        _HasMember37_38,
        _HasMember37_39,
        _HasMember37_40,
        _HasMember37_41,
        _HasMember37_42,
        _HasMember37_43,
        _HasMember37_44,
    )
)
