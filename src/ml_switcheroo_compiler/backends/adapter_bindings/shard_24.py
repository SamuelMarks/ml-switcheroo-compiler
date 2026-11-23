"""Static structural protocol bindings shard 24 (generate..gfile_copy)."""

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
class _HasMember24_00(typing.Protocol):
    """Structural protocol for objects exposing `generate`."""

    generate: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_01(typing.Protocol):
    """Structural protocol for objects exposing `generate_body`."""

    generate_body: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_02(typing.Protocol):
    """Structural protocol for objects exposing `generated_registry`."""

    generated_registry: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_03(typing.Protocol):
    """Structural protocol for objects exposing `generator`."""

    generator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_04(typing.Protocol):
    """Structural protocol for objects exposing `generator_registry`."""

    generator_registry: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_05(typing.Protocol):
    """Structural protocol for objects exposing `generic_array`."""

    generic_array: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_06(typing.Protocol):
    """Structural protocol for objects exposing `generic_asarray`."""

    generic_asarray: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_07(typing.Protocol):
    """Structural protocol for objects exposing `generic_conv_config`."""

    generic_conv_config: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_08(typing.Protocol):
    """Structural protocol for objects exposing `generic_item`."""

    generic_item: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_09(typing.Protocol):
    """Structural protocol for objects exposing `generic_visit`."""

    generic_visit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_10(typing.Protocol):
    """Structural protocol for objects exposing `generic_zeros`."""

    generic_zeros: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_11(typing.Protocol):
    """Structural protocol for objects exposing `genericconvconfig`."""

    genericconvconfig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_12(typing.Protocol):
    """Structural protocol for objects exposing `geometric`."""

    geometric: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_13(typing.Protocol):
    """Structural protocol for objects exposing `geomspace`."""

    geomspace: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_14(typing.Protocol):
    """Structural protocol for objects exposing `get`."""

    get: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_15(typing.Protocol):
    """Structural protocol for objects exposing `getDeviceCount`."""

    getDeviceCount: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_16(typing.Protocol):
    """Structural protocol for objects exposing `get_active_memory`."""

    get_active_memory: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_17(typing.Protocol):
    """Structural protocol for objects exposing `get_all_ops`."""

    get_all_ops: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_18(typing.Protocol):
    """Structural protocol for objects exposing `get_code_side_effect`."""

    get_code_side_effect: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_19(typing.Protocol):
    """Structural protocol for objects exposing `get_code_size_side_effect`."""

    get_code_size_side_effect: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_20(typing.Protocol):
    """Structural protocol for objects exposing `get_config`."""

    get_config: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_21(typing.Protocol):
    """Structural protocol for objects exposing `get_default_memory_pool`."""

    get_default_memory_pool: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_22(typing.Protocol):
    """Structural protocol for objects exposing `get_device_memory_info`."""

    get_device_memory_info: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_23(typing.Protocol):
    """Structural protocol for objects exposing `get_elapsed_time`."""

    get_elapsed_time: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_24(typing.Protocol):
    """Structural protocol for objects exposing `get_fallback_prefix`."""

    get_fallback_prefix: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_25(typing.Protocol):
    """Structural protocol for objects exposing `get_generator`."""

    get_generator: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_26(typing.Protocol):
    """Structural protocol for objects exposing `get_generator_mapping`."""

    get_generator_mapping: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_27(typing.Protocol):
    """Structural protocol for objects exposing `get_indent`."""

    get_indent: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_28(typing.Protocol):
    """Structural protocol for objects exposing `get_item`."""

    get_item: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_29(typing.Protocol):
    """Structural protocol for objects exposing `get_language`."""

    get_language: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_30(typing.Protocol):
    """Structural protocol for objects exposing `get_log_side_effect`."""

    get_log_side_effect: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_31(typing.Protocol):
    """Structural protocol for objects exposing `get_log_size_side_effect`."""

    get_log_size_side_effect: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_32(typing.Protocol):
    """Structural protocol for objects exposing `get_memory_cost`."""

    get_memory_cost: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_33(typing.Protocol):
    """Structural protocol for objects exposing `get_memory_info`."""

    get_memory_info: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_34(typing.Protocol):
    """Structural protocol for objects exposing `get_module`."""

    get_module: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_35(typing.Protocol):
    """Structural protocol for objects exposing `get_op`."""

    get_op: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_36(typing.Protocol):
    """Structural protocol for objects exposing `get_op_override`."""

    get_op_override: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_37(typing.Protocol):
    """Structural protocol for objects exposing `get_ops_map`."""

    get_ops_map: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_38(typing.Protocol):
    """Structural protocol for objects exposing `get_ops_map2`."""

    get_ops_map2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_39(typing.Protocol):
    """Structural protocol for objects exposing `get_peak_memory`."""

    get_peak_memory: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_40(typing.Protocol):
    """Structural protocol for objects exposing `get_printoptions`."""

    get_printoptions: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_41(typing.Protocol):
    """Structural protocol for objects exposing `get_ptx_side_effect`."""

    get_ptx_side_effect: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_42(typing.Protocol):
    """Structural protocol for objects exposing `get_ptx_size_side_effect`."""

    get_ptx_size_side_effect: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_43(typing.Protocol):
    """Structural protocol for objects exposing `get_sorted_nodes`."""

    get_sorted_nodes: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember24_44(typing.Protocol):
    """Structural protocol for objects exposing `gfile_copy`."""

    gfile_copy: EagerTensorProtocol


SHARD_24_READERS: dict[str, MemberReader] = {
    "generate": lambda t: (True, t.generate) if isinstance(t, _HasMember24_00) else (False, None),
    "generate_body": lambda t: (True, t.generate_body) if isinstance(t, _HasMember24_01) else (False, None),
    "generated_registry": lambda t: (True, t.generated_registry) if isinstance(t, _HasMember24_02) else (False, None),
    "generator": lambda t: (True, t.generator) if isinstance(t, _HasMember24_03) else (False, None),
    "generator_registry": lambda t: (True, t.generator_registry) if isinstance(t, _HasMember24_04) else (False, None),
    "generic_array": lambda t: (True, t.generic_array) if isinstance(t, _HasMember24_05) else (False, None),
    "generic_asarray": lambda t: (True, t.generic_asarray) if isinstance(t, _HasMember24_06) else (False, None),
    "generic_conv_config": lambda t: (True, t.generic_conv_config) if isinstance(t, _HasMember24_07) else (False, None),
    "generic_item": lambda t: (True, t.generic_item) if isinstance(t, _HasMember24_08) else (False, None),
    "generic_visit": lambda t: (True, t.generic_visit) if isinstance(t, _HasMember24_09) else (False, None),
    "generic_zeros": lambda t: (True, t.generic_zeros) if isinstance(t, _HasMember24_10) else (False, None),
    "genericconvconfig": lambda t: (True, t.genericconvconfig) if isinstance(t, _HasMember24_11) else (False, None),
    "geometric": lambda t: (True, t.geometric) if isinstance(t, _HasMember24_12) else (False, None),
    "geomspace": lambda t: (True, t.geomspace) if isinstance(t, _HasMember24_13) else (False, None),
    "get": lambda t: (True, t.get) if isinstance(t, _HasMember24_14) else (False, None),
    "getDeviceCount": lambda t: (True, t.getDeviceCount) if isinstance(t, _HasMember24_15) else (False, None),
    "get_active_memory": lambda t: (True, t.get_active_memory) if isinstance(t, _HasMember24_16) else (False, None),
    "get_all_ops": lambda t: (True, t.get_all_ops) if isinstance(t, _HasMember24_17) else (False, None),
    "get_code_side_effect": lambda t: (
        (True, t.get_code_side_effect) if isinstance(t, _HasMember24_18) else (False, None)
    ),
    "get_code_size_side_effect": lambda t: (
        (True, t.get_code_size_side_effect) if isinstance(t, _HasMember24_19) else (False, None)
    ),
    "get_config": lambda t: (True, t.get_config) if isinstance(t, _HasMember24_20) else (False, None),
    "get_default_memory_pool": lambda t: (
        (True, t.get_default_memory_pool) if isinstance(t, _HasMember24_21) else (False, None)
    ),
    "get_device_memory_info": lambda t: (
        (True, t.get_device_memory_info) if isinstance(t, _HasMember24_22) else (False, None)
    ),
    "get_elapsed_time": lambda t: (True, t.get_elapsed_time) if isinstance(t, _HasMember24_23) else (False, None),
    "get_fallback_prefix": lambda t: (True, t.get_fallback_prefix) if isinstance(t, _HasMember24_24) else (False, None),
    "get_generator": lambda t: (True, t.get_generator) if isinstance(t, _HasMember24_25) else (False, None),
    "get_generator_mapping": lambda t: (
        (True, t.get_generator_mapping) if isinstance(t, _HasMember24_26) else (False, None)
    ),
    "get_indent": lambda t: (True, t.get_indent) if isinstance(t, _HasMember24_27) else (False, None),
    "get_item": lambda t: (True, t.get_item) if isinstance(t, _HasMember24_28) else (False, None),
    "get_language": lambda t: (True, t.get_language) if isinstance(t, _HasMember24_29) else (False, None),
    "get_log_side_effect": lambda t: (True, t.get_log_side_effect) if isinstance(t, _HasMember24_30) else (False, None),
    "get_log_size_side_effect": lambda t: (
        (True, t.get_log_size_side_effect) if isinstance(t, _HasMember24_31) else (False, None)
    ),
    "get_memory_cost": lambda t: (True, t.get_memory_cost) if isinstance(t, _HasMember24_32) else (False, None),
    "get_memory_info": lambda t: (True, t.get_memory_info) if isinstance(t, _HasMember24_33) else (False, None),
    "get_module": lambda t: (True, t.get_module) if isinstance(t, _HasMember24_34) else (False, None),
    "get_op": lambda t: (True, t.get_op) if isinstance(t, _HasMember24_35) else (False, None),
    "get_op_override": lambda t: (True, t.get_op_override) if isinstance(t, _HasMember24_36) else (False, None),
    "get_ops_map": lambda t: (True, t.get_ops_map) if isinstance(t, _HasMember24_37) else (False, None),
    "get_ops_map2": lambda t: (True, t.get_ops_map2) if isinstance(t, _HasMember24_38) else (False, None),
    "get_peak_memory": lambda t: (True, t.get_peak_memory) if isinstance(t, _HasMember24_39) else (False, None),
    "get_printoptions": lambda t: (True, t.get_printoptions) if isinstance(t, _HasMember24_40) else (False, None),
    "get_ptx_side_effect": lambda t: (True, t.get_ptx_side_effect) if isinstance(t, _HasMember24_41) else (False, None),
    "get_ptx_size_side_effect": lambda t: (
        (True, t.get_ptx_size_side_effect) if isinstance(t, _HasMember24_42) else (False, None)
    ),
    "get_sorted_nodes": lambda t: (True, t.get_sorted_nodes) if isinstance(t, _HasMember24_43) else (False, None),
    "gfile_copy": lambda t: (True, t.gfile_copy) if isinstance(t, _HasMember24_44) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember24_00,
        _HasMember24_01,
        _HasMember24_02,
        _HasMember24_03,
        _HasMember24_04,
        _HasMember24_05,
        _HasMember24_06,
        _HasMember24_07,
        _HasMember24_08,
        _HasMember24_09,
        _HasMember24_10,
        _HasMember24_11,
        _HasMember24_12,
        _HasMember24_13,
        _HasMember24_14,
        _HasMember24_15,
        _HasMember24_16,
        _HasMember24_17,
        _HasMember24_18,
        _HasMember24_19,
        _HasMember24_20,
        _HasMember24_21,
        _HasMember24_22,
        _HasMember24_23,
        _HasMember24_24,
        _HasMember24_25,
        _HasMember24_26,
        _HasMember24_27,
        _HasMember24_28,
        _HasMember24_29,
        _HasMember24_30,
        _HasMember24_31,
        _HasMember24_32,
        _HasMember24_33,
        _HasMember24_34,
        _HasMember24_35,
        _HasMember24_36,
        _HasMember24_37,
        _HasMember24_38,
        _HasMember24_39,
        _HasMember24_40,
        _HasMember24_41,
        _HasMember24_42,
        _HasMember24_43,
        _HasMember24_44,
    )
)
