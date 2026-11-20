"""Static structural protocol bindings shard 57 (visit_WarpShuffleReduce..zeros_like)."""

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
class _HasMember57_00(typing.Protocol):
    """Structural protocol for objects exposing `visit_WarpShuffleReduce`."""

    visit_WarpShuffleReduce: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_01(typing.Protocol):
    """Structural protocol for objects exposing `visit_While`."""

    visit_While: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_02(typing.Protocol):
    """Structural protocol for objects exposing `visit_WhileLoop`."""

    visit_WhileLoop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_03(typing.Protocol):
    """Structural protocol for objects exposing `visit_WithField`."""

    visit_WithField: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_04(typing.Protocol):
    """Structural protocol for objects exposing `visit_Zeros`."""

    visit_Zeros: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_05(typing.Protocol):
    """Structural protocol for objects exposing `visit_Zip`."""

    visit_Zip: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_06(typing.Protocol):
    """Structural protocol for objects exposing `vjp`."""

    vjp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_07(typing.Protocol):
    """Structural protocol for objects exposing `vmap`."""

    vmap: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_08(typing.Protocol):
    """Structural protocol for objects exposing `vsplit`."""

    vsplit: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_09(typing.Protocol):
    """Structural protocol for objects exposing `vstack`."""

    vstack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_10(typing.Protocol):
    """Structural protocol for objects exposing `wait`."""

    wait: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_11(typing.Protocol):
    """Structural protocol for objects exposing `wait_drain`."""

    wait_drain: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_12(typing.Protocol):
    """Structural protocol for objects exposing `wald`."""

    wald: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_13(typing.Protocol):
    """Structural protocol for objects exposing `weibullmin`."""

    weibullmin: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_14(typing.Protocol):
    """Structural protocol for objects exposing `welch`."""

    welch: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_15(typing.Protocol):
    """Structural protocol for objects exposing `welch_config`."""

    welch_config: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_16(typing.Protocol):
    """Structural protocol for objects exposing `welchconfig`."""

    welchconfig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_17(typing.Protocol):
    """Structural protocol for objects exposing `where`."""

    where: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_18(typing.Protocol):
    """Structural protocol for objects exposing `while_loop`."""

    while_loop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_19(typing.Protocol):
    """Structural protocol for objects exposing `while_loop_eager`."""

    while_loop_eager: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_20(typing.Protocol):
    """Structural protocol for objects exposing `while_loop_tracing`."""

    while_loop_tracing: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_21(typing.Protocol):
    """Structural protocol for objects exposing `window_config`."""

    window_config: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_22(typing.Protocol):
    """Structural protocol for objects exposing `window_hamming`."""

    window_hamming: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_23(typing.Protocol):
    """Structural protocol for objects exposing `window_hann`."""

    window_hann: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_24(typing.Protocol):
    """Structural protocol for objects exposing `windowconfig`."""

    windowconfig: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_25(typing.Protocol):
    """Structural protocol for objects exposing `windowhamming`."""

    windowhamming: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_26(typing.Protocol):
    """Structural protocol for objects exposing `windowhann`."""

    windowhann: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_27(typing.Protocol):
    """Structural protocol for objects exposing `with_field`."""

    with_field: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_28(typing.Protocol):
    """Structural protocol for objects exposing `world_size`."""

    world_size: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_29(typing.Protocol):
    """Structural protocol for objects exposing `wrapkeydata`."""

    wrapkeydata: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_30(typing.Protocol):
    """Structural protocol for objects exposing `write_file`."""

    write_file: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_31(typing.Protocol):
    """Structural protocol for objects exposing `write_to_directory`."""

    write_to_directory: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_32(typing.Protocol):
    """Structural protocol for objects exposing `writefile`."""

    writefile: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_33(typing.Protocol):
    """Structural protocol for objects exposing `x`."""

    x: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_34(typing.Protocol):
    """Structural protocol for objects exposing `xdivy`."""

    xdivy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_35(typing.Protocol):
    """Structural protocol for objects exposing `xlog1py`."""

    xlog1py: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_36(typing.Protocol):
    """Structural protocol for objects exposing `xlogy`."""

    xlogy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_37(typing.Protocol):
    """Structural protocol for objects exposing `xor`."""

    xor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_38(typing.Protocol):
    """Structural protocol for objects exposing `y`."""

    y: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_39(typing.Protocol):
    """Structural protocol for objects exposing `yiq_to_rgb`."""

    yiq_to_rgb: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_40(typing.Protocol):
    """Structural protocol for objects exposing `yuv_to_rgb`."""

    yuv_to_rgb: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_41(typing.Protocol):
    """Structural protocol for objects exposing `zero_fraction`."""

    zero_fraction: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_42(typing.Protocol):
    """Structural protocol for objects exposing `zerofraction`."""

    zerofraction: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_43(typing.Protocol):
    """Structural protocol for objects exposing `zeros`."""

    zeros: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember57_44(typing.Protocol):
    """Structural protocol for objects exposing `zeros_like`."""

    zeros_like: EagerTensorProtocol


SHARD_57_READERS: dict[str, MemberReader] = {
    "visit_WarpShuffleReduce": lambda t: (
        (True, t.visit_WarpShuffleReduce) if isinstance(t, _HasMember57_00) else (False, None)
    ),
    "visit_While": lambda t: (True, t.visit_While) if isinstance(t, _HasMember57_01) else (False, None),
    "visit_WhileLoop": lambda t: (True, t.visit_WhileLoop) if isinstance(t, _HasMember57_02) else (False, None),
    "visit_WithField": lambda t: (True, t.visit_WithField) if isinstance(t, _HasMember57_03) else (False, None),
    "visit_Zeros": lambda t: (True, t.visit_Zeros) if isinstance(t, _HasMember57_04) else (False, None),
    "visit_Zip": lambda t: (True, t.visit_Zip) if isinstance(t, _HasMember57_05) else (False, None),
    "vjp": lambda t: (True, t.vjp) if isinstance(t, _HasMember57_06) else (False, None),
    "vmap": lambda t: (True, t.vmap) if isinstance(t, _HasMember57_07) else (False, None),
    "vsplit": lambda t: (True, t.vsplit) if isinstance(t, _HasMember57_08) else (False, None),
    "vstack": lambda t: (True, t.vstack) if isinstance(t, _HasMember57_09) else (False, None),
    "wait": lambda t: (True, t.wait) if isinstance(t, _HasMember57_10) else (False, None),
    "wait_drain": lambda t: (True, t.wait_drain) if isinstance(t, _HasMember57_11) else (False, None),
    "wald": lambda t: (True, t.wald) if isinstance(t, _HasMember57_12) else (False, None),
    "weibullmin": lambda t: (True, t.weibullmin) if isinstance(t, _HasMember57_13) else (False, None),
    "welch": lambda t: (True, t.welch) if isinstance(t, _HasMember57_14) else (False, None),
    "welch_config": lambda t: (True, t.welch_config) if isinstance(t, _HasMember57_15) else (False, None),
    "welchconfig": lambda t: (True, t.welchconfig) if isinstance(t, _HasMember57_16) else (False, None),
    "where": lambda t: (True, t.where) if isinstance(t, _HasMember57_17) else (False, None),
    "while_loop": lambda t: (True, t.while_loop) if isinstance(t, _HasMember57_18) else (False, None),
    "while_loop_eager": lambda t: (True, t.while_loop_eager) if isinstance(t, _HasMember57_19) else (False, None),
    "while_loop_tracing": lambda t: (True, t.while_loop_tracing) if isinstance(t, _HasMember57_20) else (False, None),
    "window_config": lambda t: (True, t.window_config) if isinstance(t, _HasMember57_21) else (False, None),
    "window_hamming": lambda t: (True, t.window_hamming) if isinstance(t, _HasMember57_22) else (False, None),
    "window_hann": lambda t: (True, t.window_hann) if isinstance(t, _HasMember57_23) else (False, None),
    "windowconfig": lambda t: (True, t.windowconfig) if isinstance(t, _HasMember57_24) else (False, None),
    "windowhamming": lambda t: (True, t.windowhamming) if isinstance(t, _HasMember57_25) else (False, None),
    "windowhann": lambda t: (True, t.windowhann) if isinstance(t, _HasMember57_26) else (False, None),
    "with_field": lambda t: (True, t.with_field) if isinstance(t, _HasMember57_27) else (False, None),
    "world_size": lambda t: (True, t.world_size) if isinstance(t, _HasMember57_28) else (False, None),
    "wrapkeydata": lambda t: (True, t.wrapkeydata) if isinstance(t, _HasMember57_29) else (False, None),
    "write_file": lambda t: (True, t.write_file) if isinstance(t, _HasMember57_30) else (False, None),
    "write_to_directory": lambda t: (True, t.write_to_directory) if isinstance(t, _HasMember57_31) else (False, None),
    "writefile": lambda t: (True, t.writefile) if isinstance(t, _HasMember57_32) else (False, None),
    "x": lambda t: (True, t.x) if isinstance(t, _HasMember57_33) else (False, None),
    "xdivy": lambda t: (True, t.xdivy) if isinstance(t, _HasMember57_34) else (False, None),
    "xlog1py": lambda t: (True, t.xlog1py) if isinstance(t, _HasMember57_35) else (False, None),
    "xlogy": lambda t: (True, t.xlogy) if isinstance(t, _HasMember57_36) else (False, None),
    "xor": lambda t: (True, t.xor) if isinstance(t, _HasMember57_37) else (False, None),
    "y": lambda t: (True, t.y) if isinstance(t, _HasMember57_38) else (False, None),
    "yiq_to_rgb": lambda t: (True, t.yiq_to_rgb) if isinstance(t, _HasMember57_39) else (False, None),
    "yuv_to_rgb": lambda t: (True, t.yuv_to_rgb) if isinstance(t, _HasMember57_40) else (False, None),
    "zero_fraction": lambda t: (True, t.zero_fraction) if isinstance(t, _HasMember57_41) else (False, None),
    "zerofraction": lambda t: (True, t.zerofraction) if isinstance(t, _HasMember57_42) else (False, None),
    "zeros": lambda t: (True, t.zeros) if isinstance(t, _HasMember57_43) else (False, None),
    "zeros_like": lambda t: (True, t.zeros_like) if isinstance(t, _HasMember57_44) else (False, None),
}
