"""Static structural protocol bindings shard 58 (zeta..zip and extended NumPy module symbols)."""

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
class _HasMember58_00(typing.Protocol):
    """Structural protocol for objects exposing `zeta`."""

    zeta: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_01(typing.Protocol):
    """Structural protocol for objects exposing `zip`."""

    zip: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_02(typing.Protocol):
    """Structural protocol for objects exposing `apply_along_axis`."""

    apply_along_axis: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_03(typing.Protocol):
    """Structural protocol for objects exposing `array_equiv`."""

    array_equiv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_04(typing.Protocol):
    """Structural protocol for objects exposing `blackman`."""

    blackman: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_05(typing.Protocol):
    """Structural protocol for objects exposing `can_cast`."""

    can_cast: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_06(typing.Protocol):
    """Structural protocol for objects exposing `finfo`."""

    finfo: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_07(typing.Protocol):
    """Structural protocol for objects exposing `hamming`."""

    hamming: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_08(typing.Protocol):
    """Structural protocol for objects exposing `hanning`."""

    hanning: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_09(typing.Protocol):
    """Structural protocol for objects exposing `histogram2d`."""

    histogram2d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_10(typing.Protocol):
    """Structural protocol for objects exposing `histogram_bin_edges`."""

    histogram_bin_edges: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_11(typing.Protocol):
    """Structural protocol for objects exposing `histogramdd`."""

    histogramdd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_12(typing.Protocol):
    """Structural protocol for objects exposing `iinfo`."""

    iinfo: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_13(typing.Protocol):
    """Structural protocol for objects exposing `kaiser`."""

    kaiser: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_14(typing.Protocol):
    """Structural protocol for objects exposing `lib`."""

    lib: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_15(typing.Protocol):
    """Structural protocol for objects exposing `logspace`."""

    logspace: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_16(typing.Protocol):
    """Structural protocol for objects exposing `pi`."""

    pi: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_17(typing.Protocol):
    """Structural protocol for objects exposing `r_`."""

    r_: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_18(typing.Protocol):
    """Structural protocol for objects exposing `setdiff1d`."""

    setdiff1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_19(typing.Protocol):
    """Structural protocol for objects exposing `setxor1d`."""

    setxor1d: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_20(typing.Protocol):
    """Structural protocol for objects exposing `nnz`."""

    nnz: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_21(typing.Protocol):
    """Structural protocol for objects exposing `num_rows`."""

    num_rows: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_22(typing.Protocol):
    """Structural protocol for objects exposing `getresponse`."""

    getresponse: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_23(typing.Protocol):
    """Structural protocol for objects exposing `request`."""

    request: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_24(typing.Protocol):
    """Structural protocol for objects exposing `read`."""

    read: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember58_25(typing.Protocol):
    """Structural protocol for objects exposing `status`."""

    status: EagerTensorProtocol


SHARD_58_READERS: dict[str, MemberReader] = {
    "zeta": lambda t: (True, t.zeta) if isinstance(t, _HasMember58_00) else (False, None),
    "zip": lambda t: (True, t.zip) if isinstance(t, _HasMember58_01) else (False, None),
    "apply_along_axis": lambda t: (True, t.apply_along_axis) if isinstance(t, _HasMember58_02) else (False, None),
    "array_equiv": lambda t: (True, t.array_equiv) if isinstance(t, _HasMember58_03) else (False, None),
    "blackman": lambda t: (True, t.blackman) if isinstance(t, _HasMember58_04) else (False, None),
    "can_cast": lambda t: (True, t.can_cast) if isinstance(t, _HasMember58_05) else (False, None),
    "finfo": lambda t: (True, t.finfo) if isinstance(t, _HasMember58_06) else (False, None),
    "hamming": lambda t: (True, t.hamming) if isinstance(t, _HasMember58_07) else (False, None),
    "hanning": lambda t: (True, t.hanning) if isinstance(t, _HasMember58_08) else (False, None),
    "histogram2d": lambda t: (True, t.histogram2d) if isinstance(t, _HasMember58_09) else (False, None),
    "histogram_bin_edges": (
        lambda t: (True, t.histogram_bin_edges) if isinstance(t, _HasMember58_10) else (False, None)
    ),
    "histogramdd": lambda t: (True, t.histogramdd) if isinstance(t, _HasMember58_11) else (False, None),
    "iinfo": lambda t: (True, t.iinfo) if isinstance(t, _HasMember58_12) else (False, None),
    "kaiser": lambda t: (True, t.kaiser) if isinstance(t, _HasMember58_13) else (False, None),
    "lib": lambda t: (True, t.lib) if isinstance(t, _HasMember58_14) else (False, None),
    "logspace": lambda t: (True, t.logspace) if isinstance(t, _HasMember58_15) else (False, None),
    "pi": lambda t: (True, t.pi) if isinstance(t, _HasMember58_16) else (False, None),
    "r_": lambda t: (True, t.r_) if isinstance(t, _HasMember58_17) else (False, None),
    "setdiff1d": lambda t: (True, t.setdiff1d) if isinstance(t, _HasMember58_18) else (False, None),
    "setxor1d": lambda t: (True, t.setxor1d) if isinstance(t, _HasMember58_19) else (False, None),
    "nnz": lambda t: (True, t.nnz) if isinstance(t, _HasMember58_20) else (False, None),
    "num_rows": lambda t: (True, t.num_rows) if isinstance(t, _HasMember58_21) else (False, None),
    "getresponse": lambda t: (True, t.getresponse) if isinstance(t, _HasMember58_22) else (False, None),
    "request": lambda t: (True, t.request) if isinstance(t, _HasMember58_23) else (False, None),
    "read": lambda t: (True, t.read) if isinstance(t, _HasMember58_24) else (False, None),
    "status": lambda t: (True, t.status) if isinstance(t, _HasMember58_25) else (False, None),
}

register_dynamic_protocols(
    (
        _HasMember58_00,
        _HasMember58_01,
        _HasMember58_02,
        _HasMember58_03,
        _HasMember58_04,
        _HasMember58_05,
        _HasMember58_06,
        _HasMember58_07,
        _HasMember58_08,
        _HasMember58_09,
        _HasMember58_10,
        _HasMember58_11,
        _HasMember58_12,
        _HasMember58_13,
        _HasMember58_14,
        _HasMember58_15,
        _HasMember58_16,
        _HasMember58_17,
        _HasMember58_18,
        _HasMember58_19,
        _HasMember58_20,
        _HasMember58_21,
        _HasMember58_22,
        _HasMember58_23,
        _HasMember58_24,
        _HasMember58_25,
    )
)
