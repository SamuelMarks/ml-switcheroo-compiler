"""Unified static attribute reader table for `EagerBackendAdapter`."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable

from ml_switcheroo_compiler.backends.eager_registry.protocols import EagerTensorProtocol
from ml_switcheroo_compiler.backends.eager_registry.types_base import BackendOperand

from ._support import wrap_shard_readers
from .shard_00 import SHARD_00_READERS
from .shard_01 import SHARD_01_READERS
from .shard_02 import SHARD_02_READERS
from .shard_03 import SHARD_03_READERS
from .shard_04 import SHARD_04_READERS
from .shard_05 import SHARD_05_READERS
from .shard_06 import SHARD_06_READERS
from .shard_07 import SHARD_07_READERS
from .shard_08 import SHARD_08_READERS
from .shard_09 import SHARD_09_READERS
from .shard_10 import SHARD_10_READERS
from .shard_11 import SHARD_11_READERS
from .shard_12 import SHARD_12_READERS
from .shard_13 import SHARD_13_READERS
from .shard_14 import SHARD_14_READERS
from .shard_15 import SHARD_15_READERS
from .shard_16 import SHARD_16_READERS
from .shard_17 import SHARD_17_READERS
from .shard_18 import SHARD_18_READERS
from .shard_19 import SHARD_19_READERS
from .shard_20 import SHARD_20_READERS
from .shard_21 import SHARD_21_READERS
from .shard_22 import SHARD_22_READERS
from .shard_23 import SHARD_23_READERS
from .shard_24 import SHARD_24_READERS
from .shard_25 import SHARD_25_READERS
from .shard_26 import SHARD_26_READERS
from .shard_27 import SHARD_27_READERS
from .shard_28 import SHARD_28_READERS
from .shard_29 import SHARD_29_READERS
from .shard_30 import SHARD_30_READERS
from .shard_31 import SHARD_31_READERS
from .shard_32 import SHARD_32_READERS
from .shard_33 import SHARD_33_READERS
from .shard_34 import SHARD_34_READERS
from .shard_35 import SHARD_35_READERS
from .shard_36 import SHARD_36_READERS
from .shard_37 import SHARD_37_READERS
from .shard_38 import SHARD_38_READERS
from .shard_39 import SHARD_39_READERS
from .shard_40 import SHARD_40_READERS
from .shard_41 import SHARD_41_READERS
from .shard_42 import SHARD_42_READERS
from .shard_43 import SHARD_43_READERS
from .shard_44 import SHARD_44_READERS
from .shard_45 import SHARD_45_READERS
from .shard_46 import SHARD_46_READERS
from .shard_47 import SHARD_47_READERS
from .shard_48 import SHARD_48_READERS
from .shard_49 import SHARD_49_READERS
from .shard_50 import SHARD_50_READERS
from .shard_51 import SHARD_51_READERS
from .shard_52 import SHARD_52_READERS
from .shard_53 import SHARD_53_READERS
from .shard_54 import SHARD_54_READERS
from .shard_55 import SHARD_55_READERS
from .shard_56 import SHARD_56_READERS
from .shard_57 import SHARD_57_READERS
from .shard_58 import SHARD_58_READERS
from .shard_59 import SHARD_59_READERS
from .shard_60 import SHARD_60_READERS

MemberReader = Callable[
    [typing.Union[types.ModuleType, BackendOperand]],  # noqa: UP007
    tuple[bool, typing.Optional[EagerTensorProtocol]],  # noqa: UP045
]

ADAPTER_ATTR_READERS: dict[str, MemberReader] = wrap_shard_readers(
    {
        **SHARD_00_READERS,
        **SHARD_01_READERS,
        **SHARD_02_READERS,
        **SHARD_03_READERS,
        **SHARD_04_READERS,
        **SHARD_05_READERS,
        **SHARD_06_READERS,
        **SHARD_07_READERS,
        **SHARD_08_READERS,
        **SHARD_09_READERS,
        **SHARD_10_READERS,
        **SHARD_11_READERS,
        **SHARD_12_READERS,
        **SHARD_13_READERS,
        **SHARD_14_READERS,
        **SHARD_15_READERS,
        **SHARD_16_READERS,
        **SHARD_17_READERS,
        **SHARD_18_READERS,
        **SHARD_19_READERS,
        **SHARD_20_READERS,
        **SHARD_21_READERS,
        **SHARD_22_READERS,
        **SHARD_23_READERS,
        **SHARD_24_READERS,
        **SHARD_25_READERS,
        **SHARD_26_READERS,
        **SHARD_27_READERS,
        **SHARD_28_READERS,
        **SHARD_29_READERS,
        **SHARD_30_READERS,
        **SHARD_31_READERS,
        **SHARD_32_READERS,
        **SHARD_33_READERS,
        **SHARD_34_READERS,
        **SHARD_35_READERS,
        **SHARD_36_READERS,
        **SHARD_37_READERS,
        **SHARD_38_READERS,
        **SHARD_39_READERS,
        **SHARD_40_READERS,
        **SHARD_41_READERS,
        **SHARD_42_READERS,
        **SHARD_43_READERS,
        **SHARD_44_READERS,
        **SHARD_45_READERS,
        **SHARD_46_READERS,
        **SHARD_47_READERS,
        **SHARD_48_READERS,
        **SHARD_49_READERS,
        **SHARD_50_READERS,
        **SHARD_51_READERS,
        **SHARD_52_READERS,
        **SHARD_53_READERS,
        **SHARD_54_READERS,
        **SHARD_55_READERS,
        **SHARD_56_READERS,
        **SHARD_57_READERS,
        **SHARD_58_READERS,
        **SHARD_59_READERS,
        **SHARD_60_READERS,
    }
)


def read_static_attr(
    target: types.ModuleType | BackendOperand,
    name: str,
) -> tuple[bool, EagerTensorProtocol | None]:
    """Read attribute `name` from `target` via static structural protocols.

    Args:
        target (types.ModuleType | BackendOperand): Target module or object.
        name (str): Static identifier name.

    Returns:
        tuple[bool, EagerTensorProtocol | None]: `(True, value)` if structurally present, else `(False, None)`.
    """
    reader = ADAPTER_ATTR_READERS.get(name)
    if reader is None:
        return False, None
    return reader(target)


__all__ = ["ADAPTER_ATTR_READERS", "MemberReader", "read_static_attr"]
