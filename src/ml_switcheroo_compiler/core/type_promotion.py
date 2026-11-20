"""Declarative type promotion lattice and precision rules for ml-switcheroo-compiler."""

from __future__ import annotations

import os
from collections.abc import Mapping

import yaml

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import DTypePromotionError

_PROMOTION_TABLE: dict[tuple[DType, DType], DType] = {}
_PROMOTION_LATTICE = _PROMOTION_TABLE


def _load_declarative_promotion_lattice() -> None:
    """Load declarative type promotion rules from ``type_promotion_rules.yaml`` as single source of truth."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "ops", "type_promotion_rules.yaml")
    if not os.path.exists(yaml_path):
        return
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    lattice = data.get("lattice", {})
    if not isinstance(lattice, Mapping):
        return
    for t1_str, targets in lattice.items():
        if not isinstance(targets, Mapping):
            continue
        try:
            dt1 = DType(str(t1_str))
        except ValueError:
            continue
        for t2_str, res_str in targets.items():
            try:
                dt2 = DType(str(t2_str))
                dt_res = DType(str(res_str))
                _PROMOTION_TABLE[(dt1, dt2)] = dt_res
                _PROMOTION_TABLE[(dt2, dt1)] = dt_res
            except ValueError:
                continue


_load_declarative_promotion_lattice()


def _clamp_x64(dtype: DType) -> DType:
    """Downcast 64-bit scalar data types to 32-bit equivalents when ``config.enable_x64`` is False.

    Args:
        dtype (DType): Candidate promoted tensor data type.

    Returns:
        DType: Unchanged 64-bit ``dtype`` when ``config.enable_x64`` is True, or its 32-bit counterpart otherwise.
    """
    if config.enable_x64:
        return dtype
    downcast: dict[DType, DType] = {
        DType.Float64: DType.Float32,
        DType.Int64: DType.Int32,
        DType.Complex128: DType.Complex64,
    }
    return downcast.get(dtype, dtype)


def promote_types(dtype1: DType | str, dtype2: DType | str) -> DType:
    """Determine the promoted ``DType`` according to the canonical promotion lattice.

    Args:
        dtype1 (DType | str): First operand data type.
        dtype2 (DType | str): Second operand data type.

    Returns:
        DType: Promoted result data type after applying ``config.enable_x64`` precision rules.

    Raises:
        DTypePromotionError: If ``dtype1`` and ``dtype2`` have no valid common promotion target.
    """
    if dtype1 == dtype2 and isinstance(dtype1, DType) and isinstance(dtype2, DType):
        return _clamp_x64(dtype1)

    res = (
        None
        if (not isinstance(dtype1, DType) or not isinstance(dtype2, DType))
        else (_PROMOTION_TABLE.get((dtype1, dtype2)) or _PROMOTION_TABLE.get((dtype2, dtype1)))
    )
    if res is None:
        raise DTypePromotionError(f"Cannot promote types: {dtype1} and {dtype2}")

    return _clamp_x64(res)
