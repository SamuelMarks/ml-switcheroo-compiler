"""Declarative type promotion lattice and precision rules for ml-switcheroo-compiler."""

from __future__ import annotations

import dataclasses
import enum
import os
from collections.abc import Mapping, Sequence

import yaml
from ml_switcheroo_ir import BoundedLattice, Eq, Err, HashableWitness, Monoid, Ok, Pretty, Result, Semigroup

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType, parse_dtype_result
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
        else (_PROMOTION_TABLE.get((dtype1, dt2 := dtype2)) or _PROMOTION_TABLE.get((dt2, dtype1)))
    )
    if res is None:
        raise DTypePromotionError(f"Cannot promote types: {dtype1} and {dtype2}")

    return _clamp_x64(res)


class PromotionLatticeKind(str, enum.Enum):
    """Discriminator tag for elements of the bounded ``DType`` promotion lattice."""

    BOTTOM = "bottom"
    DTYPE = "dtype"
    TOP = "top"


@dataclasses.dataclass(frozen=True)
class PromotionLatticeElem:
    """Element of the bounded algebraic ``DType`` precision lattice ``(⊥, DType, ⊤)``."""

    kind: PromotionLatticeKind
    dtype: DType | None = None

    @staticmethod
    def bottom() -> PromotionLatticeElem:
        """Return the bottom identity element ``⊥`` of the promotion lattice.

        Returns:
            PromotionLatticeElem: Bottom lattice element.
        """
        return PromotionLatticeElem(kind=PromotionLatticeKind.BOTTOM, dtype=None)

    @staticmethod
    def top() -> PromotionLatticeElem:
        """Return the top conflict element ``⊤`` of the promotion lattice.

        Returns:
            PromotionLatticeElem: Top conflict lattice element.
        """
        return PromotionLatticeElem(kind=PromotionLatticeKind.TOP, dtype=None)

    @staticmethod
    def of(dtype: DType) -> PromotionLatticeElem:
        """Wrap a concrete ``DType`` as a lattice element.

        Args:
            dtype (DType): Canonical ``DType`` enum member.

        Returns:
            PromotionLatticeElem: Concrete ``DType`` lattice element.
        """
        return PromotionLatticeElem(kind=PromotionLatticeKind.DTYPE, dtype=dtype)


_HASSE_EDGES: tuple[tuple[DType, DType], ...] = (
    (DType.bool, DType.uint4),
    (DType.bool, DType.int2),
    (DType.int2, DType.int4),
    (DType.bool, DType.float8_e4m3b11fnuz),
    (DType.bool, DType.float8_e4m3fn),
    (DType.bool, DType.fp8_e4m3fnuz),
    (DType.bool, DType.float8_e5m2),
    (DType.bool, DType.fp8_e5m2fnuz),
    (DType.uint4, DType.uint8),
    (DType.uint8, DType.uint16),
    (DType.uint16, DType.uint32),
    (DType.uint32, DType.uint64),
    (DType.int4, DType.int8),
    (DType.int8, DType.int16),
    (DType.int16, DType.int32),
    (DType.int32, DType.int64),
    (DType.uint4, DType.int8),
    (DType.uint8, DType.int16),
    (DType.uint16, DType.int32),
    (DType.uint32, DType.int64),
    (DType.float8_e4m3b11fnuz, DType.float16),
    (DType.float8_e4m3fn, DType.float16),
    (DType.fp8_e4m3fnuz, DType.float16),
    (DType.float8_e5m2, DType.float16),
    (DType.fp8_e5m2fnuz, DType.float16),
    (DType.int8, DType.float16),
    (DType.float16, DType.bfloat16),
    (DType.bfloat16, DType.float32),
    (DType.int16, DType.float32),
    (DType.float32, DType.float64),
    (DType.int64, DType.float64),
    (DType.uint64, DType.float64),
    (DType.float32, DType.complex64),
    (DType.float64, DType.complex128),
    (DType.complex64, DType.complex128),
)


def _build_lattice_tables() -> tuple[
    dict[tuple[PromotionLatticeElem, PromotionLatticeElem], PromotionLatticeElem],
    dict[tuple[PromotionLatticeElem, PromotionLatticeElem], PromotionLatticeElem],
]:
    """Precompute exact Hasse-diagram LUB (``join``) and GLB (``meet``) lookup tables.

    Returns:
        tuple[
            dict[tuple[PromotionLatticeElem, PromotionLatticeElem], PromotionLatticeElem],
            dict[tuple[PromotionLatticeElem, PromotionLatticeElem], PromotionLatticeElem],
        ]: Pair of ``(join_table, meet_table)`` over ``PromotionLatticeElem``.
    """
    bot = PromotionLatticeElem.bottom()
    top = PromotionLatticeElem.top()
    elems: list[PromotionLatticeElem] = [bot, top, *(PromotionLatticeElem.of(dt) for dt in DType)]
    le_map: dict[PromotionLatticeElem, set[PromotionLatticeElem]] = {el: {el, top} for el in elems}
    for el in elems:
        le_map[bot].add(el)
    for u_dt, v_dt in _HASSE_EDGES:
        le_map[PromotionLatticeElem.of(u_dt)].add(PromotionLatticeElem.of(v_dt))

    changed = True
    while changed:
        changed = False
        for x in elems:
            for y in tuple(le_map[x]):
                for z in tuple(le_map[y]):
                    if z not in le_map[x]:
                        le_map[x].add(z)
                        changed = True

    join_tbl: dict[tuple[PromotionLatticeElem, PromotionLatticeElem], PromotionLatticeElem] = {}
    meet_tbl: dict[tuple[PromotionLatticeElem, PromotionLatticeElem], PromotionLatticeElem] = {}
    for a in elems:
        for b in elems:
            ubs = [u for u in elems if u in le_map[a] and u in le_map[b]]
            lub = next(u for u in ubs if all(v in le_map[u] for v in ubs))
            lbs = [lo for lo in elems if a in le_map[lo] and b in le_map[lo]]
            glb = next(lo for lo in lbs if all(lo in le_map[v] for v in lbs))
            join_tbl[(a, b)] = lub
            meet_tbl[(a, b)] = glb
    return join_tbl, meet_tbl


_LATTICE_JOIN_TABLE, _LATTICE_MEET_TABLE = _build_lattice_tables()


def _render_promotion_elem(elem: PromotionLatticeElem) -> str:
    """Render a ``PromotionLatticeElem`` as a deterministic diagnostic string.

    Args:
        elem (PromotionLatticeElem): Lattice element to format.

    Returns:
        str: Formatted string representation.
    """
    if elem.kind == PromotionLatticeKind.BOTTOM:
        return "⊥"
    if elem.kind == PromotionLatticeKind.TOP or elem.dtype is None:
        return "⊤"
    return elem.dtype.value


COMPILER_PROMOTION_ELEM_EQ: Eq[PromotionLatticeElem] = Eq(eq=lambda a, b: a == b)
COMPILER_PROMOTION_ELEM_HASH: HashableWitness[PromotionLatticeElem] = HashableWitness(
    eq_witness=COMPILER_PROMOTION_ELEM_EQ,
    hash_value=hash,
)
COMPILER_PROMOTION_ELEM_PRETTY: Pretty[PromotionLatticeElem] = Pretty(render=_render_promotion_elem)
COMPILER_DTYPE_PROMOTION_LATTICE: BoundedLattice[PromotionLatticeElem] = BoundedLattice(
    eq_witness=COMPILER_PROMOTION_ELEM_EQ,
    bottom=PromotionLatticeElem.bottom,
    top=PromotionLatticeElem.top,
    join=lambda a, b: _LATTICE_JOIN_TABLE[(a, b)],
    meet=lambda a, b: _LATTICE_MEET_TABLE[(a, b)],
)
COMPILER_DTYPE_PROMOTION_MONOID: Monoid[PromotionLatticeElem] = Monoid(
    empty=PromotionLatticeElem.bottom,
    semigroup=Semigroup(combine=COMPILER_DTYPE_PROMOTION_LATTICE.join),
)


def promote_types_result(dtype1: DType | str, dtype2: DType | str) -> Result[DTypePromotionError, DType]:
    """Promote two ``DType`` or string aliases into an explicit ``Result`` without raising.

    Args:
        dtype1 (DType | str): First operand data type or string alias.
        dtype2 (DType | str): Second operand data type or string alias.

    Returns:
        Result[DTypePromotionError, DType]: ``Ok(DType)`` on valid promotion or ``Err(DTypePromotionError)``.
    """
    parsed1 = parse_dtype_result(dtype1)
    if isinstance(parsed1, Err):
        return Err(DTypePromotionError(f"Cannot promote types: {dtype1} and {dtype2}"))
    parsed2 = parse_dtype_result(dtype2)
    if isinstance(parsed2, Err):
        return Err(DTypePromotionError(f"Cannot promote types: {dtype1} and {dtype2}"))
    try:
        return Ok(promote_types(parsed1.value, parsed2.value))
    except DTypePromotionError as exc:
        return Err(exc)


def promote_dtypes_fold(dtypes: Sequence[DType | str]) -> Result[DTypePromotionError, DType]:
    """Fold N operand data types over ``COMPILER_DTYPE_PROMOTION_MONOID`` in an order-independent manner.

    Args:
        dtypes (Sequence[DType | str]): Non-empty sequence of ``DType`` values or string aliases.

    Returns:
        Result[DTypePromotionError, DType]: ``Ok(DType)`` with x64 clamping applied, or ``Err(DTypePromotionError)``.
    """
    if not dtypes:
        return Err(DTypePromotionError("Cannot fold empty sequence of dtypes."))
    acc: PromotionLatticeElem = COMPILER_DTYPE_PROMOTION_MONOID.empty()
    for raw_dt in dtypes:
        parsed = parse_dtype_result(raw_dt)
        if isinstance(parsed, Err):
            return Err(DTypePromotionError(f"Invalid dtype in promotion fold: {raw_dt!r}"))
        acc = COMPILER_DTYPE_PROMOTION_MONOID.semigroup.combine(acc, PromotionLatticeElem.of(parsed.value))
    if acc.kind == PromotionLatticeKind.TOP or acc.dtype is None:
        return Err(DTypePromotionError(f"Incompatible dtypes in promotion fold: {list(dtypes)!r}"))
    return Ok(_clamp_x64(acc.dtype))
