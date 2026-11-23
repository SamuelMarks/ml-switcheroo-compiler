"""Algebraic bounded lattice and collective transition inference for SPMD tensor sharding."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from ml_switcheroo_ir import BoundedLattice, Eq, HashableWitness, Monoid, Pretty, Semigroup


def _canonicalize_axes(axes: Sequence[str | None]) -> tuple[str | None, ...]:
    """Strip trailing replicated (``None``) dimensions so replicated specs share a unique canonical form.

    Args:
        axes (Sequence[str | None]): Per-dimension mesh axis names or ``None`` for replicated dimensions.

    Returns:
        tuple[str | None, ...]: Canonicalized axis tuple without trailing ``None`` entries.
    """
    trimmed: list[str | None] = list(axes)
    while trimmed and trimmed[-1] is None:
        trimmed.pop()
    return tuple(trimmed)


@dataclasses.dataclass(frozen=True)
class ShardingLatticeElem:
    """Element of the SPMD axis-sharding bounded lattice ``(⊥=Replicated, Axes, ⊤=Conflict)``."""

    axes: tuple[str | None, ...] = ()
    is_conflict: bool = False

    @staticmethod
    def bottom() -> ShardingLatticeElem:
        """Return the fully replicated bottom element ``⊥``.

        Returns:
            ShardingLatticeElem: Fully replicated sharding element.
        """
        return ShardingLatticeElem(axes=(), is_conflict=False)

    @staticmethod
    def top() -> ShardingLatticeElem:
        """Return the conflicting top element ``⊤`` requiring collective resharding.

        Returns:
            ShardingLatticeElem: Conflicting top sharding element.
        """
        return ShardingLatticeElem(axes=(), is_conflict=True)

    @staticmethod
    def from_axes(axes: Sequence[str | None]) -> ShardingLatticeElem:
        """Construct a canonical ``ShardingLatticeElem`` from per-dimension mesh axis mappings.

        Args:
            axes (Sequence[str | None]): Sequence of mesh axis names or ``None`` per tensor dimension.

        Returns:
            ShardingLatticeElem: Canonical non-conflicting sharding lattice element.
        """
        return ShardingLatticeElem(axes=_canonicalize_axes(axes), is_conflict=False)


def _join_sharding(lhs: ShardingLatticeElem, rhs: ShardingLatticeElem) -> ShardingLatticeElem:
    """Compute least upper bound (``join``) of two SPMD sharding lattice elements.

    Args:
        lhs (ShardingLatticeElem): Left sharding specification.
        rhs (ShardingLatticeElem): Right sharding specification.

    Returns:
        ShardingLatticeElem: Combined sharding or ``ShardingLatticeElem.top()`` on axis conflict.
    """
    if lhs.is_conflict or rhs.is_conflict:
        return ShardingLatticeElem.top()
    max_len = max(len(lhs.axes), len(rhs.axes))
    merged: list[str | None] = []
    for idx in range(max_len):
        a_ax = lhs.axes[idx] if idx < len(lhs.axes) else None
        b_ax = rhs.axes[idx] if idx < len(rhs.axes) else None
        if a_ax is None:
            merged.append(b_ax)
        elif b_ax is None or a_ax == b_ax:
            merged.append(a_ax)
        else:
            return ShardingLatticeElem.top()
    return ShardingLatticeElem.from_axes(merged)


def _meet_sharding(lhs: ShardingLatticeElem, rhs: ShardingLatticeElem) -> ShardingLatticeElem:
    """Compute greatest lower bound (``meet``) of two SPMD sharding lattice elements.

    Args:
        lhs (ShardingLatticeElem): Left sharding specification.
        rhs (ShardingLatticeElem): Right sharding specification.

    Returns:
        ShardingLatticeElem: Shared sharding common to both operands.
    """
    if lhs.is_conflict:
        return rhs
    if rhs.is_conflict:
        return lhs
    min_len = min(len(lhs.axes), len(rhs.axes))
    common: list[str | None] = [lhs.axes[idx] if lhs.axes[idx] == rhs.axes[idx] else None for idx in range(min_len)]
    return ShardingLatticeElem.from_axes(common)


def _render_sharding(elem: ShardingLatticeElem) -> str:
    """Render a ``ShardingLatticeElem`` as a concise human-readable layout string.

    Args:
        elem (ShardingLatticeElem): Sharding lattice element to render.

    Returns:
        str: Formatted layout string.
    """
    if elem.is_conflict:
        return "Sharding(⊤)"
    if not elem.axes:
        return "Sharding(⊥:replicated)"
    parts = [ax if ax is not None else "*" for ax in elem.axes]
    return f"Sharding({', '.join(parts)})"


def infer_collective_transition(src: ShardingLatticeElem, dst: ShardingLatticeElem) -> str | None:
    """Infer the canonical collective communication primitive needed to transition from ``src`` to ``dst``.

    Args:
        src (ShardingLatticeElem): Source tensor sharding lattice state.
        dst (ShardingLatticeElem): Target tensor sharding lattice state.

    Returns:
        str | None: ``'AllGather'``, ``'DynamicSlice'``, ``'AllToAll'``, ``'AllReduce'``, or ``None`` if identical.
    """
    if src == dst:
        return None
    if src.is_conflict or dst.is_conflict:
        return "AllReduce"
    joined = _join_sharding(src, dst)
    if joined.is_conflict:
        return "AllToAll"
    if joined == src:
        return "AllGather"
    if joined == dst:
        return "DynamicSlice"
    return "AllToAll"


COMPILER_SHARDING_EQ: Eq[ShardingLatticeElem] = Eq(eq=lambda a, b: a == b)
COMPILER_SHARDING_HASH: HashableWitness[ShardingLatticeElem] = HashableWitness(
    eq_witness=COMPILER_SHARDING_EQ,
    hash_value=hash,
)
COMPILER_SHARDING_PRETTY: Pretty[ShardingLatticeElem] = Pretty(render=_render_sharding)
COMPILER_SHARDING_LATTICE: BoundedLattice[ShardingLatticeElem] = BoundedLattice(
    eq_witness=COMPILER_SHARDING_EQ,
    bottom=ShardingLatticeElem.bottom,
    top=ShardingLatticeElem.top,
    join=_join_sharding,
    meet=_meet_sharding,
)
COMPILER_SHARDING_MONOID: Monoid[ShardingLatticeElem] = Monoid(
    empty=ShardingLatticeElem.bottom,
    semigroup=Semigroup(combine=_join_sharding),
)

__all__ = [
    "COMPILER_SHARDING_EQ",
    "COMPILER_SHARDING_HASH",
    "COMPILER_SHARDING_LATTICE",
    "COMPILER_SHARDING_MONOID",
    "COMPILER_SHARDING_PRETTY",
    "ShardingLatticeElem",
    "infer_collective_transition",
]
