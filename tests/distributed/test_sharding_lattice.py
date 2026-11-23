"""Unit tests for SPMD ShardingLatticeElem BoundedLattice laws and collective transition inference."""

from __future__ import annotations

from ml_switcheroo_compiler.distributed import (
    COMPILER_SHARDING_EQ,
    COMPILER_SHARDING_HASH,
    COMPILER_SHARDING_LATTICE,
    COMPILER_SHARDING_MONOID,
    COMPILER_SHARDING_PRETTY,
    ShardingLatticeElem,
    infer_collective_transition,
)


def test_sharding_bounded_lattice_laws_and_pretty() -> None:
    """Verify BoundedLattice axioms, canonicalization, Pretty rendering, and HashableWitness on ShardingLatticeElem."""
    lat = COMPILER_SHARDING_LATTICE
    bot = ShardingLatticeElem.bottom()
    top = ShardingLatticeElem.top()
    assert ShardingLatticeElem.from_axes([None, None]) == bot
    assert COMPILER_SHARDING_MONOID.empty() == bot

    s_dp = ShardingLatticeElem.from_axes(["dp", None])
    s_tp = ShardingLatticeElem.from_axes([None, "tp"])
    s_both = ShardingLatticeElem.from_axes(["dp", "tp"])
    s_alt = ShardingLatticeElem.from_axes(["mp", None])

    assert COMPILER_SHARDING_PRETTY.render(bot) == "Sharding(⊥:replicated)"
    assert COMPILER_SHARDING_PRETTY.render(top) == "Sharding(⊤)"
    assert COMPILER_SHARDING_PRETTY.render(s_tp) == "Sharding(*, tp)"

    samples = [bot, top, s_dp, s_tp, s_both, s_alt]
    for a in samples:
        assert COMPILER_SHARDING_EQ.eq(a, a)
        assert COMPILER_SHARDING_HASH.hash_value(a) == hash(a)
        assert lat.join(bot, a) == a
        assert lat.meet(bot, a) == bot
        assert lat.join(top, a) == top
        assert lat.meet(top, a) == a
        for b in samples:
            assert lat.join(a, b) == lat.join(b, a)
            assert lat.meet(a, b) == lat.meet(b, a)
            assert lat.join(a, lat.meet(a, b)) == a
            assert lat.meet(a, lat.join(a, b)) == a
            for c in samples:
                assert lat.join(lat.join(a, b), c) == lat.join(a, lat.join(b, c))
                assert lat.meet(lat.meet(a, b), c) == lat.meet(a, lat.meet(b, c))


def test_infer_collective_transition_all_branches() -> None:
    """Verify collective primitive selection across all lattice transitions."""
    bot = ShardingLatticeElem.bottom()
    top = ShardingLatticeElem.top()
    s_dp = ShardingLatticeElem.from_axes(["dp", None])
    s_tp = ShardingLatticeElem.from_axes([None, "tp"])
    s_conflict_axis = ShardingLatticeElem.from_axes(["tp", None])

    assert infer_collective_transition(s_dp, s_dp) is None
    assert infer_collective_transition(top, s_dp) == "AllReduce"
    assert infer_collective_transition(s_dp, top) == "AllReduce"
    assert infer_collective_transition(s_dp, bot) == "AllGather"
    assert infer_collective_transition(bot, s_dp) == "DynamicSlice"
    assert infer_collective_transition(s_dp, s_conflict_axis) == "AllToAll"
    assert infer_collective_transition(s_dp, s_tp) == "AllToAll"
