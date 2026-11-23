"""Executable algebraic law verifiers for Haskell-style typeclass witness dictionaries."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_ir import (
    BoundedLattice,
    Codec,
    Eq,
    Err,
    Invalid,
    Monoid,
    Ord,
    Ordering,
    Ring,
    Semigroup,
    Semiring,
    Valid,
    Validation,
)

_T = typing.TypeVar("_T")
_W = typing.TypeVar("_W")


def _finalize_violations(violations: list[str]) -> Validation[tuple[str, ...], None]:
    """Convert accumulated law violation strings into ``Valid(None)`` or ``Invalid(tuple(violations))``.

    Args:
        violations (list[str]): Collected law violation messages.

    Returns:
        Validation[tuple[str, ...], None]: ``Valid(None)`` when empty, else ``Invalid`` tuple.
    """
    if violations:
        return Invalid(tuple(violations))
    return Valid(None)


def verify_eq_laws(
    eq_witness: Eq[_T],
    samples: Sequence[_T],
    label: str = "Eq",
) -> Validation[tuple[str, ...], None]:
    """Verify reflexivity, symmetry, and transitivity of an ``Eq[_T]`` witness across ``samples``.

    Args:
        eq_witness (Eq[_T]): Equality typeclass witness dictionary.
        samples (Sequence[_T]): Representative domain values.
        label (str): Human-readable witness identifier for diagnostics.

    Returns:
        Validation[tuple[str, ...], None]: ``Valid(None)`` or ``Invalid`` with all law violations.
    """
    violations: list[str] = []
    for a in samples:
        if not eq_witness.eq(a, a):
            violations.append(f"{label}: reflexivity failed for {a!r}")
        for b in samples:
            if eq_witness.eq(a, b) != eq_witness.eq(b, a):
                violations.append(f"{label}: symmetry failed for ({a!r}, {b!r})")
            for c in samples:
                if eq_witness.eq(a, b) and eq_witness.eq(b, c) and not eq_witness.eq(a, c):
                    violations.append(f"{label}: transitivity failed for ({a!r}, {b!r}, {c!r})")
    return _finalize_violations(violations)


def verify_ord_laws(
    ord_witness: Ord[_T],
    samples: Sequence[_T],
    label: str = "Ord",
) -> Validation[tuple[str, ...], None]:
    """Verify ``Eq`` consistency, antisymmetry/duality, and transitivity of an ``Ord[_T]`` witness.

    Args:
        ord_witness (Ord[_T]): Total ordering typeclass witness dictionary.
        samples (Sequence[_T]): Representative domain values.
        label (str): Human-readable witness identifier for diagnostics.

    Returns:
        Validation[tuple[str, ...], None]: ``Valid(None)`` or ``Invalid`` with all law violations.
    """
    violations: list[str] = []
    eq_res = verify_eq_laws(ord_witness.eq_witness, samples, label=f"{label}.eq")
    if isinstance(eq_res, Invalid):
        violations.extend(eq_res.errors)
    for a in samples:
        for b in samples:
            cab = ord_witness.compare(a, b)
            cba = ord_witness.compare(b, a)
            if (cab == Ordering.EQ) != ord_witness.eq_witness.eq(a, b):
                violations.append(f"{label}: Eq consistency failed for ({a!r}, {b!r})")
            if (cab == Ordering.LT and cba != Ordering.GT) or (cab == Ordering.GT and cba != Ordering.LT):
                violations.append(f"{label}: antisymmetry/duality failed for ({a!r}, {b!r})")
            for c in samples:
                cbc = ord_witness.compare(b, c)
                cac = ord_witness.compare(a, c)
                if cab == Ordering.LT and cbc == Ordering.LT and cac != Ordering.LT:
                    violations.append(f"{label}: LT transitivity failed for ({a!r}, {b!r}, {c!r})")
    return _finalize_violations(violations)


def verify_semigroup_laws(
    sg: Semigroup[_T],
    eq_witness: Eq[_T],
    samples: Sequence[_T],
    label: str = "Semigroup",
) -> Validation[tuple[str, ...], None]:
    """Verify associativity ``combine(combine(a, b), c) == combine(a, combine(b, c))`` for ``sg``.

    Args:
        sg (Semigroup[_T]): Semigroup witness dictionary.
        eq_witness (Eq[_T]): Equality witness for carrier ``_T``.
        samples (Sequence[_T]): Representative domain values.
        label (str): Human-readable witness identifier for diagnostics.

    Returns:
        Validation[tuple[str, ...], None]: ``Valid(None)`` or ``Invalid`` with all law violations.
    """
    violations: list[str] = []
    for a in samples:
        for b in samples:
            for c in samples:
                lhs = sg.combine(sg.combine(a, b), c)
                rhs = sg.combine(a, sg.combine(b, c))
                if not eq_witness.eq(lhs, rhs):
                    violations.append(f"{label}: associativity failed for ({a!r}, {b!r}, {c!r})")
    return _finalize_violations(violations)


def verify_monoid_laws(
    monoid: Monoid[_T],
    eq_witness: Eq[_T],
    samples: Sequence[_T],
    label: str = "Monoid",
) -> Validation[tuple[str, ...], None]:
    """Verify associativity, left identity, and right identity for ``monoid``.

    Args:
        monoid (Monoid[_T]): Monoid witness dictionary.
        eq_witness (Eq[_T]): Equality witness for carrier ``_T``.
        samples (Sequence[_T]): Representative domain values.
        label (str): Human-readable witness identifier for diagnostics.

    Returns:
        Validation[tuple[str, ...], None]: ``Valid(None)`` or ``Invalid`` with all law violations.
    """
    violations: list[str] = []
    sg_res = verify_semigroup_laws(monoid.semigroup, eq_witness, samples, label=label)
    if isinstance(sg_res, Invalid):
        violations.extend(sg_res.errors)
    ident = monoid.empty()
    for a in samples:
        if not eq_witness.eq(monoid.semigroup.combine(ident, a), a):
            violations.append(f"{label}: left identity failed for {a!r}")
        if not eq_witness.eq(monoid.semigroup.combine(a, ident), a):
            violations.append(f"{label}: right identity failed for {a!r}")
    return _finalize_violations(violations)


def verify_semiring_laws(
    semiring: Semiring[_T],
    eq_witness: Eq[_T],
    samples: Sequence[_T],
    label: str = "Semiring",
) -> Validation[tuple[str, ...], None]:
    """Verify additive commutative monoid, multiplicative monoid, distributivity, and zero annihilation.

    Args:
        semiring (Semiring[_T]): Semiring witness dictionary.
        eq_witness (Eq[_T]): Equality witness for carrier ``_T``.
        samples (Sequence[_T]): Representative domain values.
        label (str): Human-readable witness identifier for diagnostics.

    Returns:
        Validation[tuple[str, ...], None]: ``Valid(None)`` or ``Invalid`` with all law violations.
    """
    violations: list[str] = []
    add_monoid: Monoid[_T] = Monoid(empty=semiring.zero, semigroup=Semigroup(combine=semiring.add))
    mul_monoid: Monoid[_T] = Monoid(empty=semiring.one, semigroup=Semigroup(combine=semiring.mul))

    add_res = verify_monoid_laws(add_monoid, eq_witness, samples, label=f"{label}.add")
    if isinstance(add_res, Invalid):
        violations.extend(add_res.errors)
    mul_res = verify_monoid_laws(mul_monoid, eq_witness, samples, label=f"{label}.mul")
    if isinstance(mul_res, Invalid):
        violations.extend(mul_res.errors)

    z = semiring.zero()
    for a in samples:
        if not eq_witness.eq(semiring.mul(z, a), z) or not eq_witness.eq(semiring.mul(a, z), z):
            violations.append(f"{label}: zero annihilation failed for {a!r}")
        for b in samples:
            if not eq_witness.eq(semiring.add(a, b), semiring.add(b, a)):
                violations.append(f"{label}: additive commutativity failed for ({a!r}, {b!r})")
            for c in samples:
                ld_lhs = semiring.mul(a, semiring.add(b, c))
                ld_rhs = semiring.add(semiring.mul(a, b), semiring.mul(a, c))
                if not eq_witness.eq(ld_lhs, ld_rhs):
                    violations.append(f"{label}: left distributivity failed for ({a!r}, {b!r}, {c!r})")
                rd_lhs = semiring.mul(semiring.add(a, b), c)
                rd_rhs = semiring.add(semiring.mul(a, c), semiring.mul(b, c))
                if not eq_witness.eq(rd_lhs, rd_rhs):
                    violations.append(f"{label}: right distributivity failed for ({a!r}, {b!r}, {c!r})")
    return _finalize_violations(violations)


def verify_ring_laws(
    ring: Ring[_T],
    eq_witness: Eq[_T],
    samples: Sequence[_T],
    label: str = "Ring",
) -> Validation[tuple[str, ...], None]:
    """Verify semiring laws plus additive inverse and subtraction consistency for ``ring``.

    Args:
        ring (Ring[_T]): Ring witness dictionary.
        eq_witness (Eq[_T]): Equality witness for carrier ``_T``.
        samples (Sequence[_T]): Representative domain values.
        label (str): Human-readable witness identifier for diagnostics.

    Returns:
        Validation[tuple[str, ...], None]: ``Valid(None)`` or ``Invalid`` with all law violations.
    """
    violations: list[str] = []
    sr_res = verify_semiring_laws(ring.semiring, eq_witness, samples, label=label)
    if isinstance(sr_res, Invalid):
        violations.extend(sr_res.errors)
    z = ring.semiring.zero()
    for a in samples:
        neg_a = ring.neg(a)
        if not eq_witness.eq(ring.semiring.add(a, neg_a), z) or not eq_witness.eq(ring.semiring.add(neg_a, a), z):
            violations.append(f"{label}: additive inverse failed for {a!r}")
        for b in samples:
            if not eq_witness.eq(ring.sub(a, b), ring.semiring.add(a, ring.neg(b))):
                violations.append(f"{label}: sub consistency failed for ({a!r}, {b!r})")
    return _finalize_violations(violations)


def verify_bounded_lattice_laws(
    lattice: BoundedLattice[_T],
    samples: Sequence[_T],
    label: str = "BoundedLattice",
) -> Validation[tuple[str, ...], None]:
    """Verify idempotency, commutativity, associativity, absorption, and top/bottom bounds for ``lattice``.

    Args:
        lattice (BoundedLattice[_T]): Bounded lattice witness dictionary.
        samples (Sequence[_T]): Representative lattice carrier elements.
        label (str): Human-readable witness identifier for diagnostics.

    Returns:
        Validation[tuple[str, ...], None]: ``Valid(None)`` or ``Invalid`` with all law violations.
    """
    violations: list[str] = []
    eq = lattice.eq_witness.eq
    bot = lattice.bottom()
    top = lattice.top()

    join_sg_res = verify_semigroup_laws(Semigroup(combine=lattice.join), lattice.eq_witness, samples, f"{label}.join")
    if isinstance(join_sg_res, Invalid):
        violations.extend(join_sg_res.errors)
    meet_sg_res = verify_semigroup_laws(Semigroup(combine=lattice.meet), lattice.eq_witness, samples, f"{label}.meet")
    if isinstance(meet_sg_res, Invalid):
        violations.extend(meet_sg_res.errors)

    for a in samples:
        if not eq(lattice.join(a, a), a) or not eq(lattice.meet(a, a), a):
            violations.append(f"{label}: idempotency failed for {a!r}")
        if not eq(lattice.join(bot, a), a) or not eq(lattice.meet(bot, a), bot):
            violations.append(f"{label}: bottom bound law failed for {a!r}")
        if not eq(lattice.join(top, a), top) or not eq(lattice.meet(top, a), a):
            violations.append(f"{label}: top bound law failed for {a!r}")
        for b in samples:
            if not eq(lattice.join(a, b), lattice.join(b, a)) or not eq(lattice.meet(a, b), lattice.meet(b, a)):
                violations.append(f"{label}: commutativity failed for ({a!r}, {b!r})")
            if not eq(lattice.join(a, lattice.meet(a, b)), a) or not eq(lattice.meet(a, lattice.join(a, b)), a):
                violations.append(f"{label}: absorption failed for ({a!r}, {b!r})")
    return _finalize_violations(violations)


def verify_codec_roundtrip_laws(
    codec: Codec[_W, _T],
    eq_witness: Eq[_T],
    samples: Sequence[_T],
    label: str = "Codec",
) -> Validation[tuple[str, ...], None]:
    """Verify roundtrip law ``decode(encode(a)) == Ok(a)`` across ``samples``.

    Args:
        codec (Codec[_W, _T]): Bidirectional codec witness dictionary.
        eq_witness (Eq[_T]): Equality witness for domain type ``_T``.
        samples (Sequence[_T]): Representative domain values.
        label (str): Human-readable witness identifier for diagnostics.

    Returns:
        Validation[tuple[str, ...], None]: ``Valid(None)`` or ``Invalid`` with all roundtrip violations.
    """
    violations: list[str] = []
    for a in samples:
        wire = codec.encode(a)
        decoded = codec.decode(wire)
        if isinstance(decoded, Err):
            violations.append(f"{label}: decode(encode({a!r})) returned Err({decoded.error!r})")
        elif not eq_witness.eq(decoded.value, a):
            violations.append(f"{label}: roundtrip mismatch {decoded.value!r} != {a!r}")
    return _finalize_violations(violations)


__all__ = [
    "verify_bounded_lattice_laws",
    "verify_codec_roundtrip_laws",
    "verify_eq_laws",
    "verify_monoid_laws",
    "verify_ord_laws",
    "verify_ring_laws",
    "verify_semigroup_laws",
    "verify_semiring_laws",
]
