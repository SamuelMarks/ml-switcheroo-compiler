"""Comprehensive positive and negative tests for algebraic law verifiers and scripts/verify_typeclasses.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from ml_switcheroo_ir import (
    BoundedLattice,
    Codec,
    CodecError,
    Eq,
    Err,
    Invalid,
    Monoid,
    Ok,
    Ord,
    Ordering,
    Ring,
    Semigroup,
    Semiring,
    Valid,
)
from ml_switcheroo_ir.typeclasses import INT_EQ, INT_ORD, INT_RING, INT_SEMIRING, INT_SUM_MONOID

from ml_switcheroo_compiler.ir import (
    verify_bounded_lattice_laws,
    verify_codec_roundtrip_laws,
    verify_eq_laws,
    verify_monoid_laws,
    verify_ord_laws,
    verify_ring_laws,
    verify_semigroup_laws,
    verify_semiring_laws,
)
from scripts.verify_typeclasses import main, verify_compiler_algebraic_laws


def test_all_law_verifiers_accept_lawful_instances() -> None:
    """Verify positive Valid(None) paths across Eq, Ord, Semigroup, Monoid, Semiring, Ring, Lattice, and Codec."""
    nums = (-2, -1, 0, 1, 2)
    assert isinstance(verify_eq_laws(INT_EQ, nums), Valid)
    assert isinstance(verify_ord_laws(INT_ORD, nums), Valid)
    assert isinstance(verify_semigroup_laws(INT_SUM_MONOID.semigroup, INT_EQ, nums), Valid)
    assert isinstance(verify_monoid_laws(INT_SUM_MONOID, INT_EQ, nums), Valid)
    assert isinstance(verify_semiring_laws(INT_SEMIRING, INT_EQ, nums), Valid)
    assert isinstance(verify_ring_laws(INT_RING, INT_EQ, nums), Valid)

    bool_lat: BoundedLattice[bool] = BoundedLattice(
        eq_witness=Eq(eq=lambda a, b: a == b),
        bottom=lambda: False,
        top=lambda: True,
        join=lambda a, b: a or b,
        meet=lambda a, b: a and b,
    )
    assert isinstance(verify_bounded_lattice_laws(bool_lat, (False, True)), Valid)

    int_codec: Codec[str, int] = Codec(encode=str, decode=lambda s: Ok(int(s)))
    assert isinstance(verify_codec_roundtrip_laws(int_codec, INT_EQ, nums), Valid)
    assert verify_compiler_algebraic_laws() == []


def test_all_law_verifiers_detect_unlawful_witnesses() -> None:
    """Verify every violation branch across Eq, Ord, Semigroup, Monoid, Semiring, Ring, Lattice, and Codec."""
    # Unlawful Eq: non-reflexive, asymmetric, non-transitive (abs(a - b) <= 1 is reflexive/symmetric but not transitive)
    non_refl: Eq[int] = Eq(eq=lambda a, b: a == 1 and b == 2)
    res_nr = verify_eq_laws(non_refl, (1, 2))
    assert isinstance(res_nr, Invalid) and any("reflexivity" in e for e in res_nr.errors)
    assert any("symmetry" in e for e in res_nr.errors)

    non_trans: Eq[int] = Eq(eq=lambda a, b: abs(a - b) <= 1)
    res_nt = verify_eq_laws(non_trans, (0, 1, 2))
    assert isinstance(res_nt, Invalid) and any("transitivity" in e for e in res_nt.errors)

    # Unlawful Ord: broken eq_witness, broken Eq consistency, broken antisymmetry, broken LT transitivity
    rock_paper_scissors: dict[tuple[int, int], Ordering] = {
        (0, 0): Ordering.EQ,
        (1, 1): Ordering.EQ,
        (2, 2): Ordering.EQ,
        (0, 1): Ordering.LT,
        (1, 0): Ordering.LT,
        (1, 2): Ordering.LT,
        (2, 1): Ordering.GT,
        (2, 0): Ordering.LT,
        (0, 2): Ordering.LT,
    }
    bad_ord: Ord[int] = Ord(
        eq_witness=non_trans,
        compare=lambda a, b: rock_paper_scissors[(a, b)],
    )
    res_ord = verify_ord_laws(bad_ord, (0, 1, 2))
    assert isinstance(res_ord, Invalid)
    assert any("Eq consistency" in e for e in res_ord.errors)
    assert any("antisymmetry" in e for e in res_ord.errors)
    assert any("LT transitivity" in e for e in res_ord.errors)

    # Unlawful Semigroup & Monoid (subtraction is non-associative; 99 is wrong identity)
    bad_monoid: Monoid[int] = Monoid(empty=lambda: 99, semigroup=Semigroup(combine=lambda a, b: a - b))
    res_mon = verify_monoid_laws(bad_monoid, INT_EQ, (0, 1, 2))
    assert isinstance(res_mon, Invalid)
    assert any("associativity" in e for e in res_mon.errors)
    assert any("left identity" in e for e in res_mon.errors)
    assert any("right identity" in e for e in res_mon.errors)

    # Unlawful Semiring & Ring
    bad_sr: Semiring[int] = Semiring(
        zero=lambda: 1,
        one=lambda: 2,
        add=lambda a, b: a - b,
        mul=lambda a, b: a - b + 1,
    )
    bad_ring: Ring[int] = Ring(semiring=bad_sr, neg=lambda a: a + 5, sub=lambda a, b: a + b)
    res_ring = verify_ring_laws(bad_ring, INT_EQ, (0, 1, 2))
    assert isinstance(res_ring, Invalid)
    assert any("zero annihilation" in e for e in res_ring.errors)
    assert any("additive commutativity" in e for e in res_ring.errors)
    assert any("left distributivity" in e for e in res_ring.errors)
    assert any("right distributivity" in e for e in res_ring.errors)
    assert any("additive inverse" in e for e in res_ring.errors)
    assert any("sub consistency" in e for e in res_ring.errors)

    # Unlawful BoundedLattice
    bad_lat: BoundedLattice[int] = BoundedLattice(
        eq_witness=INT_EQ,
        bottom=lambda: 1,
        top=lambda: 0,
        join=lambda a, b: a - b,
        meet=lambda a, b: (a * 2) - b,
    )
    res_lat = verify_bounded_lattice_laws(bad_lat, (0, 1, 2))
    assert isinstance(res_lat, Invalid)
    assert any("idempotency" in e for e in res_lat.errors)
    assert any("bottom bound" in e for e in res_lat.errors)
    assert any("top bound" in e for e in res_lat.errors)
    assert any("commutativity" in e for e in res_lat.errors)
    assert any("absorption" in e for e in res_lat.errors)

    # Unlawful Codecs: returning Err and returning wrong value
    err_codec: Codec[str, int] = Codec(
        encode=str,
        decode=lambda s: Err(CodecError(path=(), expected="int", actual_description=s, message="fail")),
    )
    res_ec = verify_codec_roundtrip_laws(err_codec, INT_EQ, (1,))
    assert isinstance(res_ec, Invalid) and "returned Err" in res_ec.errors[0]

    wrong_codec: Codec[str, int] = Codec(encode=str, decode=lambda s: Ok(int(s) + 1))
    res_wc = verify_codec_roundtrip_laws(wrong_codec, INT_EQ, (1,))
    assert isinstance(res_wc, Invalid) and "roundtrip mismatch" in res_wc.errors[0]


def test_verify_typeclasses_cli_reports_law_violation(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify scripts/verify_typeclasses.py surfaces algebraic law violations with exit code 1.

    Args:
        capsys (pytest.CaptureFixture[str]): Standard output/error capture fixture.
    """
    with patch("scripts.verify_typeclasses.verify_ord_laws", return_value=Invalid(("synthetic law failure",))):
        rc = main(["--backends-dir", "src/ml_switcheroo_compiler/backends"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "synthetic law failure" in captured.err
