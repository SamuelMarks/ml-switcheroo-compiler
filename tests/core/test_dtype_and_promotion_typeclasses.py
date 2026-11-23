"""Unit tests for DType typeclasses, PromotionBoundedLattice, BroadcastBoundedLattice, and Result dtype resolution."""

from __future__ import annotations

from ml_switcheroo_ir import Err, Ok, Ordering

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import (
    COMPILER_DTYPE_EQ,
    COMPILER_DTYPE_HASH,
    COMPILER_DTYPE_ORD,
    COMPILER_DTYPE_PRETTY,
    COMPILER_DTYPE_PYTHON_EMIT,
    COMPILER_DTYPE_STR_CODEC,
    DType,
    parse_dtype_result,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.type_promotion import (
    COMPILER_DTYPE_PROMOTION_LATTICE,
    COMPILER_DTYPE_PROMOTION_MONOID,
    COMPILER_PROMOTION_ELEM_EQ,
    COMPILER_PROMOTION_ELEM_HASH,
    COMPILER_PROMOTION_ELEM_PRETTY,
    PromotionLatticeElem,
    promote_dtypes_fold,
    promote_types_result,
)
from ml_switcheroo_compiler.ir import (
    COMPILER_BROADCAST_EQ,
    COMPILER_BROADCAST_HASH,
    COMPILER_BROADCAST_LATTICE,
    COMPILER_BROADCAST_MONOID,
    BroadcastShapeElem,
    broadcast_many_shapes_result,
)
from ml_switcheroo_compiler.ops.type_inference import resolve_dtype_result


class _MockDTypeHolder:
    """Stub implementing DTypeHolderProtocol for testing resolve_dtype_result."""

    def __init__(self, dt_repr: str) -> None:
        """Initialize stub with raw dtype representation.

        Args:
            dt_repr (str): Raw dtype representation string.
        """
        self.dtype = dt_repr


def test_dtype_witnesses_and_codec() -> None:
    """Verify Eq, HashableWitness, Ord, Pretty, PythonExprEmit, and Codec instances on DType."""
    assert COMPILER_DTYPE_EQ.eq(DType.Float32, DType.float32)
    assert not COMPILER_DTYPE_EQ.eq(DType.Float32, DType.Int32)
    assert COMPILER_DTYPE_HASH.hash_value(DType.Float32) == COMPILER_DTYPE_HASH.hash_value(DType.float32)

    assert COMPILER_DTYPE_ORD.compare(DType.float64, DType.float32) == Ordering.LT
    assert COMPILER_DTYPE_ORD.compare(DType.float32, DType.float64) == Ordering.GT
    assert COMPILER_DTYPE_ORD.compare(DType.float32, DType.Float32) == Ordering.EQ

    assert COMPILER_DTYPE_PRETTY.render(DType.BFloat16) == "bfloat16"
    assert COMPILER_DTYPE_PYTHON_EMIT.emit_python(DType.Float32, "") == "DType.float32"
    assert COMPILER_DTYPE_PYTHON_EMIT.emit_python(DType.Float32, "core") == "core.DType.float32"

    ok_dt = parse_dtype_result(DType.Int16)
    assert isinstance(ok_dt, Ok) and ok_dt.value == DType.Int16
    ok_alias = parse_dtype_result("bf16")
    assert isinstance(ok_alias, Ok) and ok_alias.value == DType.BFloat16
    err_dt = parse_dtype_result("not_a_real_dtype")
    assert isinstance(err_dt, Err)

    wire = COMPILER_DTYPE_STR_CODEC.encode(DType.Complex64)
    dec_ok = COMPILER_DTYPE_STR_CODEC.decode(wire)
    assert isinstance(dec_ok, Ok) and dec_ok.value == DType.Complex64
    dec_err = COMPILER_DTYPE_STR_CODEC.decode("invalid_wire_dtype")
    assert isinstance(dec_err, Err) and dec_err.error.path == ("dtype",)


def test_promotion_bounded_lattice_and_folds() -> None:
    """Verify BoundedLattice axioms, Pretty/Hash witnesses, promote_types_result, and promote_dtypes_fold."""
    lat = COMPILER_DTYPE_PROMOTION_LATTICE
    bot = lat.bottom()
    top = lat.top()
    samples = [
        bot,
        top,
        PromotionLatticeElem.of(DType.Bool),
        PromotionLatticeElem.of(DType.UInt8),
        PromotionLatticeElem.of(DType.Int8),
        PromotionLatticeElem.of(DType.Int16),
        PromotionLatticeElem.of(DType.Float16),
        PromotionLatticeElem.of(DType.Float32),
        PromotionLatticeElem.of(DType.String),
    ]
    assert COMPILER_PROMOTION_ELEM_PRETTY.render(bot) == "⊥"
    assert COMPILER_PROMOTION_ELEM_PRETTY.render(top) == "⊤"
    assert COMPILER_PROMOTION_ELEM_PRETTY.render(PromotionLatticeElem.of(DType.Float32)) == "float32"

    for a in samples:
        assert COMPILER_PROMOTION_ELEM_EQ.eq(a, a)
        assert COMPILER_PROMOTION_ELEM_HASH.hash_value(a) == hash(a)
        assert lat.join(bot, a) == a
        assert lat.join(top, a) == top
        assert lat.meet(top, a) == a
        assert lat.meet(bot, a) == bot
        for b in samples:
            assert lat.join(a, b) == lat.join(b, a)
            assert lat.meet(a, b) == lat.meet(b, a)
            assert lat.join(a, lat.meet(a, b)) == a
            assert lat.meet(a, lat.join(a, b)) == a
            for c in samples[:5]:
                assert lat.join(lat.join(a, b), c) == lat.join(a, lat.join(b, c))
                assert lat.meet(lat.meet(a, b), c) == lat.meet(a, lat.meet(b, c))

    with ConfigContext(enable_x64=True):
        res_ok = promote_types_result("int32", "float32")
        assert isinstance(res_ok, Ok) and res_ok.value == DType.Float64
        fold_ok = promote_dtypes_fold(["bool", DType.UInt8, DType.Int8, "float16"])
        assert isinstance(fold_ok, Ok) and fold_ok.value == DType.Float32

    assert isinstance(promote_types_result("bad_lhs", "float32"), Err)
    assert isinstance(promote_types_result("float32", "bad_rhs"), Err)
    assert isinstance(promote_types_result(DType.String, DType.Float32), Err)

    assert isinstance(promote_dtypes_fold([]), Err)
    assert isinstance(promote_dtypes_fold(["float32", "unknown_dt"]), Err)
    assert isinstance(promote_dtypes_fold([DType.String, DType.Int32]), Err)
    assert COMPILER_DTYPE_PROMOTION_MONOID.empty() == bot


def test_resolve_dtype_result_branches() -> None:
    """Verify resolve_dtype_result across DTypeHolderProtocol, fallback Tensor, and failure branches."""
    ok_np_style = resolve_dtype_result(_MockDTypeHolder("dtype('int32')"), None)
    assert isinstance(ok_np_style, Ok) and ok_np_style.value == DType.Int32

    ok_attr_style = resolve_dtype_result(_MockDTypeHolder("torch.float16"), None)
    assert isinstance(ok_attr_style, Ok) and ok_attr_style.value == DType.Float16

    err_holder = resolve_dtype_result(_MockDTypeHolder("dtype('mystery')"), None)
    assert isinstance(err_holder, Err)

    fallback_t = Tensor([1.0, 2.0], TensorConfig((2,), DType.Float32, None))
    ok_fallback = resolve_dtype_result(42, fallback_t)
    assert isinstance(ok_fallback, Ok) and ok_fallback.value == DType.Float32

    err_none = resolve_dtype_result(42, None)
    assert isinstance(err_none, Err)


def test_broadcast_shape_lattice_and_many_fold() -> None:
    """Verify BroadcastShapeElem BoundedLattice laws and broadcast_many_shapes_result."""
    lat = COMPILER_BROADCAST_LATTICE
    bot = BroadcastShapeElem.bottom()
    top = BroadcastShapeElem.top()
    assert BroadcastShapeElem.of((-1, 2)) == top
    assert COMPILER_BROADCAST_MONOID.empty() == bot

    elems = [
        bot,
        top,
        BroadcastShapeElem.of((1,)),
        BroadcastShapeElem.of((2,)),
        BroadcastShapeElem.of((3,)),
        BroadcastShapeElem.of((2, 1)),
        BroadcastShapeElem.of((1, 3)),
        BroadcastShapeElem.of((2, 3)),
    ]
    for a in elems:
        assert COMPILER_BROADCAST_EQ.eq(a, a)
        assert COMPILER_BROADCAST_HASH.hash_value(a) == hash(a)
        assert lat.join(bot, a) == a
        assert lat.meet(bot, a) == bot
        assert lat.join(top, a) == top
        assert lat.meet(top, a) == a
        for b in elems:
            assert lat.join(a, b) == lat.join(b, a)
            assert lat.meet(a, b) == lat.meet(b, a)
            assert lat.join(a, lat.meet(a, b)) == a
            assert lat.meet(a, lat.join(a, b)) == a
            for c in elems[:5]:
                assert lat.join(lat.join(a, b), c) == lat.join(a, lat.join(b, c))
                assert lat.meet(lat.meet(a, b), c) == lat.meet(a, lat.meet(b, c))

    ok_many = broadcast_many_shapes_result([(), (3,), (2, 1), (4, 1, 3)])
    assert isinstance(ok_many, Ok) and ok_many.value == (4, 2, 3)

    ok_empty = broadcast_many_shapes_result([])
    assert isinstance(ok_empty, Ok) and ok_empty.value == ()

    err_many = broadcast_many_shapes_result([(2, 3), (4, 5)])
    assert isinstance(err_many, Err)
