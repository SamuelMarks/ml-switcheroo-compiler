"""CLI verifier for Haskell-style backend capability typeclasses, curated YAML mappings, and algebraic laws."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Callable, Sequence

from ml_switcheroo_ir import Eq, Invalid, IRGraph, IRNode, Validation

from ml_switcheroo_compiler.backends.eager_registry.types_base import BackendOperand
from ml_switcheroo_compiler.backends.typeclass_verifier import verify_all_typeclasses
from ml_switcheroo_compiler.backends.typeclasses import (
    MinimalCoreEagerPrimitives,
    derive_core_eager_dictionary,
    verify_core_eager_algebraic_laws,
)
from ml_switcheroo_compiler.core.dtype import (
    COMPILER_DTYPE_EQ,
    COMPILER_DTYPE_ORD,
    COMPILER_DTYPE_STR_CODEC,
    DType,
)
from ml_switcheroo_compiler.core.type_promotion import (
    COMPILER_DTYPE_PROMOTION_LATTICE,
    COMPILER_DTYPE_PROMOTION_MONOID,
    COMPILER_PROMOTION_ELEM_EQ,
    PromotionLatticeElem,
)
from ml_switcheroo_compiler.distributed.sharding_lattice import (
    COMPILER_SHARDING_EQ,
    COMPILER_SHARDING_LATTICE,
    COMPILER_SHARDING_MONOID,
    ShardingLatticeElem,
)
from ml_switcheroo_compiler.grad.typeclasses import (
    SCALAR_TANGENT_EQ,
    SCALAR_TANGENT_MONOID,
    ScalarTangentElem,
)
from ml_switcheroo_compiler.ir import (
    COMPILER_BROADCAST_EQ,
    COMPILER_BROADCAST_LATTICE,
    COMPILER_BROADCAST_MONOID,
    COMPILER_POLYNOMIAL_EQ,
    COMPILER_POLYNOMIAL_RING,
    BroadcastShapeElem,
    Polynomial,
    verify_bounded_lattice_laws,
    verify_codec_roundtrip_laws,
    verify_monoid_laws,
    verify_ord_laws,
    verify_ring_laws,
)
from ml_switcheroo_compiler.serialization import (
    COMPILER_IR_BYTES_CODEC,
    COMPILER_IR_STR_CODEC,
    COMPILER_IR_YAML_CODEC,
    FROZEN_TENSOR_ARCHIVE_CODEC,
    FROZEN_TENSOR_ARCHIVE_EQ,
    FrozenTensorArchive,
)
from ml_switcheroo_compiler.transforms import LOGICAL_GRAPH_EQ

_RefTensor = tuple[tuple[int, ...], tuple[float, ...]]


def _ref_broadcast_bin(
    fn: Callable[[float, float], float],
    lhs: _RefTensor,
    rhs: _RefTensor,
) -> _RefTensor:
    """Apply scalar ``fn`` elementwise with scalar-vs-tensor broadcasting over ``_RefTensor``.

    Args:
        fn (Callable[[float, float], float]): Binary float operator.
        lhs (_RefTensor): Left shaped float buffer.
        rhs (_RefTensor): Right shaped float buffer.

    Returns:
        _RefTensor: Resulting shaped float buffer.
    """
    s_l, d_l = lhs
    s_r, d_r = rhs
    out_shape = s_r if len(d_l) == 1 else s_l
    size = max(len(d_l), len(d_r))
    vals = tuple(fn(d_l[0 if len(d_l) == 1 else idx], d_r[0 if len(d_r) == 1 else idx]) for idx in range(size))
    return (out_shape, vals)


def _ref_array(data: BackendOperand, _dtype: str | None) -> _RefTensor:
    """Ingest scalar or flat numeric sequence into a ``_RefTensor``.

    Args:
        data (BackendOperand): Input scalar or sequence.
        _dtype (str | None): Optional dtype tag.

    Returns:
        _RefTensor: Shaped float tuple tensor.
    """
    if isinstance(data, (int, float)):
        return ((1,), (float(data),))
    assert isinstance(data, Sequence) and not isinstance(data, (str, bytes))
    vals = tuple(float(v) for v in data if isinstance(v, (int, float)))
    return ((len(vals),), vals)


def _ref_matmul_2d(lhs: _RefTensor, rhs: _RefTensor) -> _RefTensor:
    """Compute exact 2D matrix multiplication over ``_RefTensor``.

    Args:
        lhs (_RefTensor): Left ``(M, K)`` matrix.
        rhs (_RefTensor): Right ``(K, N)`` matrix.

    Returns:
        _RefTensor: Product ``(M, N)`` matrix.
    """
    (m_dim, k_dim), a_vals = lhs
    (_, n_dim), b_vals = rhs
    out = tuple(
        sum(a_vals[i * k_dim + p] * b_vals[p * n_dim + j] for p in range(k_dim))
        for i in range(m_dim)
        for j in range(n_dim)
    )
    return ((m_dim, n_dim), out)


def _ref_transpose_2d(x: _RefTensor, axes: Sequence[int] | None) -> _RefTensor:
    """Transpose a 2D ``_RefTensor``.

    Args:
        x (_RefTensor): Input 2D tensor.
        axes (Sequence[int] | None): Optional axis permutation.

    Returns:
        _RefTensor: Transposed 2D tensor.
    """
    _ = axes
    (rows, cols), vals = x
    out = tuple(vals[r * cols + c] for c in range(cols) for r in range(rows))
    return ((cols, rows), out)


def _verify_reference_core_eager_laws() -> Validation[tuple[str, ...], None]:
    """Derive a ``CoreEagerDictionary`` from ``MinimalCoreEagerPrimitives`` (via ``reciprocal_fn``) and verify laws.

    Returns:
        Validation[tuple[str, ...], None]: ``Valid(None)`` or ``Invalid`` law violation messages.
    """
    primitives = MinimalCoreEagerPrimitives[_RefTensor](
        zeros_fn=lambda shp, _dt: (shp, (0.0,) * math.prod(shp)),
        ones_fn=lambda shp, _dt: (shp, (1.0,) * math.prod(shp)),
        array_fn=_ref_array,
        exp_fn=lambda x: (x[0], tuple(math.exp(v) for v in x[1])),
        log_fn=lambda x: (x[0], tuple(math.log(v) for v in x[1])),
        sin_fn=lambda x: (x[0], tuple(math.sin(v) for v in x[1])),
        cos_fn=lambda x: (x[0], tuple(math.cos(v) for v in x[1])),
        add_fn=lambda a, b: _ref_broadcast_bin(lambda u, v: u + v, a, b),
        multiply_fn=lambda a, b: _ref_broadcast_bin(lambda u, v: u * v, a, b),
        reciprocal_fn=lambda x: (x[0], tuple(1.0 / v for v in x[1])),
        matmul_fn=_ref_matmul_2d,
        reshape_fn=lambda x, shp: (shp, x[1]),
        transpose_fn=_ref_transpose_2d,
        sum_fn=lambda x, _ax, _kd: ((1,), (sum(x[1]),)),
    )
    derived_dict = derive_core_eager_dictionary(primitives)
    approx_eq: Eq[_RefTensor] = Eq(
        eq=lambda a, b: (
            a[0] == b[0]
            and len(a[1]) == len(b[1])
            and all(math.isclose(u, v, abs_tol=1e-9, rel_tol=1e-9) for u, v in zip(a[1], b[1]))
        )
    )
    return verify_core_eager_algebraic_laws(derived_dict, approx_eq, label="CORE_EAGER_MCD_DICT")


def verify_compiler_algebraic_laws() -> list[str]:
    """Execute algebraic law suites across compiler typeclass witnesses (Ord, Monoid, Ring, Lattice, Codec, Eager).

    Returns:
        list[str]: Collected algebraic law violation messages (empty when all laws hold).
    """
    dtype_samples = (DType.Bool, DType.Int8, DType.UInt8, DType.Float16, DType.Float32, DType.Complex64)
    prom_samples = (
        PromotionLatticeElem.bottom(),
        PromotionLatticeElem.top(),
        PromotionLatticeElem.of(DType.Bool),
        PromotionLatticeElem.of(DType.UInt8),
        PromotionLatticeElem.of(DType.Int8),
        PromotionLatticeElem.of(DType.Float16),
        PromotionLatticeElem.of(DType.String),
    )
    bcast_samples = (
        BroadcastShapeElem.bottom(),
        BroadcastShapeElem.top(),
        BroadcastShapeElem.of((2,)),
        BroadcastShapeElem.of((3,)),
        BroadcastShapeElem.of((2, 1)),
        BroadcastShapeElem.of((1, 3)),
    )
    shard_samples = (
        ShardingLatticeElem.bottom(),
        ShardingLatticeElem.top(),
        ShardingLatticeElem.from_axes(["dp", None]),
        ShardingLatticeElem.from_axes([None, "tp"]),
    )
    poly_samples = (
        Polynomial.from_const(0),
        Polynomial.from_const(1),
        Polynomial.from_var("B"),
        Polynomial.from_var("N").add(Polynomial.from_const(2)),
    )
    tangent_samples = (
        ScalarTangentElem.zero(),
        ScalarTangentElem.no_tangent(),
        ScalarTangentElem.of(2.0),
        ScalarTangentElem.of(-0.5),
    )
    graph_samples = (
        IRGraph(
            name="law_dag",
            inputs=["x"],
            outputs=["y"],
            nodes=[
                IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(2, 3)),
                IRNode(id="y", op_type="Relu", inputs=["x"], shape_metadata=(2, 3)),
            ],
        ),
    )
    archive_samples = (FrozenTensorArchive.from_dict({"w": (1.0, -2.5)}),)

    checks = (
        verify_ord_laws(COMPILER_DTYPE_ORD, dtype_samples, "COMPILER_DTYPE_ORD"),
        verify_codec_roundtrip_laws(COMPILER_DTYPE_STR_CODEC, COMPILER_DTYPE_EQ, dtype_samples, "DTYPE_CODEC"),
        verify_bounded_lattice_laws(COMPILER_DTYPE_PROMOTION_LATTICE, prom_samples, "PROMOTION_LATTICE"),
        verify_monoid_laws(COMPILER_DTYPE_PROMOTION_MONOID, COMPILER_PROMOTION_ELEM_EQ, prom_samples, "PROMOTION_M"),
        verify_bounded_lattice_laws(COMPILER_BROADCAST_LATTICE, bcast_samples, "BROADCAST_LATTICE"),
        verify_monoid_laws(COMPILER_BROADCAST_MONOID, COMPILER_BROADCAST_EQ, bcast_samples, "BROADCAST_MONOID"),
        verify_bounded_lattice_laws(COMPILER_SHARDING_LATTICE, shard_samples, "SHARDING_LATTICE"),
        verify_monoid_laws(COMPILER_SHARDING_MONOID, COMPILER_SHARDING_EQ, shard_samples, "SHARDING_MONOID"),
        verify_ring_laws(COMPILER_POLYNOMIAL_RING, COMPILER_POLYNOMIAL_EQ, poly_samples, "POLYNOMIAL_RING"),
        verify_monoid_laws(SCALAR_TANGENT_MONOID, SCALAR_TANGENT_EQ, tangent_samples, "SCALAR_TANGENT_MONOID"),
        verify_codec_roundtrip_laws(COMPILER_IR_STR_CODEC, LOGICAL_GRAPH_EQ, graph_samples, "IR_STR_CODEC"),
        verify_codec_roundtrip_laws(COMPILER_IR_YAML_CODEC, LOGICAL_GRAPH_EQ, graph_samples, "IR_YAML_CODEC"),
        verify_codec_roundtrip_laws(COMPILER_IR_BYTES_CODEC, LOGICAL_GRAPH_EQ, graph_samples, "IR_BYTES_CODEC"),
        verify_codec_roundtrip_laws(
            FROZEN_TENSOR_ARCHIVE_CODEC,
            FROZEN_TENSOR_ARCHIVE_EQ,
            archive_samples,
            "ARCHIVE_CODEC",
        ),
        _verify_reference_core_eager_laws(),
    )
    violations: list[str] = []
    for res in checks:
        if isinstance(res, Invalid):
            violations.extend(res.errors)
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run static AST, typeclass dictionary, curated YAML, and algebraic law verification.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments (``--backends-dir``).

    Returns:
        int: Exit code ``0`` when all capability and law obligations hold, or ``1`` on violation.
    """
    parser = argparse.ArgumentParser(
        description="Verify Haskell-style backend typeclass capabilities, YAML mappings, and algebraic laws."
    )
    parser.add_argument(
        "--backends-dir",
        default="src/ml_switcheroo_compiler/backends",
        help="Path to backends package directory.",
    )
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    backends_dir = str(args.backends_dir)
    violations = verify_all_typeclasses(backends_dir=backends_dir)
    violations.extend(verify_compiler_algebraic_laws())
    if violations:
        for msg in violations:
            sys.stderr.write(msg + os.linesep)
        return 1
    sys.stdout.write("Haskell-style backend typeclass and YAML verification passed." + os.linesep)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
