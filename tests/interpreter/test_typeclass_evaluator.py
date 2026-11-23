"""Unit tests for pure Reader/dictionary-passing evaluate_graph_with_typeclass."""

from __future__ import annotations

import numpy as np
from ml_switcheroo_ir import Err, LogicalGraph, LogicalNode, Ok
from numpy.typing import NDArray

from ml_switcheroo_compiler.backends.eager_registry.types_base import BackendOperand
from ml_switcheroo_compiler.backends.typeclasses import (
    CoreEagerTypeclass,
    MinimalCoreEagerPrimitives,
    derive_core_eager_dictionary,
)
from ml_switcheroo_compiler.interpreter import evaluate_graph_with_typeclass

FloatArray = NDArray[np.float64]


def _make_numpy_typeclass() -> CoreEagerTypeclass[FloatArray]:
    """Construct a typed FloatArray CoreEagerTypeclass dictionary via MinimalCoreEagerPrimitives.

    Returns:
        CoreEagerTypeclass[FloatArray]: Derived NumPy float64 eager dictionary witness.
    """

    def _ingest(data: BackendOperand, _dtype: str | None) -> FloatArray:
        """Convert scalar or sequence into float64 NDArray.

        Args:
            data (BackendOperand): Scalar or sequence input.
            _dtype (str | None): Optional dtype tag.

        Returns:
            FloatArray: Converted float64 array.
        """
        return np.asarray(data, dtype=np.float64)

    return derive_core_eager_dictionary(
        MinimalCoreEagerPrimitives[FloatArray](
            zeros_fn=lambda shp, _dt: np.zeros(shp, dtype=np.float64),
            ones_fn=lambda shp, _dt: np.ones(shp, dtype=np.float64),
            array_fn=_ingest,
            exp_fn=lambda x: np.asarray(np.exp(x), dtype=np.float64),
            log_fn=lambda x: np.asarray(np.log(x), dtype=np.float64),
            sin_fn=lambda x: np.asarray(np.sin(x), dtype=np.float64),
            cos_fn=lambda x: np.asarray(np.cos(x), dtype=np.float64),
            add_fn=lambda a, b: np.asarray(np.add(a, b), dtype=np.float64),
            multiply_fn=lambda a, b: np.asarray(np.multiply(a, b), dtype=np.float64),
            reciprocal_fn=lambda x: np.asarray(1.0 / x, dtype=np.float64),
            matmul_fn=lambda a, b: np.asarray(np.matmul(a, b), dtype=np.float64),
            reshape_fn=lambda x, shp: np.asarray(np.reshape(x, shp), dtype=np.float64),
            transpose_fn=lambda x, axes: np.asarray(np.transpose(x, axes=axes), dtype=np.float64),
            sum_fn=lambda x, axis, keepdims: np.asarray(np.sum(x, axis=axis, keepdims=keepdims), dtype=np.float64),
        )
    )


def test_evaluate_graph_with_typeclass_all_supported_ops() -> None:
    """Verify Reader/dictionary-passing evaluation across creation, unary, binary, shape, and reduction ops."""
    tc = _make_numpy_typeclass()
    graph = LogicalGraph(
        name="reader_eval",
        inputs=["x"],
        outputs=["final_mean", "mm", "reduced_sum"],
        nodes=[
            LogicalNode(id="x", op_type="Input"),
            LogicalNode(id="z", op_type="Zeros", attributes={"shape": [2, 2]}),
            LogicalNode(id="o", op_type="Ones", attributes={"shape": (2, 2)}),
            LogicalNode(id="c_scalar", op_type="Constant", attributes={"value": 2.0}),
            LogicalNode(id="c_vec", op_type="Constant", attributes={"value": [1.0, 2.0, 3.0, 4.0]}),
            LogicalNode(id="c_mat", op_type="Reshape", inputs=["c_vec"], attributes={"shape": (2, 2)}),
            LogicalNode(id="sum1", op_type="Add", inputs=["x", "z"]),
            LogicalNode(id="diff1", op_type="Sub", inputs=["sum1", "o"]),
            LogicalNode(id="prod1", op_type="Mul", inputs=["diff1", "c_scalar"]),
            LogicalNode(id="quot1", op_type="Div", inputs=["prod1", "c_scalar"]),
            LogicalNode(id="abs1", op_type="Abs", inputs=["quot1"]),
            LogicalNode(id="neg1", op_type="Neg", inputs=["abs1"]),
            LogicalNode(id="id1", op_type="Identity", inputs=["c_mat"]),
            LogicalNode(id="tr1", op_type="Transpose", inputs=["id1"], attributes={"perm": [1, 0]}),
            LogicalNode(id="mm", op_type="MatMul", inputs=["c_mat", "tr1"]),
            LogicalNode(
                id="reduced_sum",
                op_type="ReduceSum",
                inputs=["mm"],
                attributes={"axis": 1, "keepdims": True},
            ),
            LogicalNode(
                id="final_mean",
                op_type="Mean",
                inputs=["neg1"],
                attributes={"axis": [0, 1], "keepdims": False},
            ),
        ],
    )
    x_in = np.array([[3.0, 5.0], [7.0, 9.0]], dtype=np.float64)
    res = evaluate_graph_with_typeclass(graph, {"x": x_in}, tc)
    assert isinstance(res, Ok)
    expected_mm = np.array([[5.0, 11.0], [11.0, 25.0]], dtype=np.float64)
    np.testing.assert_allclose(res.value["mm"], expected_mm)
    np.testing.assert_allclose(res.value["reduced_sum"], np.sum(expected_mm, axis=1, keepdims=True))
    np.testing.assert_allclose(res.value["final_mean"], np.mean(-np.abs(x_in - 1.0)))

    # Graph with no explicit outputs defaults to final topological node; also cover external input without Input node
    implicit_out_g = LogicalGraph(
        name="implicit_out",
        inputs=["ext_in"],
        outputs=[],
        nodes=[LogicalNode(id="y", op_type="Sqrt", inputs=["ext_in"])],
    )
    env_without_prepop: dict[str, FloatArray] = {"ext_in": np.array([4.0, 9.0], dtype=np.float64)}
    res_imp = evaluate_graph_with_typeclass(implicit_out_g, env_without_prepop, tc)
    assert isinstance(res_imp, Ok)
    np.testing.assert_allclose(res_imp.value["y"], np.array([2.0, 3.0], dtype=np.float64))


def test_evaluate_graph_with_typeclass_error_branches() -> None:
    """Verify total Err reporting across cyclic graphs, unbound inputs, bad attributes, and unsupported ops."""
    tc = _make_numpy_typeclass()
    x_val = np.array([1.0, 2.0], dtype=np.float64)

    # 1. Cyclic graph
    cyclic = LogicalGraph(
        name="cyc",
        nodes=[
            LogicalNode(id="a", op_type="Abs", inputs=["b"]),
            LogicalNode(id="b", op_type="Abs", inputs=["a"]),
        ],
    )
    assert isinstance(evaluate_graph_with_typeclass(cyclic, {}, tc), Err)

    # 2. Missing Input tensor
    missing_in = LogicalGraph(name="m_in", nodes=[LogicalNode(id="x", op_type="Input")])
    assert isinstance(evaluate_graph_with_typeclass(missing_in, {}, tc), Err)

    # 3. Unsupported Constant attribute (string/bool)
    bad_const = LogicalGraph(
        name="bad_c",
        nodes=[LogicalNode(id="c", op_type="Constant", attributes={"value": "not_numeric"})],
    )
    assert isinstance(evaluate_graph_with_typeclass(bad_const, {}, tc), Err)

    # 4. Unbound upstream node input
    unbound = LogicalGraph(name="unb", nodes=[LogicalNode(id="y", op_type="Abs", inputs=["ghost"])])
    assert isinstance(evaluate_graph_with_typeclass(unbound, {}, tc), Err)

    # 5. Reshape without integer shape attribute (both string and non-int sequence)
    bad_reshape = LogicalGraph(
        name="bad_r",
        nodes=[LogicalNode(id="y", op_type="Reshape", inputs=["x"], attributes={"shape": "bad"})],
    )
    assert isinstance(evaluate_graph_with_typeclass(bad_reshape, {"x": x_val}, tc), Err)

    bad_reshape_seq = LogicalGraph(
        name="bad_r_seq",
        nodes=[LogicalNode(id="y", op_type="Reshape", inputs=["x"], attributes={"shape": [True, "x"]})],
    )
    assert isinstance(evaluate_graph_with_typeclass(bad_reshape_seq, {"x": x_val}, tc), Err)

    # 6. Unsupported unary op
    bad_unary = LogicalGraph(name="bad_u", nodes=[LogicalNode(id="y", op_type="UnknownUnary", inputs=["x"])])
    assert isinstance(evaluate_graph_with_typeclass(bad_unary, {"x": x_val}, tc), Err)

    # 7. Unsupported binary op
    bad_binary = LogicalGraph(
        name="bad_b",
        nodes=[LogicalNode(id="y", op_type="UnknownBinary", inputs=["x", "x"])],
    )
    assert isinstance(evaluate_graph_with_typeclass(bad_binary, {"x": x_val}, tc), Err)

    # 8. Unsupported ternary/0-arg non-creation arity
    bad_arity = LogicalGraph(
        name="bad_ar",
        nodes=[LogicalNode(id="y", op_type="Where", inputs=["x", "x", "x"])],
    )
    assert isinstance(evaluate_graph_with_typeclass(bad_arity, {"x": x_val}, tc), Err)

    # 9. Declared graph output missing from evaluated environment
    missing_out = LogicalGraph(
        name="m_out",
        outputs=["nonexistent_out"],
        nodes=[LogicalNode(id="y", op_type="Abs", inputs=["x"])],
    )
    assert isinstance(evaluate_graph_with_typeclass(missing_out, {"x": x_val}, tc), Err)
