"""Behavioral IR tracing tests for continuous probability distributions."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random.continuous import (
    ball,
    beta,
    cauchy,
    chisquare,
    dirichlet,
    double_sided_maxwell,
    exponential,
    f,
    gamma,
    generalized_normal,
    gumbel,
    laplace,
    loggamma,
    logistic,
    lognormal,
    maxwell,
    multivariate_normal,
    normal,
    orthogonal,
    pareto,
    random_gamma_p,
    rayleigh,
    t,
    triangular,
    truncated_normal,
    uniform,
    wald,
    weibull_min,
)
from ml_switcheroo_compiler.random.continuous.multivariate_normal import MultivariateNormalOptions
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape


def test_continuous_dispatched_distributions_emit_ir_nodes() -> None:
    """Verify all 21 `_dispatch_random`-backed continuous distributions emit canonical IR nodes during tracing."""
    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        traced_graph = tape.start_tracing("continuous_distributions")
        try:
            outputs = [
                ball("k", 2),
                cauchy("k"),
                chisquare("k", 4),
                double_sided_maxwell("k"),
                exponential("k"),
                f("k", 3, 5),
                generalized_normal("k", 2.0),
                gumbel("k"),
                laplace("k"),
                loggamma("k", 2.0),
                logistic("k"),
                lognormal("k"),
                maxwell("k"),
                orthogonal("k", 3),
                pareto("k", 2.0),
                random_gamma_p("k", 1.5),
                rayleigh("k", 1.0),
                t("k", 4.0),
                triangular("k", -1.0, 0.0, 1.0),
                wald("k", 1.0),
                weibull_min("k", 1.0, 2.0),
            ]
        finally:
            tape.stop_tracing()

    assert all(isinstance(out, Tensor) for out in outputs)
    emitted_ops = {node.op_type for node in traced_graph.nodes.values()}
    for expected_op in (
        "Ball",
        "Cauchy",
        "Chisquare",
        "DoubleSidedMaxwell",
        "Exponential",
        "F",
        "GeneralizedNormal",
        "Gumbel",
        "Laplace",
        "Loggamma",
        "Logistic",
        "Lognormal",
        "Maxwell",
        "Orthogonal",
        "Pareto",
        "RandomGammaP",
        "Rayleigh",
        "T",
        "Triangular",
        "Wald",
        "WeibullMin",
    ):
        assert expected_op in emitted_ops


def test_parameterized_continuous_distributions_trace_shapes_dtypes_and_attributes() -> None:
    """Verify `normal`, `uniform`, `truncated_normal`, `gamma`, `beta`, `dirichlet`, and `multivariate_normal`."""
    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        traced_graph = tape.start_tracing("parameterized_continuous")
        try:
            n32 = normal("key", (2, 2))
            n64 = normal("key", (2, 2), dtype=DType.Float64)

            u32 = uniform("key", (3, 3))
            u64 = uniform("key", (3, 3), dtype=DType.Float64, minval=-1.0, maxval=2.0)

            tn32 = truncated_normal("key", -2.0, 2.0, (2, 2))
            tn64 = truncated_normal("key", -2.0, 2.0, (2, 2), dtype=DType.Float64)

            g32 = gamma("key", 1.0, (2, 2))
            g64 = gamma("key", 1.0, (2, 2), dtype=DType.Float64)

            b32 = beta("key", 1.0, 2.0)
            b64 = beta("key", 1.0, 2.0, shape=(2, 3), dtype=DType.Float64)

            d32 = dirichlet("key", 1.0)
            d64 = dirichlet("key", 1.0, shape=(3,), dtype=DType.Float64)

            mvn_default = multivariate_normal("key", 0.0, 1.0)
            mean_t = Tensor(ProxyTensor("mean_in", (2,), "float64"), TensorConfig((2,), DType.Float64, "cpu"))
            cov_t = Tensor(ProxyTensor("cov_in", (2, 2), "float64"), TensorConfig((2, 2), DType.Float64, "cpu"))
            mvn_opts = MultivariateNormalOptions(shape=(2,), dtype=DType.Float64, method="svd")
            mvn_custom = multivariate_normal("key", mean_t, cov_t, options=mvn_opts)
        finally:
            tape.stop_tracing()

    assert n32.shape == (2, 2) and n32.dtype == DType.Float32
    assert n64.shape == (2, 2) and n64.dtype == DType.Float64
    assert u32.shape == (3, 3) and u32.dtype == DType.Float32
    assert u64.shape == (3, 3) and u64.dtype == DType.Float64
    assert tn32.shape == (2, 2) and tn64.dtype == DType.Float64
    assert g32.shape == (2, 2) and g64.dtype == DType.Float64
    assert b32.shape == () and b64.shape == (2, 3) and b64.dtype == DType.Float64
    assert d32.shape == () and d64.shape == (3,) and d64.dtype == DType.Float64
    assert mvn_default.shape == () and mvn_default.dtype == DType.Float32
    assert mvn_custom.shape == (2,) and mvn_custom.dtype == DType.Float64

    op_types = [node.op_type for node in traced_graph.nodes.values()]
    assert op_types.count("RandomNormal") == 2
    assert op_types.count("RandomUniform") == 2
    assert "Beta" in op_types
    assert "Dirichlet" in op_types
    assert op_types.count("MultivariateNormal") == 2
    mvn_nodes = [node for node in traced_graph.nodes.values() if node.op_type == "MultivariateNormal"]
    assert any(
        node.attributes == {"method": "svd"} and node.inputs == ["key", "mean_in", "cov_in"] for node in mvn_nodes
    )
