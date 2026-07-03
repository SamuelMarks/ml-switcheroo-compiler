"""Module docstring."""

from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_jax_generator_extra_coverage() -> object:
    """Function docstring."""
    g = IRGraph()
    gen = JAXCodeGenerator(g)

    def _test_node(op_type: object, inputs: object, attrs: object, expected: object) -> object:
        """Function docstring."""
        n = IRNode(id="n1", op_type=op_type, inputs=inputs, attributes=attrs, shape_metadata=None)
        res = gen.visit(n, inputs)
        assert expected in res

    _test_node("SegmentSum", ["x", "ids"], {"num_segments": 2}, "segment_sum")
    _test_node("SegmentMax", ["x", "ids"], {}, "segment_max")
    _test_node("SegmentMin", ["x", "ids"], {}, "segment_min")
    _test_node("SegmentProd", ["x", "ids"], {}, "segment_prod")

    _test_node("UnsortedSegmentSum", ["x", "ids"], {}, "segment_sum")
    _test_node("UnsortedSegmentMax", ["x", "ids"], {}, "segment_max")
    _test_node("UnsortedSegmentMin", ["x", "ids"], {}, "segment_min")
    _test_node("UnsortedSegmentProd", ["x", "ids"], {}, "segment_prod")

    _test_node("MatrixExponential", ["x"], {}, "expm")

    _test_node("Polar", ["x"], {"side": "left"}, "polar")
    _test_node("Polar", ["x"], {"side": "'right'"}, "polar")

    _test_node("Schur", ["x"], {}, "schur")
    _test_node("Cholesky", ["x"], {}, "cholesky")
    _test_node("Svd", ["x"], {"full_matrices": False, "compute_uv": False}, "svd")

    _test_node("If", ["cond"], {}, "cond")
    _test_node("Loop", ["x"], {}, "while_loop")
    _test_node("Scan", ["x", "y"], {}, "scan")
    _test_node("PowerIteration", ["x", "u"], {"num_iters": 2}, "power_iteration")
    _test_node("PowerIteration", ["x"], {}, "power_iteration")

    _test_node("ConvTranspose", ["x", "w"], {"strides": 2, "padding": "SAME"}, "conv_transpose")
    _test_node("RaggedDot", ["x", "y"], {}, "ragged_dot")

    _test_node("all_gather", ["x"], {"axis_name": "'y'"}, "all_gather")
    _test_node(
        "reduce_scatter",
        ["x"],
        {"axis_name": "'y'", "axis": 1, "op": "jax.lax.pmax"},
        "reduce_scatter",
    )
    _test_node("all_reduce", ["x"], {"axis_name": "'y'", "op": "pmax"}, "pmax")


def test_jax_generator_generate_full() -> object:
    """Function docstring."""
    g = IRGraph()
    n = IRNode(id="n1", op_type="UnknownOp", inputs=[], attributes={}, shape_metadata=None)
    g.nodes["n1"] = n

    gen = JAXCodeGenerator(g)
    # mock get_fallback_prefix and _format_zeros_like to hit them
    assert gen.get_fallback_prefix() == "jnp"
    assert "jnp.zeros({shape})" in gen._format_zeros_like("zeros", {})
    assert "jnp.full({shape}, {fill_value})" in gen._format_full({})

    code = gen.generate()
    assert "def apply_model(params, *args, **kwargs):" in code
    assert "import jax" in code

    gen._emit_constant_assignment("var_a", "1")
    assert "var_a = jnp.array(1)" in "\n".join(gen.code)
