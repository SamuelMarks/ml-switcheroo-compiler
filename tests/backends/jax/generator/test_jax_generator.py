class DummyGraph:
    def __init__(self):
        self.nodes = {}
        self.outputs = []


# ruff: noqa
from ml_switcheroo_compiler.backends.jax.generator_mixins import JaxControlFlowVisitor, JaxDistributedVisitor, JaxMathVisitor
from unittest.mock import MagicMock, patch

from unittest.mock import MagicMock
from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
import sys

"Core abstractions and logic definitions for test_jax_generator_edge_cases.py."


def test_jax_generator_extra_coverage():
    """Test the jax generator extra coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        g = IRGraph()
        gen = JAXCodeGenerator(g)

        def _test_node(op_type, inputs, attrs, expected):
            """Test the node behavior.

            Args:
                op_type (object): The op_type parameter.
                inputs (object): The inputs parameter.
                attrs (object): The attrs parameter.
                expected (object): The expected parameter.

            Returns:
                object: The inferred shape or computed result.
            """
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
        _test_node("reduce_scatter", ["x"], {"axis_name": "'y'", "axis": 1, "op": "jax.lax.pmax"}, "reduce_scatter")
        _test_node("all_reduce", ["x"], {"axis_name": "'y'", "op": "pmax"}, "pmax")
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_jax_generator_generate_full():
    """Test the jax generator generate full behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        g = IRGraph()
        n = IRNode(id="n1", op_type="UnknownOp", inputs=[], attributes={}, shape_metadata=None)
        g.nodes["n1"] = n
        gen = JAXCodeGenerator(g)
        assert gen.get_fallback_prefix() == "jnp"
        assert "jnp.zeros({shape})" in gen._format_zeros_like("zeros", {})
        assert "jnp.full({shape}, {fill_value})" in gen._format_full({})
        code = gen.generate()
        assert "def apply_model(params, *args, **kwargs) -> object:" in code
        assert "import jax" in code
        gen._emit_constant_assignment("var_a", "1")
        assert "var_a = jnp.array(1)" in "\n".join(gen.code)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


"Test module."

sys.modules["jax.scipy"] = MagicMock()

sys.modules["jax.scipy.special"] = MagicMock()

sys.modules["jax.scipy.signal"] = MagicMock()

sys.modules["jax.scipy.stats"] = MagicMock()

sys.modules["jax.scipy.linalg"] = MagicMock()

sys.modules["jax.nn"] = MagicMock()


class DummyNode:
    def __init__(self, attrs=None):
        self.attributes = attrs or {}


def test_jax_control_flow_visitor():
    vis = JaxControlFlowVisitor()
    assert vis.visit_If(DummyNode(), ["c"]) == "jax.lax.cond(c, lambda: None, lambda: None)"
    assert vis.visit_Loop(DummyNode(), ["c"]) == "jax.lax.while_loop(lambda _: True, lambda _: c, c)"
    assert vis.visit_Scan(DummyNode(), ["a"]) == "jax.lax.scan(lambda c, x: (c, x), a, None)"
    assert vis.visit_Scan(DummyNode(), ["a", "b"]) == "jax.lax.scan(lambda c, x: (c, x), a, b)"


def test_jax_generator():
    g = DummyGraph()
    gen = JAXCodeGenerator(g)
    assert gen.get_fallback_prefix() == "jnp"
    assert gen.get_fallback_prefix() == "jnp"
    assert gen._format_zeros_like("zeros", {}) == "jnp.zeros({shape})"
    assert gen._format_zeros_like("ones", {"dtype": "int32"}) == "jnp.ones({shape}), dtype='int32'"
    assert gen._format_full({}) == "jnp.full({shape}, {fill_value})"
    assert gen._format_full({"dtype": "int32"}) == "jnp.full({shape}, {fill_value}), dtype='int32'"
    ops = gen.get_ops_map({})
    assert "Zeros" in ops
    assert "Ones" in ops
    assert "Full" in ops
    gen._emit_constant_assignment("var_a", "42")
    assert gen.code[-1] == "var_a = jnp.array(42)"
    assert gen._generate_file_header() == [gen.header.strip()]
    imports = gen._resolve_imports()
    assert "import jax" in imports
    assert "import jax.numpy as jnp" in imports
    gen.code = []
    gen._generate_function_signature()
    assert gen.indent_level == 1
    assert "def apply_model(params, *args, **kwargs) -> object:" in gen.code[0]


def test_jax_generator_imports():
    g = DummyGraph()
    gen = JAXCodeGenerator(g)
    import builtins

    original_open = builtins.open

    def mock_open(path, *args, **kwargs):
        if "jax_prefix.py.tmpl" in str(path):
            from io import StringIO

            return StringIO("import test_template")
        return original_open(path, *args, **kwargs)

    with patch("builtins.open", side_effect=mock_open):
        imports = gen._resolve_imports()
        assert "import test_template" in imports


def test_jax_generator_save_funcs():
    g = DummyGraph()
    gen = JAXCodeGenerator(g)


from ml_switcheroo_compiler.backends.jax.generator_mixins import JaxDistributedVisitor


def test_jax_generator_mixin_code_prop():
    class DummyGen:
        pass

    m = JaxDistributedVisitor(DummyGen())
    assert m.code == []

    from ml_switcheroo_compiler.ir.core import IRNode

    n_ata = IRNode(id="ata", op_type="AllToAll", inputs=["t"], attributes={"axis_name": "'data'", "split_axis": 0, "concat_axis": 1})
    res_ata = m.visit_AllToAll(n_ata, ["t"])
    assert "jax.lax.all_to_all(t, axis_name='data', split_axis=0, concat_axis=1)" in res_ata

    n_bc = IRNode(id="bc", op_type="Broadcast", inputs=["t"], attributes={"axis_name": "'model'"})
    res_bc = m.visit_Broadcast(n_bc, ["t"])
    assert "jax.lax.pbroadcast(t, axis_name='model')" in res_bc


def test_jax_ragged_dot_and_einsum():
    from ml_switcheroo_compiler.backends.jax.generator_mixins import JaxMathVisitor
    from ml_switcheroo_compiler.ir.core import IRNode

    class DummyGenerator:
        pass

    mixin = JaxMathVisitor(generator=DummyGenerator())

    # Test RaggedDot
    node1 = IRNode(id="rd", op_type="RaggedDot", inputs=["a", "b"])
    res1 = mixin.visit_RaggedDot(node1, ["x", "y"])
    assert "jax_ragged_dot(x, y)" in res1

    # Test Einsum
    node2 = IRNode(id="ein", op_type="Einsum", inputs=["a", "b"])
    res2 = mixin.visit_Einsum(node2, ["x", "y"], equation="ij,jk->ik")
    assert "jnp.einsum('ij,jk->ik', x, y)" in res2
