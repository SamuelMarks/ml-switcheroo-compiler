"""Tests for test_jax_generator."""

from __future__ import annotations

import sys
import typing
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.backends.jax.generator_mixins import JaxControlFlowVisitor, JaxDistributedVisitor, JaxMathVisitor
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from tests.testing_utils import as_callable, as_tensor, as_tensor_dict, as_tensor_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyGraph:
    """Test class DummyGraph."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.nodes = as_tensor_dict({})
        self.outputs = as_tensor_list([])


def test_jax_generator_extra_coverage() -> None:
    """Test the jax generator extra coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    g = IRGraph()
    gen = JAXCodeGenerator(g)

    def _test_node(op_type: EagerTensorProtocol, inputs: EagerTensorProtocol, attrs: EagerTensorProtocol, expected: EagerTensorProtocol) -> None:
        """Test the node behavior.

        Args:
            op_type (EagerTensorProtocol): The op_type parameter.
            inputs (EagerTensorProtocol): The inputs parameter.
            attrs (EagerTensorProtocol): The attrs parameter.
            expected (EagerTensorProtocol): The expected parameter.

        Returns:
            TensorOrScalar: The inferred shape or computed result.
        """
        n = as_callable(IRNode)(id="n1", op_type=op_type, inputs=inputs, attributes=attrs, shape_metadata=None)
        res = as_callable(gen.visit)(n, inputs)
        assert expected in res

    as_callable(_test_node)("SegmentSum", ["x", "ids"], {"num_segments": 2}, "segment_sum")
    as_callable(_test_node)("SegmentMax", ["x", "ids"], {}, "segment_max")
    as_callable(_test_node)("SegmentMin", ["x", "ids"], {}, "segment_min")
    as_callable(_test_node)("SegmentProd", ["x", "ids"], {}, "segment_prod")
    as_callable(_test_node)("UnsortedSegmentSum", ["x", "ids"], {}, "segment_sum")
    as_callable(_test_node)("UnsortedSegmentMax", ["x", "ids"], {}, "segment_max")
    as_callable(_test_node)("UnsortedSegmentMin", ["x", "ids"], {}, "segment_min")
    as_callable(_test_node)("UnsortedSegmentProd", ["x", "ids"], {}, "segment_prod")
    as_callable(_test_node)("MatrixExponential", ["x"], {}, "expm")
    as_callable(_test_node)("Polar", ["x"], {"side": "left"}, "polar")
    as_callable(_test_node)("Polar", ["x"], {"side": "'right'"}, "polar")
    as_callable(_test_node)("Schur", ["x"], {}, "schur")
    as_callable(_test_node)("Cholesky", ["x"], {}, "cholesky")
    as_callable(_test_node)("Svd", ["x"], {"full_matrices": False, "compute_uv": False}, "svd")
    as_callable(_test_node)("If", ["cond"], {}, "cond")
    as_callable(_test_node)("Loop", ["x"], {}, "while_loop")
    as_callable(_test_node)("Scan", ["x", "y"], {}, "scan")
    as_callable(_test_node)("PowerIteration", ["x", "u"], {"num_iters": 2}, "power_iteration")
    as_callable(_test_node)("PowerIteration", ["x"], {}, "power_iteration")
    as_callable(_test_node)("ConvTranspose", ["x", "w"], {"strides": 2, "padding": "SAME"}, "convtranspose")
    as_callable(_test_node)("RaggedDot", ["x", "y"], {}, "ragged_dot")
    as_callable(_test_node)("all_gather", ["x"], {"axis_name": "'y'"}, "all_gather")
    as_callable(_test_node)("reduce_scatter", ["x"], {"axis_name": "'y'", "axis": 1, "op": "jax.lax.pmax"}, "reduce_scatter")
    as_callable(_test_node)("all_reduce", ["x"], {"axis_name": "'y'", "op": "pmax"}, "all_reduce")


def test_jax_generator_generate_full() -> None:
    """Test the jax generator generate full behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
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


sys.modules["jax.scipy"] = MagicMock()
sys.modules["jax.scipy.special"] = MagicMock()
sys.modules["jax.scipy.signal"] = MagicMock()
sys.modules["jax.scipy.stats"] = MagicMock()
sys.modules["jax.scipy.linalg"] = MagicMock()
sys.modules["jax.nn"] = MagicMock()


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, attrs: EagerValue = None) -> None:
        """Test function __init__.

        Args:
            attrs: Test parameter `attrs`.
        """
        self.attributes = attrs or {}


def test_jax_control_flow_visitor() -> None:
    """Test function test_jax_control_flow_visitor."""
    vis = JaxControlFlowVisitor()
    assert as_callable(vis.visit_If)(as_callable(DummyNode)(), ["c"]) == "jax.lax.cond(c, lambda: None, lambda: None)"
    assert as_callable(vis.visit_Loop)(as_callable(DummyNode)(), ["c"]) == "jax.lax.while_loop(lambda _: True, lambda _: c, c)"
    assert as_callable(vis.visit_Scan)(as_callable(DummyNode)(), ["a"]) == "jax.lax.scan(lambda c, x: (c, x), a, None)"
    assert as_callable(vis.visit_Scan)(as_callable(DummyNode)(), ["a", "b"]) == "jax.lax.scan(lambda c, x: (c, x), a, b)"


def test_jax_generator() -> None:
    """Test function test_jax_generator."""
    g = DummyGraph()
    gen = as_callable(JAXCodeGenerator)(g)
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
    setattr(gen, "code", [])
    gen._generate_function_signature()
    assert gen.indent_level == 1
    assert "def apply_model(params, *args, **kwargs) -> object:" in gen.code[0]


def test_jax_generator_imports() -> None:
    """Test function test_jax_generator_imports."""
    g = DummyGraph()
    gen = as_callable(JAXCodeGenerator)(g)
    import builtins

    original_open = builtins.open

    def mock_open(path: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function mock_open.

        Args:
            path: Test parameter `path`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        if "jax_prefix.py.tmpl" in str(path):
            from io import StringIO

            return as_tensor(StringIO("import test_template"))
        return as_tensor(as_callable(original_open)(path, *args, **kwargs))

    with patch("builtins.open", side_effect=mock_open):
        imports = gen._resolve_imports()
        assert "import test_template" in imports


def test_jax_generator_save_funcs() -> None:
    """Test function test_jax_generator_save_funcs."""
    g = DummyGraph()
    gen = as_callable(JAXCodeGenerator)(g)
    assert gen is not None


def test_jax_generator_mixin_code_prop() -> None:
    """Test function test_jax_generator_mixin_code_prop."""

    class DummyGen:
        """Test class DummyGen."""

        pass

    m = as_callable(JaxDistributedVisitor)(as_callable(DummyGen)())
    assert m.code == []
    from ml_switcheroo_compiler.ir.core import IRNode

    n_ata = IRNode(id="ata", op_type="AllToAll", inputs=["t"], attributes={"axis_name": "'data'", "split_axis": 0, "concat_axis": 1})
    res_ata = m.visit_AllToAll(n_ata, ["t"])
    assert "jax.lax.all_to_all(t, axis_name='data', split_axis=0, concat_axis=1)" in res_ata
    n_bc = IRNode(id="bc", op_type="Broadcast", inputs=["t"], attributes={"axis_name": "'model'"})
    res_bc = m.visit_Broadcast(n_bc, ["t"])
    assert "jax.lax.pbroadcast(t, axis_name='model')" in res_bc


def test_jax_ragged_dot_and_einsum() -> None:
    """Test function test_jax_ragged_dot_and_einsum."""
    from ml_switcheroo_compiler.ir.core import IRNode

    class DummyGenerator:
        """Test class DummyGenerator."""

        pass

    mixin = as_callable(JaxMathVisitor)(generator=as_callable(DummyGenerator)())
    node1 = IRNode(id="rd", op_type="RaggedDot", inputs=["a", "b"])
    res1 = mixin.visit_RaggedDot(node1, ["x", "y"])
    assert "jax_ragged_dot(x, y)" in res1
    node2 = IRNode(id="ein", op_type="Einsum", inputs=["a", "b"])
    res2 = mixin.visit_Einsum(node2, ["x", "y"], equation="ij,jk->ik")
    assert "jnp.einsum('ij,jk->ik', x, y)" in res2
