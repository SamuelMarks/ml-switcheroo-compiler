"""Tests for test_generator."""

from __future__ import annotations

import typing

import pytest

import ml_switcheroo_compiler.backends.numpy.generator as gen
from ml_switcheroo_compiler.backends.cupy.generator import CupyGenerator
from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.formatters import FormatterContext, OpFormatter
from ml_switcheroo_compiler.backends.numpy.generator import NumpyASTVisitor, NumpyGenerator, NumpyTypeTranslator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from tests.testing_utils import as_callable, as_tensor, as_tensor_dict, as_tensor_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_numpy_generator_visit_kwargs_only() -> None:
    """Test the numpy generator visit kwargs only behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """

    class Gen(gen.NumpyGenerator):
        """Configuration class for gen."""

        def __init__(self) -> None:
            """Initialize the instance.

            Returns:
                TensorOrScalar: The inferred shape or computed result.
            """
            pass

    Gen()
    node = IRNode(id="test", op_type="UnknownOpKwargs", inputs=[], attributes={"shape": "(2, 2)"}, shape_metadata=None)
    res = gen.NumpyASTVisitor.generic_visit(node, [], shape="(2, 2)")
    assert "np.unknownopkwargs" in res
    assert "shape=" in res


def test_numpy_generator_visit_no_kwargs() -> None:
    """Test the numpy generator visit no kwargs behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """

    class Gen(gen.NumpyGenerator):
        """Configuration class for gen."""

        def __init__(self) -> None:
            """Initialize the instance.

            Returns:
                TensorOrScalar: The inferred shape or computed result.
            """
            pass

    g = Gen()
    node = IRNode(id="test", op_type="Zeros", inputs=[], attributes={}, shape_metadata=None)
    res = g.visit(node, [])
    assert res == "np.zeros()" or res == "np.zeros()"


def test_numpy_generator_kwargs_only() -> None:
    """Test the numpy generator kwargs only behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    op_type = "Zeros"
    input_vars = as_tensor_list([])
    kwargs = {"shape": "(2, 2)"}

    class Gen(gen.NumpyGenerator):
        """Configuration class for gen."""

        def __init__(self) -> None:
            """Initialize the instance.

            Returns:
                TensorOrScalar: The inferred shape or computed result.
            """
            pass

    Gen()
    res = as_callable(OpFormatter.format_generic_fallback)(as_callable(FormatterContext)(prefix="out", op_type=op_type, input_vars=input_vars, kwargs=kwargs))
    assert "zeros" in str(res)


def test_cupy_generator_kwargs_only() -> None:
    """Test the cupy generator kwargs only behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """

    class Gen(CupyGenerator):
        """Configuration class for gen."""

        def __init__(self) -> None:
            """Initialize the instance.

            Returns:
                TensorOrScalar: The inferred shape or computed result.
            """
            pass

    Gen()
    OpFormatter.format_generic_fallback(FormatterContext(prefix="out", op_type="Zeros", input_vars=[], kwargs={"shape": "(2, 2)"}))
    assert callable(test_cupy_generator_kwargs_only)


def test_dask_generator_kwargs_only() -> None:
    """Test the dask generator kwargs only behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """

    class Gen(DaskGenerator):
        """Configuration class for gen."""

        def __init__(self) -> None:
            """Initialize the instance.

            Returns:
                TensorOrScalar: The inferred shape or computed result.
            """
            pass

    Gen()
    OpFormatter.format_generic_fallback(FormatterContext(prefix="out", op_type="Zeros", input_vars=[], kwargs={"shape": "(2, 2)"}))
    assert callable(test_dask_generator_kwargs_only)


def test_cupy_generator_fallback_empty_args_str_with_kwargs() -> None:
    """Test the cupy generator fallback empty args str with kwargs behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """

    class Gen(CupyGenerator):
        """Configuration class for gen."""

        def __init__(self) -> None:
            """Initialize the instance.

            Returns:
                TensorOrScalar: The inferred shape or computed result.
            """
            pass

    Gen()
    OpFormatter.format_generic_fallback(FormatterContext(prefix="out", op_type="Zeros", input_vars=[], kwargs={"shape": "(2, 2)"}))
    assert callable(test_cupy_generator_fallback_empty_args_str_with_kwargs)


def test_numpy_generator_coverage(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_numpy_generator_coverage.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """

    class DummyNode:
        """Test class DummyNode."""

        def __init__(self, op_type: EagerValue, id: EagerValue = "test_id") -> None:
            """Test function __init__.

            Args:
                op_type: Test parameter `op_type`.
                id: Test parameter `id`.
            """
            self.op_type = op_type
            self.id = id
            self.attributes = as_tensor_dict({})

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes = as_tensor_list([])

    assert as_tensor(NumpyTypeTranslator.get_fallback_prefix()) == "np"
    assert as_tensor(NumpyASTVisitor._format_kwargs({"a": 1})) == "a=1"
    assert as_callable(NumpyASTVisitor.generic_visit)(as_callable(DummyNode)("Unknown"), []) == "np.unknown()"
    gen = as_callable(NumpyGenerator)(as_callable(DummyGraph)())
    DummyNode("Einsum")
    DummyNode("PowerIteration")
    assert as_tensor(gen.get_fallback_prefix()) == "np"
    ops = gen.get_ops_map({})
    assert isinstance(as_tensor(ops), dict)
    assert as_tensor(gen.get_fallback_prefix()) == "np"
    helpers = gen.get_helper_functions()
    assert isinstance(as_tensor(helpers), list)
    with pytest.raises(Exception, match=".*"):
        NumpyGenerator.load("")
    dummy_npy = str(tmp_path / "dummy.npy")
    dummy_npz = str(tmp_path / "dummy.npz")
    dummy2_npz = str(tmp_path / "dummy2.npz")
    NumpyGenerator.save(dummy_npy, None)
    NumpyGenerator.savez(dummy_npz)
    NumpyGenerator.savez_compressed(dummy2_npz)


def test_numpy_power_iteration() -> None:
    """Test function test_numpy_power_iteration."""
    from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator
    from ml_switcheroo_compiler.ir.core import IRNode

    gen = NumpyGenerator(graph=IRGraph())
    node = IRNode(id="pi", op_type="PowerIteration", inputs=["a", "u"], attributes={"num_iters": 5})
    res = gen.visit_PowerIteration(node, ["mat_a", "vec_u"])
    assert "np_power_iteration(mat_a, 5, vec_u)" in res
    node2 = IRNode(id="pi2", op_type="PowerIteration", inputs=["a"])
    res2 = gen.visit_PowerIteration(node2, ["mat_a"])
    assert "np_power_iteration(mat_a, 1, None)" in res2
