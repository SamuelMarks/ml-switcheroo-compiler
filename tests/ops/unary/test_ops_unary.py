"""Behavioral unit tests for ops unary."""

from __future__ import annotations

import typing

import numpy as np

import ml_switcheroo_compiler.ops.unary.base as unary_base
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.unary.arithmetic import Abs, Ceil, Floor, Negative, Positive, Round, Sign, Sqrt, Square
from ml_switcheroo_compiler.ops.unary.base import UnaryMathOp
from ml_switcheroo_compiler.ops.unary.exponential import Exp, Log
from ml_switcheroo_compiler.ops.unary.special import Bitcast, Cast, Frexp
from ml_switcheroo_compiler.ops.unary.trigonometric import Cos, Sin, Tan
from tests.testing_utils import MockNodeStub, extract_test_dtype, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_unary_math_ops() -> None:
    """Test the unary math ops behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    x = 2.0
    x_arr = np.array([1.0, 2.0])
    ops = [
        (Sin(), np.sin),
        (Tan(), np.tan),
        (Cos(), np.cos),
        (Exp(), np.exp),
        (Log(), np.log),
        (Sqrt(), np.sqrt),
        (Square(), np.square),
        (Abs(), np.abs),
        (Negative(), np.negative),
        (Positive(), np.positive),
        (Sign(), np.sign),
        (Floor(), np.floor),
        (Ceil(), np.ceil),
        (Round(), np.round),
    ]
    for op, np_func in ops:
        assert require_callable(op.infer_shape)(x) == ()
        if op.op_name == "Round":
            assert np.allclose(ensure_ndarray(op.eager_eval(x_arr)), np.round(x_arr))
            assert np.allclose(ensure_ndarray(require_callable(op.eager_eval)(x_arr)), require_callable(np_func)(x_arr))


def test_unary_base_op() -> None:
    """Test the unary base op behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """

    class DummyUnary(UnaryMathOp):
        """A dummy implementation of UnaryMathOp for testing base class behaviors.

        This class overrides the abstract methods of UnaryMathOp to allow
        instantiation
        and testing of inherited methods like infer_shape and eager_eval.
        """

        op_name = "Sin"

        def vjp(self, cotangent: BackendArg, x: BackendArg, **kwargs: BackendArg) -> None:
            """Computes the vector-Jacobian product for the dummy operation.

            Args:
            cotangent (BackendArg): The cotangent vector
            x (BackendArg): The input operand
            **kwargs (BackendArg): Additional keyword arguments

            Returns:
            OpValue: The computed vector-Jacobian product.
            """

        def jvp(self, tangent: BackendArg, x: BackendArg, **kwargs: BackendArg) -> None:
            """Computes the Jacobian-vector product for the dummy operation.

            Args:
            tangent (BackendArg): The tangent vector
            x (BackendArg): The input operand
            **kwargs (BackendArg): Additional keyword arguments

            Returns:
            OpValue: The computed Jacobian-vector product.
            """

    op = DummyUnary()
    assert op.infer_shape((10,)) == (10,)
    assert np.allclose(ensure_ndarray(op.eager_eval(0.0)), 0.0)


def test_unary_special_coverage() -> None:
    """Test the unary special coverage behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    x = np.array([1.5, 2.5])
    cast_op = Cast()
    res1 = cast_op.eager_eval(x, dtype=DType.Int32)
    assert extract_test_dtype(res1) == np.int32
    res1_str = cast_op.eager_eval(x, dtype="int32")
    assert extract_test_dtype(res1_str) == np.int32
    bitcast_op = Bitcast()
    x_int = np.array([1, 2], dtype=np.int32)
    res2 = bitcast_op.eager_eval(x_int, dtype=DType.Float32)
    assert extract_test_dtype(res2) == np.float32
    res2_str = bitcast_op.eager_eval(x_int, dtype="float32")
    assert extract_test_dtype(res2_str) == np.float32
    frexp_op = Frexp()
    res_frexp = frexp_op.eager_eval(x)
    assert isinstance(res_frexp, tuple)
    assert len(res_frexp) == 2


def test_unary_special_coverage_brute() -> None:
    """Test the unary special coverage brute behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    c = Cast()
    bc = Bitcast()
    c.eager_eval(np.array([1, 2]), dtype="float32")
    _out = bc.eager_eval(np.array([1, 2], dtype=np.int32), dtype="float32")
    assert np.asarray(_out).shape == (2,) and bool(np.allclose(np.asarray(_out, dtype=float), [0.0, 0.0], atol=1e-3))


class _MockDimWithId:
    """Mock dimension exposing an id attribute."""

    def __init__(self, dim_id: str) -> None:
        """Initialize mock dimension.

        Args:
            dim_id (str): Identifier for dimension.
        """
        self.id = dim_id


class _MockShapeContainer(MockNodeStub):
    """Mock container exposing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock shape container.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        self.shape = shape


class _MockShapeMetaContainer(MockNodeStub):
    """Mock container exposing a shape_metadata attribute."""

    def __init__(self, shape_metadata: tuple[int, ...]) -> None:
        """Initialize mock shape metadata container.

        Args:
            shape_metadata (tuple[int, ...]): Shape metadata tuple.
        """
        self.shape_metadata = shape_metadata


class _MockGenerator:
    """Mock generator for variable AST visitor."""

    def get_fallback_prefix(self) -> str:
        """Get fallback prefix for AST emissions.

        Returns:
            str: Fallback prefix string.
        """
        return "mock_backend"


def test_unary_base_exhaustive() -> None:
    """Verify UnaryMathOp shape inference across empty, tuple, list, shape attr, and scalar inputs."""
    op = unary_base.UnaryMathOp()

    # 1. Empty shapes
    assert op.infer_shape() == ()

    # 2. Tuple and list inputs
    assert op.infer_shape((2, 3)) == (2, 3)
    assert op.infer_shape([4, 5]) == (4, 5)

    # 3. Object with shape attribute
    assert op.infer_shape(_MockShapeContainer((6, 7))) == (6, 7)

    # 4. Object with shape_metadata attribute
    assert op.infer_shape(_MockShapeMetaContainer((8, 9))) == (8, 9)

    # 5. Scalar input without shape attributes
    assert require_callable(op.infer_shape)(123) == ()
