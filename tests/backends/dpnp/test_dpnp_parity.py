"""Parity and execution validation test suite for Data Parallel NumPy (dpnp)."""

from __future__ import annotations

import sys
import typing
from unittest import mock

import numpy as np
import pytest
from ml_switcheroo_ir import NodeDict

from ml_switcheroo_compiler.backends.dpnp import eager as dpnp_eager
from ml_switcheroo_compiler.backends.dpnp.eager import _get_target_function, execute_op
from ml_switcheroo_compiler.backends.dpnp.generator import DPNPGenerator
from ml_switcheroo_compiler.backends.dpnp.profiler import (
    DpnpProfiler,
    _get_dpnp_peak_memory_mb,
    _get_process_memory_mb,
    _sync_dpnp_result,
)
from ml_switcheroo_compiler.backends.dpnp.types import (
    _resolve_allocation_kwargs,
    array,
    asarray,
    asnumpy,
    empty,
    from_numpy,
    full,
    get_device,
    get_usm_type,
    is_sycl_array,
    item,
    ones,
    to_numpy,
    zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def _make_unary_graph() -> IRGraph:
    """Create unary operation graph for profiling.

    Returns:
        IRGraph: Computational graph with unary Abs.
    """
    g = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[2, 2])
    n_abs = IRNode(id="out", op_type="Abs", inputs=["x"], shape_metadata=[2, 2])
    g.nodes = NodeDict({"x": n_in, "out": n_abs})
    g.inputs = ["x"]
    g.outputs = ["out"]
    return g


def test_dpnp_unary_math_parity() -> None:
    """Verify numerical parity of unary mathematical ops between DPNP eager and NumPy."""
    x = np.array([0.2, 0.5, 0.8], dtype=np.float32)

    unary_ops = [
        ("Abs", np.abs),
        ("Neg", np.negative),
        ("Sign", np.sign),
        ("Sin", np.sin),
        ("Cos", np.cos),
        ("Tan", np.tan),
        ("Asin", np.arcsin),
        ("Acos", np.arccos),
        ("Atan", np.arctan),
        ("Sinh", np.sinh),
        ("Cosh", np.cosh),
        ("Tanh", np.tanh),
        ("Asinh", np.arcsinh),
        ("Acosh", np.arccosh),
        ("Atanh", np.arctanh),
        ("Exp", np.exp),
        ("Exp2", np.exp2),
        ("Expm1", np.expm1),
        ("Log", np.log),
        ("Log2", np.log2),
        ("Log10", np.log10),
        ("Log1p", np.log1p),
        ("Sqrt", np.sqrt),
        ("Square", np.square),
        ("Cbrt", np.cbrt),
        ("Reciprocal", np.reciprocal),
        ("Floor", np.floor),
        ("Ceil", np.ceil),
        ("Trunc", np.trunc),
        ("Rint", np.rint),
        ("Round", np.round),
    ]

    for op_name, np_fn in unary_ops:
        input_data = x + 1.0 if op_name == "Acosh" else x
        actual = execute_op(op_name, input_data)
        expected = typing.cast("EagerCallableProtocol", np_fn)(input_data)
        typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(actual, expected, rtol=1e-5, atol=1e-5)

    # Boolean & integer unary
    b = np.array([True, False, True])
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("LogicalNot", b), typing.cast("EagerCallableProtocol", np.logical_not)(b))

    i = np.array([1, 2, -3], dtype=np.int32)
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("BitwiseNot", i), typing.cast("EagerCallableProtocol", np.bitwise_not)(i))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Invert", i), typing.cast("EagerCallableProtocol", np.invert)(i))

    # Floating point predicates
    nan_arr = np.array([np.nan, 1.0, np.inf])
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Isnan", nan_arr), typing.cast("EagerCallableProtocol", np.isnan)(nan_arr))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Isinf", nan_arr), typing.cast("EagerCallableProtocol", np.isinf)(nan_arr))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Isfinite", nan_arr), typing.cast("EagerCallableProtocol", np.isfinite)(nan_arr))


def test_dpnp_binary_math_parity() -> None:
    """Verify numerical parity of binary mathematical ops between DPNP eager and NumPy."""
    a = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    b = np.array([3.0, 2.0, 1.5], dtype=np.float32)

    binary_ops = [
        ("Add", np.add),
        ("Sub", np.subtract),
        ("Mul", np.multiply),
        ("Div", np.divide),
        ("TrueDivide", np.true_divide),
        ("FloorDivide", np.floor_divide),
        ("Power", np.power),
        ("Maximum", np.maximum),
        ("Minimum", np.minimum),
        ("Fmax", np.fmax),
        ("Fmin", np.fmin),
        ("Fmod", np.fmod),
        ("Remainder", np.remainder),
        ("Hypot", np.hypot),
        ("Atan2", np.arctan2),
        ("Copysign", np.copysign),
        ("Nextafter", np.nextafter),
        ("Equal", np.equal),
        ("NotEqual", np.not_equal),
        ("Greater", np.greater),
        ("GreaterEqual", np.greater_equal),
        ("Less", np.less),
        ("LessEqual", np.less_equal),
    ]

    for op_name, np_fn in binary_ops:
        actual = execute_op(op_name, a, b)
        expected = typing.cast("EagerCallableProtocol", np_fn)(a, b)
        typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(actual, expected, rtol=1e-5, atol=1e-5)

    # Logical binary
    bool_a = np.array([True, True, False, False])
    bool_b = np.array([True, False, True, False])
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("LogicalAnd", bool_a, bool_b), typing.cast("EagerCallableProtocol", np.logical_and)(bool_a, bool_b))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("LogicalOr", bool_a, bool_b), typing.cast("EagerCallableProtocol", np.logical_or)(bool_a, bool_b))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("LogicalXor", bool_a, bool_b), typing.cast("EagerCallableProtocol", np.logical_xor)(bool_a, bool_b))

    # Bitwise binary
    int_a = np.array([1, 2, 4], dtype=np.int32)
    int_b = np.array([3, 2, 1], dtype=np.int32)
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("BitwiseAnd", int_a, int_b), typing.cast("EagerCallableProtocol", np.bitwise_and)(int_a, int_b))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("BitwiseOr", int_a, int_b), typing.cast("EagerCallableProtocol", np.bitwise_or)(int_a, int_b))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("BitwiseXor", int_a, int_b), typing.cast("EagerCallableProtocol", np.bitwise_xor)(int_a, int_b))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("LeftShift", int_a, typing.cast("EagerCallableProtocol", np.array)([1, 1, 1], dtype=np.int32)), typing.cast("EagerCallableProtocol", np.left_shift)(int_a, 1))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("RightShift", int_a, typing.cast("EagerCallableProtocol", np.array)([1, 1, 1], dtype=np.int32)), typing.cast("EagerCallableProtocol", np.right_shift)(int_a, 1))


def test_dpnp_reduction_parity() -> None:
    """Verify numerical parity of reduction operations between DPNP eager and NumPy."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Sum", x), typing.cast("EagerCallableProtocol", np.sum)(x))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Prod", x), typing.cast("EagerCallableProtocol", np.prod)(x))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Mean", x), typing.cast("EagerCallableProtocol", np.mean)(x))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Std", x), typing.cast("EagerCallableProtocol", np.std)(x))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Var", x), typing.cast("EagerCallableProtocol", np.var)(x))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Max", x), typing.cast("EagerCallableProtocol", np.max)(x))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Min", x), typing.cast("EagerCallableProtocol", np.min)(x))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Argmax", x), typing.cast("EagerCallableProtocol", np.argmax)(x))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Argmin", x), typing.cast("EagerCallableProtocol", np.argmin)(x))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Cumsum", x), typing.cast("EagerCallableProtocol", np.cumsum)(x))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Cumprod", x), typing.cast("EagerCallableProtocol", np.cumprod)(x))

    b_arr = np.array([[True, True], [False, True]])
    assert bool(execute_op("All", b_arr)) is False
    assert bool(execute_op("Any", b_arr)) is True


def test_dpnp_linalg_parity() -> None:
    """Verify numerical parity of linear algebra operations between DPNP eager and NumPy."""
    m1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    m2 = np.array([[2.0, 0.0], [1.0, 2.0]], dtype=np.float64)

    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Dot", m1, m2), typing.cast("EagerCallableProtocol", np.dot)(m1, m2))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Matmul", m1, m2), typing.cast("EagerCallableProtocol", np.matmul)(m1, m2))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Tensordot", m1, m2, axes=1), typing.cast("EagerCallableProtocol", np.tensordot)(m1, m2, axes=1))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(
        typing.cast("EagerCallableProtocol", execute_op)("Outer", typing.cast("EagerCallableProtocol", np.array)([1.0, 2.0]), typing.cast("EagerCallableProtocol", np.array)([3.0, 4.0])), typing.cast("EagerCallableProtocol", np.outer)([1.0, 2.0], [3.0, 4.0])
    )
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(
        typing.cast("EagerCallableProtocol", execute_op)("Inner", typing.cast("EagerCallableProtocol", np.array)([1.0, 2.0]), typing.cast("EagerCallableProtocol", np.array)([3.0, 4.0])), typing.cast("EagerCallableProtocol", np.inner)([1.0, 2.0], [3.0, 4.0])
    )
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Kron", m1, m2), typing.cast("EagerCallableProtocol", np.kron)(m1, m2))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Trace", m1), typing.cast("EagerCallableProtocol", np.trace)(m1))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Transpose", m1), typing.cast("EagerCallableProtocol", np.transpose)(m1))

    # Decompositions & solvers on positive-definite matrix
    pos_def = np.array([[4.0, 2.0], [2.0, 5.0]], dtype=np.float64)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Cholesky", pos_def), typing.cast("EagerCallableProtocol", np.linalg.cholesky)(pos_def))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Det", pos_def), typing.cast("EagerCallableProtocol", np.linalg.det)(pos_def))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Inv", pos_def), typing.cast("EagerCallableProtocol", np.linalg.inv)(pos_def))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerCallableProtocol", execute_op)("Norm", pos_def), typing.cast("EagerCallableProtocol", np.linalg.norm)(pos_def))

    q, r = typing.cast("EagerCallableProtocol", execute_op)("Qr", pos_def)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(q @ r, pos_def, atol=1e-5)

    u, s, vt = typing.cast("EagerCallableProtocol", execute_op)("Svd", pos_def)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(u @ typing.cast("EagerCallableProtocol", np.diag)(s) @ vt, pos_def, atol=1e-5)

    b_vec = np.array([1.0, 2.0], dtype=np.float64)
    sol = execute_op("Solve", pos_def, b_vec)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerTensorProtocol", pos_def) @ typing.cast("EagerTensorProtocol", sol), b_vec, atol=1e-5)


def test_dpnp_shape_and_manipulation() -> None:
    """Verify tensor shape and manipulation operations."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Reshape", x, (4,)), typing.cast("EagerCallableProtocol", x.reshape)(4))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Flatten", x), typing.cast("EagerCallableProtocol", x.ravel)())
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("ExpandDims", x, axis=0), typing.cast("EagerCallableProtocol", np.expand_dims)(x, axis=0))
    sq_in = np.expand_dims(x, axis=0)
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Squeeze", sq_in), typing.cast("EagerCallableProtocol", np.squeeze)(sq_in))

    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Concatenate", (x, x), axis=0), typing.cast("EagerCallableProtocol", np.concatenate)((x, x), axis=0))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Stack", (x, x), axis=0), typing.cast("EagerCallableProtocol", np.stack)((x, x), axis=0))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Tile", x, (2, 1)), typing.cast("EagerCallableProtocol", np.tile)(x, (2, 1)))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Repeat", x, 2), typing.cast("EagerCallableProtocol", np.repeat)(x, 2))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Clip", x, 2.0, 3.0), typing.cast("EagerCallableProtocol", np.clip)(x, 2.0, 3.0))

    cond = np.array([[True, False], [False, True]])
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Where", cond, x, -x), typing.cast("EagerCallableProtocol", np.where)(cond, x, -x))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Flip", x), typing.cast("EagerCallableProtocol", np.flip)(x))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", execute_op)("Roll", x, 1), typing.cast("EagerCallableProtocol", np.roll)(x, 1))


def test_dpnp_eager_helpers_and_tensor_unwrap() -> None:
    """Verify argument unwrapping, linalg submodules, and registry fallback."""
    cfg = TensorConfig((2,), "float32", "cpu")
    t1 = Tensor(np.array([1.0, 2.0]), cfg)
    t2 = Tensor(np.array([3.0, 4.0]), cfg)

    # Tensor unwrapping
    res = execute_op("Add", t1, t2)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, [4.0, 6.0])

    # Resolving target functions in nested paths
    assert _get_target_function(np, "linalg.inv") is not None
    assert _get_target_function(np, "nonexistent.attr") is None
    assert _get_target_function(np, "abs") is not None

    # Eager registry fallback
    def mock_custom_op(mod: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> str:
        """Test function mock_custom_op.

        Args:
            mod: Test parameter `mod`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        del mod, args, kwargs
        return "custom_registered_result"

    with mock.patch.dict(global_eager_registry._registry, {"CustomRegistryOp": mock_custom_op}):
        assert execute_op("CustomRegistryOp", 1) == "custom_registered_result"

    # Unsupported op
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        execute_op("CompletelyUnknownNonexistentOp")


def test_dpnp_types_buffer_management() -> None:
    """Verify USM allocation, device management, and buffer conversions."""
    # Allocation with various device / usm targets
    kwargs_auto = _resolve_allocation_kwargs(device="auto", usm_type="device")
    assert kwargs_auto == {"usm_type": "device"}

    kwargs_other = _resolve_allocation_kwargs(device="auto", usm_type="unsupported_usm")
    assert kwargs_other == {}

    kwargs_gpu = _resolve_allocation_kwargs(device="gpu", usm_type="shared")
    assert kwargs_gpu == {"device": "gpu", "usm_type": "shared"}

    fake_q = mock.MagicMock()
    kwargs_q = _resolve_allocation_kwargs(sycl_queue=fake_q, usm_type="host")
    assert kwargs_q == {"sycl_queue": fake_q, "usm_type": "host"}

    # Basic creation calls
    z = zeros((3, 3), dtype="float32", device="cpu", usm_type="host")
    assert typing.cast("EagerTensorProtocol", getattr(z, "shape")) == (3, 3)

    o = ones((2, 2), dtype="float64")
    assert typing.cast("EagerTensorProtocol", getattr(o, "shape")) == (2, 2)
    assert np.all(o == 1.0)

    e = empty((2, 4))
    assert typing.cast("EagerTensorProtocol", getattr(e, "shape")) == (2, 4)

    f = full((2, 3), 7.0, dtype="float32")
    assert typing.cast("EagerTensorProtocol", getattr(f, "shape")) == (2, 3)
    assert np.all(f == 7.0)

    f_cls = typing.cast("EagerTensorProtocol", getattr(DPNPGenerator, "full"))((2, 2), 9.0)
    assert f_cls.shape == (2, 2)
    assert np.all(f_cls == 9.0)

    arr = array([10.0, 20.0], dtype="float32")
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(arr, [10.0, 20.0])

    as_arr = asarray([30.0, 40.0])
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(as_arr, [30.0, 40.0])

    # Host NumPy bridge (asnumpy, to_numpy, from_numpy)
    host_arr = asnumpy(arr)
    assert isinstance(host_arr, np.ndarray)
    assert to_numpy(arr) is not None

    class CustomAsNumpyObj:
        """Object exposing custom asnumpy."""

        def asnumpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
            """Extract numpy array.

            Returns:
                np.ndarray: Array.
            """
            return np.array([99.0])

    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", asnumpy)(typing.cast("EagerCallableProtocol", CustomAsNumpyObj)()), "__getitem__")(0)) == 99.0

    from_np = from_numpy(np.array([5.0, 6.0]), device="auto", usm_type="shared")
    assert from_np is not None

    # Device & USM inspection
    class SyclBufferObj:
        """Mock SYCL buffer with device and USM attributes."""

        device = "gpu:0"
        usm_type = "shared"
        sycl_queue = fake_q

    mock_sycl = SyclBufferObj()
    assert typing.cast("EagerCallableProtocol", get_device)(mock_sycl) == "gpu:0"
    assert typing.cast("EagerCallableProtocol", get_usm_type)(mock_sycl) == "shared"
    assert bool(typing.cast("EagerCallableProtocol", is_sycl_array)(mock_sycl)) is True

    class AltSyclObj:
        """Mock object with sycl_device."""

        sycl_device = "cpu:0"

    assert typing.cast("EagerCallableProtocol", get_device)(typing.cast("EagerCallableProtocol", AltSyclObj)()) == "cpu:0"
    assert typing.cast("EagerCallableProtocol", get_usm_type)(typing.cast("EagerCallableProtocol", AltSyclObj)()) is None
    assert is_sycl_array(np.array([1])) is False
    assert get_device(np.array([1])) == "cpu"

    # Item extraction
    assert item(np.array([3.14])) == pytest.approx(3.14)


def test_dpnp_profiler_and_timing() -> None:
    """Verify DPNP execution profiler, timing iterations, and memory metrics."""
    g = _make_unary_graph()
    fake_queue = mock.MagicMock()
    fake_queue.wait.return_value = None

    profiler = DpnpProfiler(sycl_queue=fake_queue)
    metrics = profiler.profile_graph(g, inputs={"x": [[1.0, -2.0], [-3.0, 4.0]]}, device="auto", num_iters=3, warmup_iters=1)

    assert "latencies_ms" in metrics
    assert typing.cast("EagerCallableProtocol", len)(metrics["latencies_ms"]) == 3
    assert getattr(metrics, "__getitem__")("mean_ms") >= 0.0
    assert getattr(metrics, "__getitem__")("median_ms") >= 0.0
    assert getattr(metrics, "__getitem__")("min_ms") >= 0.0
    assert getattr(metrics, "__getitem__")("max_ms") >= 0.0
    assert getattr(metrics, "__getitem__")("peak_memory_mb") >= 0.0

    # Synchronization helper coverage
    fake_res = mock.MagicMock()
    fake_res.sycl_queue = fake_queue
    _sync_dpnp_result(fake_res, sycl_queue=fake_queue)

    fake_waitable = mock.MagicMock()
    del fake_waitable.sycl_queue
    fake_waitable.wait.return_value = None
    _sync_dpnp_result(fake_waitable)
    _sync_dpnp_result([fake_waitable, fake_waitable])
    _sync_dpnp_result({"a": fake_waitable})

    # Memory inspection coverage
    assert _get_process_memory_mb() > 0.0
    mock_dpnp = mock.MagicMock()
    mock_dpnp.get_device_memory_info.return_value = (1024 * 1024 * 50, 1024 * 1024 * 10)
    with mock.patch.dict(sys.modules, {"dpnp": mock_dpnp}):
        assert _get_dpnp_peak_memory_mb() == pytest.approx(40.0)

    # Empty / failing runner path
    with mock.patch.object(profiler, "_compile_runner", return_value=None):
        empty_metrics = profiler.profile_graph(g, inputs={})
        assert empty_metrics["latencies_ms"] == []
        assert empty_metrics["mean_ms"] == 0.0


def test_dpnp_generator_visitors_and_targets() -> None:
    """Verify DPNP generator dedicated visitors and target architecture configurations."""
    g = _make_unary_graph()
    fake_queue = mock.MagicMock()

    gen_gpu = DPNPGenerator(g, device="gpu", sycl_queue=fake_queue)
    assert gen_gpu.device == "gpu"
    code = gen_gpu.generate()
    assert "import dpnp" in code
    assert "# Target SYCL Device: gpu" in code
    assert "# Active SYCL Queue configuration attached" in code

    gen_fpga = DPNPGenerator(g, device="fpga")
    code_fpga = gen_fpga.generate()
    assert "# Target SYCL Device: fpga" in code_fpga

    # Node visitors
    node_mock = mock.MagicMock()
    assert gen_gpu.visit_Add(node_mock, ["v1", "v2"]) == "dpnp.add(v1, v2)"
    assert gen_gpu.visit_Sub(node_mock, ["v1", "v2"]) == "dpnp.subtract(v1, v2)"
    assert gen_gpu.visit_Mul(node_mock, ["v1", "v2"]) == "dpnp.multiply(v1, v2)"
    assert gen_gpu.visit_Div(node_mock, ["v1", "v2"]) == "dpnp.divide(v1, v2)"
    assert gen_gpu.visit_MatMul(node_mock, ["v1", "v2"]) == "dpnp.matmul(v1, v2)"


def _test_dpnp_exhaustive_edge_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_dpnp_exhaustive_edge_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    with mock.patch("sys.platform", "linux"):
        rss = _get_process_memory_mb()
        assert rss > 0.0

    # 2. Peak memory info branch variations
    mock_dpnp = mock.MagicMock()
    mock_dpnp.get_device_memory_info.return_value = [100]  # < 2 items
    with mock.patch.dict(sys.modules, {"dpnp": mock_dpnp}):
        assert _get_dpnp_peak_memory_mb() > 0.0

    mock_dpnp.get_device_memory_info.return_value = ("non_int", "non_int")  # non-numeric
    with mock.patch.dict(sys.modules, {"dpnp": mock_dpnp}):
        assert _get_dpnp_peak_memory_mb() > 0.0

    mock_dpnp.get_device_memory_info.side_effect = RuntimeError("device error")
    with mock.patch.dict(sys.modules, {"dpnp": mock_dpnp}):
        assert _get_dpnp_peak_memory_mb() > 0.0

    # 3. Exception suppression in _sync_dpnp_result
    bad_q = mock.MagicMock()
    bad_q.wait.side_effect = RuntimeError("sync error")
    _sync_dpnp_result("dummy", sycl_queue=bad_q)

    bad_res_q = mock.MagicMock()
    bad_res_q.sycl_queue.wait.side_effect = RuntimeError("res sync error")
    _sync_dpnp_result(bad_res_q)

    bad_waitable = mock.MagicMock()
    del bad_waitable.sycl_queue
    bad_waitable.wait.side_effect = RuntimeError("wait error")
    getattr(_state, "update")({"bad_q": bad_q, "bad_res_q": bad_res_q, "bad_waitable": bad_waitable, "mock_dpnp": mock_dpnp})
    getattr(_state, "update")({"rss": rss})
    _state.update(locals())


def _test_dpnp_exhaustive_edge_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_dpnp_exhaustive_edge_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bad_waitable = _state["bad_waitable"]
    _sync_dpnp_result(bad_waitable)

    # 4. _prepare_inputs edge cases (scalars, mismatching keys, lists, ndarray, Tensor)
    profiler = DpnpProfiler()
    g = _make_unary_graph()
    cfg = TensorConfig((2,), "float32", "cpu")
    t_obj = Tensor(np.array([1.0, 2.0]), cfg)
    prepared_inputs = profiler._prepare_inputs(
        g,
        {
            "tensor_key": t_obj,
            "arr_key": np.array([3.0, 4.0]),
            "list_key": [5.0, 6.0],
            "scalar_key": 42,
        },
    )
    assert len(prepared_inputs) == 4

    # Profiler without get_device_memory_info
    class NoMemInfoModule:
        """Module without device memory info query."""

        pass

    with mock.patch.dict(sys.modules, {"dpnp": NoMemInfoModule}):
        assert _get_dpnp_peak_memory_mb() > 0.0

    # Generator auto device branch (device == 'auto' skips line 59)
    gen_auto = DPNPGenerator(g, device="auto")
    code_auto = gen_auto.generate()
    assert "# Target SYCL Device" not in code_auto
    getattr(_state, "update")({"NoMemInfoModule": NoMemInfoModule, "cfg": cfg, "code_auto": code_auto, "g": g})
    getattr(_state, "update")({"gen_auto": gen_auto, "prepared_inputs": prepared_inputs, "profiler": profiler, "t_obj": t_obj})
    _state.update(locals())


def _check_usm(kwargs: dict[str, EagerValue]) -> None:
    """Raise TypeError if usm_type is in kwargs.

    Args:
        kwargs: Test parameter `kwargs`.
    """
    if "usm_type" in kwargs:
        raise TypeError("Unsupported kwarg usm_type")


def __test_dpnp_exhaustive_edge_coverage_part_3_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of _test_dpnp_exhaustive_edge_coverage_part_3.

    Args:
        _state: Shared test state dictionary.
    """

    class FallbackModule:
        """Mock module rejecting SYCL kwargs to trigger TypeError fallback."""

        float32 = float

        @staticmethod
        def zeros(shape: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function zeros.

            Args:
                shape: Test parameter `shape`.
                kwargs: Test parameter `kwargs`.
            """
            _check_usm(kwargs)
            return typing.cast("EagerCallableProtocol", np.zeros)(shape, dtype=typing.cast("EagerCallableProtocol", kwargs.get)("dtype", float))

        @staticmethod
        def ones(shape: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function ones.

            Args:
                shape: Test parameter `shape`.
                kwargs: Test parameter `kwargs`.
            """
            _check_usm(kwargs)
            return typing.cast("EagerCallableProtocol", np.ones)(shape, dtype=typing.cast("EagerCallableProtocol", kwargs.get)("dtype", float))

        @staticmethod
        def empty(shape: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function empty.

            Args:
                shape: Test parameter `shape`.
                kwargs: Test parameter `kwargs`.
            """
            _check_usm(kwargs)
            return typing.cast("EagerCallableProtocol", np.empty)(shape, dtype=typing.cast("EagerCallableProtocol", kwargs.get)("dtype", float))

        @staticmethod
        def full(shape: EagerValue, fill_value: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function full.

            Args:
                shape: Test parameter `shape`.
                fill_value: Test parameter `fill_value`.
                kwargs: Test parameter `kwargs`.
            """
            _check_usm(kwargs)
            return typing.cast("EagerCallableProtocol", np.full)(shape, fill_value, dtype=typing.cast("EagerCallableProtocol", kwargs.get)("dtype", None))

        @staticmethod
        def array(data: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function array.

            Args:
                data: Test parameter `data`.
                kwargs: Test parameter `kwargs`.
            """
            _check_usm(kwargs)
            return typing.cast("EagerCallableProtocol", np.array)(data, dtype=typing.cast("EagerCallableProtocol", kwargs.get)("dtype", None))

        @staticmethod
        def asarray(data: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function asarray.

            Args:
                data: Test parameter `data`.
                kwargs: Test parameter `kwargs`.
            """
            _check_usm(kwargs)
            return typing.cast("EagerCallableProtocol", np.asarray)(data, dtype=typing.cast("EagerCallableProtocol", kwargs.get)("dtype", None))

        @staticmethod
        def asnumpy(a: EagerValue) -> EagerValue:
            """Test function asnumpy.

            Args:
                a: Test parameter `a`.
            """
            raise RuntimeError("asnumpy failure")

    getattr(_state, "update")({"FallbackModule": FallbackModule})
    _state.update(locals())


def __test_dpnp_exhaustive_edge_coverage_part_3_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of _test_dpnp_exhaustive_edge_coverage_part_3.

    Args:
        _state: Shared test state dictionary.
    """
    FallbackModule = _state["FallbackModule"]
    getattr(_state, "update")({"FallbackModule": FallbackModule})
    _state.update(locals())
    _state.update(locals())


def _test_dpnp_exhaustive_edge_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_dpnp_exhaustive_edge_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    getattr(_state, "__setitem__")("_state", _state)
    for _sub_fn in (
        __test_dpnp_exhaustive_edge_coverage_part_3_sub_1,
        __test_dpnp_exhaustive_edge_coverage_part_3_sub_2,
    ):
        _sub_fn(_state)
    assert len(_state) > 0


def _test_dpnp_exhaustive_edge_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_dpnp_exhaustive_edge_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    FallbackModule = _state["FallbackModule"]
    with mock.patch("ml_switcheroo_compiler.backends.dpnp.types._get_dpnp_module", return_value=FallbackModule):
        assert typing.cast("EagerTensorProtocol", getattr(zeros((2, 2), dtype="nonexistent_type", usm_type="shared"), "shape")) == (2, 2)
        assert typing.cast("EagerTensorProtocol", getattr(ones((2, 2), usm_type="shared"), "shape")) == (2, 2)
        assert typing.cast("EagerTensorProtocol", getattr(empty((2, 2), usm_type="shared"), "shape")) == (2, 2)
        assert typing.cast("EagerTensorProtocol", getattr(full((2, 2), 5.0, usm_type="shared"), "shape")) == (2, 2)
        assert typing.cast("EagerTensorProtocol", getattr(array([1, 2], usm_type="shared"), "shape")) == (2,)
        assert typing.cast("EagerTensorProtocol", getattr(asarray([1, 2], usm_type="shared"), "shape")) == (2,)

        # asnumpy fallback when dpnp_mod.asnumpy raises
        res_host = asnumpy(np.array([42]))
        assert typing.cast("EagerTensorProtocol", getattr(res_host, "__getitem__")(0)) == 42

    # 6. DPNPGenerator classmethods for buffer management
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", DPNPGenerator.zeros)((1,), dtype=None), "shape")) == (1,)
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", DPNPGenerator.zeros)((1,), dtype="float32"), "shape")) == (1,)
    assert typing.cast("EagerTensorProtocol", getattr(DPNPGenerator, "ones"))((1,), dtype=None).shape == (1,)
    assert typing.cast("EagerTensorProtocol", getattr(DPNPGenerator, "ones"))((1,), dtype="float32").shape == (1,)
    assert typing.cast("EagerTensorProtocol", getattr(DPNPGenerator, "empty"))((1,), dtype=None).shape == (1,)
    assert typing.cast("EagerTensorProtocol", getattr(DPNPGenerator, "empty"))((1,), dtype="float32").shape == (1,)
    assert typing.cast("EagerTensorProtocol", getattr(DPNPGenerator, "full"))((1,), 3.0, dtype=None).shape == (1,)
    assert typing.cast("EagerTensorProtocol", getattr(DPNPGenerator, "full"))((1,), 3.0, dtype="float32").shape == (1,)
    assert typing.cast("EagerTensorProtocol", getattr(DPNPGenerator.array([1], dtype=None), "shape")) == (1,)
    assert typing.cast("EagerTensorProtocol", getattr(DPNPGenerator.array([1], dtype="float32"), "shape")) == (1,)
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", DPNPGenerator.asarray)([1], dtype=None), "shape")) == (1,)
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", DPNPGenerator.asarray)([1], dtype="float32"), "shape")) == (1,)
    assert typing.cast("EagerTensorProtocol", getattr(DPNPGenerator, "asnumpy"))(np.array([1]))[0] == 1
    getattr(_state, "update")({"res_host": res_host})
    _state.update(locals())


def _test_dpnp_exhaustive_edge_coverage_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_dpnp_exhaustive_edge_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    assert typing.cast("EagerTensorProtocol", getattr(DPNPGenerator, "to_numpy"))(np.array([1]))[0] == 1
    assert typing.cast("EagerTensorProtocol", getattr(DPNPGenerator, "from_numpy"))(np.array([1])) is not None
    assert typing.cast("EagerTensorProtocol", getattr(DPNPGenerator, "get_device"))(np.array([1])) == "cpu"
    assert typing.cast("EagerTensorProtocol", getattr(DPNPGenerator, "get_usm_type"))(np.array([1])) is None
    assert bool(typing.cast("EagerTensorProtocol", getattr(DPNPGenerator, "is_sycl_array"))(np.array([1]))) is False

    # 7. Eager dispatch edge cases: op directly in linalg, alias resolution failure, no linalg submodule
    class LinalgOnlyModule:
        """Module with direct linalg function."""

        non_callable_root = 123

        class linalg:
            """Test class linalg."""

            @staticmethod
            def custom_linalg_op(x: EagerValue) -> EagerValue:
                """Test function custom_linalg_op.

                Args:
                    x: Test parameter `x`.
                """
                return typing.cast("EagerTensorProtocol", x) + typing.cast("EagerTensorProtocol", 100)

            non_callable = 42

    with mock.patch.dict(sys.modules, {"dpnp": LinalgOnlyModule}):
        assert typing.cast("EagerTensorProtocol", getattr(execute_op("custom_linalg_op", np.array([5])), "__getitem__")(0)) == 105
        with pytest.raises(BackendNotSupportedError):
            execute_op("non_callable_root")
        with pytest.raises(BackendNotSupportedError):
            execute_op("non_callable", np.array([5]))
    getattr(_state, "update")({"LinalgOnlyModule": LinalgOnlyModule})
    _state.update(locals())


def _test_dpnp_exhaustive_edge_coverage_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_dpnp_exhaustive_edge_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class NoLinalgModule:
        """Module lacking linalg submodule."""

        pass

    with mock.patch.dict(sys.modules, {"dpnp": NoLinalgModule}):
        with pytest.raises(BackendNotSupportedError):
            execute_op("nonexistent_op")
    with mock.patch.dict(dpnp_eager._DPNP_OP_ALIASES, {"fake_op": "missing.submodule.fn"}):
        with pytest.raises(BackendNotSupportedError):
            execute_op("fake_op", 1)
    getattr(_state, "update")({"NoLinalgModule": NoLinalgModule})


def test_dpnp_exhaustive_edge_coverage() -> None:
    """Verify edge conditions across eager dispatch, buffer management, and profiler."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_dpnp_exhaustive_edge_coverage_part_1(_state)
    _test_dpnp_exhaustive_edge_coverage_part_2(_state)
    _test_dpnp_exhaustive_edge_coverage_part_3(_state)
    _test_dpnp_exhaustive_edge_coverage_part_4(_state)
    _test_dpnp_exhaustive_edge_coverage_part_5(_state)
    _test_dpnp_exhaustive_edge_coverage_part_6(_state)
    assert len(_state) > 0
