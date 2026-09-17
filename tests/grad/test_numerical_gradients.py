"""Finite-difference numerical gradient and higher-order derivative validation tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import hvp, jvp, vjp
from ml_switcheroo_compiler.grad.symbolic_compiler import SymbolicExpressionCompiler
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op


def _create_tensor(
    data: list[float] | list[list[float]] | float | np.ndarray,
    shape: tuple[int, ...] = (),
    dtype: DType = DType.Float32,
) -> Tensor:
    """Helper to instantiate a Tensor on CPU for gradient testing.

    Args:
        data (list[float] | list[list[float]] | float | np.ndarray): Input numeric data.
        shape (tuple[int, ...]): Explicit tensor dimensions.
        dtype (DType): Tensor data type.

    Returns:
        Tensor: Constructed tensor instance.
    """
    np_dtype = np.float32 if dtype == DType.Float32 else np.float64
    arr = np.array(data, dtype=np_dtype)
    actual_shape = shape if shape else arr.shape
    return Tensor(arr, TensorConfig(actual_shape, dtype, Device("cpu")))


def _compute_finite_difference_grad(
    func: Callable[[Tensor], Tensor],
    x: Tensor,
    eps: float = 1e-3,
) -> np.ndarray:
    """Compute numerical gradient using central finite differences.

    Args:
        func (Callable[[Tensor], Tensor]): Scalar-valued tensor function.
        x (Tensor): Input tensor evaluation point.
        eps (float): Step size for perturbation.

    Returns:
        np.ndarray: Finite difference gradient array.
    """
    x_arr = np.array(x.data, dtype=np.float32)
    grad_arr = np.zeros_like(x_arr)
    it = np.nditer(x_arr, flags=["multi_index"])

    with ConfigContext(eager_mode=True):
        while not it.finished:
            idx = it.multi_index
            orig = float(x_arr[idx])

            x_arr[idx] = orig + eps
            pos_t = _create_tensor(x_arr, x.shape)
            f_pos = float(np.sum(func(pos_t).data))

            x_arr[idx] = orig - eps
            neg_t = _create_tensor(x_arr, x.shape)
            f_neg = float(np.sum(func(neg_t).data))

            grad_arr[idx] = (f_pos - f_neg) / (2.0 * eps)
            x_arr[idx] = orig
            it.iternext()

    return grad_arr


def _compute_finite_difference_hvp(
    func: Callable[[Tensor], Tensor],
    x: Tensor,
    v: Tensor,
    eps: float = 1e-3,
) -> np.ndarray:
    """Compute numerical Hessian-vector product using central differences of gradients.

    Args:
        func (Callable[[Tensor], Tensor]): Scalar-valued function.
        x (Tensor): Primal evaluation point.
        v (Tensor): Tangent vector for directional second derivative.
        eps (float): Perturbation step size.

    Returns:
        np.ndarray: Approximated Hessian-vector product array.
    """
    x_arr = np.array(x.data, dtype=np.float32)
    v_arr = np.array(v.data, dtype=np.float32)

    pos_x = _create_tensor(x_arr + eps * v_arr, x.shape)
    neg_x = _create_tensor(x_arr - eps * v_arr, x.shape)

    grad_pos = _compute_finite_difference_grad(func, pos_x, eps=eps)
    grad_neg = _compute_finite_difference_grad(func, neg_x, eps=eps)

    return (grad_pos - grad_neg) / (2.0 * eps)


def test_symbolic_compiler_higher_order_rules_manifest() -> None:
    """Verify that all declarative higher-order rules exist and validate in the manifest."""
    compiler = SymbolicExpressionCompiler()
    manifest = compiler.load_manifest()
    assert len(manifest.higher_order_rules) >= 15

    expected_ops = [
        "Add",
        "Sub",
        "Neg",
        "Negative",
        "Mul",
        "Div",
        "Exp",
        "Log",
        "Sin",
        "Cos",
        "Sinh",
        "Cosh",
        "Tanh",
        "Sqrt",
        "Pow",
        "Sigmoid",
        "Relu",
        "Abs",
        "MatMul",
        "BatchMatMul",
        "Transpose",
        "Reshape",
    ]
    for op in expected_ops:
        rule = compiler.get_higher_order_rule(op)
        assert rule is not None, f"Missing higher-order rule for {op}"
        assert rule.hvp is not None
        assert rule.jvp_order >= 2
        assert rule.vjp_order >= 2


def test_symbolic_compiler_compile_hvp_emission() -> None:
    """Test emission of higher-order derivative nodes via compile_hvp."""
    compiler = SymbolicExpressionCompiler()
    graph = IRGraph()
    node = LogicalNode(id="n0", op_type="Mul", inputs=["x", "y"], shape_metadata=[2, 2])
    graph.nodes["n0"] = node

    hvp_id = compiler.compile_hvp(graph, "Mul", node, cotangent="cot", tangents=["tx", "ty"])
    assert hvp_id is not None
    assert hvp_id in graph.nodes

    unmapped = compiler.compile_hvp(graph, "NonExistentOpXYZ", node, cotangent="cot", tangents=["tx", "ty"])
    assert unmapped is None


def test_numerical_gradient_unary_functions() -> None:
    """Validate unary math functions VJP against central finite differences."""
    unary_cases: list[tuple[Callable[[Tensor], Tensor], list[float]]] = [
        (lambda t: dispatch_op("Sin", t), [0.3, 0.7, 1.2, -0.5]),
        (lambda t: dispatch_op("Cos", t), [0.1, -0.4, 0.9, 1.5]),
        (lambda t: dispatch_op("Exp", t), [0.1, -0.2, 0.5, 0.0]),
        (lambda t: dispatch_op("Sinh", t), [0.2, -0.3, 0.4, 0.1]),
        (lambda t: dispatch_op("Cosh", t), [0.1, 0.5, -0.2, 0.3]),
        (lambda t: dispatch_op("Tanh", t), [0.3, -0.6, 0.2, -0.1]),
        (lambda t: dispatch_op("Abs", t), [0.5, -0.8, 1.2, -1.5]),
    ]

    for func, vals in unary_cases:
        x = _create_tensor(vals, (len(vals),))
        num_grad = _compute_finite_difference_grad(func, x, eps=1e-3)

        with ConfigContext(eager_mode=True):
            out, vjp_fn = vjp(func, x)
            cot = _create_tensor([1.0] * len(vals), (len(vals),))
            (sym_grad,) = vjp_fn(cot)

        sym_arr = np.array(sym_grad.data, dtype=np.float32)
        diff = np.max(np.abs(sym_arr - num_grad))
        assert diff < 5e-3, f"Gradient mismatch for {func}: max diff {diff}"


def test_numerical_gradient_binary_functions() -> None:
    """Validate binary operations VJP against central finite differences."""
    a_vals = [1.5, 2.0, 3.5]
    b_vals = [2.0, 1.5, 0.5]

    def func_add(x: Tensor) -> Tensor:
        """Elementwise addition test function."""
        b = _create_tensor(b_vals, (3,))
        return x + b

    def func_sub(x: Tensor) -> Tensor:
        """Elementwise subtraction test function."""
        b = _create_tensor(b_vals, (3,))
        return x - b

    def func_mul(x: Tensor) -> Tensor:
        """Elementwise multiplication test function."""
        b = _create_tensor(b_vals, (3,))
        return x * b

    for func in [func_add, func_sub, func_mul]:
        x = _create_tensor(a_vals, (3,))
        num_grad = _compute_finite_difference_grad(func, x, eps=1e-3)

        with ConfigContext(eager_mode=True):
            out, vjp_fn = vjp(func, x)
            cot = _create_tensor([1.0, 1.0, 1.0], (3,))
            (sym_grad,) = vjp_fn(cot)

        sym_arr = np.array(sym_grad.data, dtype=np.float32)
        diff = np.max(np.abs(sym_arr - num_grad))
        assert diff < 1e-3


def test_numerical_jvp_verification() -> None:
    """Validate JVP forward derivative against directional finite difference perturbation."""

    def func(t: Tensor) -> Tensor:
        """Composite nonlinear evaluation function."""
        return dispatch_op("Sin", t) * t

    x = _create_tensor([0.5, 1.0, 1.5], (3,))
    v = _create_tensor([1.0, 0.5, -0.5], (3,))

    eps = 1e-4
    with ConfigContext(eager_mode=True):
        x_plus = _create_tensor(np.array(x.data) + eps * np.array(v.data), (3,))
        x_minus = _create_tensor(np.array(x.data) - eps * np.array(v.data), (3,))
        f_plus = func(x_plus)
        f_minus = func(x_minus)
        num_jvp = (np.array(f_plus.data) - np.array(f_minus.data)) / (2.0 * eps)

        _, sym_jvp = jvp(func, (x,), (v,))

    sym_arr = np.array(sym_jvp.data, dtype=np.float32)
    assert np.allclose(sym_arr, num_jvp, atol=1e-3, rtol=1e-3)


def test_numerical_hvp_verification() -> None:
    """Validate HVP higher-order derivative against finite difference of gradients."""

    def func(x: Tensor) -> Tensor:
        """Cubic and trigonometric scalar function."""
        return x * x * x + dispatch_op("Sin", x)

    x = _create_tensor([1.2], (1,))
    v = _create_tensor([0.8], (1,))

    num_hvp = _compute_finite_difference_hvp(func, x, v, eps=1e-3)

    with ConfigContext(eager_mode=True):
        val, sym_hvp = hvp(func, x, v)

    sym_arr = np.array(sym_hvp.data, dtype=np.float32)
    assert np.allclose(sym_arr, num_hvp, atol=5e-2, rtol=5e-2)
