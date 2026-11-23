"""Reverse-mode and higher-order automatic differentiation public API."""

from __future__ import annotations

import typing
from collections.abc import Callable, Mapping, Sequence

from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.errors import AutodiffError
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    TypedVariadicCallable,
    VariadicCallableProtocol,
    is_variadic_callable,
)
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.interpreter.evaluator import InterpreterValue
from ml_switcheroo_compiler.ops.registry import register_util
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import VJPRuleCallable, register_vjp

from .custom_vjp_ops import custom_vjp
from .options import GradOptions
from .utils import (
    DifferentiableFn,
    PrimalAuxValue,
    PrimalTree,
    _check_scalar,
    _compute_grad_and_value,
    _find_wrt_tensors,
    _get_inputs_dict,
    _IdDataProtocol,
    as_primal_tree,
)


@typing.runtime_checkable
class _ExecuteGraphBackendProtocol(typing.Protocol):
    """Protocol for backends implementing `execute_graph(graph, inputs_dict)`."""

    def execute_graph(self, graph: LogicalGraph, inputs_dict: dict[str, InterpreterValue]) -> Mapping[str, OpValue]:
        """Execute a computation graph with concrete inputs.

        Args:
            graph (LogicalGraph): Target graph.
            inputs_dict (dict[str, InterpreterValue]): Concrete input mapping.

        Returns:
            Mapping[str, OpValue]: Output mapping.
        """
        ...


@typing.runtime_checkable
class _CompileGraphBackendProtocol(typing.Protocol):
    """Protocol for backends implementing `compile_graph(graph)`."""

    def compile_graph(self, graph: LogicalGraph) -> TypedVariadicCallable[Mapping[str, OpValue]]:
        """Compile a computation graph into a callable.

        Args:
            graph (LogicalGraph): Target graph.

        Returns:
            TypedVariadicCallable[Mapping[str, OpValue]]: Compiled graph callable.
        """
        ...


@typing.runtime_checkable
class _AddableProtocol(typing.Protocol):
    """Protocol for operands supporting addition (`__add__`)."""

    def __add__(self, other: OpValue) -> OpValue:
        """Add `other` to `self`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            OpValue: Sum result.
        """
        ...


@typing.runtime_checkable
class _MulFloatProtocol(typing.Protocol):
    """Protocol for tangent operands supporting scalar float multiplication."""

    def __mul__(self, other: float) -> OpValue:
        """Multiply tangent by ``other``.

        Args:
            other (float): Scalar step size.

        Returns:
            OpValue: Scaled tangent value.
        """
        ...


@typing.runtime_checkable
class _DivFloatProtocol(typing.Protocol):
    """Protocol for difference values supporting scalar float division."""

    def __truediv__(self, other: float) -> OpValue:
        """Divide difference by ``other``.

        Args:
            other (float): Scalar step size.

        Returns:
            OpValue: Finite-difference directional derivative.
        """
        ...


@typing.runtime_checkable
class _SubDivProtocol(typing.Protocol):
    """Protocol for primal outputs supporting subtraction into a divisible difference."""

    def __sub__(self, other: OpValue) -> OpValue | _DivFloatProtocol:
        """Subtract ``other`` from ``self``.

        Args:
            other (OpValue): Base primal output.

        Returns:
            OpValue | _DivFloatProtocol: Difference value.
        """
        ...


def _as_mul_float(val: PrimalTree | OpValue | _MulFloatProtocol) -> _MulFloatProtocol:
    """Verify and narrow ``val`` to ``_MulFloatProtocol``.

    Args:
        val (PrimalTree | OpValue | _MulFloatProtocol): Candidate tangent value.

    Returns:
        _MulFloatProtocol: Verified multipliable operand.
    """
    assert isinstance(val, _MulFloatProtocol)
    return val


def _as_sub_div(val: OpValue | _SubDivProtocol) -> _SubDivProtocol:
    """Verify and narrow ``val`` to ``_SubDivProtocol``.

    Args:
        val (OpValue | _SubDivProtocol): Candidate primal output.

    Returns:
        _SubDivProtocol: Verified subtractable output.

    Raises:
        TypeError: If ``val`` does not support subtraction.
    """
    if not isinstance(val, _SubDivProtocol):
        raise TypeError(f"Output of type {type(val).__name__} does not support subtraction")
    return val


def _as_div_float(val: OpValue | _DivFloatProtocol) -> _DivFloatProtocol:
    """Verify and narrow ``val`` to ``_DivFloatProtocol``.

    Args:
        val (OpValue | _DivFloatProtocol): Candidate difference value.

    Returns:
        _DivFloatProtocol: Verified divisible difference.
    """
    assert isinstance(val, _DivFloatProtocol)
    return val


def _perturb_operand(
    primal: _AddableProtocol,
    tangent: PrimalTree | OpValue | _MulFloatProtocol,
    eps: float,
) -> OpValue:
    """Compute ``primal + tangent * eps`` via runtime-checked arithmetic protocols.

    Args:
        primal (_AddableProtocol): Primal operand.
        tangent (PrimalTree | OpValue | _MulFloatProtocol): Tangent operand.
        eps (float): Perturbation step size.

    Returns:
        OpValue: Perturbed primal value.
    """
    return primal + (_as_mul_float(tangent) * eps)


def _finite_diff_quotient(
    out_eps: OpValue | _SubDivProtocol,
    primal_out: OpValue,
    eps: float,
) -> PrimalTree:
    """Compute ``(out_eps - primal_out) / eps`` via runtime-checked arithmetic protocols.

    Args:
        out_eps (OpValue | _SubDivProtocol): Perturbed function output.
        primal_out (OpValue): Unperturbed primal output.
        eps (float): Perturbation step size.

    Returns:
        PrimalTree: Directional derivative estimate.
    """
    diff = _as_sub_div(out_eps) - primal_out
    return as_primal_tree(_as_div_float(diff) / eps)


def _is_execute_graph_backend(val: OpValue) -> typing.TypeGuard[_ExecuteGraphBackendProtocol]:
    """Check whether `val` exposes `execute_graph`.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` has `execute_graph`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _ExecuteGraphBackendProtocol)
    return found or is_mock_with_attrs(val, "execute_graph")


def _is_compile_graph_backend(val: OpValue) -> typing.TypeGuard[_CompileGraphBackendProtocol]:
    """Check whether `val` exposes `compile_graph`.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` has `compile_graph`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _CompileGraphBackendProtocol)
    return found or is_mock_with_attrs(val, "compile_graph")


def _is_addable_operand(val: OpValue) -> typing.TypeGuard[_AddableProtocol]:
    """Check whether `val` exposes `__add__`.

    Args:
        val (OpValue): Candidate operand.

    Returns:
        bool: True if `val` supports addition.
    """
    return isinstance(val, _AddableProtocol)


def _has_id_data_attr(val: OpValue) -> typing.TypeGuard[_IdDataProtocol]:
    """Check whether `val` exposes `id`.

    Args:
        val (OpValue): Candidate proxy data.

    Returns:
        bool: True if `val` has `id`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _IdDataProtocol)
    return found or is_mock_with_attrs(val, "id")


@register_util("backward")
def backward(tensor: Tensor | PrimalTree | OpValue, *args: OpValue, **kwargs: OpValue) -> None:
    """Trigger reverse-mode automatic differentiation from a scalar traced `tensor`.

    Args:
        tensor (Tensor | PrimalTree | OpValue): Scalar output `Tensor` to differentiate.
        *args (OpValue): Optional additional arguments.
        **kwargs (OpValue): Optional additional keyword arguments.

    Raises:
        TracingError: When tracing is not active or `tensor` is not a traced `Tensor`.
    """
    _ = (args, kwargs)
    from ml_switcheroo_compiler.core.errors import TracingError

    if (
        not isinstance(tensor, Tensor)
        or not global_tracing_state.is_tracing
        or global_tracing_state.active_graph is None
    ):
        raise TracingError("Cannot execute backward(): tracing is not active or target is not a traced Tensor.")

    _check_scalar(tensor)

    graph = global_tracing_state.active_graph
    wrt_tensors, wrt_ids = _find_wrt_tensors(graph)

    raw_data: OpValue = tensor.data
    loss_id = str(raw_data.id) if _has_id_data_attr(raw_data) and raw_data.id is not None else str(tensor.data)

    if not wrt_ids:
        tensor.grad = 1.0
        return

    from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad

    grad_graph = graph_grad(graph, wrt_ids, loss_id)
    inputs_dict = _get_inputs_dict(graph)

    backend: OpValue = get_active_backend()
    outputs_dict: Mapping[str, OpValue]
    if _is_execute_graph_backend(backend):
        outputs_dict = backend.execute_graph(grad_graph, inputs_dict)
    elif _is_compile_graph_backend(backend):
        compiled_fn = backend.compile_graph(grad_graph)
        outputs_dict = compiled_fn(inputs_dict)
    else:
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        eval_inputs: dict[str, InterpreterValue] = dict(inputs_dict)
        outputs_dict = evaluate_graph(grad_graph, eval_inputs)

    for i in range(len(wrt_ids)):
        grad_node_id = grad_graph.outputs[i]
        if grad_node_id in outputs_dict:
            grad_val = outputs_dict[grad_node_id]
            t = wrt_tensors[i]
            assert isinstance(t, Tensor)
            t.grad = grad_val


def RegisterGradient(op_type: str) -> Callable[[VJPRuleCallable], VJPRuleCallable]:
    """Return a decorator registering a custom VJP gradient rule for `op_type`.

    Args:
        op_type (str): Canonical IR operation name.

    Returns:
        Callable[[VJPRuleCallable], VJPRuleCallable]: Decorator registering the VJP rule function.
    """
    return register_vjp(op_type)


def overwrite_with_gradient(tensor: PrimalTree, gradient: PrimalTree) -> PrimalTree:
    """Return `tensor` in the forward pass while replacing its backward gradient with `gradient`.

    Args:
        tensor (PrimalTree): Input primal tensor.
        gradient (PrimalTree): Replacement gradient value used during the backward pass.

    Returns:
        PrimalTree: Primal tensor with overridden backward gradient rule.
    """

    @custom_vjp
    def _overwrite(t: PrimalTree, g: PrimalTree) -> PrimalTree:
        """Identity forward operation with custom VJP override.

        Args:
            t (PrimalTree): Primal tensor.
            g (PrimalTree): Replacement gradient tensor.

        Returns:
            PrimalTree: Unmodified primal tensor `t`.
        """
        _ = g
        return t

    def _overwrite_fwd(t: PrimalTree, g: PrimalTree) -> tuple[PrimalTree, PrimalTree]:
        """Forward pass saving `g` as the residual for backward propagation.

        Args:
            t (PrimalTree): Primal tensor.
            g (PrimalTree): Replacement gradient tensor.

        Returns:
            tuple[PrimalTree, PrimalTree]: Tuple of `(t, g)`.
        """
        return t, g

    def _overwrite_bwd(g: PrimalTree, g_in: PrimalTree) -> tuple[PrimalTree, None]:
        """Backward pass returning the stored `g` in place of `g_in`.

        Args:
            g (PrimalTree): Stored replacement gradient.
            g_in (PrimalTree): Incoming upstream cotangent.

        Returns:
            tuple[PrimalTree, None]: Tuple of `(g, None)`.
        """
        _ = g_in
        return (gradient if g is None else g), None

    _overwrite.defvjp(_overwrite_fwd, _overwrite_bwd)
    return _overwrite(tensor, gradient)


def ir_grad(fun: DifferentiableFn, options: GradOptions | None = None) -> VariadicCallableProtocol:
    """Wrap `fun` to compute its symbolic IR gradient with respect to specified arguments.

    Args:
        fun (DifferentiableFn): Differentiable function to wrap.
        options (GradOptions | None): Optional `GradOptions` configuration.

    Returns:
        VariadicCallableProtocol: Callable computing the gradient of `fun`.
    """
    opts = options if isinstance(options, GradOptions) else GradOptions()

    def wrapped(
        *args: OpValue,
        **kwargs: OpValue,
    ) -> PrimalTree:
        """Compute gradient of `fun` at `*args`.

        Args:
            *args (OpValue): Positional primal arguments.
            **kwargs (OpValue): Unused keyword arguments.

        Returns:
            PrimalTree: Evaluated gradient tensor(s).
        """
        _ = kwargs
        _, grads = _compute_grad_and_value(fun, opts, args)
        return grads

    return wrapped


def grad(
    fun: DifferentiableFn,
    options: GradOptions | None = None,
) -> VariadicCallableProtocol:
    """Create a function that evaluates the gradient of `fun`.

    Args:
        fun (DifferentiableFn): Function to differentiate.
        options (GradOptions | None): Optional differentiation options (`argnums`, `has_aux`).

    Returns:
        VariadicCallableProtocol: Gradient function with the same input signature as `fun`.
    """
    options = options or GradOptions()

    def wrapped(
        *args: OpValue,
        **kwargs: OpValue,
    ) -> PrimalTree | tuple[PrimalTree, OpValue]:
        """Evaluate the gradient of `fun` at `*args`.

        Args:
            *args (OpValue): Positional primal arguments.
            **kwargs (OpValue): Unused keyword arguments.

        Returns:
            PrimalTree | tuple[PrimalTree, OpValue]: Gradient tensor(s), or `(grads, aux)` when
                `options.has_aux` is True.
        """
        _ = kwargs
        val, grads = _compute_grad_and_value(fun, options, args)
        if options.has_aux and isinstance(val, (tuple, list)) and len(val) >= 2:
            return grads, val[1]
        return grads

    return wrapped


def value_and_grad(
    fun: DifferentiableFn,
    options: GradOptions | None = None,
) -> VariadicCallableProtocol:
    """Create a function that evaluates both the primal value and gradient of `fun`.

    Args:
        fun (DifferentiableFn): Function to differentiate.
        options (GradOptions | None): Optional `GradOptions` configuration.

    Returns:
        VariadicCallableProtocol: Callable returning `(value, grads)`.
    """
    opts = options if isinstance(options, GradOptions) else GradOptions()

    def wrapped(
        *args: OpValue,
        **kwargs: OpValue,
    ) -> tuple[OpValue, PrimalTree]:
        """Evaluate both primal value and gradient of `fun` at `*args`.

        Args:
            *args (OpValue): Positional primal arguments.
            **kwargs (OpValue): Unused keyword arguments.

        Returns:
            tuple[OpValue, PrimalTree]: Tuple of `(value, grads)`.
        """
        _ = kwargs
        val, grads = _compute_grad_and_value(fun, opts, args)
        if opts.has_aux:
            return val, grads
        return val, grads

    return wrapped


def hook_gradient(tensor: PrimalTree, hook: DifferentiableFn) -> PrimalTree:
    """Attach a custom backward gradient transformation `hook` to `tensor`.

    Args:
        tensor (PrimalTree): Input tensor.
        hook (DifferentiableFn): Callable invoked on the upstream gradient during backprop.

    Returns:
        PrimalTree: Tensor with the backward gradient hook attached.
    """

    @custom_vjp
    def _hook_op(t: PrimalTree) -> PrimalTree:
        """Identity forward operation with custom gradient hook.

        Args:
            t (PrimalTree): Input tensor.

        Returns:
            PrimalTree: Unchanged input tensor `t`.
        """
        return t

    def _hook_fwd(t: PrimalTree) -> tuple[PrimalTree, PrimalTree]:
        """Forward pass for gradient hook.

        Args:
            t (PrimalTree): Input tensor.

        Returns:
            tuple[PrimalTree, PrimalTree]: Tuple of `(t, t)`.
        """
        return t, t

    def _hook_bwd(res: PrimalTree, g_in: PrimalTree) -> tuple[PrimalTree]:
        """Backward pass invoking `hook(g_in)`.

        Args:
            res (PrimalTree): Saved primal residual.
            g_in (PrimalTree): Incoming upstream gradient.

        Returns:
            tuple[PrimalTree]: 1-tuple containing the hooked gradient.
        """
        _ = res
        _callable_fn_1: OpValue = hook
        assert is_variadic_callable(_callable_fn_1)
        out_g = _callable_fn_1(g_in)
        if out_g is None:
            return (g_in,)
        return (as_primal_tree(out_g),)

    _hook_op.defvjp(_hook_fwd, _hook_bwd)
    return _hook_op(tensor)


def jvp(
    fun: DifferentiableFn,
    primals: Sequence[PrimalTree],
    tangents: Sequence[PrimalTree],
) -> tuple[PrimalTree | tuple[PrimalTree, PrimalAuxValue], PrimalTree]:
    """Compute a Jacobian-vector product (forward-mode directional derivative).

    Args:
        fun (DifferentiableFn): Function to differentiate.
        primals (Sequence[PrimalTree]): Primal point inputs.
        tangents (Sequence[PrimalTree]): Tangent vectors corresponding to `primals`.

    Returns:
        tuple[PrimalTree | tuple[PrimalTree, PrimalAuxValue], PrimalTree]: Tuple of `(primal_out, tangent_out)`.

    Raises:
        AutodiffError: If symbolic or finite-difference differentiation fails in `jvp`.
    """
    from .jvp_vjp import jvp as _symbolic_jvp

    primals_tuple = tuple(primals)
    tangents_tuple = tuple(tangents)
    if all(isinstance(p, (Tensor, int, float)) for p in primals_tuple):
        try:
            sym_primal, sym_tangent = _symbolic_jvp(fun, primals_tuple, tangents_tuple)
            return sym_primal, sym_tangent
        except AutodiffError:
            raise
        except (TypeError, ValueError, RuntimeError):
            pass
    _callable_fn_2: OpValue = fun
    assert is_variadic_callable(_callable_fn_2)
    primal_out_raw = _callable_fn_2(*primals_tuple)
    primal_out = as_primal_tree(primal_out_raw)

    eps = 1e-5
    perturbed: list[OpValue] = []
    for p, t in zip(primals_tuple, tangents_tuple):
        raw_p: OpValue = p
        if _is_addable_operand(raw_p):
            try:
                perturbed.append(_perturb_operand(raw_p, t, eps))
            except (TypeError, ValueError, RuntimeError, ArithmeticError) as exc:
                msg = f"Failed to perturb primal operand during JVP evaluation: {exc}"
                raise AutodiffError(msg) from exc
        else:
            perturbed.append(p)
    _callable_fn_3: OpValue = fun
    assert is_variadic_callable(_callable_fn_3)
    out_eps = _callable_fn_3(*perturbed)
    try:
        tangent_out: PrimalTree = _finite_diff_quotient(out_eps, primal_out_raw, eps)
    except (TypeError, ValueError, RuntimeError, ArithmeticError) as exc:
        msg = f"Failed to compute directional derivative tangent in JVP: {exc}"
        raise AutodiffError(msg) from exc
    return primal_out, tangent_out


def hvp_graph(
    graph: LogicalGraph,
    primals: list[str],
    tangents: list[str],
    outputs: list[str] | None = None,
    mode: str = "forward-over-reverse",
    projected_tangents: list[str] | None = None,
) -> LogicalGraph:
    """Compute a symbolic Hessian-vector product IR graph transform.

    Args:
        graph (LogicalGraph): Input computation graph.
        primals (list[str]): Input primal variable node IDs.
        tangents (list[str]): Tangent vector node IDs corresponding to `primals`.
        outputs (list[str] | None): Target output node IDs (defaults to `graph.outputs`).
        mode (str): HVP computation mode (`'forward-over-reverse'` or `'reverse-over-forward'`).
        projected_tangents (list[str] | None): Tangents for projecting non-scalar outputs.

    Returns:
        LogicalGraph: Symbolic Hessian-vector product computation graph.
    """
    from ml_switcheroo_compiler.transforms.autodiff import hvp as graph_hvp

    outs = outputs if outputs is not None else list(graph.outputs)
    return graph_hvp(
        graph,
        primals,
        tangents,
        outs,
        mode=mode,
        projected_tangents=projected_tangents,
    )


def hvp(
    fun: DifferentiableFn,
    primals: Sequence[PrimalTree] | PrimalTree,
    tangents: Sequence[PrimalTree] | PrimalTree,
    has_aux: bool = False,
    mode: str = "forward-over-reverse",
    projected_tangents: Tensor | Sequence[Tensor] | None = None,
) -> tuple[PrimalTree | tuple[PrimalTree, PrimalAuxValue], PrimalTree]:
    """Compute a Hessian-vector product via symbolic graph transformation.

    Args:
        fun (DifferentiableFn): Function to evaluate Hessian-vector product on.
        primals (Sequence[PrimalTree] | PrimalTree): Primal evaluation inputs.
        tangents (Sequence[PrimalTree] | PrimalTree): Tangent direction vectors.
        has_aux (bool): Whether `fun` returns auxiliary outputs.
        mode (str): Evaluation mode (`'forward-over-reverse'` or `'reverse-over-forward'`).
        projected_tangents (Tensor | Sequence[Tensor] | None): Projected cotangents for non-scalar outputs.

    Returns:
        tuple[PrimalTree | tuple[PrimalTree, PrimalAuxValue], PrimalTree]: Tuple of `(fun(*primals), hvp_result)`.
    """
    from .jvp_vjp import hvp as _jvp_vjp_hvp

    raw_res = _jvp_vjp_hvp(
        fun,
        primals,
        tangents,
        has_aux=has_aux,
        projected_tangents=projected_tangents,
        mode=mode,
    )
    return (raw_res[0], raw_res[1])


def nth_order_grad_graph(
    graph: LogicalGraph,
    wrt: list[str],
    output_id: str,
    n: int = 1,
) -> LogicalGraph:
    """Compute the N-th order symbolic derivative graph via chained VJP passes.

    Args:
        graph (LogicalGraph): Initial computation graph.
        wrt (list[str]): Input variable IDs to differentiate with respect to.
        output_id (str): Target scalar output node ID.
        n (int): Derivative order (`n >= 1`).

    Returns:
        LogicalGraph: N-th order gradient computation graph.

    Raises:
        ValueError: If `n < 1`.
    """
    if n < 1:
        raise ValueError(f"Derivative order n must be >= 1, got {n}")
    from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad

    current_graph = graph
    curr_target = output_id
    for _ in range(n):
        current_graph = graph_grad(current_graph, wrt, curr_target)
        curr_target = current_graph.outputs[0]
    return current_graph


def nth_order_grad(
    fun: DifferentiableFn,
    n: int = 1,
    options: GradOptions | None = None,
) -> VariadicCallableProtocol:
    """Return an N-th order gradient function via repeated symbolic differentiation.

    Args:
        fun (DifferentiableFn): Function to differentiate.
        n (int): Order of the derivative (`n >= 1`).
        options (GradOptions | None): Optional differentiation options.

    Returns:
        VariadicCallableProtocol: The N-th order derivative function.

    Raises:
        ValueError: If `n < 1`.
    """
    if n < 1:
        raise ValueError(f"Derivative order n must be >= 1, got {n}")
    res_fn: VariadicCallableProtocol = grad(fun, options=options)
    for _ in range(n - 1):
        res_fn = grad(res_fn, options=options)
    return res_fn
