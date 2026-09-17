# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Gradient computation and autodiff utilities."""

from __future__ import annotations

import contextlib
import math
import typing
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.ops.registry import register_util
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp

from .custom_vjp_ops import custom_vjp
from .options import GradOptions, JitOptions
from .utils import _check_scalar, _compute_grad_and_value, _find_wrt_tensors, _get_inputs_dict


@register_util("backward")
def backward(tensor, *args, **kwargs) -> None:
    """Triggers the reverse-mode auto-differentiation.

    Args:
        tensor (object): The tensor to compute gradients for.
        *args (object): Additional arguments.
        **kwargs (object): Additional keyword arguments.

    Raises:
        TracingError: When tracing is not active or target is not a traced Tensor.
    """
    from ml_switcheroo_compiler.core.errors import TracingError
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    # 1. Disallow dummy fallback; raise explicit TracingError if tracing is inactive
    if not isinstance(tensor, Tensor) or not global_tracing_state.is_tracing or global_tracing_state.active_graph is None:
        raise TracingError("Cannot execute backward(): tracing is not active or target is not a traced Tensor.")

    # 2. Validate that the target tensor is a scalar
    _check_scalar(tensor)

    # 3. Extract the active graph
    graph = global_tracing_state.active_graph

    # 4. Find all active variables/tensors in memory that require gradients and are in the graph
    wrt_tensors, wrt_ids = _find_wrt_tensors(graph)

    # 5. Extract target loss node ID
    loss_id = getattr(tensor.data, "id", None)
    if loss_id is None:
        loss_id = str(tensor.data)

    if not wrt_ids:
        # No variables require gradients
        tensor.grad = 1.0
        return

    # 6. Run the graph-level gradient generator
    from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad

    grad_graph = graph_grad(graph, wrt_ids, loss_id)

    # 7. Map input node IDs to their concrete values for evaluation
    inputs_dict = _get_inputs_dict(graph)

    # 8. Lower gradient graph to native backend execution pipeline or fallback
    backend = get_active_backend()
    if backend is not None and hasattr(backend, "execute_graph"):
        outputs_dict = backend.execute_graph(grad_graph, inputs_dict)
    elif backend is not None and hasattr(backend, "compile_graph"):
        compiled_fn = backend.compile_graph(grad_graph)
        outputs_dict = compiled_fn(inputs_dict)
    else:
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        outputs_dict = evaluate_graph(grad_graph, inputs_dict)

    # 9. Traverse the wrt list and assign computed gradient arrays to .grad attributes
    for i in range(len(wrt_ids)):
        grad_node_id = grad_graph.outputs[i]
        if grad_node_id in outputs_dict:
            grad_val = outputs_dict[grad_node_id]
            t = wrt_tensors[i]
            t.grad = grad_val


def RegisterGradient(op_type: str) -> typing.Callable:
    """Register a custom gradient for an operation.

    Args:
        op_type (str): The name of the operation.

    Returns:
        Callable: The decorator function.
    """
    return register_vjp(op_type)


def overwrite_with_gradient(tensor, gradient):
    """Overwrite the gradient of the tensor in the backward pass.

    During the forward pass, this returns the `tensor` unchanged.
    During the backward pass, it replaces the propagated gradient with `gradient`.

    Args:
        tensor (object): The input tensor.
        gradient (object): The gradient value to use in the backward pass.

    Returns: Tensor: The tensor with the overridden backward pass gradient.
    """

    @custom_vjp
    def _overwrite(t, g):
        """Overwrite the gradient during the backward pass.

        Args:
            t (object): The primal tensor.
            g (object): The gradient to overwrite with.

        Returns: Tensor: The primal tensor unchanged.
        """
        return t

    def _overwrite_fwd(t, g):
        """Forward pass for overwriting a gradient.

        Args:
            t (object): The primal tensor.
            g (object): The gradient to overwrite with.

        Returns:
            tuple[object, object]: The primal tensor and the gradient to store for backward.
        """
        return t, g

    def _overwrite_bwd(g, g_in):
        """Backward pass for overwriting a gradient.

        Args:
            g (object): The stored gradient to overwrite with.
            g_in (object): The incoming upstream gradient.

        Returns:
            tuple[object, object]: The overwritten gradient and None for the second argument.
        """
        return g, None

    _overwrite.defvjp(_overwrite_fwd, _overwrite_bwd)
    return _overwrite(tensor, gradient)


def ir_grad(fun, options=None):
    """Return a gradient wrapper.

    Args:
        fun (Callable): Function to differentiate.
        options (GradOptions): Differentiation options.

    Returns:
        Callable: The wrapper function.
    """
    options = options or GradOptions()

    def wrapped(*args, **kwargs):
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        _, grads = _compute_grad_and_value(fun, options, args)
        return grads

    return wrapped


def grad(fun, options=None):
    """Return a gradient wrapper.

    Args:
        fun (Callable): Function to differentiate.
        options (GradOptions): Differentiation options.

    Returns:
        Callable: The wrapper function.
    """
    options = options or GradOptions()

    def wrapped(*args, **kwargs):
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        val, grads = _compute_grad_and_value(fun, options, args)
        if options.has_aux:
            return grads, val[1]
        return grads

    return wrapped


def value_and_grad(fun, options=None):
    """Return a gradient wrapper.

    Args:
        fun (Callable): Function to differentiate.
        options (GradOptions): Differentiation options.

    Returns:
        Callable: The wrapper function.
    """
    options = options or GradOptions()

    def wrapped(*args, **kwargs):
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        val, grads = _compute_grad_and_value(fun, options, args)
        if options.has_aux:
            return val, grads
        return val, grads

    return wrapped


def hook_gradient(tensor, hook):
    """Register a custom gradient hook on a tensor.

    During the forward pass, this returns the `tensor` unchanged.
    During the backward pass, the `hook` function is called with the upstream gradient,
    and its return value (if not None) replaces the gradient.

    Args:
        tensor (object): The input tensor.
        hook (Callable): The hook function applied to the gradient.

    Returns:
        object: The tensor with the gradient hook attached.
    """

    @custom_vjp
    def _hook_op(t):
        """Apply the hook op.

        Args:
            t (object): The input tensor.

        Returns:
            object: The unchanged tensor.
        """
        return t

    def _hook_fwd(t):
        """Forward pass for the hook.

        Args:
            t (object): The input tensor.

        Returns:
            tuple[object, object]: The primal and None.
        """
        return t, t

    def _hook_bwd(res, g_in):
        """Backward pass for the hook.

        Args:
            res (object): The stored result.
            g_in (object): The incoming upstream gradient.

        Returns:
            tuple[object]: The processed gradient.
        """
        out_g = hook(g_in)
        if out_g is None:
            out_g = g_in
        return (out_g,)

    _hook_op.defvjp(_hook_fwd, _hook_bwd)
    return _hook_op(tensor)


def jvp(
    fun: Callable,
    primals: Sequence[object],
    tangents: Sequence[object],
) -> tuple[object, object]:
    """Compute a Jacobian-vector product (forward pushforward).

    Args:
        fun (Callable): The function to differentiate.
        primals (Sequence[object]): Primal point inputs.
        tangents (Sequence[object]): Tangent vectors corresponding to primals.

    Returns:
        tuple[object, object]: Tuple of (primal_out, tangent_out).
    """
    primals_tuple = tuple(primals)
    tangents_tuple = tuple(tangents)
    primal_out = fun(*primals_tuple)

    eps = 1e-5
    perturbed = []
    for p, t in zip(primals_tuple, tangents_tuple):
        if hasattr(p, "__add__"):
            try:
                perturbed.append(p + t * eps)
            except Exception:
                perturbed.append(p)
        else:
            perturbed.append(p)

    out_eps = fun(*perturbed)
    try:
        tangent_out = (out_eps - primal_out) / eps
    except Exception:
        tangent_out = primal_out
    return primal_out, tangent_out


def hvp_graph(
    graph: LogicalGraph,
    primals: list[str],
    tangents: list[str],
    outputs: list[str] | None = None,
    mode: str = "forward-over-reverse",
    projected_tangents: list[str] | None = None,
) -> LogicalGraph:
    """Compute symbolic Hessian-vector product IR graph transform.

    Lowers an HVP operation by applying a forward-mode JVP transformation
    directly to the reverse-mode VJP cotangent graph without runtime function re-wrapping.

    Args:
        graph (LogicalGraph): Input computation graph.
        primals (list[str]): Input primal variable node IDs.
        tangents (list[str]): Tangent vector node IDs corresponding to primals.
        outputs (Optional[list[str]]): Target output node IDs. If None, defaults to graph.outputs.
        mode (str): HVP computation mode ('forward-over-reverse' or 'reverse-over-forward').
        projected_tangents (Optional[list[str]]): Tangents for projecting non-scalar outputs.

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
    fun: Callable,
    primals: Sequence[object] | object,
    tangents: Sequence[object] | object,
    has_aux: bool = False,
    mode: str = "forward-over-reverse",
    projected_tangents: Tensor | Sequence[Tensor] | None = None,
) -> tuple[object, object]:
    """Compute a Hessian-vector product via symbolic graph transformation.

    Eliminates runtime function re-wrapping by delegating directly to symbolic
    lowering of JVP passes over the reverse-mode VJP cotangent graph.

    Args:
        fun (Callable): Function to evaluate Hessian-vector product on.
        primals (Sequence[object] | object): Primal evaluation inputs.
        tangents (Sequence[object] | object): Tangent direction vectors.
        has_aux (bool): Whether fun returns auxiliary outputs.
        mode (str): Evaluation mode ('forward-over-reverse' or 'reverse-over-forward').
        projected_tangents (Optional[Union[Tensor, Sequence[Tensor]]]): Projected cotangents.

    Returns:
        tuple[object, object]: Tuple of (fun(*primals), hvp_result).
    """
    from .jvp_vjp import hvp as _jvp_vjp_hvp

    return _jvp_vjp_hvp(
        fun,
        primals,
        tangents,
        has_aux=has_aux,
        projected_tangents=projected_tangents,
        mode=mode,
    )


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
        n (int): Derivative order (n >= 1).

    Returns:
        LogicalGraph: N-th order gradient computation graph.

    Raises:
        ValueError: If n < 1.
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
    fun: Callable,
    n: int = 1,
    options: GradOptions | None = None,
) -> Callable:
    """Return an N-th order gradient function via repeated symbolic differentiation.

    Args:
        fun (Callable): Function to differentiate.
        n (int): Order of the derivative (n >= 1).
        options (Optional[GradOptions]): Differentiation options.

    Returns:
        Callable: The N-th order derivative function.

    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError(f"Derivative order n must be >= 1, got {n}")
    curr_fn = fun
    for _ in range(n):
        curr_fn = grad(curr_fn, options=options)
    return curr_fn
