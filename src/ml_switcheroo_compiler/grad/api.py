"""Reverse-mode and higher-order automatic differentiation public API."""

from __future__ import annotations

import typing
from collections.abc import Callable, Sequence
from typing import cast

from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.interpreter.evaluator import EvalValue
from ml_switcheroo_compiler.ops.registry import register_util
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp

from .custom_vjp_ops import custom_vjp
from .options import GradOptions
from .utils import GradValue, _check_scalar, _compute_grad_and_value, _find_wrt_tensors, _get_inputs_dict


@register_util("backward")
def backward(tensor: GradValue, *args: GradValue, **kwargs: GradValue) -> None:
    """Trigger reverse-mode automatic differentiation from a scalar traced `tensor`.

    Args:
        tensor (GradValue): Scalar output `Tensor` to differentiate.
        *args (GradValue): Optional additional arguments.
        **kwargs (GradValue): Optional additional keyword arguments.

    Raises:
        TracingError: When tracing is not active or `tensor` is not a traced `Tensor`.
    """
    _ = (args, kwargs)
    from ml_switcheroo_compiler.core.errors import TracingError

    if not isinstance(tensor, Tensor) or not global_tracing_state.is_tracing or global_tracing_state.active_graph is None:
        raise TracingError("Cannot execute backward(): tracing is not active or target is not a traced Tensor.")

    _check_scalar(tensor)

    graph = global_tracing_state.active_graph
    wrt_tensors, wrt_ids = _find_wrt_tensors(graph)

    loss_id = getattr(tensor.data, "id", None)
    if loss_id is None:
        loss_id = str(tensor.data)

    if not wrt_ids:
        tensor.grad = 1.0
        return

    from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad

    grad_graph = graph_grad(graph, wrt_ids, loss_id)
    inputs_dict = _get_inputs_dict(graph)

    backend = get_active_backend()
    if backend is not None and hasattr(backend, "execute_graph"):
        outputs_dict = backend.execute_graph(grad_graph, inputs_dict)
    elif backend is not None and hasattr(backend, "compile_graph"):
        compiled_fn = backend.compile_graph(grad_graph)
        outputs_dict = compiled_fn(inputs_dict)
    else:
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        outputs_dict = evaluate_graph(grad_graph, cast("dict[str, EvalValue]", inputs_dict))

    for i in range(len(wrt_ids)):
        grad_node_id = grad_graph.outputs[i]
        if grad_node_id in outputs_dict:
            grad_val = outputs_dict[grad_node_id]
            t = wrt_tensors[i]
            setattr(t, "grad", grad_val)


def RegisterGradient(op_type: str) -> typing.Callable[..., GradValue]:
    """Return a decorator registering a custom VJP gradient rule for `op_type`.

    Args:
        op_type (str): Canonical IR operation name.

    Returns:
        Callable[..., GradValue]: Decorator registering the VJP rule function.
    """
    return typing.cast("typing.Callable[..., GradValue]", register_vjp(op_type))


def overwrite_with_gradient(tensor: GradValue, gradient: GradValue) -> GradValue:
    """Return `tensor` in the forward pass while replacing its backward gradient with `gradient`.

    Args:
        tensor (GradValue): Input primal tensor.
        gradient (GradValue): Replacement gradient value used during the backward pass.

    Returns:
        GradValue: Primal tensor with overridden backward gradient rule.
    """

    @custom_vjp
    def _overwrite(t: GradValue, g: GradValue) -> GradValue:
        """Identity forward operation with custom VJP override.

        Args:
            t (GradValue): Primal tensor.
            g (GradValue): Replacement gradient tensor.

        Returns:
            GradValue: Unmodified primal tensor `t`.
        """
        _ = g
        return t

    def _overwrite_fwd(t: GradValue, g: GradValue) -> GradValue:
        """Forward pass saving `g` as the residual for backward propagation.

        Args:
            t (GradValue): Primal tensor.
            g (GradValue): Replacement gradient tensor.

        Returns:
            GradValue: Tuple of `(t, g)`.
        """
        return t, g

    def _overwrite_bwd(g: GradValue, g_in: GradValue) -> GradValue:
        """Backward pass returning the stored `g` in place of `g_in`.

        Args:
            g (GradValue): Stored replacement gradient.
            g_in (GradValue): Incoming upstream cotangent.

        Returns:
            GradValue: Tuple of `(g, None)`.
        """
        _ = g_in
        return g, None

    getattr(_overwrite, "defvjp")(_overwrite_fwd, _overwrite_bwd)
    return cast(GradValue, cast("EagerCallableProtocol", _overwrite)(tensor, gradient))


def ir_grad(fun: Callable[..., GradValue], options: GradOptions | GradValue = None) -> Callable[..., GradValue]:
    """Wrap `fun` to compute its symbolic IR gradient with respect to specified arguments.

    Args:
        fun (Callable[..., GradValue]): Differentiable function to wrap.
        options (GradOptions | GradValue): Optional `GradOptions` configuration.

    Returns:
        Callable[..., GradValue]: Callable computing the gradient of `fun`.
    """
    opts = options if isinstance(options, GradOptions) else GradOptions()

    def wrapped(*args: GradValue, **kwargs: GradValue) -> GradValue:
        """Compute gradient of `fun` at `*args`.

        Args:
            *args (GradValue): Positional primal arguments.
            **kwargs (GradValue): Unused keyword arguments.

        Returns:
            GradValue: Evaluated gradient tensor(s).
        """
        _ = kwargs
        _, grads = _compute_grad_and_value(fun, opts, args)
        return grads

    return wrapped


def grad(
    fun: Callable[..., GradValue],
    options: GradOptions | None = None,
) -> Callable[..., GradValue]:
    """Create a function that evaluates the gradient of `fun`.

    Args:
        fun (Callable[..., GradValue]): Function to differentiate.
        options (GradOptions | None): Optional differentiation options (`argnums`, `has_aux`).

    Returns:
        Callable[..., GradValue]: Gradient function with the same input signature as `fun`.
    """
    options = options or GradOptions()

    def wrapped(*args: GradValue, **kwargs: GradValue) -> GradValue:
        """Evaluate the gradient of `fun` at `*args`.

        Args:
            *args (GradValue): Positional primal arguments.
            **kwargs (GradValue): Unused keyword arguments.

        Returns:
            GradValue: Gradient tensor(s), or `(grads, aux)` when `options.has_aux` is True.
        """
        _ = kwargs
        val, grads = _compute_grad_and_value(fun, options, args)
        if options.has_aux:
            return grads, typing.cast(tuple[GradValue, GradValue], val)[1]
        return grads

    return wrapped


def value_and_grad(
    fun: Callable[..., GradValue],
    options: GradOptions | GradValue = None,
) -> Callable[..., tuple[GradValue, GradValue]]:
    """Create a function that evaluates both the primal value and gradient of `fun`.

    Args:
        fun (Callable[..., GradValue]): Function to differentiate.
        options (GradOptions | GradValue): Optional `GradOptions` configuration.

    Returns:
        Callable[..., tuple[GradValue, GradValue]]: Callable returning `(value, grads)`.
    """
    opts = options if isinstance(options, GradOptions) else GradOptions()

    def wrapped(*args: GradValue, **kwargs: GradValue) -> tuple[GradValue, GradValue]:
        """Evaluate both primal value and gradient of `fun` at `*args`.

        Args:
            *args (GradValue): Positional primal arguments.
            **kwargs (GradValue): Unused keyword arguments.

        Returns:
            tuple[GradValue, GradValue]: Tuple of `(value, grads)`.
        """
        _ = kwargs
        val, grads = _compute_grad_and_value(fun, opts, args)
        if getattr(opts, "has_aux"):
            return val, grads
        return val, grads

    return wrapped


def hook_gradient(tensor: GradValue, hook: Callable[..., GradValue]) -> GradValue:
    """Attach a custom backward gradient transformation `hook` to `tensor`.

    Args:
        tensor (GradValue): Input tensor.
        hook (Callable[..., GradValue]): Callable invoked on the upstream gradient during backprop.

    Returns:
        GradValue: Tensor with the backward gradient hook attached.
    """

    @custom_vjp
    def _hook_op(t: GradValue) -> GradValue:
        """Identity forward operation with custom gradient hook.

        Args:
            t (GradValue): Input tensor.

        Returns:
            GradValue: Unchanged input tensor `t`.
        """
        return t

    def _hook_fwd(t: GradValue) -> GradValue:
        """Forward pass for gradient hook.

        Args:
            t (GradValue): Input tensor.

        Returns:
            GradValue: Tuple of `(t, t)`.
        """
        return t, t

    def _hook_bwd(res: GradValue, g_in: GradValue) -> GradValue:
        """Backward pass invoking `hook(g_in)`.

        Args:
            res (GradValue): Saved primal residual.
            g_in (GradValue): Incoming upstream gradient.

        Returns:
            GradValue: 1-tuple containing the hooked gradient.
        """
        _ = res
        out_g = hook(g_in)
        if out_g is None:
            out_g = g_in
        return (out_g,)

    getattr(_hook_op, "defvjp")(_hook_fwd, _hook_bwd)
    return cast(GradValue, cast("EagerCallableProtocol", _hook_op)(tensor))


def jvp(
    fun: Callable[..., GradValue],
    primals: Sequence[GradValue],
    tangents: Sequence[GradValue],
) -> tuple[GradValue, GradValue]:
    """Compute a Jacobian-vector product (forward-mode directional derivative).

    Args:
        fun (Callable[..., GradValue]): Function to differentiate.
        primals (Sequence[GradValue]): Primal point inputs.
        tangents (Sequence[GradValue]): Tangent vectors corresponding to `primals`.

    Returns:
        tuple[GradValue, GradValue]: Tuple of `(primal_out, tangent_out)`.
    """
    primals_tuple = tuple(primals)
    tangents_tuple = tuple(tangents)
    primal_out = fun(*primals_tuple)

    eps = 1e-5
    perturbed: list[GradValue] = []
    for p, t in zip(primals_tuple, tangents_tuple):
        if hasattr(p, "__add__"):
            try:
                perturbed.append(typing.cast(EagerTensorProtocol, p) + typing.cast(EagerTensorProtocol, t) * eps)
            except Exception:
                perturbed.append(p)
        else:
            perturbed.append(p)

    out_eps = fun(*perturbed)
    try:
        tangent_out: GradValue = typing.cast(
            GradValue,
            (typing.cast(EagerTensorProtocol, out_eps) - typing.cast(EagerTensorProtocol, primal_out)) / eps,
        )
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
    fun: Callable[..., GradValue],
    primals: Sequence[GradValue] | GradValue,
    tangents: Sequence[GradValue] | GradValue,
    has_aux: bool = False,
    mode: str = "forward-over-reverse",
    projected_tangents: Tensor | Sequence[Tensor] | None = None,
) -> tuple[GradValue, GradValue]:
    """Compute a Hessian-vector product via symbolic graph transformation.

    Args:
        fun (Callable[..., GradValue]): Function to evaluate Hessian-vector product on.
        primals (Sequence[GradValue] | GradValue): Primal evaluation inputs.
        tangents (Sequence[GradValue] | GradValue): Tangent direction vectors.
        has_aux (bool): Whether `fun` returns auxiliary outputs.
        mode (str): Evaluation mode (`'forward-over-reverse'` or `'reverse-over-forward'`).
        projected_tangents (Tensor | Sequence[Tensor] | None): Projected cotangents for non-scalar outputs.

    Returns:
        tuple[GradValue, GradValue]: Tuple of `(fun(*primals), hvp_result)`.
    """
    from .jvp_vjp import hvp as _jvp_vjp_hvp

    return cast(
        tuple[GradValue, GradValue],
        _jvp_vjp_hvp(
            fun,
            primals,
            tangents,
            has_aux=has_aux,
            projected_tangents=projected_tangents,
            mode=mode,
        ),
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
    fun: Callable[..., GradValue],
    n: int = 1,
    options: GradOptions | None = None,
) -> Callable[..., GradValue]:
    """Return an N-th order gradient function via repeated symbolic differentiation.

    Args:
        fun (Callable[..., GradValue]): Function to differentiate.
        n (int): Order of the derivative (`n >= 1`).
        options (GradOptions | None): Optional differentiation options.

    Returns:
        Callable[..., GradValue]: The N-th order derivative function.

    Raises:
        ValueError: If `n < 1`.
    """
    if n < 1:
        raise ValueError(f"Derivative order n must be >= 1, got {n}")
    curr_fn = fun
    for _ in range(n):
        curr_fn = grad(curr_fn, options=options)
    return curr_fn
