# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Gradient computation and autodiff utilities."""

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
def backward(tensor: object, *args: object, **kwargs: object) -> None:
    """Triggers the reverse-mode auto-differentiation.

    Args:
        tensor (object): The tensor to compute gradients for.
        *args (object): Additional arguments.
        **kwargs (object): Additional keyword arguments.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    # 1. Fallback to dummy behavior if not a Tensor, or if tracing is not active
    if not isinstance(tensor, Tensor) or not global_tracing_state.is_tracing or global_tracing_state.active_graph is None:
        if hasattr(tensor, "grad"):
            tensor.grad = 1.0
        else:
            tensor.grad = 1.0
        return

    # 2. Validate that the target tensor is a scalar
    _check_scalar(tensor)

    # 3. Extract the active graph
    graph: object = global_tracing_state.active_graph

    # 4. Find all active variables/tensors in memory that require gradients and are in the graph
    wrt_tensors, wrt_ids = _find_wrt_tensors(graph)

    # 5. Extract target loss node ID
    loss_id: object = getattr(tensor.data, "id", None)
    if loss_id is None:
        loss_id: object = str(tensor.data)

    if not wrt_ids:
        # No variables require gradients
        tensor.grad = 1.0
        return

    # 6. Run the graph-level gradient generator
    from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad

    grad_graph: object = graph_grad(graph, wrt_ids, loss_id)

    # 7. Map input node IDs to their concrete values for evaluate_graph
    inputs_dict: object = _get_inputs_dict(graph)

    # 8. Evaluate the constructed gradient graph using evaluate_graph
    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    outputs_dict: object = evaluate_graph(grad_graph, inputs_dict)

    # 9. Traverse the wrt list and assign computed NumPy gradient arrays to .grad attributes
    for i in range(len(wrt_ids)):
        grad_node_id: object = grad_graph.outputs[i]
        if grad_node_id in outputs_dict:
            grad_val: object = outputs_dict[grad_node_id]
            t: object = wrt_tensors[i]
            t.grad = grad_val


def RegisterGradient(op_type: str) -> typing.Callable:
    """Register a custom gradient for an operation.

    Args:
        op_type (str): The name of the operation.

    Returns:
        Callable: The decorator function.
    """
    return register_vjp(op_type)


def overwrite_with_gradient(tensor: object, gradient: object) -> object:
    """Overwrite the gradient of the tensor in the backward pass.

    During the forward pass, this returns the `tensor` unchanged.
    During the backward pass, it replaces the propagated gradient with `gradient`.

    Args:
        tensor (object): The input tensor.
        gradient (object): The gradient value to use in the backward pass.

    Returns: object: The tensor with the overridden backward pass gradient.
    """

    @custom_vjp
    def _overwrite(t: object, g: object) -> object:
        """Overwrite the gradient during the backward pass.

        Args:
            t (object): The primal tensor.
            g (object): The gradient to overwrite with.

        Returns: object: The primal tensor unchanged.
        """
        return t

    def _overwrite_fwd(t: object, g: object) -> tuple[object, object]:
        """Forward pass for overwriting a gradient.

        Args:
            t (object): The primal tensor.
            g (object): The gradient to overwrite with.

        Returns:
            tuple[object, object]: The primal tensor and the gradient to store for backward.
        """
        return t, g

    def _overwrite_bwd(g: object, g_in: object) -> tuple[object, object]:
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


def ir_grad(fun: Callable[..., object], options: object = None) -> Callable[..., object]:
    """Return a gradient wrapper.

    Args:
        fun (Callable): Function to differentiate.
        options (GradOptions): Differentiation options.

    Returns:
        Callable: The wrapper function.
    """
    options: object = options or GradOptions()

    def wrapped(*args: object, **kwargs: object) -> object:
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


def grad(fun: Callable[..., object], options: object = None) -> Callable[..., object]:
    """Return a gradient wrapper.

    Args:
        fun (Callable): Function to differentiate.
        options (GradOptions): Differentiation options.

    Returns:
        Callable: The wrapper function.
    """
    options: object = options or GradOptions()

    def wrapped(*args: object, **kwargs: object) -> object:
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


def value_and_grad(fun: Callable[..., object], options: object = None) -> Callable[..., object]:
    """Return a gradient wrapper.

    Args:
        fun (Callable): Function to differentiate.
        options (GradOptions): Differentiation options.

    Returns:
        Callable: The wrapper function.
    """
    options: object = options or GradOptions()

    def wrapped(*args: object, **kwargs: object) -> object:
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


def hook_gradient(tensor: object, hook: Callable[[object], object]) -> object:
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
    def _hook_op(t: object) -> object:
        """Apply the hook op.

        Args:
            t (object): The input tensor.

        Returns:
            object: The unchanged tensor.
        """
        return t

    def _hook_fwd(t: object) -> tuple[object, object]:
        """Forward pass for the hook.

        Args:
            t (object): The input tensor.

        Returns:
            tuple[object, object]: The primal and None.
        """
        return t, t

    def _hook_bwd(res: object, g_in: object) -> tuple[object]:
        """Backward pass for the hook.

        Args:
            res (object): The stored result.
            g_in (object): The incoming upstream gradient.

        Returns:
            tuple[object]: The processed gradient.
        """
        out_g: object = hook(g_in)
        if out_g is None:
            out_g: object = g_in
        return (out_g,)

    _hook_op.defvjp(_hook_fwd, _hook_bwd)
    return _hook_op(tensor)
