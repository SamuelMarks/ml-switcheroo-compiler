# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Gradient computation and autodiff utilities."""

import contextlib
import math
import typing
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

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
def backward(tensor: Any, *args: Any, **kwargs: Any) -> None:
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
            tensor.grad = 1.0  # type: ignore
        else:
            tensor.grad = 1.0
        return

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

    # 7. Map input node IDs to their concrete values for evaluate_graph
    inputs_dict = _get_inputs_dict(graph)

    # 8. Evaluate the constructed gradient graph using evaluate_graph
    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    outputs_dict = evaluate_graph(grad_graph, inputs_dict)

    # 9. Traverse the wrt list and assign computed NumPy gradient arrays to .grad attributes
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


def overwrite_with_gradient(tensor: Any, gradient: Any) -> Any:
    """Overwrite the gradient of the tensor in the backward pass.

    During the forward pass, this returns the `tensor` unchanged.
    During the backward pass, it replaces the propagated gradient with `gradient`.

    Args:
        tensor (object): The input tensor.
        gradient (object): The gradient value to use in the backward pass.

    Returns: Any: The tensor with the overridden backward pass gradient.
    """

    @custom_vjp
    def _overwrite(t: Any, g: Any) -> Any:
        """Overwrite the gradient during the backward pass.

        Args:
            t (object): The primal tensor.
            g (object): The gradient to overwrite with.

        Returns: Any: The primal tensor unchanged.
        """
        return t

    def _overwrite_fwd(t: Any, g: Any) -> tuple[Any, Any]:
        """Forward pass for overwriting a gradient.

        Args:
            t (object): The primal tensor.
            g (object): The gradient to overwrite with.

        Returns:
            tuple[Any, Any]: The primal tensor and the gradient to store for backward.
        """
        return t, g

    def _overwrite_bwd(g: Any, g_in: Any) -> tuple[Any, None]:
        """Backward pass for overwriting a gradient.

        Args:
            g (object): The stored gradient to overwrite with.
            g_in (object): The incoming upstream gradient.

        Returns:
            tuple[Any, None]: The overwritten gradient and None for the second argument.
        """
        return g, None

    _overwrite.defvjp(_overwrite_fwd, _overwrite_bwd)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return _overwrite(tensor, gradient)


def ir_grad(fun: Callable[..., Any], options: Any = None) -> Callable[..., Any]:
    """Return a gradient wrapper.

    Args:
        fun (Callable): Function to differentiate.
        options (GradOptions): Differentiation options.

    Returns:
        Callable: The wrapper function.
    """
    options = options or GradOptions()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        _, grads = _compute_grad_and_value(fun, options, args)
        return grads

    return wrapped


def grad(fun: Callable[..., Any], options: Any = None) -> Callable[..., Any]:
    """Return a gradient wrapper.

    Args:
        fun (Callable): Function to differentiate.
        options (GradOptions): Differentiation options.

    Returns:
        Callable: The wrapper function.
    """
    options = options or GradOptions()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        val, grads = _compute_grad_and_value(fun, options, args)
        if options.has_aux:
            return grads, val[1]
        return grads

    return wrapped


def value_and_grad(fun: Callable[..., Any], options: Any = None) -> Callable[..., Any]:
    """Return a gradient wrapper.

    Args:
        fun (Callable): Function to differentiate.
        options (GradOptions): Differentiation options.

    Returns:
        Callable: The wrapper function.
    """
    options = options or GradOptions()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        val, grads = _compute_grad_and_value(fun, options, args)
        if options.has_aux:
            return val, grads
        return val, grads

    return wrapped
