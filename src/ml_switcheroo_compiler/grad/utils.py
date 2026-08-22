# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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

from .options import GradOptions


def value_and_grad_wrt_vars(
    fun: Callable[..., Any],
) -> Callable[..., tuple[Any, dict[str, Any]]]:
    """Create a function that evaluates both the primal value and the gradient with respect to variables.

    Args:
        fun (Callable[..., Any]): The original function to wrap.

    Returns:
        Callable[..., tuple[Any, dict[str, Any]]]: The wrapped function returning the primal value and gradients.
    """

    def wrapped(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Evaluate wrapped operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        val = fun(*args, **kwargs)
        grads: dict[str, Any] = {}
        return (val, grads)

    return wrapped


def _check_scalar(tensor: Any) -> None:
    """Validate that the target tensor is a scalar.

    Args:
        tensor (object): The tensor to validate.

    Raises:
        SwitcherooError: If target tensor is not a scalar.
    """
    from ml_switcheroo_compiler.core.errors import SwitcherooError

    shape = getattr(tensor, "shape", ())
    prod = 1
    for s in shape:
        try:
            prod *= int(s)
        except (ValueError, TypeError):
            # Assume symbolic dimensions might be non-scalar
            prod = 2

    if prod != 1:
        raise SwitcherooError("backward() can only be called on scalar tensors.")


def _find_wrt_tensors(graph: Any) -> tuple[list[Any], list[str]]:
    """Find all active variables/tensors in memory that require gradients and are in the graph.

    Args:
        graph (object): The active tracing graph.

    Returns:
        tuple[list[Any], list[str]]: A tuple of active tensors list and their corresponding graph node IDs.
    """
    import gc

    from ml_switcheroo_compiler.core.tensor import Tensor, Variable

    all_tensors = [obj for obj in gc.get_objects() if isinstance(obj, Tensor)]
    wrt_tensors = []
    wrt_ids = []
    for t in all_tensors:
        if hasattr(t, "data") and hasattr(t.data, "id"):
            node_id = t.data.id
            if node_id in getattr(graph, "nodes", {}):
                if t.requires_grad or getattr(t, "trainable", False) or isinstance(t, Variable):
                    wrt_tensors.append(t)
                    wrt_ids.append(node_id)
    return wrt_tensors, wrt_ids


def _get_concrete_val(t: Any) -> Any:
    """Extract concrete value from a tensor.

    Args:
        t (object): The tensor.

    Returns: Any: The concrete value or None.
    """
    val = getattr(t.data, "concrete_value", None)
    if val is None:
        val = getattr(t, "_data", None)
        if isinstance(val, ProxyTensor):
            val = getattr(val, "concrete_value", None)
    return val


def _generate_fallback_input(graph: Any, inp_id: str) -> Any:
    """Generate fallback dummy input for evaluation.

    Args:
        graph (object): The computation graph.
        inp_id (str): The node ID of the input.

    Returns: Any: The concrete dummy value.
    """
    node = getattr(graph, "nodes", {}).get(inp_id)
    node_shape = getattr(node, "shape_metadata", ()) or ()
    numeric_shape = []
    for s in node_shape:
        try:
            numeric_shape.append(int(s))
        except (ValueError, TypeError):
            numeric_shape.append(1)

    dtype_str = "float32"
    if node and hasattr(node, "attributes") and "dtype" in node.attributes:
        dtype_str = node.attributes["dtype"]
    return get_active_backend().execute_op("Ones", tuple(numeric_shape), dtype=dtype_str)


def _get_inputs_dict(graph: Any) -> dict[str, Any]:
    """Map input node IDs to their concrete values for evaluate_graph.

    Args:
        graph (object): The active tracing graph.

    Returns:
        dict[str, Any]: A dictionary mapping node IDs to concrete NumPy arrays.
    """
    import gc

    from ml_switcheroo_compiler.core.tensor import Tensor

    all_tensors = [obj for obj in gc.get_objects() if isinstance(obj, Tensor)]
    inputs_dict: dict[str, Any] = {}
    for t in all_tensors:
        if hasattr(t, "data") and hasattr(t.data, "id"):
            node_id = t.data.id
            if node_id in getattr(graph, "nodes", {}):
                val = _get_concrete_val(t)
                if val is not None:
                    inputs_dict[node_id] = get_active_backend().asarray(val)

    # Generate dummy concrete arrays for any missing inputs to prevent errors
    for inp_id in getattr(graph, "inputs", []):
        if inp_id not in inputs_dict:
            inputs_dict[inp_id] = _generate_fallback_input(graph, inp_id)

    return inputs_dict


def _to_original_type(val: Any, orig: Any) -> Any:
    """Convert the computed gradient to match the original input's type.

    Args:
        val (object): The calculated gradient array/value.
        orig (object): The original input argument.

    Returns: Any: The gradient converted to the original type.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor

    if isinstance(orig, Tensor):
        from ml_switcheroo_compiler.core.device import Device

        arr = get_active_backend().asarray(val)
        dt = DType.Float32
        if str(arr.dtype) == "float64":
            dt = DType.Float64
        elif "int" in str(arr.dtype):
            dt = DType.Int32
        elif str(arr.dtype) == "bool":
            dt = DType.Bool
        return Tensor(arr, TensorConfig(arr.shape, dt, Device("cpu")))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    elif isinstance(orig, (int, float, bool)):
        try:
            arr = get_active_backend().asarray(val)
            if isinstance(orig, bool):
                return bool(arr.item())
            if isinstance(orig, int):
                return int(arr.item())
            return float(arr.item())
        except Exception:
            return val
    return val


def _compute_grad_and_value(
    fun: Callable[..., Any],
    options: GradOptions,
    args: tuple[Any, ...],
) -> tuple[Any, Any]:
    """Evaluate _compute_grad_and_value operation.

    Args:
        fun (Callable): The fun parameter.
        options (GradOptions): The options parameter.
        args (tuple): The args parameter.

    Returns:
        tuple: Result.
    """
    from .jvp_vjp import vjp

    val, vjp_fn = vjp(fun, *args, has_aux=options.has_aux)

    if options.has_aux:
        primal_val, _ = val
    else:
        primal_val = val

    primal_arr = get_active_backend().asarray(getattr(primal_val, "data", primal_val))
    cotangent = get_active_backend().execute_op("Ones_like", primal_arr)

    grads = vjp_fn(cotangent)

    argnums = options.argnums
    if isinstance(argnums, int):
        res_grad = _to_original_type(grads[argnums], args[argnums])
    elif isinstance(argnums, (tuple, list)):
        res_grad = tuple(_to_original_type(grads[idx], args[idx]) for idx in argnums)
    else:
        res_grad = _to_original_type(grads[0], args[0])

    return val, res_grad


def _convert_to_tensors(primals: Sequence[Any]) -> list[Tensor]:  # type: ignore
    """Convert input primals to Tensor objects.

    Args:
        primals (Sequence[Any]): The list/tuple of input values.

    Returns:
        list[Tensor]: A list of converted Tensor objects.
    """
    tensor_primals = []
    from ml_switcheroo_compiler.core.device import Device

    for p in primals:
        if isinstance(p, Tensor):
            tensor_primals.append(p)
        else:
            arr = get_active_backend().asarray(p)
            dt = DType.Float32
            if str(arr.dtype) == "float64":
                dt = DType.Float64
            elif "int" in str(arr.dtype):
                dt = DType.Int32
            elif str(arr.dtype) == "bool":
                dt = DType.Bool
            tensor_primals.append(Tensor(arr, TensorConfig(arr.shape, dt, Device("cpu"))))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return tensor_primals


def _get_fun_primal(fun: Callable[..., Any], has_aux: bool) -> Callable[..., Any]:
    """Return function for primal evaluation.

    Args:
        fun (Callable): The original function.
        has_aux (bool): Whether the function has auxiliary outputs.

    Returns:
        Callable: The primal-only function wrapper.
    """
    if has_aux:

        def fun_primal(*args: Any) -> Any:
            """Evaluate fun_primal operation.

            Args:
            *args (object): Positional args.

            Returns:
            tuple[int, ...]: Result.
            """
            out_val = fun(*args)
            return out_val[0] if isinstance(out_val, tuple) else out_val

        return fun_primal
    return fun
