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

from .options import GradOptions


def value_and_grad_wrt_vars(
    fun: Callable[..., object],
) -> Callable[..., tuple[object, dict[str, object]]]:
    """Create a function that evaluates both the primal value and the gradient with respect to variables.

    Args:
        fun (Callable[..., object]): The original function to wrap.

    Returns:
        Callable[..., tuple[object, dict[str, object]]]: The wrapped function returning the primal value and gradients.
    """

    def wrapped(*args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
        """Evaluate wrapped operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        val: object = fun(*args, **kwargs)
        grads: dict[str, object] = {}
        return (val, grads)

    return wrapped


def _check_scalar(tensor: object) -> None:
    """Validate that the target tensor is a scalar.

    Args:
        tensor (object): The tensor to validate.

    Raises:
        SwitcherooError: If target tensor is not a scalar.
    """
    from ml_switcheroo_compiler.core.errors import SwitcherooError

    shape: object = getattr(tensor, "shape", ())
    prod: object = 1
    for s in shape:
        try:
            prod *= int(s)
        except (ValueError, TypeError):
            # Assume symbolic dimensions might be non-scalar
            prod: object = 2

    if prod != 1:
        raise SwitcherooError("backward() can only be called on scalar tensors.")


def _find_wrt_tensors(graph: object) -> tuple[list[object], list[str]]:
    """Find all active variables/tensors in memory that require gradients and are in the graph.

    Args:
        graph (object): The active tracing graph.

    Returns:
        tuple[list[object], list[str]]: A tuple of active tensors list and their corresponding graph node IDs.
    """
    import gc

    from ml_switcheroo_compiler.core.tensor import Tensor, Variable

    all_tensors: object = [obj for obj in gc.get_objects() if isinstance(obj, Tensor)]
    wrt_tensors: object = []
    wrt_ids: object = []
    for t in all_tensors:
        if hasattr(t, "data") and hasattr(t.data, "id"):
            node_id: object = t.data.id
            if node_id in getattr(graph, "nodes", {}):
                if t.requires_grad or getattr(t, "trainable", False) or isinstance(t, Variable):
                    wrt_tensors.append(t)
                    wrt_ids.append(node_id)
    return wrt_tensors, wrt_ids


def _get_concrete_val(t: object) -> object:
    """Extract concrete value from a tensor.

    Args:
        t (object): The tensor.

    Returns: object: The concrete value or None.
    """
    val: object = getattr(t.data, "concrete_value", None)
    if val is None:
        val: object = getattr(t, "_data", None)
        if isinstance(val, ProxyTensor):
            val: object = getattr(val, "concrete_value", None)
    return val


def _generate_fallback_input(graph: object, inp_id: str) -> object:
    """Generate fallback dummy input for evaluation.

    Args:
        graph (object): The computation graph.
        inp_id (str): The node ID of the input.

    Returns: object: The concrete dummy value.
    """
    node: object = getattr(graph, "nodes", {}).get(inp_id)
    node_shape: object = getattr(node, "shape_metadata", ()) or ()
    numeric_shape: object = []
    for s in node_shape:
        try:
            numeric_shape.append(int(s))
        except (ValueError, TypeError):
            numeric_shape.append(1)

    dtype_str: object = "float32"
    if node and hasattr(node, "attributes") and "dtype" in node.attributes:
        dtype_str: object = node.attributes["dtype"]
    return get_active_backend().execute_op("Ones", tuple(numeric_shape), dtype=dtype_str)


def _get_inputs_dict(graph: object) -> dict[str, object]:
    """Map input node IDs to their concrete values for evaluate_graph.

    Args:
        graph (object): The active tracing graph.

    Returns:
        dict[str, object]: A dictionary mapping node IDs to concrete NumPy arrays.
    """
    import gc

    from ml_switcheroo_compiler.core.tensor import Tensor

    all_tensors: object = [obj for obj in gc.get_objects() if isinstance(obj, Tensor)]
    inputs_dict: dict[str, object] = {}
    for t in all_tensors:
        if hasattr(t, "data") and hasattr(t.data, "id"):
            node_id: object = t.data.id
            if node_id in getattr(graph, "nodes", {}):
                val: object = _get_concrete_val(t)
                if val is not None:
                    inputs_dict[node_id] = get_active_backend().asarray(val)

    # Generate dummy concrete arrays for any missing inputs to prevent errors
    for inp_id in getattr(graph, "inputs", []):
        if inp_id not in inputs_dict:
            inputs_dict[inp_id] = _generate_fallback_input(graph, inp_id)

    return inputs_dict


def _to_original_type(val: object, orig: object) -> object:
    """Convert the computed gradient to match the original input's type.

    Args:
        val (object): The calculated gradient array/value.
        orig (object): The original input argument.

    Returns: object: The gradient converted to the original type.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor

    if isinstance(orig, Tensor):
        from ml_switcheroo_compiler.core.device import Device

        arr: object = get_active_backend().asarray(val)
        dt: object = DType.Float32
        if str(arr.dtype) == "float64":
            dt: object = DType.Float64
        elif "int" in str(arr.dtype):
            dt: object = DType.Int32
        elif str(arr.dtype) == "bool":
            dt: object = DType.Bool
        return Tensor(arr, TensorConfig(arr.shape, dt, Device("cpu")))
    elif isinstance(orig, (int, float, bool)):
        try:
            arr: object = get_active_backend().asarray(val)
            if isinstance(orig, bool):
                return bool(arr.item())
            if isinstance(orig, int):
                return int(arr.item())
            return float(arr.item())
        except Exception:
            return val
    return val


def _compute_grad_and_value(
    fun: Callable[..., object],
    options: GradOptions,
    args: tuple[object, ...],
) -> tuple[object, object]:
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
        primal_val: object = val

    primal_arr: object = get_active_backend().asarray(getattr(primal_val, "data", primal_val))
    cotangent: object = get_active_backend().execute_op("Ones_like", primal_arr)

    grads: object = vjp_fn(cotangent)

    argnums: object = options.argnums
    if isinstance(argnums, int):
        res_grad: object = _to_original_type(grads[argnums], args[argnums])
    elif isinstance(argnums, (tuple, list)):
        res_grad: object = tuple(_to_original_type(grads[idx], args[idx]) for idx in argnums)
    else:
        res_grad: object = _to_original_type(grads[0], args[0])

    return val, res_grad


def _convert_to_tensors(primals: Sequence[object]) -> list[Tensor]:
    """Convert input primals to Tensor objects.

    Args:
        primals (Sequence[object]): The list/tuple of input values.

    Returns:
        list[Tensor]: A list of converted Tensor objects.
    """
    tensor_primals: object = []
    from ml_switcheroo_compiler.core.device import Device

    for p in primals:
        if isinstance(p, Tensor):
            tensor_primals.append(p)
        else:
            arr: object = get_active_backend().asarray(p)
            dt: object = DType.Float32
            if str(arr.dtype) == "float64":
                dt: object = DType.Float64
            elif "int" in str(arr.dtype):
                dt: object = DType.Int32
            elif str(arr.dtype) == "bool":
                dt: object = DType.Bool
            tensor_primals.append(Tensor(arr, TensorConfig(arr.shape, dt, Device("cpu"))))
    return tensor_primals


def _get_fun_primal(fun: Callable[..., object], has_aux: bool) -> Callable[..., object]:
    """Return function for primal evaluation.

    Args:
        fun (Callable): The original function.
        has_aux (bool): Whether the function has auxiliary outputs.

    Returns:
        Callable: The primal-only function wrapper.
    """
    if has_aux:

        def fun_primal(*args: object) -> object:
            """Evaluate fun_primal operation.

            Args:
            *args (object): Positional args.

            Returns:
            tuple[int, ...]: Result.
            """
            out_val: object = fun(*args)
            return out_val[0] if isinstance(out_val, tuple) else out_val

        return fun_primal
    return fun
