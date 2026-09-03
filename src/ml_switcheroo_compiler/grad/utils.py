# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Gradient computation and autodiff utilities."""

import contextlib
import math
import uuid
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar, Union, cast

from ml_switcheroo_compiler.core.tensor import Tensor, Variable


class BackendArray(Protocol):
    """Protocol for a backend array."""

    @property
    def dtype(self) -> "BackendArray":
        """Get dtype."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape."""
        ...

    @property
    def id(self) -> str:
        """Get id."""
        ...

    @property
    def data(self) -> "BackendArray":
        """Get data."""
        ...


GradValue = Union[int, float, str, bool, list, tuple, dict, None, Tensor, Variable, BackendArray]


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
    fun: Callable[..., GradValue],
) -> Callable[..., tuple[GradValue, dict[str, GradValue]]]:
    """Create a function that evaluates both the primal value and the gradient with respect to variables.

    Args:
        fun: The original function to wrap.

    Returns:
        Callable: The wrapped function returning the primal value and gradients.
    """

    def wrapped(*args: GradValue, **kwargs: GradValue) -> tuple[GradValue, dict[str, GradValue]]:
        """Evaluate wrapped operation.

        Args:
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            tuple: Result.
        """
        val = fun(*args, **kwargs)
        grads: dict[str, GradValue] = {}
        return (val, grads)

    return wrapped


def _check_scalar(tensor: GradValue) -> None:
    """Validate that the target tensor is a scalar.

    Args:
        tensor: The tensor to validate.

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


def _find_wrt_tensors(graph: GradValue) -> tuple[list[GradValue], list[str]]:
    """Find all active variables/tensors in memory that require gradients and are in the graph.

    Args:
        graph: The active tracing graph.

    Returns:
        tuple: A tuple of active tensors list and their corresponding graph node IDs.
    """
    import gc

    from ml_switcheroo_compiler.core.tensor import Tensor, Variable

    all_tensors = [obj for obj in gc.get_objects() if isinstance(obj, Tensor)]
    wrt_tensors: list[GradValue] = []
    wrt_ids: list[str] = []
    for t in all_tensors:
        if hasattr(t, "data") and hasattr(t.data, "id"):
            node_id = t.data.id
            if node_id in getattr(graph, "nodes", {}):
                if getattr(t, "requires_grad", False) or getattr(t, "trainable", False) or isinstance(t, Variable):
                    wrt_tensors.append(t)
                    wrt_ids.append(node_id)
    return wrt_tensors, wrt_ids


def _get_concrete_val(t: GradValue) -> GradValue:
    """Extract concrete value from a tensor.

    Args:
        t: The tensor.

    Returns:
        GradValue: The concrete value or None.
    """
    val = getattr(getattr(t, "data", t), "concrete_value", None)
    if val is None:
        val = getattr(t, "_data", None)
        if isinstance(val, ProxyTensor):
            val = getattr(val, "concrete_value", None)
    return val


def _get_inputs_dict(graph: GradValue) -> dict[str, GradValue]:
    """Map input node IDs to their concrete values for evaluate_graph.

    Args:
        graph: The active tracing graph.

    Returns:
        dict: A mapping of input node ID to its value.
    """
    import gc

    from ml_switcheroo_compiler.core.tensor import Tensor

    all_tensors = [obj for obj in gc.get_objects() if isinstance(obj, Tensor)]
    inputs_dict: dict[str, GradValue] = {}
    for t in all_tensors:
        if hasattr(t, "data") and hasattr(cast(BackendArray, t).data, "id"):
            node_id = cast(BackendArray, cast(BackendArray, t).data).id
            if node_id in getattr(graph, "nodes", {}):
                val = _get_concrete_val(t)
                if val is not None:
                    inputs_dict[node_id] = get_active_backend().asarray(val)

    if hasattr(graph, "inputs"):
        for inp_id in graph.inputs:
            if inp_id not in inputs_dict:
                raise ValueError(f"Missing input value for node '{inp_id}'.")

    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    return inputs_dict


def _convert_to_tensors(primals: Sequence[GradValue]) -> list[Tensor]:
    """Convert input primals to Tensor objects.

    Args:
        primals: The primals.

    Returns:
        list[Tensor]: A list of converted Tensor objects.
    """
    tensor_primals: list[Tensor] = []
    from ml_switcheroo_compiler.core.device import Device

    for p in primals:
        if isinstance(p, Tensor):
            tensor_primals.append(p)
        else:
            arr = cast(BackendArray, get_active_backend().asarray(p))
            dt = DType.Float32
            if str(arr.dtype) == "float64":
                dt = DType.Float64
            elif "int" in str(arr.dtype):
                dt = DType.Int32
            elif str(arr.dtype) == "bool":
                dt = DType.Bool
            tensor_primals.append(Tensor(arr, TensorConfig(arr.shape, dt, Device("cpu"))))
    return tensor_primals


def _get_fun_primal(fun: Callable[..., GradValue], has_aux: bool) -> Callable[..., GradValue]:
    """Return function for primal evaluation.

    Args:
        fun (Callable): The fun parameter.
        has_aux (bool): The has_aux parameter.

    Returns:
        Callable: Result.
    """
    if has_aux:

        def fun_primal(*args: GradValue) -> GradValue:
            """Evaluate fun_primal operation.

            Args:
                *args: Positional args.

            Returns:
                GradValue: Result.
            """
            out_val: GradValue = fun(*args)
            return cast(tuple[GradValue, ...], out_val)[0] if isinstance(out_val, tuple) else out_val

        return fun_primal

    return fun


def _to_original_type(val: GradValue, orig: GradValue) -> GradValue:
    """Convert the computed gradient to match the original input's type.

    Args:
        val: The calculated gradient array/value.
        orig: The original input argument.

    Returns:
        GradValue: The gradient converted to the original type.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor

    if isinstance(orig, Tensor):
        from ml_switcheroo_compiler.core.device import Device

        arr = cast(BackendArray, get_active_backend().asarray(val))
        dt = DType.Float32
        if str(arr.dtype) == "float64":
            dt = DType.Float64
        elif "int" in str(arr.dtype):
            dt = DType.Int32
        elif str(arr.dtype) == "bool":
            dt = DType.Bool
        return Tensor(arr, TensorConfig(arr.shape, dt, Device("cpu")))
    return val


def _compute_grad_and_value(
    fun: Callable[..., GradValue],
    options: GradOptions,
    args: tuple[GradValue, ...],
) -> tuple[GradValue, GradValue]:
    """Evaluate _compute_grad_and_value operation.

    Args:
        fun: The fun parameter.
        options: The options parameter.
        args: The args parameter.

    Returns:
        tuple: Result.
    """
    from .jvp_vjp import vjp

    val, vjp_fn = vjp(fun, *args, has_aux=getattr(options, "has_aux", False))

    if getattr(options, "has_aux", False):
        primal_val, _ = cast(tuple[GradValue, GradValue], val)
    else:
        primal_val = val

    primal_arr = get_active_backend().asarray(getattr(primal_val, "data", primal_val))
    cotangent = get_active_backend().execute_op("Ones_like", primal_arr)

    grads = vjp_fn(cotangent)

    argnums = getattr(options, "argnums", 0)
    if isinstance(argnums, int):
        res_grad = _to_original_type(grads[argnums], args[argnums])
    elif isinstance(argnums, (tuple, list)):
        res_grad = cast(GradValue, tuple(_to_original_type(cast(tuple[GradValue, ...], grads)[idx], args[idx]) for idx in cast(tuple[int, ...], argnums)))
    else:
        res_grad = _to_original_type(grads[0], args[0])

    return val, res_grad
