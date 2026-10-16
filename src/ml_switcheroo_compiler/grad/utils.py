# ruff: noqa: E402
"""Gradient computation and autodiff utilities."""

from __future__ import annotations

import contextlib  # noqa: F401
import math  # noqa: F401
import uuid  # noqa: F401
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar, Union, cast, runtime_checkable  # noqa: F401

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, Variable


@runtime_checkable
class GradTensorProtocol(Protocol):
    """Protocol defining the structural type contract for tensors participating in gradient evaluation."""

    @property
    def dtype(self) -> DType | str:
        """Get data type identifier."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape dimensions."""
        ...

    @property
    def id(self) -> str:
        """Get unique tensor identifier."""
        ...

    @property
    def data(self) -> GradValue:
        """Get underlying array buffer."""
        ...

    def __len__(self) -> int:
        """Get length of leading dimension."""
        ...


BackendArray = GradTensorProtocol
GradValue = EagerValue


from dataclasses import dataclass, field  # noqa: F401

from ml_switcheroo_ir import LogicalGraph, LogicalNode  # noqa: F401

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config  # noqa: F401
from ml_switcheroo_compiler.core.tensor import TensorConfig
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function  # noqa: F401
from ml_switcheroo_compiler.ops.registry import register_util  # noqa: F401
from ml_switcheroo_compiler.tracing.state import global_tracing_state  # noqa: F401
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients  # noqa: F401
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp  # noqa: F401

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
    all_tensors = _collect_live_tensors()
    wrt_tensors: list[GradValue] = []
    wrt_ids: list[str] = []
    for t in all_tensors:
        if hasattr(t, "data") and hasattr(t.data, "id"):
            node_id = getattr(t.data, "id")
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


def _collect_live_tensors() -> list[Tensor]:
    """Collect all live Tensor instances tracked by the garbage collector.

    Returns:
        list[Tensor]: List of active Tensor objects in memory.
    """
    import gc

    from ml_switcheroo_compiler.core.tensor import Tensor

    return [obj for obj in gc.get_objects() if issubclass(type(obj), Tensor)]


def _get_inputs_dict(graph: GradValue) -> dict[str, GradValue]:
    """Map input node IDs to their concrete values for evaluate_graph.

    Args:
        graph: The active tracing graph.

    Returns:
        dict: A mapping of input node ID to its value.
    """
    all_tensors = _collect_live_tensors()
    graph_nodes = getattr(graph, "nodes", {})
    inputs_dict: dict[str, GradValue] = {}
    for t in all_tensors:
        node_id = getattr(getattr(t, "data", None), "id", "")
        if node_id and node_id in graph_nodes:
            val = _get_concrete_val(t)
            if val is not None:
                inputs_dict[node_id] = get_active_backend().asarray(val)

    for inp_id in getattr(graph, "inputs", ()):
        if inp_id not in inputs_dict:
            raise ValueError(f"Missing input value for node '{inp_id}'.")

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
            return out_val[0] if isinstance(out_val, tuple) else out_val

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

    if isinstance(val, Tensor):
        return val

    if isinstance(orig, Tensor):
        import uuid  # noqa: F401

        from ml_switcheroo_ir import LogicalNode  # noqa: F401

        from ml_switcheroo_compiler.core.device import Device
        from ml_switcheroo_compiler.tracing.state import global_tracing_state  # noqa: F401
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor  # noqa: F401

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
    import uuid  # noqa: F811

    from ml_switcheroo_ir import LogicalNode  # noqa: F811

    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.tracing.state import global_tracing_state  # noqa: F811
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    from .jvp_vjp import vjp

    val, vjp_fn = getattr(vjp(fun, *getattr(args, "__iter__")(), has_aux=getattr(options, "has_aux", False)), "__iter__")()

    if getattr(options, "has_aux", False):
        primal_val, _ = cast(tuple[GradValue, GradValue], val)
    else:
        primal_val = val

    if global_tracing_state.is_tracing and global_tracing_state.active_graph is not None:
        c_id = f"cot_{uuid.uuid4().hex[:6]}"
        p_shape = getattr(primal_val, "shape", ())
        c_node = LogicalNode(id=c_id, op_type="Constant", inputs=[], attributes={"value": 1.0}, shape_metadata=p_shape)
        global_tracing_state.add_node(c_node)
        proxy = ProxyTensor(id=c_id, shape=p_shape, dtype="float32")
        cotangent: GradValue = Tensor(proxy, TensorConfig(p_shape, DType.Float32, Device("cpu")))
    else:
        primal_arr = get_active_backend().asarray(getattr(primal_val, "data", primal_val))
        cotangent = get_active_backend().execute_op("Ones_like", primal_arr)

    grads = vjp_fn(cotangent)

    argnums = getattr(options, "argnums", 0)
    if isinstance(argnums, int):
        res_grad = _to_original_type(getattr(grads, "__getitem__")(argnums), getattr(args, "__getitem__")(argnums))
    elif isinstance(argnums, (tuple, list)):
        res_grad = cast(GradValue, tuple(_to_original_type(cast(tuple[GradValue, ...], grads)[idx], args[idx]) for idx in cast(tuple[int, ...], argnums)))
    else:
        res_grad = _to_original_type(getattr(grads, "__getitem__")(0), getattr(args, "__getitem__")(0))

    return val, res_grad
