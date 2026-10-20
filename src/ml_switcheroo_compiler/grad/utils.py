"""Internal helper utilities and protocols for automatic differentiation."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Callable, Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar, TypedVariadicCallable
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, Variable
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from .options import GradOptions


@typing.runtime_checkable
class GradTensorProtocol(typing.Protocol):
    """Protocol defining the structural type contract for tensors participating in gradient evaluation."""

    @property
    def dtype(self) -> DType | str:
        """Return data type identifier.

        Returns:
            DType | str: Computed result.
        """
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape dimensions tuple.

        Returns:
            tuple[int, ...]: Computed result.
        """
        ...

    @property
    def id(self) -> str:
        """Return unique tensor identifier.

        Returns:
            str: Computed result.
        """
        ...

    @property
    def data(self) -> GradValue:
        """Return underlying array buffer.

        Returns:
            GradValue: Computed result.
        """
        ...

    def __len__(self) -> int:
        """Return length of the leading dimension.

        Returns:
            int: Computed result.
        """
        ...


BackendArray = GradTensorProtocol
GradValue = TensorOrScalar


def value_and_grad_wrt_vars(
    fun: GradValue,
) -> TypedVariadicCallable[tuple[GradValue, dict[str, GradValue]]]:
    """Create a function that evaluates both the primal value and variable gradients.

    Args:
        fun (GradValue): Original function to wrap.

    Returns:
        TypedVariadicCallable[tuple[GradValue, dict[str, GradValue]]]: Wrapped callable returning `(value, grads_dict)`.
    """

    def wrapped(*args: GradValue, **kwargs: GradValue) -> tuple[GradValue, dict[str, GradValue]]:
        """Evaluate `fun` and return `(value, grads_dict)`.

        Args:
            *args (GradValue): Positional arguments.
            **kwargs (GradValue): Keyword arguments.

        Returns:
            tuple[GradValue, dict[str, GradValue]]: Primal output value and variable gradient dictionary.
        """
        val = typing.cast(TypedVariadicCallable[GradValue], fun)(*args, **kwargs)
        grads: dict[str, GradValue] = {}
        return (val, grads)

    return wrapped


def _check_scalar(tensor: GradValue) -> None:
    """Validate that `tensor` has a scalar (unit-element) shape.

    Args:
        tensor (GradValue): Tensor to validate.

    Raises:
        SwitcherooError: If `tensor` is not a scalar tensor.
    """
    from ml_switcheroo_compiler.core.errors import SwitcherooError

    shape = getattr(tensor, "shape", ())
    prod = 1
    for s in shape:
        try:
            prod *= int(s)
        except (ValueError, TypeError):
            prod = 2

    if prod != 1:
        raise SwitcherooError("backward() can only be called on scalar tensors.")


def _find_wrt_tensors(graph: GradValue) -> tuple[list[GradValue], list[str]]:
    """Find all live trainable tensors/variables whose node IDs belong to `graph`.

    Args:
        graph (GradValue): Active tracing graph.

    Returns:
        tuple[list[GradValue], list[str]]: Tuple of `(wrt_tensors, wrt_node_ids)`.
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
    """Extract the concrete array value attached to a traced `Tensor` or `ProxyTensor`.

    Args:
        t (GradValue): Traced tensor object.

    Returns:
        GradValue: Concrete backing array value, or None if unavailable.
    """
    val = getattr(getattr(t, "data", t), "concrete_value", None)
    if val is None:
        val = getattr(t, "_data", None)
        if isinstance(val, ProxyTensor):
            val = getattr(val, "concrete_value", None)
    return val


def _collect_live_tensors() -> list[Tensor]:
    """Collect all live `Tensor` instances tracked by the garbage collector.

    Returns:
        list[Tensor]: List of active `Tensor` objects in memory.
    """
    import gc

    return [obj for obj in gc.get_objects() if issubclass(type(obj), Tensor)]


def _get_inputs_dict(graph: GradValue) -> dict[str, GradValue]:
    """Map input node IDs of `graph` to their concrete values for graph evaluation.

    Args:
        graph (GradValue): Active tracing graph.

    Returns:
        dict[str, GradValue]: Mapping of input node IDs to concrete backend arrays.

    Raises:
        ValueError: If any graph input node is missing a concrete value.
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
    """Convert a sequence of primal values into `Tensor` instances.

    Args:
        primals (Sequence[GradValue]): Input primal values.

    Returns:
        list[Tensor]: Converted `Tensor` instances.
    """
    tensor_primals: list[Tensor] = []

    for p in primals:
        if isinstance(p, Tensor):
            tensor_primals.append(p)
        else:
            arr = typing.cast(BackendArray, get_active_backend().asarray(p))
            dt = DType.Float32
            if str(arr.dtype) == "float64":
                dt = DType.Float64
            elif "int" in str(arr.dtype):
                dt = DType.Int32
            elif str(arr.dtype) == "bool":
                dt = DType.Bool
            tensor_primals.append(Tensor(arr, TensorConfig(arr.shape, dt, Device("cpu"))))
    return tensor_primals


def _get_fun_primal(fun: GradValue, has_aux: bool) -> TypedVariadicCallable[GradValue]:
    """Return a wrapper extracting only the primal output when `has_aux` is True.

    Args:
        fun (GradValue): Target function.
        has_aux (bool): Whether `fun` returns `(primal, aux)`.

    Returns:
        TypedVariadicCallable[GradValue]: Primal-only callable.
    """
    fn = typing.cast(TypedVariadicCallable[GradValue], fun)
    if has_aux:

        def fun_primal(*args: GradValue, **kwargs: GradValue) -> GradValue:
            """Evaluate `fun` and return the first element of `(primal, aux)`.

            Args:
                *args (GradValue): Positional arguments.
                **kwargs (GradValue): Keyword arguments.

            Returns:
                GradValue: Primary output value.
            """
            out_val: GradValue = fn(*args, **kwargs)
            return out_val[0] if isinstance(out_val, tuple) else out_val

        return fun_primal

    return fn


def _to_original_type(val: GradValue, orig: GradValue) -> GradValue:
    """Wrap `val` into a `Tensor` if `orig` was a `Tensor`.

    Args:
        val (GradValue): Computed gradient array or `Tensor`.
        orig (GradValue): Original primal input argument.

    Returns:
        GradValue: Gradient converted to match the type of `orig`.
    """
    if isinstance(val, Tensor):
        return val

    if isinstance(orig, Tensor):
        arr = typing.cast(BackendArray, get_active_backend().asarray(val))
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
    fun: GradValue,
    options: GradOptions,
    args: tuple[GradValue, ...],
) -> tuple[GradValue, GradValue]:
    """Compute both the primal output and selected argument gradients of `fun` at `args`.

    Args:
        fun (GradValue): Function to differentiate.
        options (GradOptions): Differentiation configuration (`argnums`, `has_aux`).
        args (tuple[GradValue, ...]): Tuple of primal arguments.

    Returns:
        tuple[GradValue, GradValue]: Tuple of `(primal_value, gradients)`.
    """
    from .jvp_vjp import _unpack_pair, vjp

    val, vjp_fn_raw = _unpack_pair(vjp(fun, *args, has_aux=getattr(options, "has_aux", False)))
    vjp_fn = typing.cast("Callable[[GradValue], GradValue]", vjp_fn_raw)

    if getattr(options, "has_aux", False):
        primal_val, _ = typing.cast(tuple[GradValue, GradValue], val)
    else:
        primal_val = val

    if global_tracing_state.is_tracing and global_tracing_state.active_graph is not None:
        c_id = f"cot_{uuid.uuid4().hex[:6]}"
        p_shape = getattr(primal_val, "shape", ())
        c_node = LogicalNode(
            id=c_id,
            op_type="Constant",
            inputs=[],
            attributes={"value": 1.0},
            shape_metadata=p_shape,
        )
        global_tracing_state.add_node(c_node)
        proxy = ProxyTensor(id=c_id, shape=p_shape, dtype="float32")
        cotangent: GradValue = Tensor(proxy, TensorConfig(p_shape, DType.Float32, Device("cpu")))
    else:
        primal_arr = get_active_backend().asarray(getattr(primal_val, "data", primal_val))
        cotangent = get_active_backend().execute_op("Ones_like", primal_arr)

    grads = vjp_fn(cotangent)
    grads_seq = tuple(grads) if isinstance(grads, (list, tuple)) else (grads,)

    argnums = getattr(options, "argnums", 0)
    if isinstance(argnums, int):
        res_grad = _to_original_type(grads_seq[argnums], args[argnums])
    elif isinstance(argnums, (tuple, list)):
        res_grad = typing.cast(
            GradValue,
            tuple(_to_original_type(grads_seq[idx], args[idx]) for idx in typing.cast(tuple[int, ...], argnums)),
        )
    else:
        res_grad = _to_original_type(grads_seq[0], args[0])

    return val, res_grad
