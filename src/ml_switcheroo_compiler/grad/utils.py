"""Internal helper utilities and protocols for automatic differentiation."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Mapping, Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    ShapeProtocol,
)
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    TensorOrScalar,
    TypedVariadicCallable,
    extract_data,
    invoke_callable,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, Variable
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from .options import GradOptions


def _is_grad_eager_tensor(val: TensorOrScalar) -> typing.TypeGuard[EagerTensorProtocol]:
    """Type guard narrowing `val` to `EagerTensorProtocol`.

    Args:
        val (TensorOrScalar): Candidate array value.

    Returns:
        bool: True if non-None.
    """
    return val is not None


def _ensure_grad_tensor(val: TensorOrScalar) -> EagerTensorProtocol:
    """Narrow `val` to `EagerTensorProtocol`.

    Args:
        val (TensorOrScalar): Candidate array value.

    Returns:
        EagerTensorProtocol: Narrowed tensor.
    """
    assert _is_grad_eager_tensor(val)
    return val


def _is_grad_shape_tuple(seq: tuple[int | str | None, ...]) -> typing.TypeGuard[tuple[int, ...]]:
    """Type guard narrowing a shape tuple to `tuple[int, ...]`.

    Args:
        seq (tuple[int | str | None, ...]): Shape tuple.

    Returns:
        bool: True if seq is a tuple.
    """
    return isinstance(seq, tuple)


def _coerce_grad_shape(seq: Sequence[int | str | None]) -> tuple[int, ...]:
    """Convert a sequence of dimensions into a shape tuple.

    Args:
        seq (Sequence[int | str | None]): Shape sequence.

    Returns:
        tuple[int, ...]: Shape tuple.
    """
    res = tuple(seq)
    assert _is_grad_shape_tuple(res)
    return res


@typing.runtime_checkable
class _IdDataProtocol(typing.Protocol):
    """Protocol for proxy tensor data exposing an `id` attribute."""

    id: str


@typing.runtime_checkable
class _ConcreteValueProtocol(typing.Protocol):
    """Protocol for proxy tensor data exposing a `concrete_value` attribute."""

    concrete_value: GradValue


@typing.runtime_checkable
class _PrivateDataProtocol(typing.Protocol):
    """Protocol for tensors exposing a `_data` attribute."""

    _data: GradValue


@typing.runtime_checkable
class _DataFieldProtocol(typing.Protocol):
    """Protocol for tensors exposing a `data` attribute."""

    data: GradValue


@typing.runtime_checkable
class _RequiresGradProtocol(typing.Protocol):
    """Protocol for tensors exposing a `requires_grad` boolean flag."""

    requires_grad: bool


@typing.runtime_checkable
class _TrainableProtocol(typing.Protocol):
    """Protocol for variables exposing a `trainable` boolean flag."""

    trainable: bool


@typing.runtime_checkable
class _GraphNodesProtocol(typing.Protocol):
    """Protocol for graphs exposing a `nodes` mapping."""

    nodes: Mapping[str, LogicalNode]


@typing.runtime_checkable
class _GraphInputsProtocol(typing.Protocol):
    """Protocol for graphs exposing an `inputs` sequence."""

    inputs: Sequence[str]


def _unwrap_data(val: GradValue) -> GradValue:
    """Return `val.data` if `val` exposes `.data`, otherwise `val`.

    Args:
        val (GradValue): Candidate tensor or array.

    Returns:
        GradValue: Unwrapped data payload.
    """
    raw: TensorOrScalar = val
    return raw.data if isinstance(raw, _DataFieldProtocol) else val


def _extract_shape(val: GradValue) -> tuple[int, ...]:
    """Return `tuple(val.shape)` if `val` exposes `.shape`, otherwise `()`.

    Args:
        val (GradValue): Candidate tensor or array.

    Returns:
        tuple[int, ...]: Shape tuple.
    """
    raw: TensorOrScalar = val
    return _coerce_grad_shape(raw.shape) if isinstance(raw, ShapeProtocol) else ()


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
        val = invoke_callable(fun, *args, **kwargs)
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

    shape = _extract_shape(tensor)
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
    raw_graph: TensorOrScalar = graph
    graph_nodes = raw_graph.nodes if isinstance(raw_graph, _GraphNodesProtocol) else {}
    wrt_tensors: list[GradValue] = []
    wrt_ids: list[str] = []
    for t in all_tensors:
        raw_t: TensorOrScalar = t
        raw_data: TensorOrScalar = t.data
        if isinstance(raw_data, _IdDataProtocol):
            node_id = str(raw_data.id)
            if node_id in graph_nodes:
                req_grad = isinstance(raw_t, _RequiresGradProtocol) and bool(raw_t.requires_grad)
                trainable = isinstance(raw_t, _TrainableProtocol) and bool(raw_t.trainable)
                if req_grad or trainable or isinstance(t, Variable):
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
    raw_data: TensorOrScalar = _unwrap_data(t)
    val: GradValue = raw_data.concrete_value if isinstance(raw_data, _ConcreteValueProtocol) else None
    if val is None:
        raw_t: TensorOrScalar = t
        priv_data = raw_t._data if isinstance(raw_t, _PrivateDataProtocol) else None
        raw_priv: TensorOrScalar = priv_data
        val = raw_priv.concrete_value if isinstance(raw_priv, _ConcreteValueProtocol) else priv_data
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
    raw_graph: TensorOrScalar = graph
    graph_nodes = raw_graph.nodes if isinstance(raw_graph, _GraphNodesProtocol) else {}
    graph_inputs = raw_graph.inputs if isinstance(raw_graph, _GraphInputsProtocol) else ()
    inputs_dict: dict[str, GradValue] = {}
    for t in all_tensors:
        raw_data: TensorOrScalar = _unwrap_data(t)
        node_id = str(raw_data.id) if isinstance(raw_data, _IdDataProtocol) else ""
        if node_id and node_id in graph_nodes:
            val = _get_concrete_val(t)
            if val is not None:
                inputs_dict[node_id] = get_active_backend().asarray(val)

    for inp_id in graph_inputs:
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
            arr = _ensure_grad_tensor(get_active_backend().asarray(p))
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
    if has_aux:

        def fun_primal(*args: GradValue, **kwargs: GradValue) -> GradValue:
            """Evaluate `fun` and return the first element of `(primal, aux)`.

            Args:
                *args (GradValue): Positional arguments.
                **kwargs (GradValue): Keyword arguments.

            Returns:
                GradValue: Primary output value.
            """
            out_val: GradValue = invoke_callable(fun, *args, **kwargs)
            return out_val[0] if isinstance(out_val, tuple) else out_val

        return fun_primal

    def fun_direct(*args: GradValue, **kwargs: GradValue) -> GradValue:
        """Evaluate `fun` directly.

        Args:
            *args (GradValue): Positional arguments.
            **kwargs (GradValue): Keyword arguments.

        Returns:
            GradValue: Evaluated result.
        """
        return invoke_callable(fun, *args, **kwargs)

    return fun_direct


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
        arr = _ensure_grad_tensor(get_active_backend().asarray(val))
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
    raw_args: tuple[GradValue, ...],
) -> tuple[GradValue, GradValue]:
    """Compute both the primal output and selected argument gradients of `fun` at `raw_args`.

    Args:
        fun (GradValue): Function to differentiate.
        options (GradOptions): Differentiation configuration (`argnums`, `has_aux`).
        raw_args (tuple[GradValue, ...]): Tuple of primal arguments.

    Returns:
        tuple[GradValue, GradValue]: Tuple of `(primal_value, gradients)`.
    """
    from .jvp_vjp import _unpack_pair, vjp

    args: tuple[GradValue, ...] = tuple(a if isinstance(a, Tensor) else extract_data(a) for a in raw_args)
    val, vjp_fn_raw = _unpack_pair(vjp(fun, *args, has_aux=options.has_aux))
    vjp_fn = vjp_fn_raw

    if options.has_aux:
        primal_val, _ = _unpack_pair(val)
    else:
        primal_val = val

    if global_tracing_state.is_tracing and global_tracing_state.active_graph is not None:
        c_id = f"cot_{uuid.uuid4().hex[:6]}"
        p_shape = _extract_shape(primal_val)
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
        primal_arr = get_active_backend().asarray(_unwrap_data(primal_val))
        cotangent = get_active_backend().execute_op("Ones_like", primal_arr)

    grads = invoke_callable(vjp_fn, cotangent)
    grads_seq = tuple(grads) if isinstance(grads, (list, tuple)) else (grads,)

    argnums: GradValue = options.argnums
    if isinstance(argnums, int):
        res_grad: GradValue = _to_original_type(grads_seq[argnums], args[argnums])
    elif isinstance(argnums, (tuple, list)):
        res_grad = tuple(_to_original_type(grads_seq[int(idx)], args[int(idx)]) for idx in argnums if isinstance(idx, int))
    else:
        res_grad = _to_original_type(grads_seq[0], args[0])

    return val, res_grad
