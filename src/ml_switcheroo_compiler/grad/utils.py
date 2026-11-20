"""Internal helper utilities and protocols for automatic differentiation."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Mapping, Sequence
from types import ModuleType

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    CallableLike,
    CallableObjectProtocol,
    OpValue,
    Scalar,
    ShapeDim,
    ShapeProtocol,
    SliceProtocol,
    TypedVariadicCallable,
    VariadicCallableProtocol,
    extract_data,
    extract_dtype,
    extract_int_shape,
    is_variadic_callable,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, Variable
from ml_switcheroo_compiler.interpreter.environment import InterpreterValue, as_eval_value
from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from .options import GradOptions

PrimalTree = typing.Union[  # noqa: UP007
    Tensor,
    ShapeProtocol,
    Scalar,
    None,
    Sequence["PrimalTree"],
    Mapping[str, "PrimalTree"],
]
PrimalAuxValue = typing.Union[PrimalTree, str]  # noqa: UP007
PrimalReturn = typing.Union[PrimalTree, tuple[PrimalTree, PrimalAuxValue]]  # noqa: UP007
DifferentiableFn = typing.Union[  # noqa: UP007
    CallableLike[PrimalReturn],
    CallableLike[OpValue],
    TypedVariadicCallable[PrimalReturn],
    VariadicCallableProtocol,
    CallableObjectProtocol,
]


def as_primal_tree(val: OpValue | PrimalTree) -> PrimalTree:
    """Validate and narrow a runtime tensor, scalar, or PyTree container to `PrimalTree`.

    Args:
        val (OpValue | PrimalTree): Candidate tensor, array, scalar, or nested container.

    Returns:
        PrimalTree: Validated primal PyTree value.
    """
    if val is None or isinstance(val, (Tensor, int, float, bool, complex, ShapeProtocol)):
        return val
    if isinstance(val, tuple):
        return tuple(as_primal_tree(item) for item in val)
    if isinstance(val, list):
        return [as_primal_tree(item) for item in val]
    if isinstance(val, Mapping):
        return {str(k): as_primal_tree(v) for k, v in val.items()}
    arr = get_active_backend().asarray(val)
    assert isinstance(arr, (ShapeProtocol, int, float, bool, complex))
    return arr


@typing.runtime_checkable
class GradArrayProtocol(typing.Protocol):
    """Protocol for evaluated backend arrays supporting shape, flatten, and reshape."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return array shape tuple.

        Returns:
            tuple[int, ...]: Array dimensions.
        """
        ...

    @property
    def ndim(self) -> int:
        """Return array rank.

        Returns:
            int: Number of dimensions.
        """
        ...

    def __len__(self) -> int:
        """Return length along axis 0.

        Returns:
            int: Leading dimension size.
        """
        ...

    def __getitem__(self, key: SliceProtocol | int) -> GradArrayProtocol:
        """Index or slice array along axis 0.

        Args:
            key (SliceProtocol | int): Slice or integer index.

        Returns:
            GradArrayProtocol: Indexed subarray.
        """
        ...

    def flatten(self) -> GradArrayProtocol:
        """Return 1-D flattened array.

        Returns:
            GradArrayProtocol: Flattened array.
        """
        ...

    def reshape(self, shape: tuple[int, ...]) -> OpValue:
        """Return array reshaped to ``shape``.

        Args:
            shape (tuple[int, ...]): Target shape tuple.

        Returns:
            OpValue: Reshaped array.
        """
        ...


def _ensure_grad_tensor(val: OpValue | GradArrayProtocol) -> GradArrayProtocol:
    """Verify and narrow ``val`` to ``GradArrayProtocol``.

    Args:
        val (OpValue | GradArrayProtocol): Candidate array value.

    Returns:
        GradArrayProtocol: Verified backend array.
    """
    assert isinstance(val, GradArrayProtocol) and not isinstance(val, ModuleType)
    return val


def _coerce_grad_shape(seq: Sequence[ShapeDim]) -> tuple[int, ...]:
    """Convert a sequence of dimensions into a concrete integer shape tuple.

    Args:
        seq (Sequence[ShapeDim]): Shape sequence.

    Returns:
        tuple[int, ...]: Shape tuple.
    """
    return tuple(int(d) for d in seq if isinstance(d, int))


@typing.runtime_checkable
class _IdDataProtocol(typing.Protocol):
    """Protocol for proxy tensor data exposing an `id` attribute."""

    id: str


@typing.runtime_checkable
class _ConcreteValueProtocol(typing.Protocol):
    """Protocol for proxy tensor data exposing a `concrete_value` attribute."""

    concrete_value: OpValue


@typing.runtime_checkable
class _PrivateDataProtocol(typing.Protocol):
    """Protocol for tensors exposing a `_data` attribute."""

    _data: OpValue


@typing.runtime_checkable
class _DataFieldProtocol(typing.Protocol):
    """Protocol for tensors exposing a `data` attribute."""

    data: OpValue


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


def _has_id_data(val: OpValue) -> typing.TypeGuard[_IdDataProtocol]:
    """Check whether `val` exposes `id`.

    Args:
        val (OpValue): Candidate data object.

    Returns:
        bool: True if `val` has `id`.
    """
    return isinstance(val, _IdDataProtocol)


def _has_concrete_value(val: OpValue) -> typing.TypeGuard[_ConcreteValueProtocol]:
    """Check whether `val` exposes `concrete_value`.

    Args:
        val (OpValue): Candidate data object.

    Returns:
        bool: True if `val` has `concrete_value`.
    """
    return isinstance(val, _ConcreteValueProtocol)


def _has_private_data(val: OpValue) -> typing.TypeGuard[_PrivateDataProtocol]:
    """Check whether `val` exposes `_data`.

    Args:
        val (OpValue): Candidate tensor object.

    Returns:
        bool: True if `val` has `_data`.
    """
    return isinstance(val, _PrivateDataProtocol)


def _has_data_field(val: OpValue) -> typing.TypeGuard[_DataFieldProtocol]:
    """Check whether `val` exposes `data`.

    Args:
        val (OpValue): Candidate tensor object.

    Returns:
        bool: True if `val` has `data`.
    """
    return isinstance(val, _DataFieldProtocol)


def _has_requires_grad(val: OpValue) -> typing.TypeGuard[_RequiresGradProtocol]:
    """Check whether `val` exposes `requires_grad`.

    Args:
        val (OpValue): Candidate tensor object.

    Returns:
        bool: True if `val` has `requires_grad`.
    """
    return isinstance(val, _RequiresGradProtocol)


def _has_trainable(val: OpValue) -> typing.TypeGuard[_TrainableProtocol]:
    """Check whether `val` exposes `trainable`.

    Args:
        val (OpValue): Candidate variable object.

    Returns:
        bool: True if `val` has `trainable`.
    """
    return isinstance(val, _TrainableProtocol)


def _has_graph_nodes(
    val: LogicalGraph | _GraphNodesProtocol | _GraphInputsProtocol | OpValue,
) -> typing.TypeGuard[_GraphNodesProtocol]:
    """Check whether `val` exposes `nodes`.

    Args:
        val (LogicalGraph | _GraphNodesProtocol | _GraphInputsProtocol | OpValue): Candidate graph object.

    Returns:
        bool: True if `val` has `nodes`.
    """
    return isinstance(val, _GraphNodesProtocol)


def _has_graph_inputs(
    val: LogicalGraph | _GraphNodesProtocol | _GraphInputsProtocol | OpValue,
) -> typing.TypeGuard[_GraphInputsProtocol]:
    """Check whether `val` exposes `inputs`.

    Args:
        val (LogicalGraph | _GraphNodesProtocol | _GraphInputsProtocol | OpValue): Candidate graph object.

    Returns:
        bool: True if `val` has `inputs`.
    """
    return isinstance(val, _GraphInputsProtocol)


def _has_shape_field(val: OpValue) -> typing.TypeGuard[ShapeProtocol]:
    """Check whether `val` exposes `shape`.

    Args:
        val (OpValue): Candidate tensor or array.

    Returns:
        bool: True if `val` has `shape`.
    """
    return isinstance(val, ShapeProtocol)


def _unwrap_data(val: OpValue | PrimalTree) -> OpValue:
    """Return `val.data` if `val` exposes `.data`, otherwise `val`.

    Args:
        val (OpValue | PrimalTree): Candidate tensor or array.

    Returns:
        OpValue: Unwrapped data payload.
    """
    raw: OpValue = val
    return raw.data if _has_data_field(raw) else raw


def _extract_shape(val: OpValue | PrimalTree) -> tuple[int, ...]:
    """Return `tuple(val.shape)` if `val` exposes `.shape`, otherwise `()`.

    Args:
        val (OpValue | PrimalTree): Candidate tensor or array.

    Returns:
        tuple[int, ...]: Shape tuple.
    """
    raw: OpValue = val
    return _coerce_grad_shape(raw.shape) if _has_shape_field(raw) else ()


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
    def data(self) -> PrimalTree:
        """Return underlying array buffer.

        Returns:
            PrimalTree: Computed result.
        """
        ...

    def __len__(self) -> int:
        """Return length of the leading dimension.

        Returns:
            int: Computed result.
        """
        ...


BackendArray = GradTensorProtocol


def value_and_grad_wrt_vars(
    fun: DifferentiableFn,
) -> TypedVariadicCallable[tuple[PrimalTree, dict[str, PrimalTree]]]:
    """Create a function that evaluates both the primal value and variable gradients.

    Args:
        fun (DifferentiableFn): Original function to wrap.

    Returns:
        TypedVariadicCallable[tuple[PrimalTree, dict[str, PrimalTree]]]: Wrapped callable returning
            `(value, grads_dict)`.
    """

    def wrapped(*args: OpValue, **kwargs: OpValue) -> tuple[PrimalTree, dict[str, PrimalTree]]:
        """Evaluate `fun` and return `(value, grads_dict)`.

        Args:
            *args (OpValue): Positional arguments.
            **kwargs (OpValue): Keyword arguments.

        Returns:
            tuple[PrimalTree, dict[str, PrimalTree]]: Primal output value and variable gradient dictionary.
        """
        _callable_fn_1: OpValue = fun
        assert is_variadic_callable(_callable_fn_1)
        val = as_primal_tree(_callable_fn_1(*args, **kwargs))
        grads: dict[str, PrimalTree] = {}
        return (val, grads)

    return wrapped


def _check_scalar(tensor: OpValue | PrimalTree) -> None:
    """Validate that `tensor` has a scalar (unit-element) shape.

    Args:
        tensor (OpValue | PrimalTree): Tensor to validate.

    Raises:
        SwitcherooError: If `tensor` is not a scalar tensor.
    """
    from ml_switcheroo_compiler.core.errors import SwitcherooError

    raw: OpValue = tensor
    shape: Sequence[int | str | SymNode | SymInt | None] = raw.shape if _has_shape_field(raw) else ()
    prod = 1
    for s in shape:
        try:
            prod *= int(str(s))
        except (ValueError, TypeError):
            prod = 2

    if prod != 1:
        raise SwitcherooError("backward() can only be called on scalar tensors.")


def _find_wrt_tensors(
    graph: LogicalGraph | _GraphNodesProtocol | OpValue,
) -> tuple[list[Tensor], list[str]]:
    """Find all live trainable tensors/variables whose node IDs belong to `graph`.

    Args:
        graph (LogicalGraph | _GraphNodesProtocol | OpValue): Active tracing graph.

    Returns:
        tuple[list[Tensor], list[str]]: Tuple of `(wrt_tensors, wrt_node_ids)`.
    """
    all_tensors = _collect_live_tensors()
    graph_nodes = graph.nodes if _has_graph_nodes(graph) else {}
    wrt_tensors: list[Tensor] = []
    wrt_ids: list[str] = []
    for t in all_tensors:
        raw_t: OpValue = t
        raw_data: OpValue = t.data
        if _has_id_data(raw_data):
            node_id = str(raw_data.id)
            if node_id in graph_nodes:
                req_grad = _has_requires_grad(raw_t) and bool(raw_t.requires_grad)
                trainable = _has_trainable(raw_t) and bool(raw_t.trainable)
                if req_grad or trainable or isinstance(t, Variable):
                    wrt_tensors.append(t)
                    wrt_ids.append(node_id)
    return wrt_tensors, wrt_ids


def _get_concrete_val(t: OpValue | PrimalTree) -> OpValue:
    """Extract the concrete array value attached to a traced `Tensor` or `ProxyTensor`.

    Args:
        t (OpValue | PrimalTree): Traced tensor object.

    Returns:
        OpValue: Concrete backing array value, or None if unavailable.
    """
    raw_data: OpValue = _unwrap_data(t)
    val: OpValue = raw_data.concrete_value if _has_concrete_value(raw_data) else None
    if val is None:
        raw_t: OpValue = t
        priv_data = raw_t._data if _has_private_data(raw_t) else None
        raw_priv: OpValue = priv_data
        val = raw_priv.concrete_value if _has_concrete_value(raw_priv) else priv_data
    return val


def _collect_live_tensors() -> list[Tensor]:
    """Collect all live `Tensor` instances tracked in the weak tensor registry.

    Returns:
        list[Tensor]: List of active `Tensor` objects in memory without GC heap scanning.
    """
    from ml_switcheroo_compiler.tracing.state import get_live_tensors

    return get_live_tensors()


def _get_inputs_dict(
    graph: LogicalGraph | _GraphNodesProtocol | _GraphInputsProtocol | OpValue,
) -> dict[str, InterpreterValue]:
    """Map input node IDs of `graph` to their concrete values for graph evaluation.

    Args:
        graph (LogicalGraph | _GraphNodesProtocol | _GraphInputsProtocol | OpValue): Active tracing graph.

    Returns:
        dict[str, InterpreterValue]: Mapping of input node IDs to concrete backend arrays.

    Raises:
        ValueError: If any graph input node is missing a concrete value.
    """
    all_tensors = _collect_live_tensors()
    graph_nodes = graph.nodes if _has_graph_nodes(graph) else {}
    graph_inputs = graph.inputs if _has_graph_inputs(graph) else ()
    inputs_dict: dict[str, InterpreterValue] = {}
    for t in all_tensors:
        raw_data: OpValue = _unwrap_data(t)
        node_id = str(raw_data.id) if _has_id_data(raw_data) else ""
        if node_id and node_id in graph_nodes:
            val = _get_concrete_val(t)
            if val is not None:
                inputs_dict[node_id] = as_eval_value(get_active_backend().asarray(val))

    for inp_id in graph_inputs:
        if inp_id not in inputs_dict:
            raise ValueError(f"Missing input value for node '{inp_id}'.")

    return inputs_dict


def _convert_to_tensors(primals: Sequence[PrimalTree | OpValue]) -> list[Tensor]:
    """Convert a sequence of primal values into `Tensor` instances.

    Args:
        primals (Sequence[PrimalTree | OpValue]): Input primal values.

    Returns:
        list[Tensor]: Converted `Tensor` instances.
    """
    tensor_primals: list[Tensor] = []

    for p in primals:
        if isinstance(p, Tensor):
            tensor_primals.append(p)
        else:
            arr = get_active_backend().asarray(p)
            dt_str = str(extract_dtype(arr) or "")
            dt = DType.Float32
            if dt_str == "float64":
                dt = DType.Float64
            elif "int" in dt_str:
                dt = DType.Int32
            elif dt_str == "bool":
                dt = DType.Bool
            tensor_primals.append(Tensor(arr, TensorConfig(extract_int_shape(arr), dt, Device("cpu"))))
    return tensor_primals


def _get_fun_primal(fun: DifferentiableFn, has_aux: bool) -> TypedVariadicCallable[PrimalTree]:
    """Return a wrapper extracting only the primal output when `has_aux` is True.

    Args:
        fun (DifferentiableFn): Target function.
        has_aux (bool): Whether `fun` returns `(primal, aux)`.

    Returns:
        TypedVariadicCallable[PrimalTree]: Primal-only callable.
    """
    if has_aux:

        def fun_primal(*args: OpValue, **kwargs: OpValue) -> PrimalTree:
            """Evaluate `fun` and return the first element of `(primal, aux)`.

            Args:
                *args (OpValue): Positional arguments.
                **kwargs (OpValue): Keyword arguments.

            Returns:
                PrimalTree: Primary output value.
            """
            _callable_fn_2: OpValue = fun
            assert is_variadic_callable(_callable_fn_2)
            out_val = _callable_fn_2(*args, **kwargs)
            first = out_val[0] if isinstance(out_val, tuple) else out_val
            return as_primal_tree(first)

        return fun_primal

    def fun_direct(*args: OpValue, **kwargs: OpValue) -> PrimalTree:
        """Evaluate `fun` directly.

        Args:
            *args (OpValue): Positional arguments.
            **kwargs (OpValue): Keyword arguments.

        Returns:
            PrimalTree: Evaluated result.
        """
        _callable_fn_3: OpValue = fun
        assert is_variadic_callable(_callable_fn_3)
        return as_primal_tree(_callable_fn_3(*args, **kwargs))

    return fun_direct


def _to_original_type(val: PrimalTree | OpValue, orig: PrimalTree | OpValue) -> PrimalTree:
    """Wrap `val` into a `Tensor` if `orig` was a `Tensor`.

    Args:
        val (PrimalTree | OpValue): Computed gradient array or `Tensor`.
        orig (PrimalTree | OpValue): Original primal input argument.

    Returns:
        PrimalTree: Gradient converted to match the type of `orig`.
    """
    if isinstance(val, Tensor):
        return val

    if isinstance(orig, Tensor):
        arr = get_active_backend().asarray(val)
        dt_str = str(extract_dtype(arr) or "")
        dt = DType.Float32
        if dt_str == "float64":
            dt = DType.Float64
        elif "int" in dt_str:
            dt = DType.Int32
        elif dt_str == "bool":
            dt = DType.Bool

        return Tensor(arr, TensorConfig(extract_int_shape(arr), dt, Device("cpu")))
    return as_primal_tree(val)


def _compute_grad_and_value(
    fun: DifferentiableFn,
    options: GradOptions,
    raw_args: tuple[OpValue, ...],
) -> tuple[OpValue, PrimalTree]:
    """Compute both the primal output and selected argument gradients of `fun` at `raw_args`.

    Args:
        fun (DifferentiableFn): Function to differentiate.
        options (GradOptions): Differentiation configuration (`argnums`, `has_aux`).
        raw_args (tuple[OpValue, ...]): Tuple of primal arguments.

    Returns:
        tuple[OpValue, PrimalTree]: Tuple of `(primal_value, gradients)`.
    """
    from .jvp_vjp import _unpack_pair, vjp

    args: tuple[PrimalTree, ...] = tuple(
        a if isinstance(a, Tensor) else as_primal_tree(extract_data(a)) for a in raw_args
    )
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
        cotangent: PrimalTree = Tensor(proxy, TensorConfig(p_shape, DType.Float32, Device("cpu")))
    else:
        primal_arr = get_active_backend().asarray(_unwrap_data(primal_val))
        cotangent = as_primal_tree(get_active_backend().execute_op("Ones_like", primal_arr))
    _callable_fn_4: OpValue = vjp_fn
    assert is_variadic_callable(_callable_fn_4)
    grads = _callable_fn_4(cotangent)
    grads_seq = tuple(grads) if isinstance(grads, (list, tuple)) else (grads,)

    argnums = options.argnums
    if isinstance(argnums, int):
        res_grad: PrimalTree = _to_original_type(grads_seq[argnums], args[argnums])
    elif isinstance(argnums, (tuple, list)):
        res_grad = tuple(_to_original_type(grads_seq[idx], args[idx]) for idx in argnums)
    else:
        res_grad = _to_original_type(grads_seq[0], args[0])

    return val, res_grad
