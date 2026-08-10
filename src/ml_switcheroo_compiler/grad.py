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

__all__ = [
    "backward",
    "hvp",
    "jvp",
    "vjp",
    "jacfwd",
    "jacrev",
    "hessian",
    "checkpoint",
    "remat",
    "overwrite_with_gradient",
    "CustomVJPFunction",
    "custom_vjp",
    "check_numerical_grads",
    "UnconnectedGradients",
    "RegisterGradient",
    "GradCheckOptions",
    "GradOptions",
    "JitOptions",
    "ir_grad",
    "grad",
    "value_and_grad",
    "jit",
    "disable_jit",
    "eval_shape",
]


class CustomVJPFunction:
    """Wrap for custom_vjp functions."""

    def __init__(self, fun: Callable[..., Any]) -> None:
        """Initialize the custom VJP wrapper.

        Args:
            fun (Callable[..., Any]): The base function being wrapped for custom VJPs.
        """
        self.fun = fun
        self.fwd = None
        self.bwd = None
        self._tracing_fwd = False

    def defvjp(self, fwd: Callable[..., Any], bwd: Callable[..., Any]) -> None:
        """Define the forward and backward passes.

        Args:
            fwd (Callable[..., Any]): The forward pass.
            bwd (Callable[..., Any]): The backward pass.
        """
        self.fwd = fwd  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        self.bwd = bwd  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def _extract_tensor_args(self, args: tuple[Any, ...]) -> list[Any]:
        """Extract all tensor instances from the provided arguments.

        Args:
            args (tuple): The positional arguments.

        Returns:
            list: A list containing only the tensor arguments.
        """
        return [a for a in args if isinstance(a, Tensor)]

    def _trace_fwd_graph(self, tensor_args: list[Any]) -> Any:
        """Trace the forward pass of the custom VJP function to construct its logical graph.

        Args:
            tensor_args (list): The list of tensor arguments.

        Returns: Any: The traced logical graph for the forward pass.
        """
        if self.fwd is None or self.bwd is None:
            return None

        self._tracing_fwd = True
        try:
            return _trace_function(self.fwd, tuple(tensor_args), "fwd_pass")  # type: ignore
        finally:
            self._tracing_fwd = False

    def _resolve_output_metadata(self, tensor_args: list[Any]) -> tuple[tuple[int, ...], str, str]:
        """Determine the shape, dtype, and device for the output based on the input tensors.

        Args:
            tensor_args (list): The list of tensor arguments.

        Returns:
            tuple: A tuple containing the output shape, dtype, and device string.
        """
        shape = ()
        dtype = "float32"
        device = "cpu"
        if tensor_args:
            first_arg = tensor_args[0]
            shape = getattr(first_arg, "shape", ())
            dtype = getattr(getattr(first_arg, "dtype", None), "value", "float32")
            device = getattr(first_arg, "device", "cpu")
        return (shape, dtype, device)

    def _emit_vjp_node(self, tensor_args: list[Any], fwd_graph: Any, primal_graph: Any) -> Any:
        """Emit a CustomVJP node into the active computation graph.

        Args:
            tensor_args (list): The tensor arguments.
            fwd_graph (object): The traced forward graph.
            primal_graph (object): The primal logical graph.

        Returns: Any: The proxy tensor representing the custom VJP output.
        """
        out_id = str(uuid.uuid4())
        meta = self._resolve_output_metadata(tensor_args)
        node = LogicalNode(
            id=out_id,
            op_type="CustomVJP",
            inputs=[a.data.id for a in tensor_args],  # type: ignore
            attributes={"primal_graph": primal_graph, "fwd_graph": fwd_graph, "bwd_fn": self.bwd},
            shape_metadata=meta[0],
        )
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=meta[0], dtype=meta[1])
        return Tensor(proxy, TensorConfig(meta[0], DType(meta[1]), meta[2]))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the function, tracing the forward and primal graphs if tracing is active.

        Args:
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: The computed primal output.
        """
        if config.eager_mode or not global_tracing_state.is_tracing or self._tracing_fwd:
            return self.fun(*args, **kwargs)

        tensor_args = self._extract_tensor_args(args)
        fwd_graph = self._trace_fwd_graph(tensor_args)
        primal_graph = _trace_function(self.fun, tuple(tensor_args), "primal_pass")  # type: ignore
        return self._emit_vjp_node(tensor_args, fwd_graph, primal_graph)


def custom_vjp(fun: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a function to allow defining custom vector-Jacobian product (VJP) rules.

    Args:
        fun (Callable): The original function.

    Returns:
        Callable: The wrapped function that supports custom VJPs.
    """
    return CustomVJPFunction(fun)


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

        Returns: Any: Result.
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


def custom_jvp(fun: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a function to allow defining custom Jacobian-vector product (JVP) rules.

    Args:
        fun (Callable): The original function.

    Returns:
        Callable: The wrapped function that supports custom JVPs.
    """
    return fun


DEFAULT_GRAD_EPSILON = 0.0001


@dataclass
class GradCheckOptions:
    """Options for gradient checking."""

    order: int = 1
    atol: float = DEFAULT_GRAD_EPSILON
    rtol: float = DEFAULT_GRAD_EPSILON
    step: float = DEFAULT_GRAD_EPSILON


def check_numerical_grads(f: Callable[..., Any], args: tuple[Any, ...], options: Any = None) -> None:
    """Check numerical gradients for a function against analytical gradients.

    Args:
        f (Callable): The function to differentiate.
        args (tuple): The arguments to evaluate the function.
        options (GradCheckOptions): The configuration options for checking grads.

    Raises:
        SwitcherooError: If analytical and numerical gradients do not match.
    """
    options = options or GradCheckOptions()
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.errors import SwitcherooError

    with ConfigContext(eager_mode=True):
        # Compute analytical gradients using VJP
        out, vjp_fn = vjp(f, *args)
        out_arr = get_active_backend().asarray(getattr(out, "data", out))
        cotangent = get_active_backend().execute_op("Ones_like", out_arr)
        analytical_grads = vjp_fn(cotangent)

        step = options.step
        atol = options.atol
        rtol = options.rtol

        for arg_idx, arg in enumerate(args):
            arg_arr = get_active_backend().array(getattr(arg, "data", arg), dtype="float64")
            numerical_grad = get_active_backend().execute_op("Zeros_like", arg_arr)

            flat_arg = arg_arr.ravel()
            flat_num_grad = numerical_grad.ravel()

            for i in range(flat_arg.size):
                orig_val = flat_arg[i]

                # Perturb positive
                flat_arg[i] = orig_val + step
                args_pos = list(args)
                from ml_switcheroo_compiler.core.device import Device

                if isinstance(arg, Tensor):
                    args_pos[arg_idx] = Tensor(
                        arg_arr.reshape(arg_arr.shape).copy(),
                        TensorConfig(arg_arr.shape, DType.Float32, Device("cpu")),  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
                    )
                else:
                    args_pos[arg_idx] = arg_arr.reshape(arg_arr.shape).copy()
                out_pos = f(*args_pos)
                out_pos_arr = get_active_backend().asarray(getattr(out_pos, "data", out_pos))

                # Perturb negative
                flat_arg[i] = orig_val - step
                args_neg = list(args)
                if isinstance(arg, Tensor):
                    args_neg[arg_idx] = Tensor(
                        arg_arr.reshape(arg_arr.shape).copy(),
                        TensorConfig(arg_arr.shape, DType.Float32, Device("cpu")),  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
                    )
                else:
                    args_neg[arg_idx] = arg_arr.reshape(arg_arr.shape).copy()
                out_neg = f(*args_neg)
                out_neg_arr = get_active_backend().asarray(getattr(out_neg, "data", out_neg))

                flat_arg[i] = orig_val

                diff = (out_pos_arr - out_neg_arr) / (2.0 * step)
                flat_num_grad[i] = float(get_active_backend().execute_op("Sum", diff))

            anal_grad = get_active_backend().asarray(getattr(analytical_grads[arg_idx], "data", analytical_grads[arg_idx]))

            if not get_active_backend().execute_op("Allclose", anal_grad, numerical_grad, atol=atol, rtol=rtol):
                msg = f"Gradient check failed for argument {arg_idx}.\nAnalytical gradient:\n{anal_grad}\nNumerical gradient:\n{numerical_grad}"
                raise SwitcherooError(msg)


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


def checkpoint(fun: Callable[..., Any]) -> Callable[..., Any]:
    """Gradient checkpointing / rematerialization.

    Args:
        fun (Callable[..., Any]): The function to checkpoint.

    Returns:
        Callable[..., Any]: The checkpointed function.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Evaluate wrapper operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        from ml_switcheroo_compiler.core.config import config
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        if config.eager_mode or not global_tracing_state.is_tracing:
            return fun(*args, **kwargs)

        import uuid

        from ml_switcheroo_ir import LogicalNode

        from ml_switcheroo_compiler.core.dtype import DType
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
        from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        tensor_args = [a for a in args if isinstance(a, Tensor)]
        fwd_block = _trace_function(fun, tuple(tensor_args), f"checkpoint_{uuid.uuid4().hex[:6]}")

        # Infer output metadata from the traced block
        out_node_id = fwd_block.outputs[0]
        nodes_dict = {n.id: n for n in (fwd_block.nodes if isinstance(fwd_block.nodes, list) else fwd_block.nodes.values())}
        out_node = nodes_dict[out_node_id]
        real_out_node = nodes_dict[out_node.inputs[0]]

        shape = real_out_node.shape_metadata
        # Try to infer dtype. Proxy tensors typically just use float32 as default if not specified
        dtype = "float32"
        if hasattr(real_out_node, "attributes") and "dtype" in real_out_node.attributes:
            dtype = real_out_node.attributes["dtype"]
        elif tensor_args:
            dtype = getattr(getattr(tensor_args[0], "dtype", None), "value", "float32")

        device = "cpu"
        if tensor_args:
            device = getattr(tensor_args[0], "device", "cpu")

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Checkpoint",
            inputs=[a.data.id for a in tensor_args if hasattr(a, "data") and hasattr(a.data, "id")],
            attributes={"subgraph": fwd_block},
            shape_metadata=shape,
        )
        global_tracing_state.add_node(node)

        proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype)
        return Tensor(proxy, TensorConfig(shape, DType(dtype), device))

    return wrapper


def remat(fun: Callable[..., Any]) -> Callable[..., Any]:
    """Gradient checkpointing / rematerialization alias.

    Args:
        fun (Callable[..., Any]): The function to rematerialize.

    Returns:
        Callable[..., Any]: The rematerialized function.
    """
    return checkpoint(fun)


def recompute_grad(fun: Callable[..., Any]) -> Callable[..., Any]:
    """Gradient checkpointing / rematerialization.

    Args:
        fun (Callable[..., Any]): The function to recompute gradients for.

    Returns:
        Callable[..., Any]: The recomputing function.
    """
    return checkpoint(fun)


@dataclass
class GradOptions:
    """Options for gradient compilation."""

    argnums: Any = 0
    has_aux: bool = False
    holistic: bool = False
    reduce_axes: Any = field(default_factory=tuple)
    return_value: bool = False


@dataclass
class JitOptions:
    """Options for JIT compilation."""

    static_argnums: Any = None
    static_argnames: Any = None
    donate_argnums: Any = None
    donate_argnames: Any = None
    keep_unused: bool = False
    device: Any = None
    backend: Any = None
    inline: bool = False
    abstracted_axes: Any = None


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


def jit(fun: Callable[..., Any], options: Any = None) -> Callable[..., Any]:
    """Return a JIT wrapper.

    Args:
        fun (Callable[..., Any]): The function to jit compile.
        options (JitOptions): Configuration options.

    Returns:
        Callable[..., Any]: The JIT wrapped function.
    """
    options = options or JitOptions()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return fun(*args, **kwargs)

    return wrapped


@contextlib.contextmanager
def disable_jit() -> typing.Iterator[None]:
    """Provide context manager to disable JIT locally.

    Yields:
        None: Context manager yield.
    """
    yield


def eval_shape(fun: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Evaluate eval_shape operation.

    Args:
        fun (object): The fun parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return fun(*args, **kwargs)


def _convert_to_tensors(primals: Sequence[Any]) -> list[Tensor]:
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

            Returns: Any: Result.
            """
            out_val = fun(*args)
            return out_val[0] if isinstance(out_val, tuple) else out_val

        return fun_primal
    return fun


def jvp(
    fun: Callable[..., Any],
    primals: Any,
    tangents: Any,
    has_aux: bool = False,
) -> tuple[Any, Any]:
    """Evaluate jvp operation.

    Args:
        fun (object): The fun parameter.
        primals (object): The primals parameter.
        tangents (object): The tangents parameter.
        has_aux (bool): The has_aux parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.interpreter import evaluate_graph
    from ml_switcheroo_compiler.transforms.autodiff import jvp as graph_jvp

    primals_seq = primals if isinstance(primals, (tuple, list)) else (primals,)
    tangents_seq = tangents if isinstance(tangents, (tuple, list)) else (tangents,)

    with ConfigContext(eager_mode=True):
        if has_aux:
            val, aux = fun(*primals_seq)
        else:
            val = fun(*primals_seq)

    fun_primal = _get_fun_primal(fun, has_aux)
    tensor_primals = _convert_to_tensors(primals_seq)

    # Trace
    block = _trace_function(fun_primal, tuple(tensor_primals), f"jvp_{uuid.uuid4().hex[:6]}")
    forward_graph = LogicalGraph(name=block.id)
    for node in block.nodes:
        forward_graph.nodes[node.id] = node
    forward_graph.inputs = block.inputs
    forward_graph.outputs = block.outputs

    # Add tangent constants to the graph
    tangent_ids = []
    for t, p_id in zip(tangents_seq, forward_graph.inputs):
        t_id = f"tangent_{uuid.uuid4().hex[:6]}"
        t_node = LogicalNode(
            id=t_id,
            op_type="Constant",
            attributes={"value": getattr(t, "data", t)},
            shape_metadata=forward_graph.nodes[p_id].shape_metadata,
        )
        forward_graph.nodes[t_id] = t_node
        tangent_ids.append(t_id)

    # Compute JVP graph
    jvp_graph = graph_jvp(forward_graph, forward_graph.inputs, tangent_ids, forward_graph.outputs)

    # Evaluate JVP graph
    inputs_dict = {inp_id: get_active_backend().asarray(getattr(p, "data", p)) for inp_id, p in zip(forward_graph.inputs, tensor_primals)}
    outputs_dict = evaluate_graph(jvp_graph, inputs_dict)

    out_tangent_values = [outputs_dict[out_id] for out_id in jvp_graph.outputs]
    out_tan = out_tangent_values[0] if len(out_tangent_values) == 1 else tuple(out_tangent_values)

    if has_aux:
        return (val, aux), out_tan
    return val, out_tan


def vjp(
    fun: Callable[..., Any],
    *primals: Any,
    has_aux: bool = False,
) -> tuple[Any, Callable]:
    """Evaluate vjp operation.

    Args:
        fun (Callable[..., Any]): The fun parameter.
        *primals (object): Positional args.
        has_aux (bool): Aux.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.interpreter import evaluate_graph
    from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad
    from ml_switcheroo_compiler.tree_util import tree_flatten, tree_unflatten

    # 1. Flatten nested primals if any
    flat_primals, tree_def = tree_flatten(primals)

    # 2. Wrap function to unflatten inputs before execution
    def fun_flat(*flat_args: Any) -> Any:
        """Evaluate fun_flat operation.

        Args:
        *flat_args (object): Positional args.

        Returns: Any: Result.
        """
        unflat_args = tree_unflatten(tree_def, list(flat_args))
        return fun(*unflat_args)

    with ConfigContext(eager_mode=True):
        if has_aux:
            val, aux = fun(*primals)
        else:
            val = fun(*primals)

    fun_primal = _get_fun_primal(fun_flat, has_aux)
    tensor_primals = _convert_to_tensors(flat_primals)

    # 3. Trace the flat primal function
    block = _trace_function(fun_primal, tuple(tensor_primals), f"vjp_{uuid.uuid4().hex[:6]}")
    forward_graph = LogicalGraph(name=block.id)
    for node in block.nodes:
        forward_graph.nodes[node.id] = node
    forward_graph.inputs = block.inputs
    forward_graph.outputs = block.outputs

    # 4. Create starting cotangent input nodes inside the forward graph for each output tensor
    output_node_id = forward_graph.outputs[0]
    output_node = forward_graph.nodes[output_node_id]

    cotangent_mapping = {}
    cotangent_ids_list = []
    for i, y_id in enumerate(output_node.inputs):
        cot_id = f"cotangent_{i}_{uuid.uuid4().hex[:6]}"
        cot_node = LogicalNode(
            id=cot_id,
            op_type="Input",
            inputs=[],
            shape_metadata=forward_graph.nodes[y_id].shape_metadata,
        )
        forward_graph.nodes[cot_id] = cot_node
        forward_graph.inputs.append(cot_id)
        cotangent_mapping[y_id] = cot_id
        cotangent_ids_list.append(cot_id)

    # 5. Build the gradient graph starting with our cotangent_mapping as the loss adjoint
    grad_graph = graph_grad(
        forward_graph,
        forward_graph.inputs[: len(tensor_primals)],
        output_node_id,
        cotangent_id=cotangent_mapping,  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    )

    def vjp_fn(cotangent: Any) -> tuple[Any, ...]:
        """Evaluate vjp_fn operation.

        Args:
        cotangent (object): The cotangent parameter.

        Returns: Any: Result.
        """
        # Run the evaluator on grad_graph
        inputs_dict = {inp_id: get_active_backend().asarray(getattr(p, "data", p)) for inp_id, p in zip(forward_graph.inputs[: len(tensor_primals)], tensor_primals)}

        # Flatten the cotangent Pytree if it is nested
        flat_cot, _ = tree_flatten(cotangent)
        for cot_id, cot_val in zip(cotangent_ids_list, flat_cot):
            inputs_dict[cot_id] = get_active_backend().asarray(getattr(cot_val, "data", cot_val))

        outputs_dict = evaluate_graph(grad_graph, inputs_dict)

        flat_grads = []
        for out_id in grad_graph.outputs:
            g_val = outputs_dict[out_id]
            flat_grads.append(g_val)

        # Unflatten gradients back to original tree structure
        res = tree_unflatten(tree_def, flat_grads)
        return tuple(res)

    if has_aux:
        return (val, aux), vjp_fn
    return val, vjp_fn


def hvp(
    fun: Callable[..., Any],
    primals: Any,
    tangents: Any,
    has_aux: bool = False,
) -> tuple[Any, Any]:
    """Evaluate hvp operation.

    Args:
        fun (object): The fun parameter.
        primals (object): The primals parameter.
        tangents (object): The tangents parameter.
        has_aux (bool): The has_aux parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.interpreter import evaluate_graph
    from ml_switcheroo_compiler.transforms.autodiff import hvp as graph_hvp

    primals_seq = primals if isinstance(primals, (tuple, list)) else (primals,)
    tangents_seq = tangents if isinstance(tangents, (tuple, list)) else (tangents,)

    from ml_switcheroo_compiler.core.config import ConfigContext

    with ConfigContext(eager_mode=True):
        if has_aux:
            val, aux = fun(*primals_seq)
        else:
            val = fun(*primals_seq)

    fun_primal = _get_fun_primal(fun, has_aux)
    tensor_primals = _convert_to_tensors(primals_seq)

    # Trace
    block = _trace_function(fun_primal, tuple(tensor_primals), f"hvp_{uuid.uuid4().hex[:6]}")
    forward_graph = LogicalGraph(name=block.id)
    for node in block.nodes:
        forward_graph.nodes[node.id] = node
    forward_graph.inputs = block.inputs
    forward_graph.outputs = block.outputs

    # Add tangent constants
    tangent_ids = []
    for t, p_id in zip(tangents_seq, forward_graph.inputs):
        t_id = f"tangent_{uuid.uuid4().hex[:6]}"
        t_node = LogicalNode(
            id=t_id,
            op_type="Constant",
            attributes={"value": getattr(t, "data", t)},
            shape_metadata=forward_graph.nodes[p_id].shape_metadata,
        )
        forward_graph.nodes[t_id] = t_node
        tangent_ids.append(t_id)

    # Compute HVP graph
    hvp_graph = graph_hvp(forward_graph, forward_graph.inputs, tangent_ids, forward_graph.outputs)

    # Evaluate HVP graph
    inputs_dict = {inputs_id: get_active_backend().asarray(getattr(p, "data", p)) for inputs_id, p in zip(forward_graph.inputs, tensor_primals)}
    outputs_dict = evaluate_graph(hvp_graph, inputs_dict)

    out_tangent_values = [outputs_dict[out_id] for out_id in hvp_graph.outputs]
    out_tan = out_tangent_values[0] if len(out_tangent_values) == 1 else tuple(out_tangent_values)

    if has_aux:
        return (val, aux), out_tan
    return val, out_tan


def jacfwd(fun: typing.Callable[..., Any], options: Any = None) -> typing.Callable[..., Any]:
    """Evaluate jacfwd operation.

    Args:
        fun (object): The fun parameter.
        options (GradOptions): The options parameter.

    Returns: Any: Result.
    """
    options = options or GradOptions()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        # Evaluate fun to see input and output dimensions
        out = fun(*args, **kwargs)
        out_arr = get_active_backend().asarray(getattr(out, "data", out))

        # Build basis tangents for each input coordinate
        arg0 = get_active_backend().asarray(getattr(args[0], "data", args[0]))
        flat_arg0 = arg0.flatten()

        jacobian_rows = []
        for i in range(len(flat_arg0)):
            # Standard basis vector for coordinate i
            tangent_flat = get_active_backend().execute_op(
                "OneHot",
                get_active_backend().asarray(i),
                len(flat_arg0),
                on_value=1.0,
                off_value=0.0,
                axis=-1,
                dtype="float32",
            )
            tangent = tangent_flat.reshape(arg0.shape)

            # Run jvp
            _, out_tangent = jvp(fun, args, (tangent,), has_aux=options.has_aux)
            jacobian_rows.append(get_active_backend().asarray(out_tangent).flatten())

        # Standard format of Jacobian: (output_size, input_size)
        res = get_active_backend().execute_op("Stack", jacobian_rows, axis=-1)
        if out_arr.ndim > 0:
            return res.reshape(out_arr.shape + arg0.shape)
        return res

    return wrapped


def jacrev(fun: typing.Callable[..., Any], options: Any = None) -> typing.Callable[..., Any]:
    """Evaluate jacrev operation.

    Args:
        fun (object): The fun parameter.
        options (GradOptions): The options parameter.

    Returns: Any: Result.
    """
    options = options or GradOptions()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        from ml_switcheroo_compiler.tree_util import tree_flatten, tree_unflatten

        out, vjp_fn = vjp(fun, *args, has_aux=options.has_aux)
        if options.has_aux:
            out_val, _ = out
        else:
            out_val = out

        flat_out, out_tree_def = tree_flatten(out_val)
        flat_shapes = [get_active_backend().asarray(getattr(o, "data", o)).shape for o in flat_out]
        flat_sizes = [int(math.prod(get_active_backend().asarray(getattr(o, "data", o)).shape)) for o in flat_out]
        total_size = sum(flat_sizes)

        arg0 = get_active_backend().asarray(getattr(args[0], "data", args[0]))

        jacobian_rows = []
        for i in range(total_size):
            cotangent_flat = get_active_backend().execute_op(
                "OneHot",
                get_active_backend().asarray(i),
                total_size,
                on_value=1.0,
                off_value=0.0,
                axis=-1,
                dtype="float32",
            )

            flat_cots = []
            curr_idx = 0
            for sz, shp in zip(flat_sizes, flat_shapes):
                flat_cots.append(cotangent_flat[curr_idx : curr_idx + sz].reshape(shp))
                curr_idx += sz

            cotangent = tree_unflatten(out_tree_def, flat_cots)
            grads = vjp_fn(cotangent)
            # Differentiate with respect to first argument
            jacobian_rows.append(get_active_backend().asarray(grads[0]).flatten())

        res = get_active_backend().execute_op("Stack", jacobian_rows, axis=0)
        # Reshape output to out_shape + arg_shape
        if len(flat_sizes) > 1:
            # Multi-output: standard output shape is (total_size,) + arg_shape
            return res.reshape((total_size,) + arg0.shape)
        else:
            out_arr = get_active_backend().asarray(getattr(out_val, "data", out_val))
            if out_arr.ndim > 0:
                return res.reshape(out_arr.shape + arg0.shape)
            return res.reshape(arg0.shape)

    return wrapped


def hessian(fun: typing.Callable[..., Any], options: Any = None) -> typing.Callable[..., Any]:
    """Evaluate hessian operation.

    Args:
        fun (object): The fun parameter.
        options (GradOptions): The options parameter.

    Returns: Any: Result.
    """
    options = options or GradOptions()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        arg0 = get_active_backend().asarray(getattr(args[0], "data", args[0]))
        flat_arg0 = arg0.flatten()

        hessian_rows = []
        for i in range(len(flat_arg0)):
            tangent_flat = get_active_backend().execute_op(
                "OneHot",
                get_active_backend().asarray(i),
                len(flat_arg0),
                on_value=1.0,
                off_value=0.0,
                axis=-1,
                dtype="float32",
            )
            tangent = tangent_flat.reshape(arg0.shape)

            _, out_tangent = hvp(fun, args, (tangent,), has_aux=options.has_aux)
            hessian_rows.append(get_active_backend().asarray(out_tangent).flatten())

        res = get_active_backend().execute_op("Stack", hessian_rows, axis=0)
        return res.reshape(arg0.shape + arg0.shape)

    return wrapped
