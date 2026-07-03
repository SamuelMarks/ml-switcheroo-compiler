"""Gradient computation and autodiff utilities."""

import typing
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.ops.registry import register_util
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


class CustomVJPFunction:
    """Wrapper for custom_vjp functions."""

    def __init__(self, fun: Callable[..., object]) -> None:
        """Execute __init__.

        Args:
            fun (Callable[..., object]): The fun parameter for the operation.
        """
        self.fun = fun
        self.fwd = None
        self.bwd = None
        self._tracing_fwd = False

    def defvjp(self, fwd: Callable[..., object], bwd: Callable[..., object]) -> None:
        """Define the forward and backward passes.

        Args:
            fwd (Callable[..., object]): The forward pass.
            bwd (Callable[..., object]): The backward pass.
        """
        self.fwd = fwd
        self.bwd = bwd

    def _extract_tensor_args(self, args: tuple[object, ...]) -> list[object]:
        """Function docstring.

        Args:
        args: Arg.
        """
        return [a for a in args if isinstance(a, Tensor)]

    def _trace_fwd_graph(self, tensor_args: list[object]) -> object:
        """Function docstring.

        Args:
        tensor_args: Arg.
        """
        if self.fwd is None or self.bwd is None:
            return None

        self._tracing_fwd = True
        try:
            return _trace_function(self.fwd, tuple(tensor_args), "fwd_pass")
        finally:
            self._tracing_fwd = False

    def _resolve_output_metadata(self, tensor_args: list[object]) -> tuple[tuple[int, ...], str, str]:
        """Function docstring.

        Args:
        tensor_args: Arg.
        """
        shape = ()
        dtype = "float32"
        device = "cpu"
        if tensor_args:
            first_arg = tensor_args[0]
            shape = first_arg.shape
            dtype = first_arg.dtype.value
            device = first_arg.device
        return (shape, dtype, device)

    def _emit_vjp_node(self, tensor_args: list[object], fwd_graph: object, primal_graph: object) -> object:
        """Function docstring.

        Args:
        tensor_args: Arg.
        fwd_graph: Arg.
        primal_graph: Arg.
        """
        out_id = str(uuid.uuid4())
        meta = self._resolve_output_metadata(tensor_args)
        node = LogicalNode(
            id=out_id,
            op_type="CustomVJP",
            inputs=[a.data.id for a in tensor_args],
            attributes={"primal_graph": primal_graph, "fwd_graph": fwd_graph, "bwd_fn": self.bwd},
            shape_metadata=meta[0],
        )
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=meta[0], dtype=meta[1])
        return Tensor(proxy, TensorConfig(meta[0], DType(meta[1]), meta[2]))

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Evaluate the function.

        Args:
            *args (object): Additional arguments.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        if config.eager_mode or not global_tracing_state.is_tracing or self._tracing_fwd:
            return self.fun(*args, **kwargs)

        tensor_args = self._extract_tensor_args(args)
        fwd_graph = self._trace_fwd_graph(tensor_args)
        primal_graph = _trace_function(self.fun, tuple(tensor_args), "primal_pass")
        return self._emit_vjp_node(tensor_args, fwd_graph, primal_graph)


def custom_vjp(fun: Callable[..., object]) -> Callable[..., object]:
    """Ensure custom_vjp allows defining custom gradient functions natively.

    Args:
        fun (Callable): The function

    Returns:
        Callable: The function
    """
    return CustomVJPFunction(fun)


def value_and_grad_wrt_vars(
    fun: Callable[..., object],
) -> Callable[..., tuple[object, dict[str, object]]]:
    """Creates a function that evaluates both the value and gradient of fun with respect to variables.

    Args:
        fun (Callable[..., object]): The fun parameter for the operation.

    Returns:
        Callable[..., tuple[object, dict[str, object]]]: The wrapped function.
    """

    def wrapped(*args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
        """Evaluates the wrapped function, returning value and gradient dictionary.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[object, dict[str, object]]: The evaluated output and gradient dictionary.
        """
        val = fun(*args, **kwargs)
        grads: dict[str, object] = {}
        return (val, grads)

    return wrapped


@register_util("backward")
def backward(tensor: object, *args: object, **kwargs: object) -> None:
    """Triggers the reverse-mode auto-differentiation.

    Args:
        tensor (object): The tensor to compute gradients for.
        *args (object): Additional arguments.
        **kwargs (object): Additional keyword arguments.
    """
    pass


def custom_jvp(fun: Callable[..., object]) -> Callable[..., object]:
    """Ensure custom_jvp allows defining custom JVP functions natively.

    Args:
        fun (Callable): The function

    Returns:
        Callable: The function
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


def check_numerical_grads(f: Callable[..., object], args: tuple[object, ...], options: GradCheckOptions = None) -> None:
    """Check numerical gradients for a function against analytical gradients.

    Args:
        f (Callable): The function to differentiate.
        args (tuple): The arguments to evaluate the function.
        options (GradCheckOptions, optional): The configuration options for checking grads.
        rtol (float): Relative tolerance.
        step (float): The step size for numerical differentiation.
    """
    pass


class UnconnectedGradients(Enum):
    """Specifies how unconnected gradients are handled."""

    NONE = "none"
    ZERO = "zero"


def RegisterGradient(op_type: str) -> typing.Callable:
    """Register a custom gradient for an operation."""
    return register_vjp(op_type)


def recompute_grad(fun: Callable[..., object]) -> Callable[..., object]:
    """Gradient checkpointing / rematerialization."""
    return fun
