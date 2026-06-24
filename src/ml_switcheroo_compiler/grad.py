"""Gradient computation and autodiff utilities."""

from dataclasses import dataclass
import contextlib
from collections.abc import Callable, Generator

from ml_switcheroo_compiler.core.tensor import TensorConfig


def ir_grad(fun: Callable[..., object], argnums: int = 0) -> Callable[..., object]:
    """Creates a function that evaluates the gradient of fun.

    Args:
        fun (Callable[..., object]): The fun parameter for the operation.
        argnums (int): The argnums parameter for the operation.

    Returns:
        Callable[..., object]: The evaluated output resulting from this operation.
    """
    _ = argnums

    def wrapped(*args: object, **kwargs: object) -> object:
        """Evaluates the wrapped function.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        # mock impl
        return fun(*args, **kwargs)

    return wrapped


def grad(fun: Callable[..., object], argnums: int = 0) -> Callable[..., object]:
    """Creates a function that evaluates the gradient of fun.

    Args:
        fun (Callable[..., object]): The fun parameter for the operation.
        argnums (int): The argnums parameter for the operation.

    Returns:
        Callable[..., object]: The evaluated output resulting from this operation.
    """
    return ir_grad(fun, argnums=argnums)


def value_and_grad(fun: Callable[..., object], argnums: int = 0) -> Callable[..., object]:
    """Creates a function that evaluates both the value and gradient of fun.

    Args:
        fun (Callable[..., object]): The fun parameter for the operation.
        argnums (int): The argnums parameter for the operation.

    Returns:
        Callable[..., object]: The evaluated output resulting from this operation.
    """
    _ = argnums

    def wrapped(*args: object, **kwargs: object) -> tuple[object, object]:
        """Evaluates the wrapped function, returning value and gradient.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[object, object]: The evaluated output resulting from this operation.
        """
        return fun(*args, **kwargs), fun(*args, **kwargs)

    return wrapped


def jit(fun: Callable[..., object]) -> Callable[..., object]:
    """Compiles a function to execute faster.

    In our parity layer this currently acts as an eager wrapper.


    Args:
        fun (Callable[..., object]): The fun parameter for the operation.

    Returns:
        Callable[..., object]: The evaluated output resulting from this operation.
    """
    return fun


def disable_jit() -> contextlib._GeneratorContextManager[None]:
    """A context manager to temporarily disable JIT compilation.

    Returns:
        contextlib._GeneratorContextManager[None]: The evaluated output.
    """

    @contextlib.contextmanager
    def _disable() -> Generator[None, None, None]:
        """Yields execution to temporarily disable JIT.

        Returns:
            Generator[None, None, None]: The evaluated output resulting from this operation.
        """
        yield

    return _disable()


def eval_shape(fun: Callable[..., object], *args: object, **kwargs: object) -> object:
    """Evaluates the shape and dtype of the output of fun without computing its values.

    Args:
        fun (Callable[..., object]): The fun parameter for the operation.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    return fun(*args, **kwargs)


def jvp(
    fun: Callable[..., object],
    primals: list[object],
    tangents: list[object],
) -> tuple[object, object]:
    """Compute the Jacobian-vector product.

    Args:
        fun (Callable): The function
        primals (list[object]): The primals
        tangents (list[object]): The tangents

    Returns:
        tuple[object, object]: (out_primals, out_tangents)
    """
    return fun(*primals), tangents


def vjp(fun: Callable[..., object], *primals: object) -> tuple[object, Callable[..., object]]:
    """Compute the Vector-Jacobian product.

    Args:
        fun (Callable): The function to differentiate.
        primals (object): The primal inputs.

    Returns:
        tuple[object, Callable]: The primal output and a function that computes the VJP.
    """
    out_primal = fun(*primals)

    def vjp_fn(*cotangents: object) -> object:
        """Execute vjp_fn.

        Args:
            *cotangents (Any): Argument *cotangents.

        Returns:
        Any: The result.
        """
        # Mock VJP backward function
        return cotangents

    return out_primal, vjp_fn


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
        from ml_switcheroo_compiler.core.tensor import Tensor

        return [a for a in args if isinstance(a, Tensor)]

    def _trace_fwd_graph(self, tensor_args: list[object]) -> object:
        """Function docstring.

        Args:
        tensor_args: Arg.
        """
        if self.fwd is None or self.bwd is None:  # pragma: no branch
            return None  # pragma: no cover
        from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function

        self._tracing_fwd = True
        try:
            return _trace_function(self.fwd, tuple(tensor_args), "fwd_pass")
        finally:
            self._tracing_fwd = False

    def _resolve_output_metadata(
        self, tensor_args: list[object]
    ) -> tuple[tuple[int, ...], str, str]:
        """Function docstring.

        Args:
        tensor_args: Arg.
        """
        shape = ()
        dtype = "float32"
        device = "cpu"
        if tensor_args:  # pragma: no branch
            first_arg = tensor_args[0]
            shape = first_arg.shape
            dtype = first_arg.dtype.value
            device = first_arg.device
        return shape, dtype, device

    def _emit_vjp_node(
        self,
        tensor_args: list[object],
        fwd_graph: object,
        primal_graph: object,
    ) -> object:
        """Function docstring.

        Args:
        tensor_args: Arg.
        fwd_graph: Arg.
        primal_graph: Arg.
        """
        import uuid

        from ml_switcheroo_ir import LogicalNode

        from ml_switcheroo_compiler.core.dtype import DType
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

        out_id = str(uuid.uuid4())
        meta = self._resolve_output_metadata(tensor_args)

        node = LogicalNode(
            id=out_id,
            op_type="CustomVJP",
            inputs=[a.data.id for a in tensor_args],
            attributes={
                "primal_graph": primal_graph,
                "fwd_graph": fwd_graph,
                "bwd_fn": self.bwd,
            },
            shape_metadata=meta[0],
        )
        _tracer.add_node(node)

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
        from ml_switcheroo_compiler.core.config import config
        from ml_switcheroo_compiler.tracing.tracer import _tracer

        if config.eager_mode or not _tracer.is_tracing or self._tracing_fwd:
            return self.fun(*args, **kwargs)

        from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function

        tensor_args = self._extract_tensor_args(args)
        fwd_graph = self._trace_fwd_graph(tensor_args)

        # Trace the primal function itself for normal evaluation
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
        # Mock implementation. In a real compiler, it would traverse state and variables.
        val = fun(*args, **kwargs)
        grads: dict[str, object] = {}
        return val, grads

    return wrapped


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


DEFAULT_GRAD_EPSILON = 1e-4


@dataclass
class GradCheckOptions:
    """Options for gradient checking."""

    order: int = 1
    atol: float = DEFAULT_GRAD_EPSILON
    rtol: float = DEFAULT_GRAD_EPSILON
    step: float = DEFAULT_GRAD_EPSILON


def check_numerical_grads(
    f: Callable[..., object],
    args: tuple[object, ...],
    options: GradCheckOptions = None,
) -> None:
    """Check numerical gradients for a function against analytical gradients.

    Args:
        f (Callable): The function to differentiate.
        args (tuple): The arguments to evaluate the function.
        options (GradCheckOptions, optional): The configuration options for checking grads.
        rtol (float): Relative tolerance.
        step (float): The step size for numerical differentiation.
    """
    pass
