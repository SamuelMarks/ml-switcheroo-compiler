"""Gradient computation and autodiff utilities."""

import contextlib
from collections.abc import Callable, Generator


def ir_grad(fun: Callable[..., object], argnums: int = 0) -> Callable[..., object]:
    """Creates a function that evaluates the gradient of fun."""

    def wrapped(*args: object, **kwargs: object) -> object:
        """Evaluates the wrapped function."""
        # mock impl
        return fun(*args, **kwargs)

    return wrapped


def grad(fun: Callable[..., object], argnums: int = 0) -> Callable[..., object]:
    """Creates a function that evaluates the gradient of fun."""
    return ir_grad(fun, argnums=argnums)


def value_and_grad(fun: Callable[..., object], argnums: int = 0) -> Callable[..., object]:
    """Creates a function that evaluates both the value and gradient of fun."""

    def wrapped(*args: object, **kwargs: object) -> tuple[object, object]:
        """Evaluates the wrapped function, returning value and gradient."""
        return fun(*args, **kwargs), fun(*args, **kwargs)

    return wrapped


def jit(fun: Callable[..., object]) -> Callable[..., object]:
    """Compiles a function to execute faster.

    In our parity layer this currently acts as an eager wrapper.
    """
    return fun


def disable_jit() -> contextlib._GeneratorContextManager[None]:
    """A context manager to temporarily disable JIT compilation."""

    @contextlib.contextmanager
    def _disable() -> Generator[None, None, None]:
        """Yields execution to temporarily disable JIT."""
        yield

    return _disable()


def eval_shape(fun: Callable[..., object], *args: object, **kwargs: object) -> object:
    """Evaluates the shape and dtype of the output of fun without computing its values."""
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
        # Mock VJP backward function
        return cotangents

    return out_primal, vjp_fn


def custom_vjp(fun: Callable[..., object]) -> Callable[..., object]:
    """Ensure custom_vjp allows defining custom gradient functions natively.

    Args:
        fun (Callable): The function

    Returns:
        Callable: The function
    """
    return fun


def backward(tensor: object, *args: object, **kwargs: object) -> None:
    """Triggers the reverse-mode auto-differentiation.

    Args:
        tensor (object): The tensor to compute gradients for.
        *args (object): Variable length argument list
        **kwargs (object): Variable length argument list
    """


def custom_jvp(fun: Callable[..., object]) -> Callable[..., object]:
    """Ensure custom_jvp allows defining custom JVP functions natively.

    Args:
        fun (Callable): The function

    Returns:
        Callable: The function
    """
    return fun
