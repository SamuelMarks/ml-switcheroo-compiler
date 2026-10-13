# ruff: noqa: E402, E501, F403
"""Gradient computation and autodiff utilities."""

import contextlib
import math  # noqa: F401
import typing
import uuid  # noqa: F401
from collections.abc import Callable, Sequence  # noqa: F401
from dataclasses import dataclass, field  # noqa: F401

from ml_switcheroo_ir import LogicalGraph, LogicalNode  # noqa: F401

from ml_switcheroo_compiler.backends.registry import get_active_backend  # noqa: F401
from ml_switcheroo_compiler.core.config import config  # noqa: F401
from ml_switcheroo_compiler.core.dtype import DType  # noqa: F401
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # noqa: F401
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function  # noqa: F401
from ml_switcheroo_compiler.ops.registry import register_util  # noqa: F401
from ml_switcheroo_compiler.tracing.state import global_tracing_state  # noqa: F401
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor  # noqa: F401
from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients  # noqa: F401
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp  # noqa: F401


@dataclass
class JitOptions:
    """Options for JIT compilation."""

    static_argnums = None
    static_argnames = None
    donate_argnums = None
    donate_argnames = None
    keep_unused: bool = False
    device = None
    backend = None
    inline: bool = False
    abstracted_axes = None


def jit(fun, options=None):
    """Return a JIT wrapper.

    Args:
        fun (Callable[..., object]): The function to jit compile.
        options (JitOptions): Configuration options.

    Returns:
        Callable[..., object]: The JIT wrapped function.
    """
    options = options or JitOptions()

    def wrapped(*args, **kwargs):
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
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


def eval_shape(fun, *args, **kwargs):
    """Evaluate eval_shape operation.

    Args:
        fun (object): The fun parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return fun(*args, **kwargs)
