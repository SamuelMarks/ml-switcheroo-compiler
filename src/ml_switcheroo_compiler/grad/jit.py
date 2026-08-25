# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Gradient computation and autodiff utilities."""

import contextlib
import math
import typing
import uuid
from collections.abc import Callable, Sequence
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


@dataclass
class JitOptions:
    """Options for JIT compilation."""

    static_argnums: object = None
    static_argnames: object = None
    donate_argnums: object = None
    donate_argnames: object = None
    keep_unused: bool = False
    device: object = None
    backend: object = None
    inline: bool = False
    abstracted_axes: object = None


def jit(fun: Callable[..., object], options: object = None) -> Callable[..., object]:
    """Return a JIT wrapper.

    Args:
        fun (Callable[..., object]): The function to jit compile.
        options (JitOptions): Configuration options.

    Returns:
        Callable[..., object]: The JIT wrapped function.
    """
    options: object = options or JitOptions()

    def wrapped(*args: object, **kwargs: object) -> object:
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


def eval_shape(fun: Callable[..., object], *args: object, **kwargs: object) -> object:
    """Evaluate eval_shape operation.

    Args:
        fun (object): The fun parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return fun(*args, **kwargs)
