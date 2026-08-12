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

from .jvp_vjp import vjp
from .options import DEFAULT_GRAD_EPSILON, GradCheckOptions


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
