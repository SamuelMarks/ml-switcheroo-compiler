"""Gradient computation and autodiff utilities."""

from __future__ import annotations

import contextlib  # noqa: F401
import math  # noqa: F401
import uuid  # noqa: F401
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field  # noqa: F401
from typing import cast

from ml_switcheroo_ir import LogicalGraph, LogicalNode  # noqa: F401

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config  # noqa: F401
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function  # noqa: F401
from ml_switcheroo_compiler.ops.registry import register_util  # noqa: F401
from ml_switcheroo_compiler.tracing.state import global_tracing_state  # noqa: F401
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor  # noqa: F401
from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients  # noqa: F401
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp  # noqa: F401

from .jvp_vjp import jvp, vjp
from .options import DEFAULT_GRAD_EPSILON, GradCheckOptions  # noqa: F401
from .utils import GradValue


def check_numerical_grads(f: Callable[..., EagerValue], args: EagerValue, options: EagerValue = None) -> None:
    """Check numerical gradients for a function against analytical gradients.

    Args:
        f (Callable): The function to differentiate.
        args (tuple): The arguments to evaluate the function.
        options (GradCheckOptions): The configuration options for checking grads.

    Raises:
        SwitcherooError: If analytical and numerical gradients do not match.
    """
    options = cast(EagerValue, options or GradCheckOptions())
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.errors import SwitcherooError

    with ConfigContext(eager_mode=True):
        # Compute analytical gradients using VJP
        out, vjp_fn = getattr(vjp(f, *getattr(args, "__iter__")()), "__iter__")()
        out_arr = cast(EagerTensorProtocol, get_active_backend().asarray(getattr(out, "data", out)))
        cotangent = get_active_backend().execute_op("Ones_like", out_arr)
        analytical_grads = vjp_fn(cotangent)

        step = getattr(options, "step")
        atol = getattr(options, "atol")
        rtol = getattr(options, "rtol")

        for arg_idx, arg in enumerate(getattr(args, "__iter__")()):
            arg_arr = cast(EagerTensorProtocol, get_active_backend().array(getattr(arg, "data", arg), dtype="float64"))
            numerical_grad = cast(EagerTensorProtocol, get_active_backend().execute_op("Zeros_like", arg_arr))

            flat_arg = arg_arr.ravel()
            flat_num_grad = numerical_grad.ravel()

            for i in range(flat_arg.size):
                orig_val = flat_arg[i]

                # Perturb positive
                flat_arg[i] = orig_val + step
                args_pos = list(getattr(args, "__iter__")())
                from ml_switcheroo_compiler.core.device import Device

                if isinstance(arg, Tensor):
                    args_pos[arg_idx] = Tensor(
                        arg_arr.reshape(arg_arr.shape).copy(),
                        TensorConfig(arg_arr.shape, DType.Float32, Device("cpu")),
                    )
                else:
                    args_pos[arg_idx] = arg_arr.reshape(arg_arr.shape).copy()
                out_pos = f(*args_pos)
                out_pos_arr = cast(EagerTensorProtocol, get_active_backend().asarray(getattr(out_pos, "data", out_pos)))

                # Perturb negative
                flat_arg[i] = orig_val - step
                args_neg = list(getattr(args, "__iter__")())
                if isinstance(arg, Tensor):
                    args_neg[arg_idx] = Tensor(
                        arg_arr.reshape(arg_arr.shape).copy(),
                        TensorConfig(arg_arr.shape, DType.Float32, Device("cpu")),
                    )
                else:
                    args_neg[arg_idx] = arg_arr.reshape(arg_arr.shape).copy()
                out_neg = f(*args_neg)
                out_neg_arr = cast(EagerTensorProtocol, get_active_backend().asarray(getattr(out_neg, "data", out_neg)))

                flat_arg[i] = orig_val

                diff = (out_pos_arr - out_neg_arr) / (2.0 * step)
                flat_num_grad[i] = float(cast(EagerTensorProtocol, get_active_backend().execute_op("Sum", diff)))

            anal_grad = get_active_backend().asarray(getattr(getattr(analytical_grads, "__getitem__")(arg_idx), "data", getattr(analytical_grads, "__getitem__")(arg_idx)))

            if not get_active_backend().execute_op("Allclose", anal_grad, numerical_grad, atol=atol, rtol=rtol):
                msg = f"Gradient check failed for argument {arg_idx}.\nAnalytical gradient:\n{anal_grad}\nNumerical gradient:\n{numerical_grad}"
                raise SwitcherooError(msg)


def check_jvp_vjp_duality(
    f: Callable[..., GradValue],
    primals: Sequence[GradValue] | GradValue,
    tangents: Sequence[GradValue] | GradValue | None = None,
    cotangents: GradValue | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Validate the adjoint duality relationship between JVP and VJP.

    Asserts that:
        ⟨J(x) · v, w⟩ == ⟨v, J*(x) · w⟩
    where:
        J(x) · v is the forward-mode directional derivative (JVP),
        J*(x) · w is the reverse-mode adjoint vector pullback (VJP),
        v is the input perturbation vector (tangents),
        w is the output adjoint vector (cotangents).

    Args:
        f (Callable[..., Tensor]): Mathematical function to differentiate.
        primals (Union[Sequence[Tensor], Tensor]): Primal point(s) at which to evaluate derivatives.
        tangents (Optional[Union[Sequence[Tensor], Tensor]]): Tangent vector(s) v matching primals structure.
        cotangents (Optional[Tensor]): Cotangent vector(s) w matching the output structure of f.
        atol (float): Absolute numerical tolerance.
        rtol (float): Relative numerical tolerance.

    Raises:
        SwitcherooError: If duality identity fails within tolerance.
    """
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.errors import SwitcherooError

    with ConfigContext(eager_mode=True):
        backend = get_active_backend()
        primals_seq = list(primals) if isinstance(primals, (list, tuple)) else [primals]

        # Evaluate primal forward
        out = f(*primals_seq)
        out_arr = cast(EagerTensorProtocol, backend.asarray(getattr(out, "data", out)))

        # Establish tangents v
        tangents_seq: list[GradValue] = []
        if tangents is None:
            for p in primals_seq:
                p_arr = cast(EagerTensorProtocol, backend.asarray(getattr(p, "data", p)))
                tan_arr = backend.execute_op("Ones_like", p_arr)
                if isinstance(p, Tensor):
                    tangents_seq.append(Tensor(tan_arr, TensorConfig(p_arr.shape, p.dtype, p.device)))
                else:
                    tangents_seq.append(tan_arr)
        else:
            tangents_seq = list(cast(Sequence[GradValue], tangents)) if isinstance(tangents, (list, tuple)) else [cast(GradValue, tangents)]

        # Establish cotangents w
        cotangents_val: GradValue
        if cotangents is None:
            cotangents_arr = backend.execute_op("Ones_like", out_arr)
            if isinstance(out, Tensor):
                cotangents_val = Tensor(
                    cotangents_arr,
                    TensorConfig(out_arr.shape, out.dtype, out.device),
                )
            else:
                cotangents_val = cotangents_arr
        else:
            cotangents_val = cotangents

        # 1. Compute JVP: J(x) · v
        _, jvp_out = getattr(jvp(f, tuple(primals_seq), tuple(tangents_seq)), "__iter__")()
        jvp_arr = cast(EagerTensorProtocol, backend.asarray(getattr(jvp_out, "data", jvp_out)))
        cot_arr = cast(EagerTensorProtocol, backend.asarray(getattr(cotangents_val, "data", cotangents_val)))

        # LHS = ⟨J(x) · v, w⟩
        lhs = float(cast(EagerTensorProtocol, backend.execute_op("Sum", jvp_arr * cot_arr)))

        # 2. Compute VJP: J*(x) · w
        _, vjp_fn = getattr(vjp(f, *getattr(primals_seq, "__iter__")()), "__iter__")()
        vjp_out = vjp_fn(cotangents_val)
        vjp_list = list(vjp_out) if isinstance(vjp_out, (list, tuple)) else [vjp_out]

        # RHS = ⟨v, J*(x) · w⟩
        rhs = 0.0
        for t, v_adj in zip(tangents_seq, vjp_list):
            t_arr = cast(EagerTensorProtocol, backend.asarray(getattr(t, "data", t)))
            v_adj_val = getattr(v_adj, "data", v_adj)
            v_arr = cast(EagerTensorProtocol, backend.asarray(v_adj_val))
            rhs += float(cast(EagerTensorProtocol, backend.execute_op("Sum", t_arr * v_arr)))

        diff = abs(lhs - rhs)
        tol = atol + rtol * max(abs(lhs), abs(rhs))
        if diff > tol:
            msg = "JVP/VJP duality check failed:\n" + f"⟨J(x) · v, w⟩ = {lhs}\n" + f"⟨v, J*(x) · w⟩ = {rhs}\n" + f"Absolute difference: {diff} (tolerance: {tol})"
            raise SwitcherooError(msg)
