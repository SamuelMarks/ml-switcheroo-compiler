"""Numerical gradient checking and JVP/VJP adjoint duality verification."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Sequence

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from .jvp_vjp import jvp, vjp
from .options import DEFAULT_GRAD_EPSILON, GradCheckOptions
from .utils import GradValue

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.shape import TypedVariadicCallable, VariadicCallableProtocol

__all__ = [
    "DEFAULT_GRAD_EPSILON",
    "GradCheckOptions",
    "check_jvp_vjp_duality",
    "check_numerical_grads",
]


def check_numerical_grads(
    f: GradValue,
    args: GradValue,
    options: GradCheckOptions | GradValue = None,
) -> None:
    """Check finite-difference numerical gradients of `f` against analytical VJP gradients.

    Args:
        f (GradValue): Differentiable function under test.
        args (GradValue): Sequence of input tensor arguments at which to evaluate `f`.
        options (GradCheckOptions | GradValue): Optional `GradCheckOptions` specifying step size and tolerances.

    Raises:
        SwitcherooError: If analytical and numerical gradients diverge beyond tolerance.
    """
    opts = options if isinstance(options, GradCheckOptions) else GradCheckOptions()
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.errors import SwitcherooError

    call_f = typing.cast("VariadicCallableProtocol", f)
    args_seq: list[GradValue] = list(args) if isinstance(args, Iterable) else [args]

    with ConfigContext(eager_mode=True):
        vjp_res = vjp(typing.cast("TypedVariadicCallable[GradValue]", f), *args_seq)
        out: GradValue = typing.cast("tuple[GradValue, VariadicCallableProtocol]", vjp_res)[0]
        vjp_fn: VariadicCallableProtocol = typing.cast("tuple[GradValue, VariadicCallableProtocol]", vjp_res)[1]
        out_arr = typing.cast(EagerTensorProtocol, get_active_backend().asarray(getattr(out, "data", out)))
        cotangent = get_active_backend().execute_op("Ones_like", out_arr)
        analytical_grads_raw = vjp_fn(cotangent)
        analytical_grads: list[GradValue] = list(analytical_grads_raw) if isinstance(analytical_grads_raw, (list, tuple)) else [analytical_grads_raw]

        step = opts.step
        atol = opts.atol
        rtol = opts.rtol

        for arg_idx, arg in enumerate(args_seq):
            arg_arr = typing.cast(EagerTensorProtocol, get_active_backend().array(getattr(arg, "data", arg), dtype="float64"))
            numerical_grad = typing.cast(EagerTensorProtocol, get_active_backend().execute_op("Zeros_like", arg_arr))

            flat_arg = arg_arr.ravel()
            flat_num_grad = numerical_grad.ravel()

            for i in range(flat_arg.size):
                orig_val = flat_arg[i]

                flat_arg[i] = orig_val + step
                args_pos = list(args_seq)
                from ml_switcheroo_compiler.core.device import Device

                if isinstance(arg, Tensor):
                    args_pos[arg_idx] = Tensor(
                        arg_arr.reshape(arg_arr.shape).copy(),
                        TensorConfig(arg_arr.shape, DType.Float32, Device("cpu")),
                    )
                else:
                    args_pos[arg_idx] = arg_arr.reshape(arg_arr.shape).copy()
                out_pos = call_f(*args_pos)
                out_pos_arr: EagerTensorProtocol = get_active_backend().asarray(getattr(out_pos, "data", out_pos))  # type: ignore[assignment]

                flat_arg[i] = orig_val - step
                args_neg = list(args_seq)
                if isinstance(arg, Tensor):
                    args_neg[arg_idx] = Tensor(
                        arg_arr.reshape(arg_arr.shape).copy(),
                        TensorConfig(arg_arr.shape, DType.Float32, Device("cpu")),
                    )
                else:
                    args_neg[arg_idx] = arg_arr.reshape(arg_arr.shape).copy()
                out_neg = call_f(*args_neg)
                out_neg_arr: EagerTensorProtocol = get_active_backend().asarray(getattr(out_neg, "data", out_neg))  # type: ignore[assignment]

                flat_arg[i] = orig_val

                diff = (out_pos_arr - out_neg_arr) / (2.0 * step)
                sum_diff: EagerTensorProtocol = get_active_backend().execute_op("Sum", diff)  # type: ignore[assignment]
                flat_num_grad[i] = float(sum_diff)

            anal_item = analytical_grads[arg_idx]
            anal_grad = get_active_backend().asarray(getattr(anal_item, "data", anal_item))

            if not get_active_backend().execute_op("Allclose", anal_grad, numerical_grad, atol=atol, rtol=rtol):
                msg = chr(10).join(
                    [
                        f"Gradient check failed for argument {arg_idx}.",
                        "Analytical gradient:",
                        str(anal_grad),
                        "Numerical gradient:",
                        str(numerical_grad),
                    ]
                )
                raise SwitcherooError(msg)


def check_jvp_vjp_duality(
    f: GradValue,
    primals: Sequence[GradValue] | GradValue,
    tangents: Sequence[GradValue] | GradValue | None = None,
    cotangents: GradValue | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Validate the adjoint duality identity `<J(x) · v, w> == <v, J*(x) · w>` between JVP and VJP.

    Args:
        f (GradValue): Differentiable mathematical function.
        primals (Sequence[GradValue] | GradValue): Primal input point(s).
        tangents (Sequence[GradValue] | GradValue | None): Tangent vector(s) `v` matching `primals`.
        cotangents (GradValue | None): Cotangent vector `w` matching the output of `f`.
        atol (float): Absolute numerical tolerance.
        rtol (float): Relative numerical tolerance.

    Raises:
        SwitcherooError: If the inner-product duality identity fails within tolerance.
    """
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.errors import SwitcherooError

    call_f: VariadicCallableProtocol = f  # type: ignore[assignment]
    with ConfigContext(eager_mode=True):
        backend = get_active_backend()
        primals_seq: list[GradValue] = list(primals) if isinstance(primals, (list, tuple)) else [primals]

        out = call_f(*primals_seq)
        out_arr: EagerTensorProtocol = backend.asarray(getattr(out, "data", out))  # type: ignore[assignment]

        tangents_seq: list[GradValue] = []
        if tangents is None:
            for p in primals_seq:
                p_arr: EagerTensorProtocol = backend.asarray(getattr(p, "data", p))  # type: ignore[assignment]
                tan_arr = backend.execute_op("Ones_like", p_arr)
                if isinstance(p, Tensor):
                    tangents_seq.append(Tensor(tan_arr, TensorConfig(p_arr.shape, p.dtype, p.device)))
                else:
                    tangents_seq.append(tan_arr)
        else:
            tangents_seq = list(tangents) if isinstance(tangents, (list, tuple)) else [tangents]

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

        jvp_pair = jvp(typing.cast("TypedVariadicCallable[GradValue]", f), tuple(primals_seq), tuple(tangents_seq))
        jvp_out: GradValue = typing.cast(tuple[GradValue, GradValue], jvp_pair)[1]
        jvp_arr = typing.cast(EagerTensorProtocol, backend.asarray(getattr(jvp_out, "data", jvp_out)))
        cot_arr = typing.cast(EagerTensorProtocol, backend.asarray(getattr(cotangents_val, "data", cotangents_val)))

        lhs_sum = typing.cast(EagerTensorProtocol, backend.execute_op("Sum", jvp_arr * cot_arr))
        lhs = float(lhs_sum)

        vjp_pair = vjp(typing.cast("TypedVariadicCallable[GradValue]", f), *primals_seq)
        vjp_fn: VariadicCallableProtocol = typing.cast("tuple[GradValue, VariadicCallableProtocol]", vjp_pair)[1]
        vjp_out = vjp_fn(cotangents_val)
        vjp_list = list(vjp_out) if isinstance(vjp_out, (list, tuple)) else [vjp_out]

        rhs = 0.0
        for t, v_adj in zip(tangents_seq, vjp_list):
            t_arr = typing.cast(EagerTensorProtocol, backend.asarray(getattr(t, "data", t)))
            v_adj_val = getattr(v_adj, "data", v_adj)
            v_arr = typing.cast(EagerTensorProtocol, backend.asarray(v_adj_val))
            rhs_sum = typing.cast(EagerTensorProtocol, backend.execute_op("Sum", t_arr * v_arr))
            rhs += float(rhs_sum)

        diff = abs(lhs - rhs)
        tol = atol + rtol * max(abs(lhs), abs(rhs))
        if diff > tol:
            msg = chr(10).join(
                [
                    "JVP/VJP duality check failed:",
                    f"⟨J(x) · v, w⟩ = {lhs}",
                    f"⟨v, J*(x) · w⟩ = {rhs}",
                    f"Absolute difference: {diff} (tolerance: {tol})",
                ]
            )
            raise SwitcherooError(msg)
