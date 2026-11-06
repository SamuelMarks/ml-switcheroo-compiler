"""Numerical gradient checking and JVP/VJP adjoint duality verification."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Sequence

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar, VariadicCallableProtocol, invoke_callable
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from .jvp_vjp import _unpack_pair, jvp, vjp
from .options import DEFAULT_GRAD_EPSILON, GradCheckOptions
from .utils import GradValue, _ensure_grad_tensor, _extract_shape, _unwrap_data

__all__ = [
    "DEFAULT_GRAD_EPSILON",
    "GradCheckOptions",
    "check_jvp_vjp_duality",
    "check_numerical_grads",
]


def _is_vjp_pullback(val: GradValue) -> typing.TypeGuard[VariadicCallableProtocol]:
    """Type guard narrowing `val` to `VariadicCallableProtocol`.

    Args:
        val (GradValue): Candidate VJP pullback function.

    Returns:
        bool: True if callable.
    """
    return callable(val)


def _to_grad_float(val: TensorOrScalar | GradValue) -> float:
    """Coerce a scalar tensor or numeric value to float.

    Args:
        val (TensorOrScalar | GradValue): Scalar value.

    Returns:
        float: Coerced float value.
    """
    if isinstance(val, (int, float)):
        return float(val)
    raw = _ensure_grad_tensor(val)
    item_val = raw.item()
    return float(item_val) if isinstance(item_val, (int, float, str, bytes)) else 0.0


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

    call_f = f
    args_seq: list[GradValue] = list(args) if isinstance(args, Iterable) else [args]

    with ConfigContext(eager_mode=True):
        vjp_res = vjp(f, *args_seq)
        out, vjp_fn_cand = _unpack_pair(vjp_res)
        assert _is_vjp_pullback(vjp_fn_cand)
        vjp_fn: VariadicCallableProtocol = vjp_fn_cand
        out_arr = _ensure_grad_tensor(get_active_backend().asarray(_unwrap_data(out)))
        cotangent = get_active_backend().execute_op("Ones_like", out_arr)
        analytical_grads_raw = vjp_fn(cotangent)
        analytical_grads: list[GradValue] = list(analytical_grads_raw) if isinstance(analytical_grads_raw, (list, tuple)) else [analytical_grads_raw]

        step = opts.step
        atol = opts.atol
        rtol = opts.rtol

        for arg_idx, arg in enumerate(args_seq):
            arg_arr = _ensure_grad_tensor(get_active_backend().array(_unwrap_data(arg), dtype="float64"))
            numerical_grad = _ensure_grad_tensor(get_active_backend().execute_op("Zeros_like", arg_arr))

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
                out_pos = invoke_callable(call_f, *args_pos)
                out_pos_arr = _ensure_grad_tensor(get_active_backend().asarray(_unwrap_data(out_pos)))

                flat_arg[i] = orig_val - step
                args_neg = list(args_seq)
                if isinstance(arg, Tensor):
                    args_neg[arg_idx] = Tensor(
                        arg_arr.reshape(arg_arr.shape).copy(),
                        TensorConfig(arg_arr.shape, DType.Float32, Device("cpu")),
                    )
                else:
                    args_neg[arg_idx] = arg_arr.reshape(arg_arr.shape).copy()
                out_neg = invoke_callable(call_f, *args_neg)
                out_neg_arr = _ensure_grad_tensor(get_active_backend().asarray(_unwrap_data(out_neg)))

                flat_arg[i] = orig_val

                diff = (out_pos_arr - out_neg_arr) / (2.0 * step)
                sum_diff = get_active_backend().execute_op("Sum", diff)
                flat_num_grad[i] = _to_grad_float(sum_diff)

            anal_item = analytical_grads[arg_idx]
            anal_grad = get_active_backend().asarray(_unwrap_data(anal_item))

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
        del args_seq


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

    with ConfigContext(eager_mode=True):
        backend = get_active_backend()
        primals_seq: list[GradValue] = list(primals) if isinstance(primals, (list, tuple)) else [primals]

        out = invoke_callable(f, *primals_seq)
        out_arr = backend.asarray(_unwrap_data(out))

        tangents_seq: list[GradValue] = []
        if tangents is None:
            for p in primals_seq:
                p_arr = backend.asarray(_unwrap_data(p))
                tan_arr = backend.execute_op("Ones_like", p_arr)
                if isinstance(p, Tensor):
                    tangents_seq.append(Tensor(tan_arr, TensorConfig(_extract_shape(p_arr), p.dtype, p.device)))
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
                    TensorConfig(_extract_shape(out_arr), out.dtype, out.device),
                )
            else:
                cotangents_val = cotangents_arr
        else:
            cotangents_val = cotangents

        jvp_pair = jvp(f, tuple(primals_seq), tuple(tangents_seq))
        _, jvp_out = _unpack_pair(jvp_pair)
        jvp_arr = _ensure_grad_tensor(backend.asarray(_unwrap_data(jvp_out)))
        cot_arr = _ensure_grad_tensor(backend.asarray(_unwrap_data(cotangents_val)))

        lhs_sum = _ensure_grad_tensor(backend.execute_op("Sum", jvp_arr * cot_arr))
        lhs = float(lhs_sum)

        vjp_pair = vjp(f, *primals_seq)
        _, vjp_fn_cand = _unpack_pair(vjp_pair)
        assert _is_vjp_pullback(vjp_fn_cand)
        vjp_fn: VariadicCallableProtocol = vjp_fn_cand
        vjp_out = vjp_fn(cotangents_val)
        vjp_list = list(vjp_out) if isinstance(vjp_out, (list, tuple)) else [vjp_out]

        rhs = 0.0
        for t, v_adj in zip(tangents_seq, vjp_list):
            t_arr = backend.asarray(_unwrap_data(t))
            v_adj_val = _unwrap_data(v_adj)
            v_arr = _ensure_grad_tensor(backend.asarray(v_adj_val))
            rhs_sum = _ensure_grad_tensor(backend.execute_op("Sum", t_arr * v_arr))
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
