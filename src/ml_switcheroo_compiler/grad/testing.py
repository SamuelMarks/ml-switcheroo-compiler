"""Numerical gradient checking and JVP/VJP adjoint duality verification."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Sequence
from types import ModuleType

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import OpValue, is_variadic_callable
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from .jvp_vjp import _unpack_pair, jvp, vjp
from .options import DEFAULT_GRAD_EPSILON, GradCheckOptions
from .utils import DifferentiableFn, PrimalTree, _extract_shape, _unwrap_data, as_primal_tree


@typing.runtime_checkable
class _FlatBufferProtocol(typing.Protocol):
    """Protocol for 1-D raveled buffer views supporting element indexing and mutation."""

    @property
    def size(self) -> int:
        """Return total element count.

        Returns:
            int: Element count.
        """
        ...

    def __getitem__(self, idx: int) -> float:
        """Read scalar element at ``idx``.

        Args:
            idx (int): Flat element index.

        Returns:
            float: Element value.
        """
        ...

    def __setitem__(self, idx: int, val: float) -> None:
        """Write scalar element ``val`` at ``idx``.

        Args:
            idx (int): Flat element index.
            val (float): Replacement value.
        """
        ...


@typing.runtime_checkable
class _MutableRavelArrayProtocol(typing.Protocol):
    """Protocol for mutable numerical arrays used in finite-difference gradient checking."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return array shape tuple.

        Returns:
            tuple[int, ...]: Array dimensions.
        """
        ...

    def ravel(self) -> _FlatBufferProtocol:
        """Return contiguous 1-D view of array.

        Returns:
            _FlatBufferProtocol: Flat mutable buffer view.
        """
        ...

    def reshape(self, shape: tuple[int, ...]) -> _MutableRavelArrayProtocol:
        """Return array reshaped to ``shape``.

        Args:
            shape (tuple[int, ...]): Target shape tuple.

        Returns:
            _MutableRavelArrayProtocol: Reshaped array.
        """
        ...

    def copy(self) -> OpValue:
        """Return an independent copy of the array.

        Returns:
            OpValue: Copied array buffer.
        """
        ...


def _as_mutable_ravel_array(val: OpValue | _MutableRavelArrayProtocol) -> _MutableRavelArrayProtocol:
    """Verify and narrow ``val`` to ``_MutableRavelArrayProtocol``.

    Args:
        val (OpValue | _MutableRavelArrayProtocol): Candidate array buffer.

    Returns:
        _MutableRavelArrayProtocol: Verified mutable array.
    """
    assert isinstance(val, _MutableRavelArrayProtocol) and not isinstance(val, ModuleType)
    return val


__all__ = [
    "DEFAULT_GRAD_EPSILON",
    "GradCheckOptions",
    "check_jvp_vjp_duality",
    "check_numerical_grads",
]


def _as_supports_float(val: OpValue | PrimalTree | typing.SupportsFloat) -> typing.SupportsFloat:
    """Verify and narrow ``val`` to ``typing.SupportsFloat``.

    Args:
        val (OpValue | PrimalTree | typing.SupportsFloat): Candidate scalar value.

    Returns:
        typing.SupportsFloat: Verified float-convertible value.
    """
    assert isinstance(val, typing.SupportsFloat)
    return val


def _to_grad_float(val: OpValue | PrimalTree) -> float:
    """Coerce a scalar tensor or numeric value to float.

    Args:
        val (OpValue | PrimalTree): Scalar value.

    Returns:
        float: Coerced float value.
    """
    return float(_as_supports_float(val))


def check_numerical_grads(
    f: DifferentiableFn,
    args: PrimalTree | Sequence[PrimalTree],
    options: GradCheckOptions | None = None,
) -> None:
    """Check finite-difference numerical gradients of `f` against analytical VJP gradients.

    Args:
        f (DifferentiableFn): Differentiable function under test.
        args (PrimalTree | Sequence[PrimalTree]): Input tensor argument(s) at which to evaluate `f`.
        options (GradCheckOptions | None): Optional `GradCheckOptions` specifying step size and tolerances.

    Raises:
        SwitcherooError: If analytical and numerical gradients diverge beyond tolerance.
    """
    opts = options if isinstance(options, GradCheckOptions) else GradCheckOptions()
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.errors import SwitcherooError

    call_f = f
    args_seq: list[PrimalTree] = (
        [as_primal_tree(x) for x in args]
        if isinstance(args, Iterable) and not isinstance(args, Tensor)
        else [as_primal_tree(args)]
    )

    with ConfigContext(eager_mode=True):
        vjp_res = vjp(f, *args_seq)
        out, vjp_fn_cand = _unpack_pair(vjp_res)
        out_arr = get_active_backend().asarray(_unwrap_data(out))
        cotangent = get_active_backend().execute_op("Ones_like", out_arr)
        assert is_variadic_callable(vjp_fn_cand)
        assert is_variadic_callable(call_f)
        analytical_grads_raw = vjp_fn_cand(cotangent)
        analytical_grads: list[OpValue] = (
            list(analytical_grads_raw) if isinstance(analytical_grads_raw, (list, tuple)) else [analytical_grads_raw]
        )

        for arg_idx, arg in enumerate(args_seq):
            arg_arr = _as_mutable_ravel_array(get_active_backend().array(_unwrap_data(arg), dtype="float64"))
            raw_num_grad = get_active_backend().execute_op("Zeros_like", arg_arr)
            numerical_grad = _as_mutable_ravel_array(raw_num_grad)

            flat_arg = arg_arr.ravel()
            flat_num_grad = numerical_grad.ravel()

            from ml_switcheroo_compiler.core.device import Device

            for i in range(flat_arg.size):
                orig_val = flat_arg[i]

                flat_arg[i] = orig_val + opts.step
                args_pos: list[OpValue] = list(args_seq)
                args_pos[arg_idx] = (
                    Tensor(
                        arg_arr.reshape(arg_arr.shape).copy(),
                        TensorConfig(arg_arr.shape, DType.Float32, Device("cpu")),
                    )
                    if isinstance(arg, Tensor)
                    else arg_arr.reshape(arg_arr.shape).copy()
                )
                out_pos_arr = get_active_backend().asarray(_unwrap_data(call_f(*args_pos)))

                flat_arg[i] = orig_val - opts.step
                args_neg: list[OpValue] = list(args_seq)
                args_neg[arg_idx] = (
                    Tensor(
                        arg_arr.reshape(arg_arr.shape).copy(),
                        TensorConfig(arg_arr.shape, DType.Float32, Device("cpu")),
                    )
                    if isinstance(arg, Tensor)
                    else arg_arr.reshape(arg_arr.shape).copy()
                )
                out_neg_arr = get_active_backend().asarray(_unwrap_data(call_f(*args_neg)))

                flat_arg[i] = orig_val

                diff = get_active_backend().execute_op(
                    "Divide", get_active_backend().execute_op("Subtract", out_pos_arr, out_neg_arr), 2.0 * opts.step
                )
                flat_num_grad[i] = _to_grad_float(get_active_backend().execute_op("Sum", diff))

            anal_item = analytical_grads[arg_idx]
            anal_grad = get_active_backend().asarray(_unwrap_data(anal_item))

            if not get_active_backend().execute_op("Allclose", anal_grad, raw_num_grad, atol=opts.atol, rtol=opts.rtol):
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
    f: DifferentiableFn,
    primals: Sequence[PrimalTree] | PrimalTree,
    tangents: Sequence[PrimalTree] | PrimalTree | None = None,
    cotangents: PrimalTree = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Validate the adjoint duality identity `<J(x) · v, w> == <v, J*(x) · w>` between JVP and VJP.

    Args:
        f (DifferentiableFn): Differentiable mathematical function.
        primals (Sequence[PrimalTree] | PrimalTree): Primal input point(s).
        tangents (Sequence[PrimalTree] | PrimalTree | None): Tangent vector(s) `v` matching `primals`.
        cotangents (PrimalTree): Cotangent vector `w` matching the output of `f`.
        atol (float): Absolute numerical tolerance.
        rtol (float): Relative numerical tolerance.

    Raises:
        SwitcherooError: If the inner-product duality identity fails within tolerance.
    """
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.errors import SwitcherooError

    with ConfigContext(eager_mode=True):
        backend = get_active_backend()
        primals_seq: list[PrimalTree] = list(primals) if isinstance(primals, (list, tuple)) else [primals]
        _callable_fn_4: OpValue = f
        assert is_variadic_callable(_callable_fn_4)
        out = _callable_fn_4(*primals_seq)
        out_arr = backend.asarray(_unwrap_data(out))

        tangents_seq: list[PrimalTree] = []
        if tangents is None:
            for p in primals_seq:
                p_arr = backend.asarray(_unwrap_data(p))
                tan_arr = backend.execute_op("Ones_like", p_arr)
                if isinstance(p, Tensor):
                    tangents_seq.append(Tensor(tan_arr, TensorConfig(_extract_shape(p_arr), p.dtype, p.device)))
                else:
                    tangents_seq.append(as_primal_tree(tan_arr))
        else:
            tangents_seq = list(tangents) if isinstance(tangents, (list, tuple)) else [tangents]

        cotangents_val: PrimalTree
        if cotangents is None:
            cotangents_arr = backend.execute_op("Ones_like", out_arr)
            if isinstance(out, Tensor):
                cotangents_val = Tensor(
                    cotangents_arr,
                    TensorConfig(_extract_shape(out_arr), out.dtype, out.device),
                )
            else:
                cotangents_val = as_primal_tree(cotangents_arr)
        else:
            cotangents_val = cotangents

        jvp_pair = jvp(f, tuple(primals_seq), tuple(tangents_seq))
        _, jvp_out = _unpack_pair(jvp_pair)
        jvp_arr = backend.asarray(_unwrap_data(jvp_out))
        cot_arr = backend.asarray(_unwrap_data(cotangents_val))

        lhs = _to_grad_float(backend.execute_op("Sum", backend.execute_op("Multiply", jvp_arr, cot_arr)))

        vjp_pair = vjp(f, *primals_seq)
        _, vjp_fn_cand = _unpack_pair(vjp_pair)
        _callable_fn_5: OpValue = vjp_fn_cand
        assert is_variadic_callable(_callable_fn_5)
        vjp_out = _callable_fn_5(cotangents_val)
        vjp_list = list(vjp_out) if isinstance(vjp_out, (list, tuple)) else [vjp_out]

        rhs = 0.0
        for t, v_adj in zip(tangents_seq, vjp_list):
            t_arr = backend.asarray(_unwrap_data(t))
            v_arr = backend.asarray(_unwrap_data(v_adj))
            rhs += _to_grad_float(backend.execute_op("Sum", backend.execute_op("Multiply", t_arr, v_arr)))

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
