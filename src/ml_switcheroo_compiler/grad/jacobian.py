"""Higher-order Jacobian (`jacfwd`, `jacrev`), Hessian (`hessian`), and HVP (`hvp`) transforms."""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.shape import OpValue, TypedVariadicCallable, is_variadic_callable
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.interpreter.environment import InterpreterValue, as_eval_value

from .custom_jvp_ops import _to_constant_attr
from .options import GradOptions
from .utils import (
    DifferentiableFn,
    PrimalAuxValue,
    PrimalTree,
    _convert_to_tensors,
    _ensure_grad_tensor,
    _get_fun_primal,
    _unwrap_data,
    as_primal_tree,
)


def hvp(
    fun: DifferentiableFn,
    primals: Sequence[PrimalTree] | PrimalTree,
    tangents: Sequence[PrimalTree] | PrimalTree,
    has_aux: bool = False,
    projected_tangents: Tensor | Sequence[Tensor] | None = None,
    mode: str = "forward-over-reverse",
) -> tuple[PrimalTree | tuple[PrimalTree, PrimalAuxValue], PrimalTree]:
    """Compute the Hessian-vector product of `fun` at `primals` along `tangents`.

    Args:
        fun (DifferentiableFn): Scalar or projected function to differentiate twice.
        primals (Sequence[PrimalTree] | PrimalTree): Primal point(s).
        tangents (Sequence[PrimalTree] | PrimalTree): Tangent vector(s) matching `primals`.
        has_aux (bool): Whether `fun` returns auxiliary data `(val, aux)`.
        projected_tangents (Tensor | Sequence[Tensor] | None): Optional cotangent projection vectors.
        mode (str): HVP mode (`'forward-over-reverse'` or `'reverse-over-forward'`).

    Returns:
        tuple[PrimalTree | tuple[PrimalTree, PrimalAuxValue], PrimalTree]: Tuple of `(primal_out, hvp_out)` or
            `((primal_out, aux), hvp_out)`.
    """
    from ml_switcheroo_compiler.interpreter import evaluate_graph
    from ml_switcheroo_compiler.transforms.autodiff import hvp as graph_hvp

    from .jvp_vjp import _build_traced_logical_graph, _to_list_or_self, _unpack_pair

    fn_callable = fun
    primals_seq = primals if isinstance(primals, (tuple, list)) else (primals,)
    tangents_seq = tangents if isinstance(tangents, (tuple, list)) else (tangents,)

    aux: PrimalAuxValue = None
    with ConfigContext(eager_mode=True):
        assert is_variadic_callable(fn_callable)
        if has_aux:
            raw_val, raw_aux = _unpack_pair(fn_callable(*primals_seq))
            val = as_primal_tree(raw_val)
            aux = raw_aux if isinstance(raw_aux, str) else as_primal_tree(raw_aux)
        else:
            val = as_primal_tree(fn_callable(*primals_seq))

    fun_primal = _get_fun_primal(fn_callable, has_aux)
    tensor_primals = _convert_to_tensors(primals_seq)
    forward_graph = _build_traced_logical_graph(fun_primal, tensor_primals, "hvp")

    tangent_ids: list[str] = []
    for t, p_id in zip(tangents_seq, forward_graph.inputs):
        t_id = f"tangent_{uuid.uuid4().hex[:6]}"
        raw_t = t.data if isinstance(t, Tensor) else t
        t_val = _to_list_or_self(raw_t)
        t_node = LogicalNode(
            id=t_id,
            op_type="Constant",
            attributes={"value": _to_constant_attr(t_val)},
            shape_metadata=forward_graph.nodes[p_id].shape_metadata,
        )
        forward_graph.nodes[t_id] = t_node
        tangent_ids.append(t_id)

    proj_tangent_ids: list[str] | None = None
    if projected_tangents is not None:
        proj_seq = projected_tangents if isinstance(projected_tangents, (tuple, list)) else (projected_tangents,)
        proj_tangent_ids = []
        for pt, o_id in zip(proj_seq, forward_graph.outputs):
            pt_id = f"proj_tangent_{uuid.uuid4().hex[:6]}"
            raw_pt = pt.data if isinstance(pt, Tensor) else pt
            pt_val = _to_list_or_self(raw_pt)
            pt_node = LogicalNode(
                id=pt_id,
                op_type="Constant",
                attributes={"value": _to_constant_attr(pt_val)},
                shape_metadata=forward_graph.nodes[o_id].shape_metadata,
            )
            forward_graph.nodes[pt_id] = pt_node
            proj_tangent_ids.append(pt_id)

    hvp_graph = graph_hvp(
        forward_graph,
        forward_graph.inputs,
        tangent_ids,
        forward_graph.outputs,
        mode=mode,
        projected_tangents=proj_tangent_ids,
    )

    inputs_dict: dict[str, InterpreterValue] = {
        inputs_id: as_eval_value(get_active_backend().asarray(_unwrap_data(p)))
        for inputs_id, p in zip(forward_graph.inputs, tensor_primals)
    }
    outputs_dict = evaluate_graph(hvp_graph, inputs_dict)

    out_tangent_values = [as_primal_tree(outputs_dict[out_id]) for out_id in hvp_graph.outputs]
    out_tan: PrimalTree = out_tangent_values[0] if len(out_tangent_values) == 1 else tuple(out_tangent_values)

    if has_aux:
        return ((val, aux), out_tan)
    return (val, out_tan)


def jacfwd(fun: DifferentiableFn, options: GradOptions | None = None) -> TypedVariadicCallable[PrimalTree]:
    """Create a function computing the full Jacobian of `fun` using forward-mode JVP pushforwards.

    Args:
        fun (DifferentiableFn): Function whose Jacobian is to be computed.
        options (GradOptions | None): Optional `GradOptions` configuration.

    Returns:
        TypedVariadicCallable[PrimalTree]: Callable returning the forward-mode Jacobian tensor.
    """
    from . import jvp_vjp as _jv_mod

    opts = options if isinstance(options, GradOptions) else GradOptions()
    fn_callable = fun

    def wrapped(*args: OpValue, **kwargs: OpValue) -> PrimalTree:
        """Evaluate the forward-mode Jacobian of `fun` at `*args`.

        Args:
            *args (OpValue): Primal input arguments.
            **kwargs (OpValue): Keyword arguments passed to `fun`.

        Returns:
            PrimalTree: Evaluated Jacobian tensor.
        """
        with ConfigContext(eager_mode=True):
            _callable_fn_3: OpValue = fn_callable
            assert is_variadic_callable(_callable_fn_3)
            out = _callable_fn_3(*args, **kwargs)
            out_val = _jv_mod._unpack_pair(out)[0] if opts.has_aux else out
        out_arr = _jv_mod._ensure_grad_tensor(_jv_mod.get_active_backend().asarray(_unwrap_data(out_val)))

        arg0 = _jv_mod._ensure_grad_tensor(_jv_mod.get_active_backend().asarray(_unwrap_data(args[0])))
        flat_arg0 = arg0.flatten()

        primal_args = tuple(as_primal_tree(a) for a in args)
        jacobian_rows = []
        for i in range(len(flat_arg0)):
            tangent_flat = _jv_mod._ensure_grad_tensor(
                _jv_mod.get_active_backend().execute_op(
                    "OneHot",
                    _jv_mod.get_active_backend().asarray(i),
                    len(flat_arg0),
                    on_value=1.0,
                    off_value=0.0,
                    axis=-1,
                    dtype="float32",
                )
            )
            tangent = as_primal_tree(tangent_flat.reshape(arg0.shape))

            _, out_tangent = _jv_mod._unpack_pair(
                _jv_mod.jvp(fn_callable, primal_args, (tangent,), has_aux=opts.has_aux)
            )
            jacobian_rows.append(
                _jv_mod._ensure_grad_tensor(_jv_mod.get_active_backend().asarray(_unwrap_data(out_tangent))).flatten()
            )

        res = _jv_mod.get_active_backend().execute_op("Stack", jacobian_rows, axis=-1)
        if out_arr.ndim > 0:
            return as_primal_tree(
                _jv_mod.get_active_backend().execute_op(
                    "Reshape",
                    res,
                    tuple(out_arr.shape) + tuple(arg0.shape),
                )
            )
        return as_primal_tree(res)

    return wrapped


def jacrev(fun: DifferentiableFn, options: GradOptions | None = None) -> TypedVariadicCallable[PrimalTree]:
    """Create a function computing the full Jacobian of `fun` using reverse-mode VJP pullbacks.

    Args:
        fun (DifferentiableFn): Function whose Jacobian is to be computed.
        options (GradOptions | None): Optional `GradOptions` configuration.

    Returns:
        TypedVariadicCallable[PrimalTree]: Callable returning the reverse-mode Jacobian tensor.
    """
    from . import jvp_vjp as _jv_mod

    opts = options if isinstance(options, GradOptions) else GradOptions()

    def wrapped(*args: OpValue, **kwargs: OpValue) -> PrimalTree:
        """Evaluate the reverse-mode Jacobian of `fun` at `*args`.

        Args:
            *args (OpValue): Primal input arguments.
            **kwargs (OpValue): Unused keyword arguments.

        Returns:
            PrimalTree: Evaluated Jacobian tensor.
        """
        _ = kwargs
        from ml_switcheroo_compiler.tree_util import tree_flatten, tree_unflatten

        primal_args = tuple(as_primal_tree(a) for a in args)
        out, vjp_fn_raw = _jv_mod._unpack_pair(_jv_mod.vjp(fun, *primal_args, has_aux=opts.has_aux))
        vjp_fn = vjp_fn_raw
        out_val = _jv_mod._unpack_pair(out)[0] if opts.has_aux else out

        flat_out, out_tree_def = tree_flatten(out_val)
        flat_shapes = [
            _ensure_grad_tensor(_jv_mod.get_active_backend().asarray(_unwrap_data(o))).shape for o in flat_out
        ]
        flat_sizes = [int(math.prod(shp)) for shp in flat_shapes]
        total_size = sum(flat_sizes)

        arg0 = _ensure_grad_tensor(_jv_mod.get_active_backend().asarray(_unwrap_data(args[0])))
        jacobian_rows = []

        for i in range(total_size):
            cotangent_flat = _ensure_grad_tensor(
                _jv_mod.get_active_backend().execute_op(
                    "OneHot",
                    _jv_mod.get_active_backend().asarray(i),
                    total_size,
                    on_value=1.0,
                    off_value=0.0,
                    axis=-1,
                    dtype="float32",
                )
            )

            flat_cots = []
            curr_idx = 0
            for sz, shp in zip(flat_sizes, flat_shapes):
                flat_cots.append(cotangent_flat[curr_idx : curr_idx + sz].reshape(shp))
                curr_idx += sz

            cotangent = tree_unflatten(out_tree_def, flat_cots)
            _callable_fn_4: OpValue = vjp_fn
            assert is_variadic_callable(_callable_fn_4)
            grads = _callable_fn_4(cotangent)
            first_grad = grads[0] if isinstance(grads, (list, tuple)) else grads
            jacobian_rows.append(
                _ensure_grad_tensor(_jv_mod.get_active_backend().asarray(_unwrap_data(first_grad))).flatten()
            )

        res = _jv_mod.get_active_backend().execute_op("Stack", jacobian_rows, axis=0)
        if len(flat_sizes) > 1:
            return as_primal_tree(
                _jv_mod.get_active_backend().execute_op("Reshape", res, (total_size,) + tuple(arg0.shape))
            )
        out_arr = _ensure_grad_tensor(_jv_mod.get_active_backend().asarray(_unwrap_data(out_val)))
        if out_arr.ndim > 0:
            return as_primal_tree(
                _jv_mod.get_active_backend().execute_op(
                    "Reshape",
                    res,
                    tuple(out_arr.shape) + tuple(arg0.shape),
                )
            )
        return as_primal_tree(_jv_mod.get_active_backend().execute_op("Reshape", res, arg0.shape))

    return wrapped


def hessian(fun: DifferentiableFn, options: GradOptions | None = None) -> TypedVariadicCallable[PrimalTree]:
    """Create a function computing the full Hessian matrix of scalar function `fun`.

    Args:
        fun (DifferentiableFn): Scalar-valued function to differentiate twice.
        options (GradOptions | None): Optional `GradOptions` configuration.

    Returns:
        TypedVariadicCallable[PrimalTree]: Callable returning the Hessian tensor of shape `arg.shape + arg.shape`.
    """
    from . import jvp_vjp as _jv_mod

    opts = options if isinstance(options, GradOptions) else GradOptions()

    def wrapped(*args: OpValue, **kwargs: OpValue) -> PrimalTree:
        """Evaluate the Hessian matrix of `fun` at `*args`.

        Args:
            *args (OpValue): Primal input arguments.
            **kwargs (OpValue): Unused keyword arguments.

        Returns:
            PrimalTree: Evaluated Hessian matrix tensor.
        """
        _ = kwargs
        arg0 = _ensure_grad_tensor(_jv_mod.get_active_backend().asarray(_unwrap_data(args[0])))
        flat_arg0 = arg0.flatten()

        primal_args = tuple(as_primal_tree(a) for a in args)
        hessian_rows = []
        for i in range(len(flat_arg0)):
            tangent_flat = _ensure_grad_tensor(
                _jv_mod.get_active_backend().execute_op(
                    "OneHot",
                    _jv_mod.get_active_backend().asarray(i),
                    len(flat_arg0),
                    on_value=1.0,
                    off_value=0.0,
                    axis=-1,
                    dtype="float32",
                )
            )
            tangent = as_primal_tree(tangent_flat.reshape(arg0.shape))

            _, out_tangent = _jv_mod._unpack_pair(_jv_mod.hvp(fun, primal_args, (tangent,), has_aux=opts.has_aux))
            hessian_rows.append(_ensure_grad_tensor(_jv_mod.get_active_backend().asarray(out_tangent)).flatten())

        res = _jv_mod.get_active_backend().execute_op("Stack", hessian_rows, axis=0)
        return as_primal_tree(
            _jv_mod.get_active_backend().execute_op(
                "Reshape",
                res,
                tuple(arg0.shape) + tuple(arg0.shape),
            )
        )

    return wrapped
