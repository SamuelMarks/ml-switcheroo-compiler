"""Forward-mode (JVP) and reverse-mode (VJP) autodiff transformations."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend as get_active_backend
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import OpValue, TypedVariadicCallable, is_variadic_callable
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.interpreter.environment import InterpreterValue, as_eval_value
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from ._jvp_vjp_helpers import (
    _build_traced_logical_graph as _build_traced_logical_graph,
)
from ._jvp_vjp_helpers import (
    _CompileGraphBackendProtocol as _CompileGraphBackendProtocol,
)
from ._jvp_vjp_helpers import (
    _execute_autodiff_graph as _execute_autodiff_graph,
)
from ._jvp_vjp_helpers import (
    _ExecuteGraphBackendProtocol as _ExecuteGraphBackendProtocol,
)
from ._jvp_vjp_helpers import (
    _extract_node_id as _extract_node_id,
)
from ._jvp_vjp_helpers import (
    _has_id_block_attr as _has_id_block_attr,
)
from ._jvp_vjp_helpers import (
    _has_tolist_method as _has_tolist_method,
)
from ._jvp_vjp_helpers import (
    _IdBlockProtocol as _IdBlockProtocol,
)
from ._jvp_vjp_helpers import (
    _is_compile_graph_backend as _is_compile_graph_backend,
)
from ._jvp_vjp_helpers import (
    _is_execute_graph_backend as _is_execute_graph_backend,
)
from ._jvp_vjp_helpers import (
    _to_list_or_self as _to_list_or_self,
)
from ._jvp_vjp_helpers import (
    _ToListProtocol as _ToListProtocol,
)
from ._jvp_vjp_helpers import (
    _unpack_pair as _unpack_pair,
)
from .custom_jvp_ops import (
    CustomJVPFunction,
    _run_custom_jvp_rule,
    _to_constant_attr,
    custom_jvp,
)
from .jacobian import hessian, hvp, jacfwd, jacrev
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


def jvp(
    fun: DifferentiableFn,
    primals: Sequence[PrimalTree] | PrimalTree,
    tangents: Sequence[PrimalTree] | PrimalTree,
    has_aux: bool = False,
) -> tuple[PrimalTree | tuple[PrimalTree, PrimalAuxValue], PrimalTree]:
    """Compute the Jacobian-vector product of `fun` at `primals` along `tangents`.

    Args:
        fun (DifferentiableFn): Function to differentiate.
        primals (Sequence[PrimalTree] | PrimalTree): Primal input tensor or sequence of tensors.
        tangents (Sequence[PrimalTree] | PrimalTree): Tangent vector or sequence of tangent vectors.
        has_aux (bool): Whether `fun` returns `(output, aux)`.

    Returns:
        tuple[PrimalTree | tuple[PrimalTree, PrimalAuxValue], PrimalTree]: `(primal_out, tangent_out)` or
            `((primal_out, aux), tangent_out)`.
    """
    from ml_switcheroo_compiler.transforms.autodiff import jvp as graph_jvp

    fn_callable = fun
    primals_seq = primals if isinstance(primals, (tuple, list)) else (primals,)
    tangents_seq = tangents if isinstance(tangents, (tuple, list)) else (tangents,)

    if isinstance(fun, CustomJVPFunction) and fun.jvp_rule is not None:
        return _run_custom_jvp_rule(fun, primals_seq, tangents_seq)

    with ConfigContext(eager_mode=True):
        _callable_fn_1: OpValue = fn_callable
        assert is_variadic_callable(_callable_fn_1)
        out_eager = _callable_fn_1(*primals_seq)
        raw_val, raw_aux = _unpack_pair(out_eager) if has_aux else (out_eager, None)
        val = as_primal_tree(raw_val)
        aux: PrimalAuxValue = raw_aux if isinstance(raw_aux, str) else as_primal_tree(raw_aux)

    fun_primal = _get_fun_primal(fn_callable, has_aux)
    tensor_primals = _convert_to_tensors(primals_seq)
    forward_graph = _build_traced_logical_graph(fun_primal, tensor_primals, "jvp")

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

    jvp_graph = graph_jvp(forward_graph, forward_graph.inputs, tangent_ids, forward_graph.outputs)

    backend = get_active_backend()
    inputs_dict: dict[str, InterpreterValue] = {
        inp_id: as_eval_value(backend.asarray(_unwrap_data(p)))
        for inp_id, p in zip(forward_graph.inputs, tensor_primals)
    }
    outputs_dict = _execute_autodiff_graph(jvp_graph, inputs_dict)

    out_tangent_values = [as_primal_tree(outputs_dict[out_id]) for out_id in jvp_graph.outputs]
    out_tan: PrimalTree = out_tangent_values[0] if len(out_tangent_values) == 1 else tuple(out_tangent_values)

    return ((val, aux), out_tan) if has_aux else (val, out_tan)


def _attach_cotangent_inputs(forward_graph: LogicalGraph) -> tuple[str, dict[str, str], list[str]]:
    """Create cotangent input nodes inside the forward graph for each output tensor.

    Args:
        forward_graph (LogicalGraph): The traced forward graph to augment.

    Returns:
        tuple[str, dict[str, str], list[str]]: Tuple of `(output_node_id, cotangent_mapping, cotangent_ids_list)`.
    """
    output_node_id = forward_graph.outputs[0]
    output_node = forward_graph.nodes[output_node_id]

    cotangent_mapping: dict[str, str] = {}
    cotangent_ids_list: list[str] = []
    for i, y_id in enumerate(output_node.inputs):
        cot_id = f"cotangent_{i}_{uuid.uuid4().hex[:6]}"
        cot_node = LogicalNode(
            id=cot_id,
            op_type="Input",
            inputs=[],
            shape_metadata=forward_graph.nodes[y_id].shape_metadata,
        )
        forward_graph.nodes[cot_id] = cot_node
        forward_graph.inputs.append(cot_id)
        cotangent_mapping[y_id] = cot_id
        cotangent_ids_list.append(cot_id)
    return output_node_id, cotangent_mapping, cotangent_ids_list


def _inline_vjp_into_active_trace(
    grad_graph: LogicalGraph,
    primal_input_ids: list[str],
    tensor_primals: list[Tensor],
    cotangent_ids_list: list[str],
    flat_cot: Sequence[OpValue],
) -> list[OpValue]:
    """Symbolically inline a gradient graph into the currently active tracing graph.

    Args:
        grad_graph (LogicalGraph): Constructed VJP gradient graph.
        primal_input_ids (list[str]): Primal input node IDs in the forward graph.
        tensor_primals (list[Tensor]): Tensor primal inputs.
        cotangent_ids_list (list[str]): Cotangent input node IDs.
        flat_cot (Sequence[OpValue]): Flattened cotangent values.

    Returns:
        list[OpValue]: List of proxy gradient `Tensor` objects corresponding to `grad_graph` outputs.
    """
    from ml_switcheroo_compiler.core.device import Device

    id_map: dict[str, str] = {}
    for inp_id, p in zip(primal_input_ids, tensor_primals):
        id_map[inp_id] = _extract_node_id(p)

    for cot_id, cot_val in zip(cotangent_ids_list, flat_cot):
        id_map[cot_id] = _extract_node_id(cot_val)

    for nid, n in grad_graph.nodes.items():
        if n.op_type != "Input":
            new_id = f"vjp_{uuid.uuid4().hex[:6]}_{nid}"
            id_map[nid] = new_id
            cloned = LogicalNode(
                id=new_id,
                op_type=n.op_type,
                inputs=[id_map.get(inp, inp) for inp in n.inputs],
                attributes=dict(n.attributes or {}),
                shape_metadata=n.shape_metadata,
            )
            global_tracing_state.add_node(cloned)

    flat_grads: list[OpValue] = []
    for out_id in grad_graph.outputs:
        mapped_id = id_map.get(out_id, out_id)
        out_node = grad_graph.nodes.get(out_id)
        out_shape: tuple[int, ...] = (
            tuple(int(d) for d in out_node.shape_metadata if isinstance(d, int))
            if out_node is not None and out_node.shape_metadata is not None
            else ()
        )
        proxy = ProxyTensor(id=mapped_id, shape=out_shape, dtype="float32")
        flat_grads.append(Tensor(proxy, TensorConfig(out_shape, DType.Float32, Device("cpu"))))
    return flat_grads


def vjp(
    fun: DifferentiableFn,
    *primals: PrimalTree | OpValue,
    has_aux: bool = False,
) -> tuple[PrimalTree | tuple[PrimalTree, PrimalAuxValue], TypedVariadicCallable[tuple[PrimalTree, ...]]]:
    """Compute the reverse-mode vector-Jacobian product (VJP) pullback of `fun` at `primals`.

    Args:
        fun (DifferentiableFn): Differentiable function.
        *primals (PrimalTree | OpValue): Primal input arguments.
        has_aux (bool): Whether `fun` returns `(output, aux)`.

    Returns:
        tuple[PrimalTree | tuple[PrimalTree, PrimalAuxValue], TypedVariadicCallable[tuple[PrimalTree, ...]]]:
            `(primal_out, vjp_fn)` or `((primal_out, aux), vjp_fn)`.
    """
    from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad
    from ml_switcheroo_compiler.tree_util import tree_flatten, tree_unflatten

    fn_callable = fun
    flat_primals, tree_def = tree_flatten(primals)

    def fun_flat(*flat_args: OpValue, **kwargs: OpValue) -> PrimalTree:
        """Unflatten `flat_args` and evaluate `fun`.

        Args:
            *flat_args (OpValue): Flattened primal leaves.
            **kwargs (OpValue): Keyword arguments.

        Returns:
            PrimalTree: Output of `fun(*unflat_args)`.
        """
        del kwargs
        unflat_args = tree_unflatten(tree_def, list(flat_args))
        seq_args = tuple(unflat_args) if isinstance(unflat_args, Iterable) else (unflat_args,)
        _callable_fn_2: OpValue = fn_callable
        assert is_variadic_callable(_callable_fn_2)
        return as_primal_tree(_callable_fn_2(*seq_args))

    with ConfigContext(eager_mode=True):
        _callable_fn_3: OpValue = fn_callable
        assert is_variadic_callable(_callable_fn_3)
        out_eager = _callable_fn_3(*primals)
        raw_val, raw_aux = _unpack_pair(out_eager) if has_aux else (out_eager, None)
        val = as_primal_tree(raw_val)
        aux: PrimalAuxValue = raw_aux if isinstance(raw_aux, str) else as_primal_tree(raw_aux)

    fun_primal = _get_fun_primal(fun_flat, has_aux)
    tensor_primals = _convert_to_tensors(flat_primals)
    forward_graph = _build_traced_logical_graph(fun_primal, tensor_primals, "vjp")
    primal_input_ids = list(forward_graph.inputs[: len(tensor_primals)])
    output_node_id, cotangent_mapping, cotangent_ids_list = _attach_cotangent_inputs(forward_graph)

    grad_graph = graph_grad(
        forward_graph,
        primal_input_ids,
        output_node_id,
        cotangent_id=cotangent_mapping,
    )

    def vjp_fn(*cot_args: OpValue, **cot_kwargs: OpValue) -> tuple[PrimalTree, ...]:
        """Evaluate the pullback function on upstream cotangent.

        Args:
            *cot_args (OpValue): Positional upstream cotangent argument.
            **cot_kwargs (OpValue): Unused keyword arguments.

        Returns:
            tuple[PrimalTree, ...]: Tuple of cotangents with respect to `primals`.
        """
        _ = cot_kwargs
        cotangent = cot_args[0] if cot_args else None
        flat_cot, _ = tree_flatten(cotangent)
        if global_tracing_state.is_tracing and global_tracing_state.active_graph is not None:
            flat_grads = _inline_vjp_into_active_trace(
                grad_graph,
                primal_input_ids,
                tensor_primals,
                cotangent_ids_list,
                flat_cot,
            )
            unflat = tree_unflatten(tree_def, flat_grads)
            seq_unflat = tuple(unflat) if isinstance(unflat, Iterable) else (unflat,)
            return tuple(as_primal_tree(x) for x in seq_unflat)

        backend = get_active_backend()
        inputs_dict: dict[str, InterpreterValue] = {
            inp_id: as_eval_value(backend.asarray(_unwrap_data(p)))
            for inp_id, p in zip(primal_input_ids, tensor_primals)
        }
        for cot_id, cot_val in zip(cotangent_ids_list, flat_cot):
            inputs_dict[cot_id] = as_eval_value(backend.asarray(_unwrap_data(cot_val)))

        outputs_dict = _execute_autodiff_graph(grad_graph, inputs_dict)
        flat_out_grads: list[OpValue] = [outputs_dict[out_id] for out_id in grad_graph.outputs]
        unflat = tree_unflatten(tree_def, flat_out_grads)
        seq_unflat = tuple(unflat) if isinstance(unflat, Iterable) else (unflat,)
        return tuple(as_primal_tree(x) for x in seq_unflat)

    return ((val, aux), vjp_fn) if has_aux else (val, vjp_fn)


__all__ = [
    "CustomJVPFunction",
    "GradOptions",
    "_ensure_grad_tensor",
    "_unpack_pair",
    "custom_jvp",
    "get_active_backend",
    "hessian",
    "hvp",
    "jacfwd",
    "jacrev",
    "jvp",
    "vjp",
]
