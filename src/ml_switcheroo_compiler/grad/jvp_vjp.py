# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
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

from .options import GradOptions
from .utils import _convert_to_tensors, _get_fun_primal


def custom_jvp(fun: Callable[..., object]) -> Callable[..., object]:
    """Wrap a function to allow defining custom Jacobian-vector product (JVP) rules.

    Args:
        fun (Callable): The original function.

    Returns:
        Callable: The wrapped function that supports custom JVPs.
    """
    return fun


def jvp(
    fun: Callable[..., object],
    primals: object,
    tangents: object,
    has_aux: bool = False,
) -> tuple[object, object]:
    """Evaluate jvp operation.

    Args:
        fun (object): The fun parameter.
        primals (object): The primals parameter.
        tangents (object): The tangents parameter.
        has_aux (bool): The has_aux parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.interpreter import evaluate_graph
    from ml_switcheroo_compiler.transforms.autodiff import jvp as graph_jvp

    primals_seq: object = primals if isinstance(primals, (tuple, list)) else (primals,)
    tangents_seq: object = tangents if isinstance(tangents, (tuple, list)) else (tangents,)

    with ConfigContext(eager_mode=True):
        if has_aux:
            val, aux = fun(*primals_seq)
        else:
            val: object = fun(*primals_seq)

    fun_primal: object = _get_fun_primal(fun, has_aux)
    tensor_primals: object = _convert_to_tensors(primals_seq)

    # Trace
    block: object = _trace_function(fun_primal, tuple(tensor_primals), f"jvp_{uuid.uuid4().hex[:6]}")
    forward_graph: object = LogicalGraph(name=block.id)
    for node in block.nodes:
        forward_graph.nodes[node.id] = node
    forward_graph.inputs = block.inputs
    forward_graph.outputs = block.outputs

    # Add tangent constants to the graph
    tangent_ids: object = []
    for t, p_id in zip(tangents_seq, forward_graph.inputs):
        t_id: object = f"tangent_{uuid.uuid4().hex[:6]}"
        t_node: object = LogicalNode(
            id=t_id,
            op_type="Constant",
            attributes={"value": getattr(t, "data", t)},
            shape_metadata=forward_graph.nodes[p_id].shape_metadata,
        )
        forward_graph.nodes[t_id] = t_node
        tangent_ids.append(t_id)

    # Compute JVP graph
    jvp_graph: object = graph_jvp(forward_graph, forward_graph.inputs, tangent_ids, forward_graph.outputs)

    # Evaluate JVP graph
    inputs_dict: object = {inp_id: get_active_backend().asarray(getattr(p, "data", p)) for inp_id, p in zip(forward_graph.inputs, tensor_primals)}
    outputs_dict: object = evaluate_graph(jvp_graph, inputs_dict)

    out_tangent_values: object = [outputs_dict[out_id] for out_id in jvp_graph.outputs]
    out_tan: object = out_tangent_values[0] if len(out_tangent_values) == 1 else tuple(out_tangent_values)

    if has_aux:
        return (val, aux), out_tan
    return val, out_tan


def vjp(
    fun: Callable[..., object],
    *primals: object,
    has_aux: bool = False,
) -> tuple[object, Callable[..., object]]:
    """Evaluate vjp operation.

    Args:
        fun (Callable[..., object]): The fun parameter.
        *primals (object): Positional args.
        has_aux (bool): Aux.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.interpreter import evaluate_graph
    from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad
    from ml_switcheroo_compiler.tree_util import tree_flatten, tree_unflatten

    # 1. Flatten nested primals if any
    flat_primals, tree_def = tree_flatten(primals)

    # 2. Wrap function to unflatten inputs before execution
    def fun_flat(*flat_args: object) -> object:
        """Evaluate fun_flat operation.

        Args:
        *flat_args (object): Positional args.

        Returns:
            tuple[int, ...]: Result.
        """
        unflat_args: object = tree_unflatten(tree_def, list(flat_args))
        return fun(*unflat_args)

    with ConfigContext(eager_mode=True):
        if has_aux:
            val, aux = fun(*primals)
        else:
            val: object = fun(*primals)

    fun_primal: object = _get_fun_primal(fun_flat, has_aux)
    tensor_primals: object = _convert_to_tensors(flat_primals)

    # 3. Trace the flat primal function
    block: object = _trace_function(fun_primal, tuple(tensor_primals), f"vjp_{uuid.uuid4().hex[:6]}")
    forward_graph: object = LogicalGraph(name=block.id)
    for node in block.nodes:
        forward_graph.nodes[node.id] = node
    forward_graph.inputs = block.inputs
    forward_graph.outputs = block.outputs

    # 4. Create starting cotangent input nodes inside the forward graph for each output tensor
    output_node_id: object = forward_graph.outputs[0]
    output_node: object = forward_graph.nodes[output_node_id]

    cotangent_mapping: object = {}
    cotangent_ids_list: object = []
    for i, y_id in enumerate(output_node.inputs):
        cot_id: object = f"cotangent_{i}_{uuid.uuid4().hex[:6]}"
        cot_node: object = LogicalNode(
            id=cot_id,
            op_type="Input",
            inputs=[],
            shape_metadata=forward_graph.nodes[y_id].shape_metadata,
        )
        forward_graph.nodes[cot_id] = cot_node
        forward_graph.inputs.append(cot_id)
        cotangent_mapping[y_id] = cot_id
        cotangent_ids_list.append(cot_id)

    # 5. Build the gradient graph starting with our cotangent_mapping as the loss adjoint
    grad_graph: object = graph_grad(
        forward_graph,
        forward_graph.inputs[: len(tensor_primals)],
        output_node_id,
        cotangent_id=cotangent_mapping,
    )

    def vjp_fn(cotangent: object) -> tuple[object, ...]:
        """Evaluate vjp_fn operation.

        Args:
        cotangent (object): The cotangent parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        # Run the evaluator on grad_graph
        inputs_dict: object = {inp_id: get_active_backend().asarray(getattr(p, "data", p)) for inp_id, p in zip(forward_graph.inputs[: len(tensor_primals)], tensor_primals)}

        # Flatten the cotangent Pytree if it is nested
        flat_cot, _ = tree_flatten(cotangent)
        for cot_id, cot_val in zip(cotangent_ids_list, flat_cot):
            inputs_dict[cot_id] = get_active_backend().asarray(getattr(cot_val, "data", cot_val))

        outputs_dict: object = evaluate_graph(grad_graph, inputs_dict)

        flat_grads: object = []
        for out_id in grad_graph.outputs:
            g_val: object = outputs_dict[out_id]
            flat_grads.append(g_val)

        # Unflatten gradients back to original tree structure
        res: object = tree_unflatten(tree_def, flat_grads)
        return tuple(res)

    if has_aux:
        return (val, aux), vjp_fn
    return val, vjp_fn


def hvp(
    fun: Callable[..., object],
    primals: object,
    tangents: object,
    has_aux: bool = False,
) -> tuple[object, object]:
    """Evaluate hvp operation.

    Args:
        fun (object): The fun parameter.
        primals (object): The primals parameter.
        tangents (object): The tangents parameter.
        has_aux (bool): The has_aux parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.interpreter import evaluate_graph
    from ml_switcheroo_compiler.transforms.autodiff import hvp as graph_hvp

    primals_seq: object = primals if isinstance(primals, (tuple, list)) else (primals,)
    tangents_seq: object = tangents if isinstance(tangents, (tuple, list)) else (tangents,)

    from ml_switcheroo_compiler.core.config import ConfigContext

    with ConfigContext(eager_mode=True):
        if has_aux:
            val, aux = fun(*primals_seq)
        else:
            val: object = fun(*primals_seq)

    fun_primal: object = _get_fun_primal(fun, has_aux)
    tensor_primals: object = _convert_to_tensors(primals_seq)

    # Trace
    block: object = _trace_function(fun_primal, tuple(tensor_primals), f"hvp_{uuid.uuid4().hex[:6]}")
    forward_graph: object = LogicalGraph(name=block.id)
    for node in block.nodes:
        forward_graph.nodes[node.id] = node
    forward_graph.inputs = block.inputs
    forward_graph.outputs = block.outputs

    # Add tangent constants
    tangent_ids: object = []
    for t, p_id in zip(tangents_seq, forward_graph.inputs):
        t_id: object = f"tangent_{uuid.uuid4().hex[:6]}"
        t_node: object = LogicalNode(
            id=t_id,
            op_type="Constant",
            attributes={"value": getattr(t, "data", t)},
            shape_metadata=forward_graph.nodes[p_id].shape_metadata,
        )
        forward_graph.nodes[t_id] = t_node
        tangent_ids.append(t_id)

    # Compute HVP graph
    hvp_graph: object = graph_hvp(forward_graph, forward_graph.inputs, tangent_ids, forward_graph.outputs)

    # Evaluate HVP graph
    inputs_dict: object = {inputs_id: get_active_backend().asarray(getattr(p, "data", p)) for inputs_id, p in zip(forward_graph.inputs, tensor_primals)}
    outputs_dict: object = evaluate_graph(hvp_graph, inputs_dict)

    out_tangent_values: object = [outputs_dict[out_id] for out_id in hvp_graph.outputs]
    out_tan: object = out_tangent_values[0] if len(out_tangent_values) == 1 else tuple(out_tangent_values)

    if has_aux:
        return (val, aux), out_tan
    return val, out_tan


def jacfwd(fun: typing.Callable[..., object], options: object = None) -> typing.Callable[..., object]:
    """Evaluate jacfwd operation.

    Args:
        fun (object): The fun parameter.
        options (GradOptions): The options parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    options: object = options or GradOptions()

    def wrapped(*args: object, **kwargs: object) -> object:
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        # Evaluate fun to see input and output dimensions
        out: object = fun(*args, **kwargs)
        out_arr: object = get_active_backend().asarray(getattr(out, "data", out))

        # Build basis tangents for each input coordinate
        arg0: object = get_active_backend().asarray(getattr(args[0], "data", args[0]))
        flat_arg0: object = arg0.flatten()

        jacobian_rows: object = []
        for i in range(len(flat_arg0)):
            # Standard basis vector for coordinate i
            tangent_flat: object = get_active_backend().execute_op(
                "OneHot",
                get_active_backend().asarray(i),
                len(flat_arg0),
                on_value=1.0,
                off_value=0.0,
                axis=-1,
                dtype="float32",
            )
            tangent: object = tangent_flat.reshape(arg0.shape)

            # Run jvp
            _, out_tangent = jvp(fun, args, (tangent,), has_aux=options.has_aux)
            jacobian_rows.append(get_active_backend().asarray(out_tangent).flatten())

        # Standard format of Jacobian: (output_size, input_size)
        res: object = get_active_backend().execute_op("Stack", jacobian_rows, axis=-1)
        if out_arr.ndim > 0:
            return res.reshape(out_arr.shape + arg0.shape)
        return res

    return wrapped


def jacrev(fun: typing.Callable[..., object], options: object = None) -> typing.Callable[..., object]:
    """Evaluate jacrev operation.

    Args:
        fun (object): The fun parameter.
        options (GradOptions): The options parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    options: object = options or GradOptions()

    def wrapped(*args: object, **kwargs: object) -> object:
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.tree_util import tree_flatten, tree_unflatten

        out, vjp_fn = vjp(fun, *args, has_aux=options.has_aux)
        if options.has_aux:
            out_val, _ = out
        else:
            out_val: object = out

        flat_out, out_tree_def = tree_flatten(out_val)
        flat_shapes: object = [get_active_backend().asarray(getattr(o, "data", o)).shape for o in flat_out]
        flat_sizes: object = [int(math.prod(get_active_backend().asarray(getattr(o, "data", o)).shape)) for o in flat_out]
        total_size: object = sum(flat_sizes)

        arg0: object = get_active_backend().asarray(getattr(args[0], "data", args[0]))

        jacobian_rows: object = []
        for i in range(total_size):
            cotangent_flat: object = get_active_backend().execute_op(
                "OneHot",
                get_active_backend().asarray(i),
                total_size,
                on_value=1.0,
                off_value=0.0,
                axis=-1,
                dtype="float32",
            )

            flat_cots: object = []
            curr_idx: object = 0
            for sz, shp in zip(flat_sizes, flat_shapes):
                flat_cots.append(cotangent_flat[curr_idx : curr_idx + sz].reshape(shp))
                curr_idx += sz

            cotangent: object = tree_unflatten(out_tree_def, flat_cots)
            grads: object = vjp_fn(cotangent)
            # Differentiate with respect to first argument
            jacobian_rows.append(get_active_backend().asarray(grads[0]).flatten())

        res: object = get_active_backend().execute_op("Stack", jacobian_rows, axis=0)
        # Reshape output to out_shape + arg_shape
        if len(flat_sizes) > 1:
            # Multi-output: standard output shape is (total_size,) + arg_shape
            return res.reshape((total_size,) + arg0.shape)
        else:
            out_arr: object = get_active_backend().asarray(getattr(out_val, "data", out_val))
            if out_arr.ndim > 0:
                return res.reshape(out_arr.shape + arg0.shape)
            return res.reshape(arg0.shape)

    return wrapped


def hessian(fun: typing.Callable[..., object], options: object = None) -> typing.Callable[..., object]:
    """Evaluate hessian operation.

    Args:
        fun (object): The fun parameter.
        options (GradOptions): The options parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    options: object = options or GradOptions()

    def wrapped(*args: object, **kwargs: object) -> object:
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        arg0: object = get_active_backend().asarray(getattr(args[0], "data", args[0]))
        flat_arg0: object = arg0.flatten()

        hessian_rows: object = []
        for i in range(len(flat_arg0)):
            tangent_flat: object = get_active_backend().execute_op(
                "OneHot",
                get_active_backend().asarray(i),
                len(flat_arg0),
                on_value=1.0,
                off_value=0.0,
                axis=-1,
                dtype="float32",
            )
            tangent: object = tangent_flat.reshape(arg0.shape)

            _, out_tangent = hvp(fun, args, (tangent,), has_aux=options.has_aux)
            hessian_rows.append(get_active_backend().asarray(out_tangent).flatten())

        res: object = get_active_backend().execute_op("Stack", hessian_rows, axis=0)
        return res.reshape(arg0.shape + arg0.shape)

    return wrapped
