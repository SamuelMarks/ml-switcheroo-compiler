# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Gradient computation and autodiff utilities."""

import contextlib
import math
import typing
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Optional, Union

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import ConfigContext, config
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


class CustomJVPFunction:
    """Wrap a function to allow defining custom Jacobian-vector product (JVP) rules."""

    def __init__(self, fun: Callable) -> None:
        """Initialize CustomJVPFunction.

        Args:
            fun (Callable): The base function being wrapped.
        """
        self.fun = fun
        self.jvp_rule: Optional[Callable] = None

    def defjvp(self, jvp_rule: Callable) -> Callable:
        """Define the custom Jacobian-vector product (JVP) rule.

        Args:
            jvp_rule (Callable): The JVP rule. Signature can be (primals, tangents) -> (val, out_tangent)
                or (*primals, *tangents).

        Returns:
            Callable: The registered jvp_rule.
        """
        self.jvp_rule = jvp_rule
        return jvp_rule

    def __eq__(self, other: object) -> bool:
        """Check equality with another function or CustomJVPFunction.

        Args:
            other (object): Other object to compare.

        Returns:
            bool: True if equal, False otherwise.
        """
        if isinstance(other, CustomJVPFunction):
            return self.fun == other.fun
        return self.fun == other

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Execute the function, emitting a CustomJVP node when tracing with a registered rule.

        Args:
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: The computed output.
        """
        if config.eager_mode or not global_tracing_state.is_tracing or self.jvp_rule is None:
            return self.fun(*args, **kwargs)

        tensor_args = [a for a in args if isinstance(a, Tensor)]
        if not tensor_args:
            return self.fun(*args, **kwargs)

        from ml_switcheroo_compiler.core.config import ConfigContext

        with ConfigContext(eager_mode=True):
            sample_out = self.fun(*args, **kwargs)

        out_shape = getattr(sample_out, "shape", ())
        out_dtype = getattr(getattr(sample_out, "dtype", None), "value", "float32")
        device = getattr(tensor_args[0], "device", "cpu")

        in_ids: list[str] = []
        for a in tensor_args:
            if hasattr(a, "data") and hasattr(a.data, "id"):
                in_ids.append(a.data.id)
            else:
                cid = str(uuid.uuid4())
                arr = getattr(a, "data", a)
                c_node = LogicalNode(
                    id=cid,
                    op_type="Constant",
                    inputs=[],
                    attributes={"value": arr},
                    shape_metadata=getattr(arr, "shape", ()),
                )
                global_tracing_state.add_node(c_node)
                in_ids.append(cid)

        out_id = f"custom_jvp_{uuid.uuid4().hex[:6]}"
        node = LogicalNode(
            id=out_id,
            op_type="CustomJVP",
            inputs=in_ids,
            attributes={"jvp_rule": self.jvp_rule, "fun": self.fun},
            shape_metadata=out_shape,
        )
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype)
        return Tensor(proxy, TensorConfig(out_shape, DType(out_dtype), device))


def custom_jvp(fun: Callable) -> CustomJVPFunction:
    """Wrap a function to allow defining custom Jacobian-vector product (JVP) rules.

    Args:
        fun (Callable): The original function.

    Returns:
        CustomJVPFunction: The wrapped function that supports custom JVPs.
    """
    return CustomJVPFunction(fun)


def jvp(
    fun,
    primals,
    tangents,
    has_aux: bool = False,
):
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

    primals_seq = primals if isinstance(primals, (tuple, list)) else (primals,)
    tangents_seq = tangents if isinstance(tangents, (tuple, list)) else (tangents,)

    if isinstance(fun, CustomJVPFunction) and fun.jvp_rule is not None:
        with ConfigContext(eager_mode=True):
            try:
                res = fun.jvp_rule(primals_seq, tangents_seq)
            except TypeError:
                res = fun.jvp_rule(*primals_seq, *tangents_seq)

            if isinstance(res, (tuple, list)) and len(res) == 2:
                val, out_tan = res
            else:
                val = fun(*primals_seq)
                out_tan = res

        if has_aux:
            return val, out_tan
        return val, out_tan

    with ConfigContext(eager_mode=True):
        if has_aux:
            val, aux = fun(*primals_seq)
        else:
            val = fun(*primals_seq)

    fun_primal = _get_fun_primal(fun, has_aux)
    tensor_primals = _convert_to_tensors(primals_seq)

    # Trace
    block = _trace_function(fun_primal, tuple(tensor_primals), f"jvp_{uuid.uuid4().hex[:6]}")
    forward_graph = LogicalGraph(name=block.id)
    for node in block.nodes:
        forward_graph.nodes[node.id] = node
    forward_graph.inputs = block.inputs
    forward_graph.outputs = block.outputs

    # Add tangent constants to the graph
    tangent_ids = []
    for t, p_id in zip(tangents_seq, forward_graph.inputs):
        t_id = f"tangent_{uuid.uuid4().hex[:6]}"
        t_node = LogicalNode(
            id=t_id,
            op_type="Constant",
            attributes={"value": getattr(t, "data", t)},
            shape_metadata=forward_graph.nodes[p_id].shape_metadata,
        )
        forward_graph.nodes[t_id] = t_node
        tangent_ids.append(t_id)

    # Compute JVP graph
    jvp_graph = graph_jvp(forward_graph, forward_graph.inputs, tangent_ids, forward_graph.outputs)

    # Evaluate JVP graph
    backend = get_active_backend()
    inputs_dict = {inp_id: backend.asarray(getattr(p, "data", p)) for inp_id, p in zip(forward_graph.inputs, tensor_primals)}
    if backend is not None and hasattr(backend, "execute_graph"):
        outputs_dict = backend.execute_graph(jvp_graph, inputs_dict)
    elif backend is not None and hasattr(backend, "compile_graph"):
        compiled_fn = backend.compile_graph(jvp_graph)
        outputs_dict = compiled_fn(inputs_dict)
    else:
        outputs_dict = evaluate_graph(jvp_graph, inputs_dict)

    out_tangent_values = [outputs_dict[out_id] for out_id in jvp_graph.outputs]
    out_tan = out_tangent_values[0] if len(out_tangent_values) == 1 else tuple(out_tangent_values)

    if has_aux:
        return (val, aux), out_tan
    return val, out_tan


def vjp(
    fun,
    *primals,
    has_aux: bool = False,
):
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
    def fun_flat(*flat_args):
        """Evaluate fun_flat operation.

        Args:
        *flat_args (object): Positional args.

        Returns:
            tuple[int, ...]: Result.
        """
        unflat_args = tree_unflatten(tree_def, list(flat_args))
        return fun(*unflat_args)

    with ConfigContext(eager_mode=True):
        if has_aux:
            val, aux = fun(*primals)
        else:
            val = fun(*primals)

    fun_primal = _get_fun_primal(fun_flat, has_aux)
    tensor_primals = _convert_to_tensors(flat_primals)

    # 3. Trace the flat primal function
    block = _trace_function(fun_primal, tuple(tensor_primals), f"vjp_{uuid.uuid4().hex[:6]}")
    forward_graph = LogicalGraph(name=block.id)
    for node in block.nodes:
        forward_graph.nodes[node.id] = node
    forward_graph.inputs = block.inputs
    forward_graph.outputs = block.outputs

    # 4. Create starting cotangent input nodes inside the forward graph for each output tensor
    output_node_id = forward_graph.outputs[0]
    output_node = forward_graph.nodes[output_node_id]

    cotangent_mapping = {}
    cotangent_ids_list = []
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

    # 5. Build the gradient graph starting with our cotangent_mapping as the loss adjoint
    grad_graph = graph_grad(
        forward_graph,
        forward_graph.inputs[: len(tensor_primals)],
        output_node_id,
        cotangent_id=cotangent_mapping,
    )

    def vjp_fn(cotangent):
        """Evaluate vjp_fn operation.

        Args:
        cotangent (object): The cotangent parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        if global_tracing_state.is_tracing and global_tracing_state.active_graph is not None:
            # Inline grad_graph symbolically into the active tracing graph
            id_map: dict[str, str] = {}
            for inp_id, p in zip(forward_graph.inputs[: len(tensor_primals)], tensor_primals):
                p_data = getattr(p, "data", p)
                id_map[inp_id] = getattr(p_data, "id", str(p_data))

            flat_cot, _ = tree_flatten(cotangent)
            for cot_id, cot_val in zip(cotangent_ids_list, flat_cot):
                c_data = getattr(cot_val, "data", cot_val)
                id_map[cot_id] = getattr(c_data, "id", str(c_data))

            for nid, n in grad_graph.nodes.items():
                if getattr(n, "op_type", "") != "Input":
                    new_id = f"vjp_{uuid.uuid4().hex[:6]}_{nid}"
                    id_map[nid] = new_id
                    new_inputs = [id_map.get(inp, inp) for inp in n.inputs]
                    cloned = LogicalNode(
                        id=new_id,
                        op_type=n.op_type,
                        inputs=new_inputs,
                        attributes=dict(getattr(n, "attributes", {})),
                        shape_metadata=getattr(n, "shape_metadata", ()),
                    )
                    global_tracing_state.add_node(cloned)

            flat_grads = []
            for out_id in grad_graph.outputs:
                mapped_id = id_map.get(out_id, out_id)
                out_node = grad_graph.nodes.get(out_id)
                out_shape = getattr(out_node, "shape_metadata", ())
                proxy = ProxyTensor(id=mapped_id, shape=out_shape, dtype="float32")
                from ml_switcheroo_compiler.core.device import Device

                flat_grads.append(Tensor(proxy, TensorConfig(out_shape, DType.Float32, Device("cpu"))))

            res = tree_unflatten(tree_def, flat_grads)
            return tuple(res)

        # Run the evaluator or native execution on grad_graph
        backend = get_active_backend()
        inputs_dict = {inp_id: backend.asarray(getattr(p, "data", p)) for inp_id, p in zip(forward_graph.inputs[: len(tensor_primals)], tensor_primals)}

        # Flatten the cotangent Pytree if it is nested
        flat_cot, _ = tree_flatten(cotangent)
        for cot_id, cot_val in zip(cotangent_ids_list, flat_cot):
            inputs_dict[cot_id] = backend.asarray(getattr(cot_val, "data", cot_val))

        if backend is not None and hasattr(backend, "execute_graph"):
            outputs_dict = backend.execute_graph(grad_graph, inputs_dict)
        elif backend is not None and hasattr(backend, "compile_graph"):
            compiled_fn = backend.compile_graph(grad_graph)
            outputs_dict = compiled_fn(inputs_dict)
        else:
            outputs_dict = evaluate_graph(grad_graph, inputs_dict)

        flat_grads = []
        for out_id in grad_graph.outputs:
            g_val = outputs_dict[out_id]
            flat_grads.append(g_val)

        # Unflatten gradients back to original tree structure
        res = tree_unflatten(tree_def, flat_grads)
        return tuple(res)

    if has_aux:
        return (val, aux), vjp_fn
    return val, vjp_fn


def hvp(
    fun,
    primals,
    tangents,
    has_aux: bool = False,
    projected_tangents: Optional[Union[Tensor, Sequence[Tensor]]] = None,
    mode: str = "forward-over-reverse",
):
    """Evaluate hvp operation.

    Args:
        fun (object): The fun parameter.
        primals (object): The primals parameter.
        tangents (object): The tangents parameter.
        has_aux (bool): The has_aux parameter.
        projected_tangents (Optional[Any]): Projected cotangents/tangents for non-scalar outputs.
        mode (str): HVP computation mode ('forward-over-reverse' or 'reverse-over-forward').

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.interpreter import evaluate_graph
    from ml_switcheroo_compiler.transforms.autodiff import hvp as graph_hvp

    primals_seq = primals if isinstance(primals, (tuple, list)) else (primals,)
    tangents_seq = tangents if isinstance(tangents, (tuple, list)) else (tangents,)

    from ml_switcheroo_compiler.core.config import ConfigContext

    with ConfigContext(eager_mode=True):
        if has_aux:
            val, aux = fun(*primals_seq)
        else:
            val = fun(*primals_seq)

    fun_primal = _get_fun_primal(fun, has_aux)
    tensor_primals = _convert_to_tensors(primals_seq)

    # Trace
    block = _trace_function(fun_primal, tuple(tensor_primals), f"hvp_{uuid.uuid4().hex[:6]}")
    forward_graph = LogicalGraph(name=block.id)
    for node in block.nodes:
        forward_graph.nodes[node.id] = node
    forward_graph.inputs = block.inputs
    forward_graph.outputs = block.outputs

    # Add tangent constants
    tangent_ids = []
    for t, p_id in zip(tangents_seq, forward_graph.inputs):
        t_id = f"tangent_{uuid.uuid4().hex[:6]}"
        t_node = LogicalNode(
            id=t_id,
            op_type="Constant",
            attributes={"value": getattr(t, "data", t)},
            shape_metadata=forward_graph.nodes[p_id].shape_metadata,
        )
        forward_graph.nodes[t_id] = t_node
        tangent_ids.append(t_id)

    proj_tangent_ids = None
    if projected_tangents is not None:
        proj_seq = projected_tangents if isinstance(projected_tangents, (tuple, list)) else (projected_tangents,)
        proj_tangent_ids = []
        for pt, o_id in zip(proj_seq, forward_graph.outputs):
            pt_id = f"proj_tangent_{uuid.uuid4().hex[:6]}"
            pt_node = LogicalNode(
                id=pt_id,
                op_type="Constant",
                attributes={"value": getattr(pt, "data", pt)},
                shape_metadata=forward_graph.nodes[o_id].shape_metadata,
            )
            forward_graph.nodes[pt_id] = pt_node
            proj_tangent_ids.append(pt_id)

    # Compute HVP graph
    hvp_graph = graph_hvp(
        forward_graph,
        forward_graph.inputs,
        tangent_ids,
        forward_graph.outputs,
        mode=mode,
        projected_tangents=proj_tangent_ids,
    )

    # Evaluate HVP graph
    inputs_dict = {inputs_id: get_active_backend().asarray(getattr(p, "data", p)) for inputs_id, p in zip(forward_graph.inputs, tensor_primals)}
    outputs_dict = evaluate_graph(hvp_graph, inputs_dict)

    out_tangent_values = [outputs_dict[out_id] for out_id in hvp_graph.outputs]
    out_tan = out_tangent_values[0] if len(out_tangent_values) == 1 else tuple(out_tangent_values)

    if has_aux:
        return (val, aux), out_tan
    return val, out_tan


def jacfwd(fun, options=None):
    """Evaluate jacfwd operation.

    Args:
        fun (object): The fun parameter.
        options (GradOptions): The options parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    options = options or GradOptions()

    def wrapped(*args, **kwargs):
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        # Evaluate fun to see input and output dimensions
        with ConfigContext(eager_mode=True):
            out = fun(*args, **kwargs)
            if options.has_aux:
                out_val, _ = out
            else:
                out_val = out
        out_arr = get_active_backend().asarray(getattr(out_val, "data", out_val))

        # Build basis tangents for each input coordinate
        arg0 = get_active_backend().asarray(getattr(args[0], "data", args[0]))
        flat_arg0 = arg0.flatten()

        jacobian_rows = []
        for i in range(len(flat_arg0)):
            # Standard basis vector for coordinate i
            tangent_flat = get_active_backend().execute_op(
                "OneHot",
                get_active_backend().asarray(i),
                len(flat_arg0),
                on_value=1.0,
                off_value=0.0,
                axis=-1,
                dtype="float32",
            )
            tangent = tangent_flat.reshape(arg0.shape)

            # Run jvp
            _, out_tangent = jvp(fun, args, (tangent,), has_aux=options.has_aux)
            jacobian_rows.append(get_active_backend().asarray(out_tangent).flatten())

        # Standard format of Jacobian: (output_size, input_size)
        res = get_active_backend().execute_op("Stack", jacobian_rows, axis=-1)
        if out_arr.ndim > 0:
            return res.reshape(out_arr.shape + arg0.shape)
        return res

    return wrapped


def jacrev(fun, options=None):
    """Evaluate jacrev operation.

    Args:
        fun (object): The fun parameter.
        options (GradOptions): The options parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    options = options or GradOptions()

    def wrapped(*args, **kwargs):
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
            out_val = out

        flat_out, out_tree_def = tree_flatten(out_val)
        flat_shapes = [get_active_backend().asarray(getattr(o, "data", o)).shape for o in flat_out]
        flat_sizes = [int(math.prod(get_active_backend().asarray(getattr(o, "data", o)).shape)) for o in flat_out]
        total_size = sum(flat_sizes)

        arg0 = get_active_backend().asarray(getattr(args[0], "data", args[0]))

        jacobian_rows = []
        for i in range(total_size):
            cotangent_flat = get_active_backend().execute_op(
                "OneHot",
                get_active_backend().asarray(i),
                total_size,
                on_value=1.0,
                off_value=0.0,
                axis=-1,
                dtype="float32",
            )

            flat_cots = []
            curr_idx = 0
            for sz, shp in zip(flat_sizes, flat_shapes):
                flat_cots.append(cotangent_flat[curr_idx : curr_idx + sz].reshape(shp))
                curr_idx += sz

            cotangent = tree_unflatten(out_tree_def, flat_cots)
            grads = vjp_fn(cotangent)
            # Differentiate with respect to first argument
            jacobian_rows.append(get_active_backend().asarray(grads[0]).flatten())

        res = get_active_backend().execute_op("Stack", jacobian_rows, axis=0)
        # Reshape output to out_shape + arg_shape
        if len(flat_sizes) > 1:
            # Multi-output: standard output shape is (total_size,) + arg_shape
            return res.reshape((total_size,) + arg0.shape)
        else:
            out_arr = get_active_backend().asarray(getattr(out_val, "data", out_val))
            if out_arr.ndim > 0:
                return res.reshape(out_arr.shape + arg0.shape)
            return res.reshape(arg0.shape)

    return wrapped


def hessian(fun, options=None):
    """Evaluate hessian operation.

    Args:
        fun (object): The fun parameter.
        options (GradOptions): The options parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    options = options or GradOptions()

    def wrapped(*args, **kwargs):
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        arg0 = get_active_backend().asarray(getattr(args[0], "data", args[0]))
        flat_arg0 = arg0.flatten()

        hessian_rows = []
        for i in range(len(flat_arg0)):
            tangent_flat = get_active_backend().execute_op(
                "OneHot",
                get_active_backend().asarray(i),
                len(flat_arg0),
                on_value=1.0,
                off_value=0.0,
                axis=-1,
                dtype="float32",
            )
            tangent = tangent_flat.reshape(arg0.shape)

            _, out_tangent = hvp(fun, args, (tangent,), has_aux=options.has_aux)
            hessian_rows.append(get_active_backend().asarray(out_tangent).flatten())

        res = get_active_backend().execute_op("Stack", hessian_rows, axis=0)
        return res.reshape(arg0.shape + arg0.shape)

    return wrapped
