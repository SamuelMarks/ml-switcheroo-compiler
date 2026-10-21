"""Forward-mode (JVP), reverse-mode (VJP), Jacobian, and Hessian autodiff transformations."""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Mapping, Sequence

from ml_switcheroo_compiler.backends.eager_registry import as_callable, as_tensor
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable, invoke_callable
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import LogicalGraph, LogicalNode, as_attrs, as_node_list
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from .options import GradOptions
from .utils import GradValue, _convert_to_tensors, _get_fun_primal


def _unpack_pair(val: GradValue) -> tuple[GradValue, GradValue]:
    """Unpack a 2-element iterable or tuple into `(first, second)`.

    Args:
        val (GradValue): 2-element sequence or iterable.

    Returns:
        tuple[GradValue, GradValue]: Unpacked pair of elements.
    """
    seq = list(getattr(val, "__iter__")())
    return seq[0], seq[1]


class CustomJVPFunction:
    """Wrap a function to allow defining custom Jacobian-vector product (JVP) rules."""

    def __init__(self, fun: GradValue) -> None:
        """Initialize `CustomJVPFunction`.

        Args:
            fun (GradValue): Base primal function being wrapped.
        """
        self.fun: TypedVariadicCallable[GradValue] = as_callable(fun)
        self.jvp_rule: TypedVariadicCallable[GradValue] | None = None
        self.jvp_subgraph: LogicalGraph | None = None

    def defjvp(
        self,
        jvp_rule: GradValue,
        jvp_subgraph: LogicalGraph | None = None,
    ) -> TypedVariadicCallable[GradValue]:
        """Define the custom Jacobian-vector product (JVP) rule.

        Args:
            jvp_rule (GradValue): Custom JVP rule callable.
            jvp_subgraph (Optional[LogicalGraph]): Optional declarative forward/tangent IR subgraph.

        Returns:
            TypedVariadicCallable[GradValue]: The registered `jvp_rule`.
        """
        rule_fn = as_callable(jvp_rule)
        self.jvp_rule = rule_fn
        if jvp_subgraph is not None:
            self.jvp_subgraph = jvp_subgraph
        elif isinstance(jvp_rule, LogicalGraph):
            self.jvp_subgraph = jvp_rule
        elif hasattr(jvp_rule, "graph") and isinstance(getattr(jvp_rule, "graph", None), LogicalGraph):
            self.jvp_subgraph = getattr(jvp_rule, "graph")
        else:
            self.jvp_subgraph = None
        return rule_fn

    def defjvp_subgraph(self, jvp_subgraph: LogicalGraph) -> None:
        """Attach a declarative forward/tangent IR subgraph directly.

        Args:
            jvp_subgraph (LogicalGraph): Custom forward/tangent IR subgraph.
        """
        self.jvp_subgraph = jvp_subgraph

    def __eq__(self, other: object) -> bool:
        """Check equality with another function or `CustomJVPFunction`.

        Args:
            other (CustomJVPFunction | TypedVariadicCallable[GradValue]): Other object to compare.

        Returns:
            bool: True if equal, False otherwise.
        """
        if isinstance(other, CustomJVPFunction):
            return self.fun == other.fun
        return bool(self.fun == other)

    def __call__(self, *args: GradValue, **kwargs: GradValue) -> GradValue:
        """Execute the function, emitting a `CustomJVP` node when tracing with a registered rule.

        Args:
            *args (GradValue): Positional arguments.
            **kwargs (GradValue): Keyword arguments.

        Returns:
            GradValue: Computed output tensor or value.
        """
        if config.eager_mode or not global_tracing_state.is_tracing or self.jvp_rule is None:
            return self.fun(*args, **kwargs)

        tensor_args = [a for a in args if isinstance(a, Tensor)]
        if not tensor_args:
            return self.fun(*args, **kwargs)

        with ConfigContext(eager_mode=True):
            sample_out = self.fun(*args, **kwargs)

        out_shape = getattr(sample_out, "shape", ())
        out_dtype = getattr(getattr(sample_out, "dtype", None), "value", "float32")
        device = getattr(tensor_args[0], "device", "cpu")

        in_ids: list[str] = []
        for a in tensor_args:
            if hasattr(a, "data") and hasattr(a.data, "id"):
                in_ids.append(getattr(a.data, "id"))
            else:
                cid = str(uuid.uuid4())
                arr = getattr(a, "data", a)
                c_node = LogicalNode(
                    id=cid,
                    op_type="Constant",
                    inputs=[],
                    attributes=as_attrs({"value": arr}),
                    shape_metadata=getattr(arr, "shape", ()),
                )
                global_tracing_state.add_node(c_node)
                in_ids.append(cid)

        out_id = f"custom_jvp_{uuid.uuid4().hex[:6]}"
        jvp_attrs: dict[str, TypedVariadicCallable[GradValue] | LogicalGraph | None] = {
            "jvp_rule": self.jvp_rule,
            "fun": self.fun,
        }
        subgraphs: dict[str, LogicalGraph] = {}
        if self.jvp_subgraph is not None:
            jvp_attrs["jvp_graph"] = self.jvp_subgraph
            jvp_attrs["custom_forward_subgraph"] = self.jvp_subgraph
            subgraphs["jvp"] = self.jvp_subgraph

        node = LogicalNode(
            id=out_id,
            op_type="CustomJVP",
            inputs=in_ids,
            attributes=as_attrs(jvp_attrs),
            subgraphs=subgraphs,
            shape_metadata=out_shape,
        )
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype)
        return Tensor(proxy, TensorConfig(out_shape, DType(out_dtype), device))


def custom_jvp(fun: GradValue) -> CustomJVPFunction:
    """Wrap a function to allow defining custom Jacobian-vector product (JVP) rules.

    Args:
        fun (GradValue): The original primal function.

    Returns:
        CustomJVPFunction: Wrapped callable supporting `.defjvp(...)`.
    """
    return CustomJVPFunction(fun)


def _run_custom_jvp_rule(
    fun: CustomJVPFunction,
    primals_seq: Sequence[GradValue],
    tangents_seq: Sequence[GradValue],
) -> tuple[GradValue, GradValue]:
    """Execute a registered custom JVP rule in eager mode.

    Args:
        fun (CustomJVPFunction): `CustomJVPFunction` with a registered `jvp_rule`.
        primals_seq (Sequence[GradValue]): Sequence of primal arguments.
        tangents_seq (Sequence[GradValue]): Sequence of tangent arguments.

    Returns:
        tuple[GradValue, GradValue]: Tuple of `(primal_output, tangent_output)`.
    """
    with ConfigContext(eager_mode=True):
        assert fun.jvp_rule is not None
        try:
            res = fun.jvp_rule(primals_seq, tangents_seq)
        except TypeError:
            res = fun.jvp_rule(*primals_seq, *tangents_seq)

        if isinstance(res, (tuple, list)) and len(res) == 2:
            val, out_tan = res
        else:
            val = fun(*primals_seq)
            out_tan = res
    return val, out_tan


def _build_traced_logical_graph(
    fun_primal: TypedVariadicCallable[GradValue],
    tensor_primals: list[Tensor],
    prefix: str,
) -> LogicalGraph:
    """Trace a primal function and construct a `LogicalGraph` from the traced block.

    Args:
        fun_primal (TypedVariadicCallable[GradValue]): Primal function to trace.
        tensor_primals (list[Tensor]): Tensor primal inputs.
        prefix (str): Name prefix for the traced graph.

    Returns:
        LogicalGraph: Populated forward logical graph.
    """
    block = _trace_function(fun_primal, tuple(tensor_primals), f"{prefix}_{uuid.uuid4().hex[:6]}")
    forward_graph = LogicalGraph(name=getattr(block, "id", getattr(block, "name", prefix)))
    nodes_iter: Iterable[LogicalNode] = block.nodes.values() if hasattr(block.nodes, "values") else as_node_list(block.nodes)
    for node in nodes_iter:
        forward_graph.nodes[node.id] = node
    forward_graph.inputs = block.inputs
    forward_graph.outputs = block.outputs
    return forward_graph


def _execute_autodiff_graph(graph: LogicalGraph, inputs_dict: dict[str, GradValue]) -> dict[str, GradValue]:
    """Execute an autodiff graph on the active backend or fallback interpreter.

    Args:
        graph (LogicalGraph): The JVP or VJP logical graph to execute.
        inputs_dict (dict[str, GradValue]): Mapping of input node IDs to concrete arrays.

    Returns:
        dict[str, GradValue]: Mapping of output node IDs to evaluated results.
    """
    from ml_switcheroo_compiler.interpreter import evaluate_graph

    backend = get_active_backend()
    if hasattr(backend, "execute_graph"):
        raw_out = backend.execute_graph(graph, inputs_dict)
        return dict(raw_out) if isinstance(raw_out, Mapping) else {}
    if hasattr(backend, "compile_graph"):
        compiled_fn = backend.compile_graph(graph)
        raw_comp = compiled_fn(inputs_dict)
        return dict(raw_comp) if isinstance(raw_comp, Mapping) else {}
    return evaluate_graph(graph, inputs_dict)


def jvp(
    fun: GradValue,
    primals: GradValue,
    tangents: GradValue,
    has_aux: bool = False,
) -> GradValue:
    """Compute the Jacobian-vector product of `fun` at `primals` along `tangents`.

    Args:
        fun (GradValue): Function to differentiate.
        primals (GradValue): Primal input tensor or sequence of tensors.
        tangents (GradValue): Tangent vector or sequence of tangent vectors.
        has_aux (bool): Whether `fun` returns `(output, aux)`.

    Returns:
        GradValue: `(primal_out, tangent_out)` or `((primal_out, aux), tangent_out)`.
    """
    from ml_switcheroo_compiler.transforms.autodiff import jvp as graph_jvp

    fn_callable = fun
    primals_seq = primals if isinstance(primals, (tuple, list)) else (primals,)
    tangents_seq = tangents if isinstance(tangents, (tuple, list)) else (tangents,)

    if isinstance(fun, CustomJVPFunction) and fun.jvp_rule is not None:
        return _run_custom_jvp_rule(fun, primals_seq, tangents_seq)

    with ConfigContext(eager_mode=True):
        out_eager = invoke_callable(fn_callable, *primals_seq)
        val, aux = _unpack_pair(out_eager) if has_aux else (out_eager, None)

    fun_primal = _get_fun_primal(fn_callable, has_aux)
    tensor_primals = _convert_to_tensors(primals_seq)
    forward_graph = _build_traced_logical_graph(fun_primal, tensor_primals, "jvp")

    tangent_ids: list[str] = []
    for t, p_id in zip(tangents_seq, forward_graph.inputs):
        t_id = f"tangent_{uuid.uuid4().hex[:6]}"
        raw_t = t.data if isinstance(t, Tensor) else t
        t_val = getattr(raw_t, "tolist", lambda raw_t=raw_t: raw_t)()
        t_node = LogicalNode(
            id=t_id,
            op_type="Constant",
            attributes={"value": t_val},
            shape_metadata=forward_graph.nodes[p_id].shape_metadata,
        )
        forward_graph.nodes[t_id] = t_node
        tangent_ids.append(t_id)

    jvp_graph = graph_jvp(forward_graph, forward_graph.inputs, tangent_ids, forward_graph.outputs)

    backend = get_active_backend()
    inputs_dict = {inp_id: backend.asarray(getattr(p, "data", p)) for inp_id, p in zip(forward_graph.inputs, tensor_primals)}
    outputs_dict = _execute_autodiff_graph(jvp_graph, inputs_dict)

    out_tangent_values = [outputs_dict[out_id] for out_id in jvp_graph.outputs]
    out_tan = out_tangent_values[0] if len(out_tangent_values) == 1 else tuple(out_tangent_values)

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
    flat_cot: list[GradValue],
) -> list[GradValue]:
    """Symbolically inline a gradient graph into the currently active tracing graph.

    Args:
        grad_graph (LogicalGraph): Constructed VJP gradient graph.
        primal_input_ids (list[str]): Primal input node IDs in the forward graph.
        tensor_primals (list[Tensor]): Tensor primal inputs.
        cotangent_ids_list (list[str]): Cotangent input node IDs.
        flat_cot (list[GradValue]): Flattened cotangent values.

    Returns:
        list[GradValue]: List of proxy gradient `Tensor` objects corresponding to `grad_graph` outputs.
    """
    from ml_switcheroo_compiler.core.device import Device

    id_map: dict[str, str] = {}
    for inp_id, p in zip(primal_input_ids, tensor_primals):
        p_data = getattr(p, "data", p)
        id_map[inp_id] = getattr(p_data, "id", str(p_data))

    for cot_id, cot_val in zip(cotangent_ids_list, flat_cot):
        c_data = getattr(cot_val, "data", cot_val)
        id_map[cot_id] = getattr(c_data, "id", str(c_data))

    for nid, n in grad_graph.nodes.items():
        if getattr(n, "op_type", "") != "Input":
            new_id = f"vjp_{uuid.uuid4().hex[:6]}_{nid}"
            id_map[nid] = new_id
            cloned = LogicalNode(
                id=new_id,
                op_type=n.op_type,
                inputs=[id_map.get(inp, inp) for inp in n.inputs],
                attributes=dict(getattr(n, "attributes", {})),
                shape_metadata=getattr(n, "shape_metadata", ()),
            )
            global_tracing_state.add_node(cloned)

    flat_grads: list[GradValue] = []
    for out_id in grad_graph.outputs:
        mapped_id = id_map.get(out_id, out_id)
        out_node = grad_graph.nodes.get(out_id)
        out_shape = getattr(out_node, "shape_metadata", ())
        proxy = ProxyTensor(id=mapped_id, shape=out_shape, dtype="float32")
        flat_grads.append(Tensor(proxy, TensorConfig(out_shape, DType.Float32, Device("cpu"))))
    return flat_grads


def vjp(
    fun: GradValue,
    *primals: GradValue,
    has_aux: bool = False,
) -> GradValue:
    """Compute the reverse-mode vector-Jacobian product (VJP) pullback of `fun` at `primals`.

    Args:
        fun (GradValue): Differentiable function.
        *primals (GradValue): Primal input arguments.
        has_aux (bool): Whether `fun` returns `(output, aux)`.

    Returns:
        GradValue: `(primal_out, vjp_fn)` or `((primal_out, aux), vjp_fn)`.
    """
    from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad
    from ml_switcheroo_compiler.tree_util import tree_flatten, tree_unflatten

    fn_callable = fun
    flat_primals, tree_def = tree_flatten(primals)

    def fun_flat(*flat_args: GradValue, **kwargs: GradValue) -> GradValue:
        """Unflatten `flat_args` and evaluate `fun`.

        Args:
            *flat_args (GradValue): Flattened primal leaves.
            **kwargs (GradValue): Keyword arguments.

        Returns:
            GradValue: Output of `fun(*unflat_args)`.
        """
        del kwargs
        unflat_args = tree_unflatten(tree_def, list(flat_args))
        seq_args = tuple(unflat_args) if isinstance(unflat_args, Iterable) else (unflat_args,)
        return invoke_callable(fn_callable, *seq_args)

    with ConfigContext(eager_mode=True):
        out_eager = invoke_callable(fn_callable, *primals)
        val, aux = _unpack_pair(out_eager) if has_aux else (out_eager, None)

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

    def vjp_fn(cotangent: GradValue) -> GradValue:
        """Evaluate the pullback function on upstream `cotangent`.

        Args:
            cotangent (GradValue): Upstream cotangent matching the output of `fun`.

        Returns:
            GradValue: Tuple of cotangents with respect to `primals`.
        """
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
            return tuple(unflat) if isinstance(unflat, Iterable) else (unflat,)

        backend = get_active_backend()
        inputs_dict = {inp_id: backend.asarray(getattr(p, "data", p)) for inp_id, p in zip(primal_input_ids, tensor_primals)}
        for cot_id, cot_val in zip(cotangent_ids_list, flat_cot):
            inputs_dict[cot_id] = backend.asarray(getattr(cot_val, "data", cot_val))

        outputs_dict = _execute_autodiff_graph(grad_graph, inputs_dict)
        flat_grads = [outputs_dict[out_id] for out_id in grad_graph.outputs]
        unflat = tree_unflatten(tree_def, flat_grads)
        return tuple(unflat) if isinstance(unflat, Iterable) else (unflat,)

    return ((val, aux), vjp_fn) if has_aux else (val, vjp_fn)


def hvp(
    fun: GradValue,
    primals: GradValue,
    tangents: GradValue,
    has_aux: bool = False,
    projected_tangents: Tensor | Sequence[Tensor] | None = None,
    mode: str = "forward-over-reverse",
) -> tuple[GradValue, GradValue]:
    """Compute the Hessian-vector product of `fun` at `primals` along `tangents`.

    Args:
        fun (GradValue): Scalar or projected function to differentiate twice.
        primals (GradValue): Primal point(s).
        tangents (GradValue): Tangent vector(s) matching `primals`.
        has_aux (bool): Whether `fun` returns auxiliary data `(val, aux)`.
        projected_tangents (Optional[Union[Tensor, Sequence[Tensor]]]): Optional cotangent projection vectors.
        mode (str): HVP mode (`'forward-over-reverse'` or `'reverse-over-forward'`).

    Returns:
        GradValue: Tuple of `(primal_out, hvp_out)` or `((primal_out, aux), hvp_out)`.
    """
    from ml_switcheroo_compiler.interpreter import evaluate_graph
    from ml_switcheroo_compiler.transforms.autodiff import hvp as graph_hvp

    fn_callable = fun
    primals_seq = primals if isinstance(primals, (tuple, list)) else (primals,)
    tangents_seq = tangents if isinstance(tangents, (tuple, list)) else (tangents,)

    with ConfigContext(eager_mode=True):
        if has_aux:
            val, aux = _unpack_pair(invoke_callable(fn_callable, *primals_seq))
        else:
            val = invoke_callable(fn_callable, *primals_seq)

    fun_primal = _get_fun_primal(fn_callable, has_aux)
    tensor_primals = _convert_to_tensors(primals_seq)

    block = _trace_function(fun_primal, tuple(tensor_primals), f"hvp_{uuid.uuid4().hex[:6]}")
    forward_graph = LogicalGraph(name=getattr(block, "id", getattr(block, "name", "hvp")))
    nodes_iter: Iterable[LogicalNode] = block.nodes.values() if hasattr(block.nodes, "values") else as_node_list(block.nodes)
    for node in nodes_iter:
        forward_graph.nodes[node.id] = node
    forward_graph.inputs = block.inputs
    forward_graph.outputs = block.outputs

    tangent_ids = []
    for t, p_id in zip(tangents_seq, forward_graph.inputs):
        t_id = f"tangent_{uuid.uuid4().hex[:6]}"
        raw_t = t.data if isinstance(t, Tensor) else t
        t_val = getattr(raw_t, "tolist", lambda raw_t=raw_t: raw_t)()
        t_node = LogicalNode(
            id=t_id,
            op_type="Constant",
            attributes={"value": t_val},
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
            raw_pt = pt.data if isinstance(pt, Tensor) else pt
            pt_val = getattr(raw_pt, "tolist", lambda raw_pt=raw_pt: raw_pt)()
            pt_node = LogicalNode(
                id=pt_id,
                op_type="Constant",
                attributes={"value": pt_val},
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

    inputs_dict = {inputs_id: get_active_backend().asarray(getattr(p, "data", p)) for inputs_id, p in zip(forward_graph.inputs, tensor_primals)}
    outputs_dict = evaluate_graph(hvp_graph, inputs_dict)

    out_tangent_values = [outputs_dict[out_id] for out_id in hvp_graph.outputs]
    out_tan = out_tangent_values[0] if len(out_tangent_values) == 1 else tuple(out_tangent_values)

    if has_aux:
        return ((val, aux), out_tan)
    return (val, out_tan)


def jacfwd(fun: GradValue, options: GradOptions | GradValue = None) -> TypedVariadicCallable[GradValue]:
    """Create a function computing the full Jacobian of `fun` using forward-mode JVP pushforwards.

    Args:
        fun (GradValue): Function whose Jacobian is to be computed.
        options (GradOptions | GradValue): Optional `GradOptions` configuration.

    Returns:
        TypedVariadicCallable[GradValue]: Callable returning the forward-mode Jacobian tensor.
    """
    opts = options if isinstance(options, GradOptions) else GradOptions()
    fn_callable = fun

    def wrapped(*args: GradValue, **kwargs: GradValue) -> GradValue:
        """Evaluate the forward-mode Jacobian of `fun` at `*args`.

        Args:
            *args (GradValue): Primal input arguments.
            **kwargs (GradValue): Keyword arguments passed to `fun`.

        Returns:
            GradValue: Evaluated Jacobian tensor.
        """
        with ConfigContext(eager_mode=True):
            out = invoke_callable(fn_callable, *args, **kwargs)
            out_val = _unpack_pair(out)[0] if getattr(opts, "has_aux") else out
        out_arr = get_active_backend().asarray(getattr(out_val, "data", out_val))

        arg0 = get_active_backend().asarray(getattr(args[0], "data", args[0]))
        flat_arg0 = getattr(arg0, "flatten")()

        jacobian_rows = []
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
            tangent = getattr(tangent_flat, "reshape")(getattr(arg0, "shape"))

            _, out_tangent = _unpack_pair(jvp(fn_callable, args, (tangent,), has_aux=getattr(opts, "has_aux")))
            jacobian_rows.append(getattr(get_active_backend().asarray(out_tangent), "flatten")())

        res = get_active_backend().execute_op("Stack", jacobian_rows, axis=-1)
        if getattr(out_arr, "ndim") > 0:
            return get_active_backend().execute_op(
                "Reshape",
                res,
                tuple(getattr(out_arr, "shape")) + tuple(getattr(arg0, "shape")),
            )
        return res

    return wrapped


def jacrev(fun: GradValue, options: GradOptions | GradValue = None) -> TypedVariadicCallable[GradValue]:
    """Create a function computing the full Jacobian of `fun` using reverse-mode VJP pullbacks.

    Args:
        fun (GradValue): Function whose Jacobian is to be computed.
        options (GradOptions | GradValue): Optional `GradOptions` configuration.

    Returns:
        TypedVariadicCallable[GradValue]: Callable returning the reverse-mode Jacobian tensor.
    """
    opts = options if isinstance(options, GradOptions) else GradOptions()

    def wrapped(*args: GradValue, **kwargs: GradValue) -> GradValue:
        """Evaluate the reverse-mode Jacobian of `fun` at `*args`.

        Args:
            *args (GradValue): Primal input arguments.
            **kwargs (GradValue): Unused keyword arguments.

        Returns:
            GradValue: Evaluated Jacobian tensor.
        """
        _ = kwargs
        from ml_switcheroo_compiler.tree_util import tree_flatten, tree_unflatten

        out, vjp_fn_raw = _unpack_pair(vjp(fun, *args, has_aux=getattr(opts, "has_aux")))
        vjp_fn = vjp_fn_raw
        out_val = _unpack_pair(out)[0] if getattr(opts, "has_aux") else out

        flat_out, out_tree_def = tree_flatten(out_val)
        flat_shapes = [getattr(get_active_backend().asarray(getattr(o, "data", o)), "shape") for o in flat_out]
        flat_sizes = [int(math.prod(getattr(get_active_backend().asarray(getattr(o, "data", o)), "shape"))) for o in flat_out]
        total_size = sum(flat_sizes)

        arg0 = get_active_backend().asarray(getattr(args[0], "data", args[0]))
        jacobian_rows = []

        for i in range(total_size):
            cotangent_flat = as_tensor(
                get_active_backend().execute_op(
                    "OneHot",
                    get_active_backend().asarray(i),
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
            grads = invoke_callable(vjp_fn, cotangent)
            first_grad = grads[0] if isinstance(grads, (list, tuple)) else grads
            jacobian_rows.append(getattr(get_active_backend().asarray(first_grad), "flatten")())

        res = get_active_backend().execute_op("Stack", jacobian_rows, axis=0)
        if len(flat_sizes) > 1:
            return get_active_backend().execute_op("Reshape", res, (total_size,) + tuple(getattr(arg0, "shape")))
        out_arr = get_active_backend().asarray(getattr(out_val, "data", out_val))
        if getattr(out_arr, "ndim") > 0:
            return get_active_backend().execute_op(
                "Reshape",
                res,
                tuple(getattr(out_arr, "shape")) + tuple(getattr(arg0, "shape")),
            )
        return get_active_backend().execute_op("Reshape", res, getattr(arg0, "shape"))

    return wrapped


def hessian(fun: GradValue, options: GradOptions | GradValue = None) -> TypedVariadicCallable[GradValue]:
    """Create a function computing the full Hessian matrix of scalar function `fun`.

    Args:
        fun (TypedVariadicCallable[GradValue]): Scalar-valued function to differentiate twice.
        options (GradOptions | GradValue): Optional `GradOptions` configuration.

    Returns:
        TypedVariadicCallable[GradValue]: Callable returning the Hessian tensor of shape `arg.shape + arg.shape`.
    """
    opts = options if isinstance(options, GradOptions) else GradOptions()

    def wrapped(*args: GradValue, **kwargs: GradValue) -> GradValue:
        """Evaluate the Hessian matrix of `fun` at `*args`.

        Args:
            *args (GradValue): Primal input arguments.
            **kwargs (GradValue): Unused keyword arguments.

        Returns:
            GradValue: Evaluated Hessian matrix tensor.
        """
        _ = kwargs
        arg0 = get_active_backend().asarray(getattr(args[0], "data", args[0]))
        flat_arg0 = getattr(arg0, "flatten")()

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
            tangent = getattr(tangent_flat, "reshape")(getattr(arg0, "shape"))

            _, out_tangent = _unpack_pair(hvp(fun, args, (tangent,), has_aux=getattr(opts, "has_aux")))
            hessian_rows.append(getattr(get_active_backend().asarray(out_tangent), "flatten")())

        res = get_active_backend().execute_op("Stack", hessian_rows, axis=0)
        return get_active_backend().execute_op(
            "Reshape",
            res,
            tuple(getattr(arg0, "shape")) + tuple(getattr(arg0, "shape")),
        )

    return wrapped
