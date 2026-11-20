"""Custom forward-mode JVP rule wrapper (`custom_jvp` and `CustomJVPFunction`)."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Sequence

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import OpValue, is_variadic_callable
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.callback_registry import register_custom_rule_handle
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from .utils import (
    DifferentiableFn,
    PrimalTree,
    _extract_shape,
    _IdDataProtocol,
    _unwrap_data,
    as_primal_tree,
)

CustomJVPRule = typing.Union[DifferentiableFn, LogicalGraph, None]  # noqa: UP007
_EqT = typing.TypeVar("_EqT")


def _to_constant_attr(val: OpValue) -> AttributeValue:
    """Normalize a runtime constant operand into a serializable `AttributeValue`.

    Args:
        val (OpValue): Constant scalar, sequence, or array buffer.

    Returns:
        AttributeValue: Serializable IR attribute representation.
    """
    from ml_switcheroo_ir import ArrayBufferProtocol

    if val is None or isinstance(val, (int, float, bool, str, bytes, ArrayBufferProtocol)):
        return val
    if isinstance(val, (list, tuple)):
        return [_to_constant_attr(x) for x in val]
    return str(val)


@typing.runtime_checkable
class _GraphHolderProtocol(typing.Protocol):
    """Protocol for objects exposing a `.graph` attribute."""

    graph: LogicalGraph


@typing.runtime_checkable
class _DTypeAttrProtocol(typing.Protocol):
    """Protocol for objects exposing a `.dtype` attribute."""

    dtype: OpValue


@typing.runtime_checkable
class _DTypeValueProtocol(typing.Protocol):
    """Protocol for DType enum values exposing `.value`."""

    value: str


def _has_graph_holder(val: OpValue) -> typing.TypeGuard[_GraphHolderProtocol]:
    """Check whether `val` exposes `graph`.

    Args:
        val (OpValue): Candidate rule callable.

    Returns:
        bool: True if `val` has `graph`.
    """
    return isinstance(val, _GraphHolderProtocol)


def _has_dtype_attr(val: OpValue) -> typing.TypeGuard[_DTypeAttrProtocol]:
    """Check whether `val` exposes `dtype`.

    Args:
        val (OpValue): Candidate array or tensor.

    Returns:
        bool: True if `val` has `dtype`.
    """
    return isinstance(val, _DTypeAttrProtocol)


def _has_dtype_value(val: OpValue) -> typing.TypeGuard[_DTypeValueProtocol]:
    """Check whether `val` exposes `value`.

    Args:
        val (OpValue): Candidate dtype object.

    Returns:
        bool: True if `val` has `value`.
    """
    return isinstance(val, _DTypeValueProtocol)


def _has_id_data_attr(val: OpValue) -> typing.TypeGuard[_IdDataProtocol]:
    """Check whether `val` exposes `id`.

    Args:
        val (OpValue): Candidate proxy data.

    Returns:
        bool: True if `val` has `id`.
    """
    return isinstance(val, _IdDataProtocol)


def _resolve_dtype_value(val: OpValue | PrimalTree) -> str:
    """Resolve the string dtype value from `val.dtype` if present, else `'float32'`.

    Args:
        val (OpValue | PrimalTree): Candidate tensor or array.

    Returns:
        str: Resolved dtype string.
    """
    raw: OpValue = val
    return str(raw.dtype.value) if _has_dtype_attr(raw) and _has_dtype_value(raw.dtype) else "float32"


class CustomJVPFunction:
    """Wrap a function to allow defining custom Jacobian-vector product (JVP) rules."""

    def __init__(self, fun: DifferentiableFn) -> None:
        """Initialize `CustomJVPFunction`.

        Args:
            fun (DifferentiableFn): Base primal function being wrapped.
        """
        self.fun: DifferentiableFn = fun
        self.jvp_rule: CustomJVPRule = None
        self.jvp_subgraph: LogicalGraph | None = None

    def defjvp(
        self,
        jvp_rule: CustomJVPRule,
        jvp_subgraph: LogicalGraph | None = None,
    ) -> CustomJVPRule:
        """Define the custom Jacobian-vector product (JVP) rule.

        Args:
            jvp_rule (CustomJVPRule): Custom JVP rule callable or `LogicalGraph`.
            jvp_subgraph (LogicalGraph | None): Optional declarative forward/tangent IR subgraph.

        Returns:
            CustomJVPRule: The registered `jvp_rule`.
        """
        self.jvp_rule = jvp_rule
        raw_rule: OpValue = jvp_rule
        if jvp_subgraph is not None:
            self.jvp_subgraph = jvp_subgraph
        elif isinstance(jvp_rule, LogicalGraph):
            self.jvp_subgraph = jvp_rule
        elif _has_graph_holder(raw_rule) and isinstance(raw_rule.graph, LogicalGraph):
            self.jvp_subgraph = raw_rule.graph
        else:
            self.jvp_subgraph = None
        return jvp_rule

    def defjvp_subgraph(self, jvp_subgraph: LogicalGraph) -> None:
        """Attach a declarative forward/tangent IR subgraph directly.

        Args:
            jvp_subgraph (LogicalGraph): Custom forward/tangent IR subgraph.
        """
        self.jvp_subgraph = jvp_subgraph

    def __eq__(self, other: _EqT) -> bool:
        """Check equality with another function or `CustomJVPFunction`.

        Args:
            other (_EqT): Other object to compare.

        Returns:
            bool: True if equal, False otherwise.
        """
        if isinstance(other, CustomJVPFunction):
            return bool(self.fun == other.fun)
        return bool(self.fun == other)

    def __call__(self, *args: PrimalTree | OpValue, **kwargs: PrimalTree | OpValue) -> PrimalTree:
        """Execute the function, emitting a `CustomJVP` node when tracing with a registered rule.

        Args:
            *args (PrimalTree | OpValue): Positional arguments.
            **kwargs (PrimalTree | OpValue): Keyword arguments.

        Returns:
            PrimalTree: Computed output tensor or value.
        """
        if config.eager_mode or not global_tracing_state.is_tracing or self.jvp_rule is None:
            _callable_fn_1: OpValue = self.fun
            assert is_variadic_callable(_callable_fn_1)
            return as_primal_tree(_callable_fn_1(*args, **kwargs))

        tensor_args = [a for a in args if isinstance(a, Tensor)]
        if not tensor_args:
            _callable_fn_2: OpValue = self.fun
            assert is_variadic_callable(_callable_fn_2)
            return as_primal_tree(_callable_fn_2(*args, **kwargs))

        with ConfigContext(eager_mode=True):
            _callable_fn_3: OpValue = self.fun
            assert is_variadic_callable(_callable_fn_3)
            sample_out = _callable_fn_3(*args, **kwargs)

        out_shape = _extract_shape(sample_out)
        out_dtype = _resolve_dtype_value(sample_out)
        device = tensor_args[0].device

        in_ids: list[str] = []
        for a in tensor_args:
            raw_data: OpValue = _unwrap_data(a)
            if _has_id_data_attr(raw_data):
                in_ids.append(str(raw_data.id))
            else:
                cid = str(uuid.uuid4())
                arr = _unwrap_data(a)
                c_node = LogicalNode(
                    id=cid,
                    op_type="Constant",
                    inputs=[],
                    attributes={"value": _to_constant_attr(arr)},
                    shape_metadata=_extract_shape(arr),
                )
                global_tracing_state.add_node(c_node)
                in_ids.append(cid)

        out_id = f"custom_jvp_{uuid.uuid4().hex[:6]}"
        subgraphs: dict[str, LogicalGraph] = {}
        if self.jvp_subgraph is not None:
            subgraphs["jvp"] = self.jvp_subgraph

        rule_attr: AttributeValue = (
            self.jvp_rule
            if isinstance(self.jvp_rule, LogicalGraph) or self.jvp_rule is None
            else register_custom_rule_handle(self.jvp_rule)
        )
        node_attrs: dict[str, AttributeValue] = {
            "jvp_rule": rule_attr,
            "fun": register_custom_rule_handle(self.fun),
        }
        if self.jvp_subgraph is not None:
            node_attrs["jvp_graph"] = self.jvp_subgraph
            node_attrs["custom_forward_subgraph"] = self.jvp_subgraph
        node = LogicalNode(
            id=out_id,
            op_type="CustomJVP",
            inputs=in_ids,
            attributes=node_attrs,
            subgraphs=subgraphs,
            shape_metadata=out_shape,
        )
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype)
        return Tensor(proxy, TensorConfig(out_shape, DType(out_dtype), device))


def custom_jvp(fun: DifferentiableFn) -> CustomJVPFunction:
    """Wrap a function to allow defining custom Jacobian-vector product (JVP) rules.

    Args:
        fun (DifferentiableFn): The original primal function.

    Returns:
        CustomJVPFunction: Wrapped callable supporting `.defjvp(...)`.
    """
    return CustomJVPFunction(fun)


def _run_custom_jvp_rule(
    fun: CustomJVPFunction,
    primals_seq: Sequence[PrimalTree | OpValue],
    tangents_seq: Sequence[PrimalTree | OpValue],
) -> tuple[PrimalTree, PrimalTree]:
    """Execute a registered custom JVP rule in eager mode.

    Args:
        fun (CustomJVPFunction): `CustomJVPFunction` with a registered `jvp_rule`.
        primals_seq (Sequence[PrimalTree | OpValue]): Sequence of primal arguments.
        tangents_seq (Sequence[PrimalTree | OpValue]): Sequence of tangent arguments.

    Returns:
        tuple[PrimalTree, PrimalTree]: Tuple of `(primal_output, tangent_output)`.
    """
    with ConfigContext(eager_mode=True):
        assert fun.jvp_rule is not None
        try:
            _callable_fn_4: OpValue = fun.jvp_rule
            assert is_variadic_callable(_callable_fn_4)
            res = _callable_fn_4(primals_seq, tangents_seq)
        except TypeError:
            _callable_fn_5: OpValue = fun.jvp_rule
            assert is_variadic_callable(_callable_fn_5)
            res = _callable_fn_5(*primals_seq, *tangents_seq)

        if isinstance(res, (tuple, list)) and len(res) == 2:
            val = as_primal_tree(res[0])
            out_tan = as_primal_tree(res[1])
        else:
            val = fun(*primals_seq)
            out_tan = as_primal_tree(res)
    return val, out_tan
