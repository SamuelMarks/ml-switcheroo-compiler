"""Provides higher-order control flow primitives for tracing and eager execution.

This module implements functional control flow operators such as conditional branching,
while loops, scanning, vectorizing maps (vmap), and parallel maps (pmap). These
operators support both eager execution (using NumPy/Python loops) and tracing into an
intermediate representation (IR) graph for compilation
"""

from __future__ import annotations


import uuid
from typing import Any, Callable

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRBlock
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.ops.vmap import vmap
from ml_switcheroo_compiler.tracing import ProxyTensor, _tracer


def _trace_true_branch(true_fn: Callable) -> IRBlock:
    return _trace_function(true_fn, (), "true_branch")


def _trace_false_branch(false_fn: Callable) -> IRBlock:
    return _trace_function(false_fn, (), "false_branch")


def cond(
    pred: Tensor,
    true_fn: Callable[[], Any],
    false_fn: Callable[[], Any],
) -> object:
    """Conditionally executes one of two functions based on a boolean predicate.

    In eager mode, this directly evaluates the predicate and runs the corresponding
    branch function. In tracing mode, it traces both branches into subgraphs and
    emits an 'If' logical node

    Args:
        pred (Tensor): A scalar boolean tensor determining which branch to execute
        true_fn (Callable[[], Any]): The function to execute if the predicate is True
        false_fn (Callable[[], Any]): The function to execute if the predicate is False

    Returns:
    object: The result of executing the selected branch (a Tensor or structure of
    Tensors)

    Raises:
    RuntimeError: If called outside of a tracing context when eager mode is
    disabled
    """
    if config.eager_mode:
        if bool(pred.data):
            return true_fn()
        return false_fn()
    if not _tracer.is_tracing:
        msg = "Cannot emit Cond node outside of a tracing context."
        raise RuntimeError(msg)

    true_graph = _trace_true_branch(true_fn)
    false_graph = _trace_false_branch(false_fn)

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="If",
        inputs=[pred.data.id],
        attributes={
            "then_branch": true_graph,
            "else_branch": false_graph,
        },
        shape_metadata=(),
    )
    _tracer.add_node(node)

    proxy = ProxyTensor(id=out_id, shape=(), dtype="float32")
    return Tensor(proxy, TensorConfig((), DType.Float32, pred.device))


def _trace_while_cond(cond_fn: Callable, args: tuple[Tensor, ...]) -> IRBlock:
    return _trace_function(cond_fn, args, "cond")


def _trace_while_body(body_fn: Callable, args: tuple[Tensor, ...]) -> IRBlock:
    return _trace_function(body_fn, args, "body")


def _approximate_while_return(init_val: object, out_id: str) -> object:
    if isinstance(init_val, Tensor):
        proxy = ProxyTensor(
            id=out_id,
            shape=init_val.shape,
            dtype=init_val.dtype.value,
        )
        return Tensor(proxy, TensorConfig(init_val.shape, init_val.dtype, init_val.device))
    return init_val


def while_loop(
    cond_fn: Callable[[Any], Tensor],
    body_fn: Callable[[Any], Any],
    init_val: object,
) -> object:
    """Executes a body function repeatedly while a condition function evaluates to True.

    In eager mode, this runs a standard Python while loop. In tracing mode, it
    traces
    both the condition and body functions into subgraphs and emits a 'Loop' logical
    node

    Args:
        cond_fn (Callable[[Any], Tensor]): A function that takes the current state and
        returns a scalar boolean Tensor
        body_fn (Callable[[Any], Any]): A function that takes the current state and
        returns the updated state of the same structure and shape
        init_val (object): The initial state, which can be a Tensor or a sequence of
        Tensors

    Returns:
    object: The final state after the loop terminates

    Raises:
    RuntimeError: If called outside of a tracing context when eager mode is
    disabled
    """
    if config.eager_mode:
        val = init_val
        while bool(cond_fn(val).data):
            val = body_fn(val)
        return val
    if not _tracer.is_tracing:
        msg = "Cannot emit While node outside of a tracing context."
        raise RuntimeError(msg)

    import typing

    args = (
        (init_val,)
        if isinstance(init_val, Tensor)
        else tuple(typing.cast(typing.Iterable[Tensor], init_val))
    )

    cond_graph = _trace_while_cond(cond_fn, args)
    body_graph = _trace_while_body(body_fn, args)

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Loop",
        inputs=[a.data.id for a in args],
        attributes={
            "body": body_graph,
            "cond": cond_graph,
        },
        shape_metadata=(),
    )
    _tracer.add_node(node)

    return _approximate_while_return(init_val, out_id)


def scan(
    f: Callable[[Any, Any], tuple[Any, Any]],
    init: object,
    xs: object,
    length: int | None = None,
) -> tuple[Any, Any]:
    """Scans a function over the leading dimension of sequence tensors while carrying.

    state

    In eager mode, this sequentially applies the function along the 0-th dimension
    of
    the input sequence, accumulating the results. In tracing mode, it emits a 'Scan'
    logical node

    Args:
        f (Callable[[Any, Any], tuple[Any, Any]]): A function mapping `(carry, x)` to
        `(new_carry, y)`
        init (object): The initial carry state
        xs (object): The sequence of values to scan over, typically a Tensor
        length (int | None): An optional length to scan

    Returns:
    tuple[Any, Any]: A tuple containing the final carry state and the stacked
        outputs (ys) from each step

    Raises:
    RuntimeError: If called outside of a tracing context when eager mode is
    disabled
    """
    if config.eager_mode:
        return _scan_eager(f, init, xs, length)

    return _scan_tracing(f, init, xs)


def _scan_eager(f: Callable, init: object, xs: object, length: int | None) -> tuple[object, object]:
    """Execute _scan_eager.

    Args:
        f (Any): Argument f.
        init (Any): Argument init.
        xs (Any): Argument xs.
        length (Any): Argument length.

    Returns:
    Any: The result.
    """
    carry = init
    ys = []
    # xs is assumed to be a Tensor with a batch dimension 0
    scan_length = length if length is not None else (xs.shape[0] if xs is not None else 0)
    for i in range(scan_length):
        # Extract slice
        x = (
            Tensor(xs.data[i], TensorConfig(xs.shape[1:], xs.dtype, xs.device))
            if xs is not None
            else None
        )
        carry, y = f(carry, x)
        ys.append(y.data if hasattr(y, "data") else y)

    out_tensor = _stack_scan_outputs(ys, init, y if scan_length > 0 else init)
    return carry, out_tensor


def _stack_scan_outputs(ys: list, init: object, last_y: object) -> Tensor:
    """Execute _stack_scan_outputs.

    Args:
        ys (Any): Argument ys.
        init (Any): Argument init.
        last_y (Any): Argument last_y.

    Returns:
    Any: The result.
    """
    if len(ys) > 0 and isinstance(ys[0], tuple):
        stacked_ys = get_active_backend().execute_op("Stack", ys)
        return Tensor(
            stacked_ys,
            TensorConfig(
                stacked_ys.shape,
                last_y.dtype if hasattr(last_y, "dtype") else init.dtype,
                last_y.device if hasattr(last_y, "device") else init.device,
            ),
        )
    else:
        stacked_ys = get_active_backend().array(ys)
        from ml_switcheroo_compiler.core.dtype import DType

        return Tensor(
            stacked_ys,
            TensorConfig(stacked_ys.shape, DType(str(stacked_ys.dtype)), config.default_device),
        )


def _scan_tracing(f: Callable, init: object, xs: object) -> tuple[object, object]:
    """Execute _scan_tracing.

    Args:
        f (Callable): The function to scan.
        init (Any): Argument init.
        xs (Any): Argument xs.

    Returns:
    Any: The result.
    """
    if not _tracer.is_tracing:
        msg = "Cannot emit Scan node outside of a tracing context."
        raise RuntimeError(msg)

    # Need a dummy x for tracing `f`
    x_shape = xs.shape[1:] if xs is not None and len(xs.shape) > 0 else ()
    proxy_x = ProxyTensor(id="dummy_x", shape=x_shape, dtype=xs.dtype.value)
    dummy_x = Tensor(proxy_x, TensorConfig(x_shape, xs.dtype, xs.device))

    body_graph = _trace_function(f, (init, dummy_x), "scan_body")

    def _flatten_inputs(obj: object) -> list[str]:
        if isinstance(obj, Tensor):
            return [obj.data.id]
        elif isinstance(obj, (list, tuple)):
            res = []
            for item in obj:
                res.extend(_flatten_inputs(item))
            return res
        return []

    init_ids = _flatten_inputs(init)

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Scan",
        inputs=init_ids + [xs.data.id],
        attributes={"body": body_graph},
        shape_metadata=(),
    )
    _tracer.add_node(node)

    # Return a dummy proxy
    proxy = ProxyTensor(id=out_id, shape=xs.shape, dtype=xs.dtype.value)
    out_tensor = Tensor(proxy, TensorConfig(xs.shape, xs.dtype, xs.device))

    # We must construct a structure matching `init` for the returned carry
    # However, currently it just returns `init` as the carry, which might not correctly propagate traced IDs
    # But for now, returning `init` is what the original code did.
    return init, out_tensor


def _create_pmap_dummy_args(args: tuple[object, ...]) -> list[object]:
    dummy_args = []
    for a in args:
        if isinstance(a, Tensor):
            new_shape = a.shape[1:] if len(a.shape) > 0 else ()
            proxy = ProxyTensor(id=str(uuid.uuid4()), shape=new_shape, dtype=a.dtype.value)
            dummy_args.append(Tensor(proxy, TensorConfig(new_shape, a.dtype, a.device)))
        else:
            dummy_args.append(a)
    return dummy_args


def _emit_pmap_node(
    func: Callable, args: tuple[object, ...], dummy_args: list[object], axis_name: str | None
) -> object:
    import typing

    body_graph = _trace_function(
        func, typing.cast(tuple[Tensor, ...], tuple(dummy_args)), "pmap_body"
    )

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Pmap",
        inputs=[typing.cast(typing.Any, a).data.id for a in args if isinstance(a, Tensor)],
        attributes={"axis_name": axis_name, "body": body_graph},
        shape_metadata=(),
    )
    _tracer.add_node(node)

    arg = typing.cast(typing.Any, args[0])
    proxy = ProxyTensor(id=out_id, shape=arg.shape, dtype=arg.dtype.value)
    return Tensor(proxy, TensorConfig(arg.shape, arg.dtype, arg.device))


def pmap(func: Callable, axis_name: str | None = None) -> Callable:
    """Creates a parallel mapped version of a function for distributed execution.

    In eager mode, this falls back to a vectorized map (vmap). In tracing mode,
    it records a 'Pmap' logical node in the IR

    Args:
        func (Callable): The function to map in parallel
        axis_name (str | None): The name of the mapped axis, used for collective
        operations within the function. Defaults to None

    Returns:
    Callable: A parallel mapped version of the input function
    """

    def wrapped(*args: object) -> object:
        """Wrapped.

        Args:
            *args (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if config.eager_mode:
            return vmap(func)(*args)
        if not _tracer.is_tracing:
            msg = "Cannot emit Pmap outside of a tracing context."
            raise RuntimeError(msg)

        dummy_args = _create_pmap_dummy_args(args)
        return _emit_pmap_node(func, args, dummy_args, axis_name)

    return wrapped


def stop_gradient(x: object) -> object:
    """Stops the flow of gradients during reverse-mode differentiation.

    Args:
        x (object): The input x tensor.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    import uuid

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    if config.eager_mode or not _tracer.is_tracing:
        return x

    if isinstance(x, Tensor) and isinstance(x.data, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(
            id=out_id,
            op_type="StopGradient",
            inputs=[x.data.id],
            shape_metadata=x.shape,
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
        return Tensor(proxy, TensorConfig(x.shape, x.dtype, x.device))
    if isinstance(x, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(
            id=out_id,
            op_type="StopGradient",
            inputs=[x.id],
            shape_metadata=x.shape,
        )
        _tracer.add_node(node)
        return ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype)

    return x


@register_op("Assert")
class AssertOp(OpDef):
    """An operation definition for asserting a condition within the computational graph."""

    def infer_shape(self, condition: object, **kwargs: object) -> object:
        """Infer the output shape of the operation."""
        return ()


def assert_value(condition: object, message: str = "") -> None:
    """Assert a condition. In eager mode, records it. In tracing mode, emits an Assert node."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.assertions import record_assertion
    from ml_switcheroo_compiler.tracing import _tracer
    import uuid
    from ml_switcheroo_ir import LogicalNode
    from ml_switcheroo_compiler.core.tensor import Tensor

    if config.eager_mode or not _tracer.is_tracing:
        record_assertion(condition, message)
        return

    inp_id = condition.data.id if isinstance(condition, Tensor) else condition.id
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Assert",
        inputs=[inp_id],
        attributes={"message": message},
        shape_metadata=(),
    )
    _tracer.add_node(node)
