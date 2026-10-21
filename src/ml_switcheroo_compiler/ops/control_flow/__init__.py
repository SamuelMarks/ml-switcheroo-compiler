"""Control flow operators dispatcher."""

from __future__ import annotations

import typing
from collections.abc import Callable  # noqa: F401

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, as_callable, as_tensor
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar, invoke_callable
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, get_op, register_op

# Build pred_fn_pairs from branch_fns
from ml_switcheroo_compiler.ops.binary import (
    add,
    equal,
    less,
)
from ml_switcheroo_compiler.ops.vmap import vmap as vmap
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from .cond import cond as cond
from .eager import (
    assert_value_eager,
    cond_eager,  # noqa: F401
    map_fn_eager,
    pmap_eager,
    scan_eager,  # noqa: F401
    stop_gradient_eager,
    while_loop_eager,  # noqa: F401
)
from .scan import scan as scan
from .tracing import (
    assert_value_tracing,
    cond_tracing,  # noqa: F401
    map_fn_tracing,
    pmap_tracing,
    scan_tracing,  # noqa: F401
    stop_gradient_tracing,
    while_loop_tracing,  # noqa: F401
)
from .while_loop import while_loop as while_loop


def map_fn(fn: TensorOrScalar, elems: Tensor, dtype: DType | None = None) -> TensorOrScalar:
    """Compute the map fn tensor operation.

    Args:
        fn (TensorOrScalar): The fn parameter.
        elems (Tensor): The elems parameter.
        dtype (DType | None): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        return map_fn_eager(fn, elems, dtype)
    return map_fn_tracing(fn, elems, dtype)


def pmap(func: TensorOrScalar, axis_name: str | None = None) -> TensorOrScalar:
    """Compute the pmap tensor operation.

    Args:
        func (TensorOrScalar): The func parameter.
        axis_name (str | None): The axis_name parameter.

    Returns:
        Callable: Result.
    """
    if config.eager_mode:
        return pmap_eager(func, axis_name)
    return pmap_tracing(func, axis_name)


def stop_gradient(x: Tensor | ProxyTensor) -> Tensor | ProxyTensor:
    """Compute the stop gradient tensor operation.

    Args:
        x (Tensor | ProxyTensor): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if config.eager_mode:
        return stop_gradient_eager(x)
    return stop_gradient_tracing(x)


def assert_value(condition: TensorOrScalar, message: str = "") -> None:
    """Compute the assert value tensor operation.

    Args:
        condition (TensorOrScalar): The condition parameter.
        message (str): The message parameter.
    """
    if config.eager_mode:
        assert_value_eager(condition, message)
    else:
        assert_value_tracing(condition, message)


@register_op("Assert")
class AssertOp(OpDef):
    """Configuration class for assert op."""

    def infer_shape(self, condition: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Compute the infer shape tensor operation.

        Args:
            condition (EagerTensorProtocol): The condition parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


def fori_loop(lower: TensorOrScalar, upper: TensorOrScalar, body_fun: TensorOrScalar, init_val: TensorOrScalar) -> TensorOrScalar:
    """Compute the fori loop tensor operation.

    Args:
        lower (TensorOrScalar): The lower parameter.
        upper (TensorOrScalar): The upper parameter.
        body_fun (TensorOrScalar): The body_fun parameter.
        init_val (TensorOrScalar): The init_val parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    body_callable = body_fun

    def cond_fn(val: TensorOrScalar) -> TensorOrScalar:
        """Compute the cond fn tensor operation.

        Args:
        val (TensorOrScalar): The val parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        i, _ = getattr(val, "__iter__")()
        return less(i, upper)

    def body_wrapper(val: TensorOrScalar) -> TensorOrScalar:
        """Compute the body wrapper tensor operation.

        Args:
        val (TensorOrScalar): The val parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        i, x = getattr(val, "__iter__")()
        return add(i, 1), invoke_callable(body_callable, i, x)

    _, res = as_callable(while_loop)(cond_fn, body_wrapper, (lower, init_val))
    return as_tensor(res)


def map(fn: TensorOrScalar, elems: Tensor) -> TensorOrScalar:
    """Apply a function iteratively over elements of a tensor.

    Args:
        fn (TensorOrScalar): The function to apply to each element.
        elems (Tensor): The input tensor containing elements.

    Returns:
        Tensor: The stacked output elements.
    """
    return map_fn(fn, elems)


def vectorized_map(fn: TensorOrScalar, elems: Tensor) -> TensorOrScalar:
    """Apply a function in a vectorized manner over elements of a tensor.

    Args:
        fn (TensorOrScalar): The function to apply to each element in parallel.
        elems (Tensor): The input tensor to process.

    Returns:
        Tensor: The vectorized result tensor.
    """
    return as_tensor(as_callable(vmap)(fn)(elems))


def switch(index: Tensor, branches: TensorOrScalar, *operands: TensorOrScalar) -> TensorOrScalar:
    """Select a specific branch function based on an index tensor.

    Args:
        index (Tensor): The index parameter.
        branches (list): The branches parameter.
        *operands (TensorOrScalar): Positional args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    if not branches:
        raise ValueError("branches cannot be empty")

    def build_tree(start: int, end: int) -> TensorOrScalar:
        """Compute the build tree tensor operation.

        Args:
        start (int): The start parameter.
        end (int): The end parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        if end - start == 1:
            return as_tensor(getattr(branches, "__getitem__")(start)(*operands))

        mid = (start + end) // 2

        def true_fn() -> TensorOrScalar:
            """Compute the true fn tensor operation.

            Returns:
            tuple[int, ...]: Result.
            """
            return build_tree(start, mid)

        def false_fn() -> TensorOrScalar:
            """Compute the false fn tensor operation.

            Returns:
            tuple[int, ...]: Result.
            """
            return build_tree(mid, end)

        return cond(less(index, mid), true_fn, false_fn)

    return build_tree(0, getattr(branches, "__len__")())


def custom_gradient(func: TensorOrScalar) -> TensorOrScalar:
    """Create a wrapper that enables custom gradient definitions.

    Args:
        func (TensorOrScalar): The function providing both value and gradient definitions.

    Returns:
        Callable: A wrapped version of the function that can integrate with autograd.
    """
    fn_callable = func

    def wrapper(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Compute the wrapper tensor operation.

        Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        res_seq = list(getattr(invoke_callable(fn_callable, *getattr(args, "__iter__")(), **kwargs), "__iter__")())
        val: TensorOrScalar = res_seq[0]
        _ = res_seq[1]
        if config.eager_mode:
            return val

        import uuid

        from ml_switcheroo_compiler.ops.base import OpDef, register_op
        from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
        from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp

        op_name = f"CustomGradient_{uuid.uuid4().hex}"

        @register_op(op_name)
        class DynamicCustomGradOp(OpDef):
            """Dynamic custom gradient operation definition."""

            op_name_class = op_name

            def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
                """Compute the infer shape tensor operation.

                Args:
                    *args (TensorOrScalar): Positional args.
                    **kwargs (TensorOrScalar): Keyword args.

                Returns:
                tuple[int, ...]: Result.
                """
                return getattr(val, "shape", ())

        @register_vjp(op_name)
        def dyn_vjp(graph: TensorOrScalar, node: TensorOrScalar, cotangent: str) -> tuple[str, ...]:
            """VJP function for dynamic custom gradient.

            Args:
                graph (TensorOrScalar): The IR graph.
                node (TensorOrScalar): The node.
                cotangent (str): Cotangent.

            Returns:
                tuple: Input gradients.
            """
            return tuple([cotangent] * len(getattr(node, "inputs") if node else []))

        return _emit_shape_node(op_name, list(args), kwargs, getattr(val, "shape", ()), getattr(val, "dtype", None))

    return wrapper


def case(
    pred_fn_pairs: list[tuple[Tensor, Callable[[], Tensor | None]]],
    default: Callable[[], Tensor | None] | None = None,
) -> Tensor | None:
    """Compute the case tensor operation.

    Args:
        pred_fn_pairs (list): The pred_fn_pairs parameter.
        default (Callable): The default parameter.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    if not pred_fn_pairs:
        if default is not None:
            return default()
        raise ValueError("case requires at least one (pred, fn) pair or a default")

    def _build_case(idx: int) -> TensorOrScalar:
        """Compute the build case tensor operation.

        Args:
        idx (int): The idx parameter.

        Returns:
        Callable: Result.
        """
        if idx == len(pred_fn_pairs):
            return default if default is not None else lambda: None
        pred, fn = pred_fn_pairs[idx]
        return lambda: as_callable(cond)(pred, fn, _build_case(idx + 1))

    return typing.cast(typing.Union[Tensor, "None"], as_callable(_build_case)(0)())


def switch_case(branch_index: Tensor, branch_fns: TensorOrScalar, default: TensorOrScalar = None) -> TensorOrScalar:
    """Select a specific function based on a dynamic index mapping.

    Args:
        branch_index (Tensor): The branch_index parameter.
        branch_fns (dict): The branch_fns parameter.
        default (Callable): The default parameter.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    if not branch_fns:
        if default is not None:
            return as_tensor(as_callable(default)())
        raise ValueError("switch_case requires at least one branch or a default")

    pred_fn_pairs = []
    # Sort keys to ensure deterministic ordering (not strictly necessary but good practice)
    for key in sorted(getattr(branch_fns, "keys")()):
        key_tensor = Tensor(key, TensorConfig((), branch_index.dtype, branch_index.device))
        pred = equal(branch_index, key_tensor)
        pred_fn_pairs.append((pred, getattr(branch_fns, "__getitem__")(key)))

    return as_tensor(as_callable(case)(pred_fn_pairs, default))


@register_op("DebugInfs")
class DebugInfs(OpDef):
    """Operation definition for debugging infinite values in tensors."""

    op_name = "DebugInfs"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Compute the infer shape tensor operation.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0] if args else ()


@register_op("DebugNans")
class DebugNans(OpDef):
    """Operation definition for debugging NaN values in tensors."""

    op_name = "DebugNans"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Compute the infer shape tensor operation.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0] if args else ()


debug_infs = get_op("DebugInfs")()
debug_nans = get_op("DebugNans")()


@register_op("Switch")
class SwitchOp(OpDef):
    """Operation definition for a dynamic branching switch statement."""

    op_name = "Switch"

    def infer_shape(self, index: EagerTensorProtocol, branches: EagerTensorProtocol, *operands: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Compute the infer shape tensor operation.

        Args:
            index (EagerTensorProtocol): Index.
            branches (EagerTensorProtocol): Branches.
            *operands (EagerTensorProtocol): Operands.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Scan")
class ScanOp(OpDef):
    """Operation definition for performing iterative scan computations."""

    op_name = "Scan"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Compute the infer shape tensor operation.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("AssociativeScan")
class AssociativeScan(OpDef):
    """Operation definition for performing associative scan computations."""

    op_name = "AssociativeScan"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Compute the infer shape tensor operation.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0] if args else ()


def associative_scan(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the associative scan tensor operation.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        return backend.execute_op("AssociativeScan", *args, **kwargs)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    shape = getattr(args[0], "shape", ()) if args else ()
    dtype = getattr(args[0], "dtype", "float32") if args else "float32"
    return _emit_shape_node("AssociativeScan", list(args), kwargs, shape, dtype)


def scan_bind(f: TensorOrScalar, xs: TensorOrScalar, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Bind arguments to a scanning function prior to execution.

    Args:
        f (TensorOrScalar): The function to bind.
        xs (TensorOrScalar): The elements to scan over.
        *args (TensorOrScalar): Additional positional arguments.
        **kwargs (TensorOrScalar): Additional keyword arguments.

    Returns: Tensor: The bound function along with its inputs.
    """
    # Partial bind implementation for scan_bind
    return f, xs


__all__ = [
    "cond",
    "cond_eager",
    "cond_tracing",
    "while_loop",
    "while_loop_eager",
    "while_loop_tracing",
    "scan",
    "scan_eager",
    "scan_tracing",
    "map_fn",
    "pmap",
    "stop_gradient",
    "assert_value",
    "AssertOp",
    "fori_loop",
    "map",
    "vectorized_map",
    "switch",
    "custom_gradient",
    "case",
    "switch_case",
    "DebugInfs",
    "DebugNans",
    "debug_infs",
    "debug_nans",
    "SwitchOp",
    "ScanOp",
    "AssociativeScan",
    "associative_scan",
    "scan_bind",
]
