"""Control flow operators dispatcher."""

from __future__ import annotations

from typing import Callable, Optional, cast  # noqa: F401

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
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


def map_fn(fn: Callable[..., EagerValue], elems: Tensor, dtype: DType | None = None) -> EagerValue:
    """Evaluate map_fn operation.

    Args:
        fn (Callable): The fn parameter.
        elems (Tensor): The elems parameter.
        dtype (Any): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        return map_fn_eager(fn, elems, dtype)
    return map_fn_tracing(fn, elems, dtype)


def pmap(func: Callable[..., EagerValue], axis_name: str | None = None) -> EagerValue:
    """Evaluate pmap operation.

    Args:
        func (Callable): The func parameter.
        axis_name (Any): The axis_name parameter.

    Returns:
        Callable: Result.
    """
    if config.eager_mode:
        return pmap_eager(func, axis_name)
    return pmap_tracing(func, axis_name)


def stop_gradient(x: Tensor | ProxyTensor) -> Tensor | ProxyTensor:
    """Evaluate stop_gradient operation.

    Args:
        x (Any): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if config.eager_mode:
        return stop_gradient_eager(x)
    return stop_gradient_tracing(x)


def assert_value(condition: EagerValue, message: str = "") -> None:
    """Evaluate assert_value operation.

    Args:
        condition (Any): The condition parameter.
        message (str): The message parameter.
    """
    if config.eager_mode:
        assert_value_eager(condition, message)
    else:
        assert_value_tracing(condition, message)


@register_op("Assert")
class AssertOp(OpDef):
    """Configuration class for assert op."""

    def infer_shape(self, condition: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Evaluate infer_shape operation.

        Args:
            condition (Any): The condition parameter.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


def fori_loop(lower: EagerValue, upper: EagerValue, body_fun: Callable[..., EagerValue], init_val: EagerValue) -> EagerValue:
    """Evaluate fori_loop operation.

    Args:
        lower (Any): The lower parameter.
        upper (Any): The upper parameter.
        body_fun (Any): The body_fun parameter.
        init_val (Any): The init_val parameter.

    Returns:
            tuple[int, ...]: Result.
    """

    def cond_fn(val: EagerValue) -> EagerValue:
        """Evaluate cond_fn operation.

        Args:
        val (Any): The val parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        i, _ = getattr(val, "__iter__")()
        return less(i, upper)

    def body_wrapper(val: EagerValue) -> EagerValue:
        """Evaluate body_wrapper operation.

        Args:
        val (Any): The val parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        i, x = getattr(val, "__iter__")()
        return add(i, 1), body_fun(i, x)

    _, res = cast("Callable[..., EagerTensorProtocol]", while_loop)(cond_fn, body_wrapper, (lower, init_val))
    return cast(EagerValue, res)


def map(fn: Callable[..., EagerValue], elems: Tensor) -> EagerValue:
    """Apply a function iteratively over elements of a tensor.

    Args:
        fn (Callable): The function to apply to each element.
        elems (Tensor): The input tensor containing elements.

    Returns:
        Tensor: The stacked output elements.
    """
    return map_fn(fn, elems)


def vectorized_map(fn: Callable[..., EagerValue], elems: Tensor) -> EagerValue:
    """Apply a function in a vectorized manner over elements of a tensor.

    Args:
        fn (Callable): The function to apply to each element in parallel.
        elems (Tensor): The input tensor to process.

    Returns:
        Tensor: The vectorized result tensor.
    """
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", vmap)(fn)(elems))


def switch(index: Tensor, branches: EagerValue, *operands: EagerValue) -> EagerValue:
    """Select a specific branch function based on an index tensor.

    Args:
        index (Tensor): The index parameter.
        branches (list): The branches parameter.
        *operands (Any): Positional args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    if not branches:
        raise ValueError("branches cannot be empty")

    def build_tree(start: int, end: int) -> EagerValue:
        """Evaluate build_tree operation.

        Args:
        start (int): The start parameter.
        end (int): The end parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        if end - start == 1:
            return cast(EagerValue, getattr(branches, "__getitem__")(start)(*operands))

        mid = (start + end) // 2

        def true_fn() -> EagerValue:
            """Evaluate true_fn operation.

            Returns:
            tuple[int, ...]: Result.
            """
            return build_tree(start, mid)

        def false_fn() -> EagerValue:
            """Evaluate false_fn operation.

            Returns:
            tuple[int, ...]: Result.
            """
            return build_tree(mid, end)

        return cond(less(index, mid), true_fn, false_fn)

    return build_tree(0, getattr(branches, "__len__")())


def custom_gradient(func: Callable[..., EagerValue]) -> EagerValue:
    """Create a wrapper that enables custom gradient definitions.

    Args:
        func (Callable): The function providing both value and gradient definitions.

    Returns:
        Callable: A wrapped version of the function that can integrate with autograd.
    """

    def wrapper(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Evaluate wrapper operation.

        Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        val, grad_fn = getattr(func(*getattr(args, "__iter__")(), **kwargs), "__iter__")()
        if config.eager_mode:
            return cast(EagerValue, val)

        import uuid

        from ml_switcheroo_compiler.ops.base import OpDef, register_op
        from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
        from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp

        op_name = f"CustomGradient_{uuid.uuid4().hex}"

        @register_op(op_name)
        class DynamicCustomGradOp(OpDef):
            """Dynamic custom gradient operation definition."""

            op_name_class = op_name

            def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Evaluate infer_shape operation.

                Args:
                    *args (Any): Positional args.
                    **kwargs (Any): Keyword args.

                Returns:
                tuple[int, ...]: Result.
                """
                return getattr(val, "shape", ())

        @register_vjp(op_name)
        def dyn_vjp(graph: EagerValue, node: EagerValue, cotangent: str) -> tuple[str, ...]:
            """VJP function for dynamic custom gradient.

            Args:
                graph (Any): The IR graph.
                node (Any): The node.
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
    """Evaluate case operation.

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

    def _build_case(idx: int) -> EagerValue:
        """Evaluate _build_case operation.

        Args:
        idx (int): The idx parameter.

        Returns:
        Callable: Result.
        """
        if idx == len(pred_fn_pairs):
            return default if default is not None else lambda: None
        pred, fn = pred_fn_pairs[idx]
        return lambda: cast("Callable[..., EagerTensorProtocol]", cond)(pred, fn, _build_case(idx + 1))

    return cast(Optional[Tensor], cast("Callable[..., EagerTensorProtocol]", _build_case)(0)())


def switch_case(branch_index: Tensor, branch_fns: EagerValue, default: EagerValue = None) -> EagerValue:
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
            return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", default)())
        raise ValueError("switch_case requires at least one branch or a default")

    pred_fn_pairs = []
    # Sort keys to ensure deterministic ordering (not strictly necessary but good practice)
    for key in sorted(getattr(branch_fns, "keys")()):
        key_tensor = Tensor(key, TensorConfig((), branch_index.dtype, branch_index.device))
        pred = equal(branch_index, key_tensor)
        pred_fn_pairs.append((pred, getattr(branch_fns, "__getitem__")(key)))

    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", case)(pred_fn_pairs, default))


@register_op("DebugInfs")
class DebugInfs(OpDef):
    """Operation definition for debugging infinite values in tensors."""

    op_name = "DebugInfs"

    def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Evaluate infer_shape operation.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0] if args else ()


@register_op("DebugNans")
class DebugNans(OpDef):
    """Operation definition for debugging NaN values in tensors."""

    op_name = "DebugNans"

    def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Evaluate infer_shape operation.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

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

    def infer_shape(self, index: EagerTensorProtocol, branches: EagerTensorProtocol, *operands: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Evaluate infer_shape operation.

        Args:
            index (Any): Index.
            branches (Any): Branches.
            *operands (Any): Operands.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Scan")
class ScanOp(OpDef):
    """Operation definition for performing iterative scan computations."""

    op_name = "Scan"

    def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Evaluate infer_shape operation.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("AssociativeScan")
class AssociativeScan(OpDef):
    """Operation definition for performing associative scan computations."""

    op_name = "AssociativeScan"

    def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Evaluate infer_shape operation.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0] if args else ()


def associative_scan(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate associative_scan operation.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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


def scan_bind(f: Callable[..., EagerValue], xs: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Bind arguments to a scanning function prior to execution.

    Args:
        f (Any): The function to bind.
        xs (Any): The elements to scan over.
        *args (Any): Additional positional arguments.
        **kwargs (Any): Additional keyword arguments.

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
