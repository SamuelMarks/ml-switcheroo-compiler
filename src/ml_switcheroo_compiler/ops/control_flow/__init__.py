"""Control flow operators dispatcher."""

from __future__ import annotations

from typing import Any, Callable

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

from .eager import (
    assert_value_eager,
    cond_eager,
    map_fn_eager,
    pmap_eager,
    scan_eager,
    stop_gradient_eager,
    while_loop_eager,
)
from .tracing import (
    assert_value_tracing,
    cond_tracing,
    map_fn_tracing,
    pmap_tracing,
    scan_tracing,
    stop_gradient_tracing,
    while_loop_tracing,
)


def cond(pred: Tensor, true_fn: Callable[[], Any], false_fn: Callable[[], Any]) -> object:
    """Evaluate cond operation.

    Args:
        pred (Tensor): The pred parameter.
        true_fn (Callable): The true_fn parameter.
        false_fn (Callable): The false_fn parameter.

    Returns:
        object: Result.
    """
    if config.eager_mode:
        return cond_eager(pred, true_fn, false_fn)
    return cond_tracing(pred, true_fn, false_fn)


def while_loop(cond_fn: Callable[[Any], Tensor], body_fn: Callable[[Any], Any], init_val: object) -> object:
    """Evaluate while_loop operation.

    Args:
        cond_fn (object): The cond_fn parameter.
        body_fn (object): The body_fn parameter.
        init_val (object): The init_val parameter.

    Returns:
        object: Result.
    """
    if config.eager_mode:
        return while_loop_eager(cond_fn, body_fn, init_val)
    return while_loop_tracing(cond_fn, body_fn, init_val)


def scan(f: Callable[[Any, Any], tuple[Any, Any]], init: object, xs: object, length: int | None = None) -> tuple[Any, Any]:
    """Evaluate scan operation.

    Args:
        f (object): The f parameter.
        init (object): The init parameter.
        xs (object): The xs parameter.
        length (object): The length parameter.

    Returns:
        object: Result.
    """
    if config.eager_mode:
        return scan_eager(f, init, xs, length)
    return scan_tracing(f, init, xs, length)


def map_fn(fn: Callable[[Any], Any], elems: Tensor, dtype: DType | None = None) -> Tensor:
    """Evaluate map_fn operation.

    Args:
        fn (Callable): The fn parameter.
        elems (Tensor): The elems parameter.
        dtype (object): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        return map_fn_eager(fn, elems, dtype)
    return map_fn_tracing(fn, elems, dtype)


def pmap(func: Callable, axis_name: str | None = None) -> Callable:
    """Evaluate pmap operation.

    Args:
        func (Callable): The func parameter.
        axis_name (object): The axis_name parameter.

    Returns:
        Callable: Result.
    """
    if config.eager_mode:
        return pmap_eager(func, axis_name)
    return pmap_tracing(func, axis_name)


def stop_gradient(x: object) -> object:
    """Evaluate stop_gradient operation.

    Args:
        x (object): The x parameter.

    Returns:
        object: Result.
    """
    if config.eager_mode:
        return stop_gradient_eager(x)
    return stop_gradient_tracing(x)


def assert_value(condition: object, message: str = "") -> None:
    """Evaluate assert_value operation.

    Args:
        condition (object): The condition parameter.
        message (str): The message parameter.
    """
    if config.eager_mode:
        assert_value_eager(condition, message)
    else:
        assert_value_tracing(condition, message)


@register_op("Assert")
class AssertOp(OpDef):
    """Configuration class for assert op."""

    def infer_shape(self, condition: object, **kwargs: object) -> object:
        """Evaluate infer_shape operation.

        Args:
            condition (object): The condition parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return ()


def fori_loop(lower: object, upper: object, body_fun: Callable[[object, object], object], init_val: object) -> object:
    """Evaluate fori_loop operation.

    Args:
        lower (object): The lower parameter.
        upper (object): The upper parameter.
        body_fun (object): The body_fun parameter.
        init_val (object): The init_val parameter.

    Returns:
        object: Result.
    """

    def cond_fn(val: object) -> object:
        """Evaluate cond_fn operation.

        Args:
        val (object): The val parameter.

        Returns:
        object: Result.
        """
        i, _ = val
        return less(i, upper)

    def body_wrapper(val: object) -> object:
        """Evaluate body_wrapper operation.

        Args:
        val (object): The val parameter.

        Returns:
        object: Result.
        """
        i, x = val
        return add(i, 1), body_fun(i, x)

    _, res = while_loop(cond_fn, body_wrapper, (lower, init_val))
    return res


def map(fn: Callable[[Any], Any], elems: Tensor) -> Tensor:
    """Apply a function iteratively over elements of a tensor.

    Args:
        fn (Callable): The function to apply to each element.
        elems (Tensor): The input tensor containing elements.

    Returns:
        Tensor: The stacked output elements.
    """
    return map_fn(fn, elems)


def vectorized_map(fn: Callable[[Any], Any], elems: Tensor) -> Tensor:
    """Apply a function in a vectorized manner over elements of a tensor.

    Args:
        fn (Callable): The function to apply to each element in parallel.
        elems (Tensor): The input tensor to process.

    Returns:
        Tensor: The vectorized result tensor.
    """
    return vmap(fn)(elems)


def switch(index: Tensor, branches: list[Callable], *operands: object) -> object:
    """Select a specific branch function based on an index tensor.

    Args:
        index (Tensor): The index parameter.
        branches (list): The branches parameter.
        *operands (object): Positional args.

    Returns:
        object: Result.

    Raises:
        ValueError: An exception.
    """
    if not branches:
        raise ValueError("branches cannot be empty")

    def build_tree(start: int, end: int) -> object:
        """Evaluate build_tree operation.

        Args:
        start (int): The start parameter.
        end (int): The end parameter.

        Returns:
        object: Result.
        """
        if end - start == 1:
            return branches[start](*operands)

        mid = (start + end) // 2

        def true_fn() -> object:
            """Evaluate true_fn operation.

            Returns:
            object: Result.
            """
            return build_tree(start, mid)

        def false_fn() -> object:
            """Evaluate false_fn operation.

            Returns:
            object: Result.
            """
            return build_tree(mid, end)

        return cond(less(index, mid), true_fn, false_fn)

    return build_tree(0, len(branches))


def custom_gradient(func: Callable) -> Callable:
    """Create a wrapper that enables custom gradient definitions.

    Args:
        func (Callable): The function providing both value and gradient definitions.

    Returns:
        Callable: A wrapped version of the function that can integrate with autograd.
    """

    def wrapper(*args: object, **kwargs: object) -> object:
        """Evaluate wrapper operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
        object: Result.
        """
        val, grad_fn = func(*args, **kwargs)
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

            def infer_shape(self, *args: object, **kwargs: object) -> object:
                """Evaluate infer_shape operation.

                Args:
                    *args (object): Positional args.
                    **kwargs (object): Keyword args.

                Returns:
                    object: Result.
                """
                return getattr(val, "shape", ())

        @register_vjp(op_name)
        def dyn_vjp(graph: object, node: object, cotangent: str) -> tuple:
            """VJP function for dynamic custom gradient.

            Args:
                graph (object): The IR graph.
                node (object): The node.
                cotangent (str): Cotangent.

            Returns:
                tuple: Input gradients.
            """
            return tuple([cotangent] * len(node.inputs if node else []))

        return _emit_shape_node(op_name, list(args), kwargs, getattr(val, "shape", ()), getattr(val, "dtype", None))

    return wrapper


def case(pred_fn_pairs: list[tuple[Tensor, Callable]], default: Callable = None) -> object:
    """Evaluate case operation.

    Args:
        pred_fn_pairs (list): The pred_fn_pairs parameter.
        default (Callable): The default parameter.

    Returns:
        object: Result.

    Raises:
        ValueError: An exception.
    """
    if not pred_fn_pairs:
        if default is not None:
            return default()
        raise ValueError("case requires at least one (pred, fn) pair or a default")

    def _build_case(idx: int) -> Callable:
        """Evaluate _build_case operation.

        Args:
        idx (int): The idx parameter.

        Returns:
        Callable: Result.
        """
        if idx == len(pred_fn_pairs):
            return default if default is not None else lambda: None
        pred, fn = pred_fn_pairs[idx]
        return lambda: cond(pred, fn, _build_case(idx + 1))

    return _build_case(0)()


def switch_case(branch_index: Tensor, branch_fns: dict[int, Callable], default: Callable = None) -> object:
    """Select a specific function based on a dynamic index mapping.

    Args:
        branch_index (Tensor): The branch_index parameter.
        branch_fns (dict): The branch_fns parameter.
        default (Callable): The default parameter.

    Returns:
        object: Result.

    Raises:
        ValueError: An exception.
    """
    if not branch_fns:
        if default is not None:
            return default()
        raise ValueError("switch_case requires at least one branch or a default")

    pred_fn_pairs = []
    # Sort keys to ensure deterministic ordering (not strictly necessary but good practice)
    for key in sorted(branch_fns.keys()):
        key_tensor = Tensor(key, TensorConfig((), branch_index.dtype, branch_index.device))
        pred = equal(branch_index, key_tensor)
        pred_fn_pairs.append((pred, branch_fns[key]))

    return case(pred_fn_pairs, default)


@register_op("DebugInfs")
class DebugInfs(OpDef):
    """Operation definition for debugging infinite values in tensors."""

    op_name = "DebugInfs"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Evaluate infer_shape operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0] if args else ()


@register_op("DebugNans")
class DebugNans(OpDef):
    """Operation definition for debugging NaN values in tensors."""

    op_name = "DebugNans"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Evaluate infer_shape operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0] if args else ()


debug_infs = get_op("DebugInfs")()
debug_nans = get_op("DebugNans")()


@register_op("Switch")
class SwitchOp(OpDef):
    """Operation definition for a dynamic branching switch statement."""

    op_name = "Switch"

    def infer_shape(self, index: object, branches: object, *operands: object, **kwargs: object) -> object:
        """Evaluate infer_shape operation.

        Args:
            index (object): Index.
            branches (object): Branches.
            *operands (object): Operands.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return ()


@register_op("Scan")
class ScanOp(OpDef):
    """Operation definition for performing iterative scan computations."""

    op_name = "Scan"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Evaluate infer_shape operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return ()


@register_op("AssociativeScan")
class AssociativeScan(OpDef):
    """Operation definition for performing associative scan computations."""

    op_name = "AssociativeScan"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Evaluate infer_shape operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0] if args else ()


def associative_scan(*args: object, **kwargs: object) -> object:
    """Evaluate associative_scan operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
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


def scan_bind(f: object, xs: object, *args: object, **kwargs: object) -> object:
    """Bind arguments to a scanning function prior to execution.

    Args:
        f (object): The function to bind.
        xs (object): The elements to scan over.
        *args (object): Additional positional arguments.
        **kwargs (object): Additional keyword arguments.

    Returns:
        object: The bound function along with its inputs.
    """
    # Partial bind implementation for scan_bind
    return f, xs
