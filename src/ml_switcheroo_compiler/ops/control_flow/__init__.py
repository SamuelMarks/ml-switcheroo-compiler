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
    """Evaluate and process the cond operation.

    Args:
        pred (Tensor): Required parameter for pred.
        true_fn (Callable): Required parameter for true_fn.
        false_fn (Callable): Required parameter for false_fn.

    Returns:
        object: The evaluated or processed output.
    """
    if config.eager_mode:
        return cond_eager(pred, true_fn, false_fn)
    return cond_tracing(pred, true_fn, false_fn)


def while_loop(cond_fn: Callable[[Any], Tensor], body_fn: Callable[[Any], Any], init_val: object) -> object:
    """Evaluate and process the while loop operation.

    Args:
        cond_fn (Callable): Required parameter for cond_fn.
        body_fn (Callable): Required parameter for body_fn.
        init_val (object): Required parameter for init_val.

    Returns:
        object: The evaluated or processed output.
    """
    if config.eager_mode:
        return while_loop_eager(cond_fn, body_fn, init_val)
    return while_loop_tracing(cond_fn, body_fn, init_val)


def scan(f: Callable[[Any, Any], tuple[Any, Any]], init: object, xs: object, length: int | None = None) -> tuple[Any, Any]:
    """Evaluate and process the scan operation.

    Args:
        f (Callable): Required parameter for f.
        init (object): Required parameter for init.
        xs (object): Required parameter for xs.
        length (Any): Required parameter for length.

    Returns:
        tuple: The evaluated or processed output.
    """
    if config.eager_mode:
        return scan_eager(f, init, xs, length)
    return scan_tracing(f, init, xs, length)


def map_fn(fn: Callable[[Any], Any], elems: Tensor, dtype: DType | None = None) -> Tensor:
    """Evaluate and process the map fn operation.

    Args:
        fn (Callable): Required parameter for fn.
        elems (Tensor): Required parameter for elems.
        dtype (Any): Required parameter for dtype.

    Returns:
        Tensor: The evaluated or processed output.
    """
    if config.eager_mode:
        return map_fn_eager(fn, elems, dtype)
    return map_fn_tracing(fn, elems, dtype)


def pmap(func: Callable, axis_name: str | None = None) -> Callable:
    """Evaluate and process the pmap operation.

    Args:
        func (Callable): Required parameter for func.
        axis_name (Any): Required parameter for axis_name.

    Returns:
        Callable: The evaluated or processed output.
    """
    if config.eager_mode:
        return pmap_eager(func, axis_name)
    return pmap_tracing(func, axis_name)


def stop_gradient(x: object) -> object:
    """Evaluate and process the stop gradient operation.

    Args:
        x (object): Required parameter for x.

    Returns:
        object: The evaluated or processed output.
    """
    if config.eager_mode:
        return stop_gradient_eager(x)
    return stop_gradient_tracing(x)


def assert_value(condition: object, message: str = "") -> None:
    """Evaluate and process the assert value operation.

    Args:
        condition (object): Required parameter for condition.
        message (str): Required parameter for message.

    Returns:
        Any: The evaluated or processed output.
    """
    if config.eager_mode:
        assert_value_eager(condition, message)
    else:
        assert_value_tracing(condition, message)


@register_op("Assert")
class AssertOp(OpDef):
    """Configuration class for assert op."""

    def infer_shape(self, condition: object, **kwargs: object) -> object:
        """Evaluate and process the infer shape operation.

        Args:
            condition (object): Required parameter for condition.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            object: The evaluated or processed output.
        """
        return ()


def fori_loop(lower: object, upper: object, body_fun: Callable[[object, object], object], init_val: object) -> object:
    """Evaluate a bounded integer loop.

    Args:
        lower (object): The lower bound for the loop.
        upper (object): The upper bound for the loop.
        body_fun (Callable): The function to execute on each iteration.
        init_val (object): The initial value for the loop state.

    Returns:
        object: The final state after the loop terminates.
    """

    def cond_fn(val: object) -> object:
        """Evaluate and process the cond fn operation.

        Args:
            val (object): Required parameter for val.

        Returns:
            object: The evaluated or processed output.
        """
        i, _ = val
        return less(i, upper)

    def body_wrapper(val: object) -> object:
        """Evaluate and process the body wrapper operation.

        Args:
            val (object): Required parameter for val.

        Returns:
            object: The evaluated or processed output.
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
        index (Tensor): The zero-based index of the branch to select.
        branches (list): A list of callable branch functions.
        *operands (object): Arguments to pass to the selected branch function.

    Returns:
        object: The output of the selected branch function.
    """
    if not branches:
        raise ValueError("branches cannot be empty")

    def build_tree(start: int, end: int) -> object:
        """Evaluate and process the build tree operation.

        Args:
            start (int): Required parameter for start.
            end (int): Required parameter for end.

        Returns:
            object: The evaluated or processed output.
        """
        if end - start == 1:
            return branches[start](*operands)

        mid = (start + end) // 2

        def true_fn() -> object:
            """Evaluate and process the true fn operation.

            Returns:
                object: The evaluated or processed output.
            """
            return build_tree(start, mid)

        def false_fn() -> object:
            """Evaluate and process the false fn operation.

            Returns:
                object: The evaluated or processed output.
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
        """Evaluate and process the wrapper operation."""
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
            op_name_class = op_name

            def infer_shape(self, *args: object, **kwargs: object) -> object:
                """Infer shape."""
                return getattr(val, "shape", ())

        @register_vjp(op_name)
        def dyn_vjp(graph: object, node: object, cotangent: str) -> tuple:
            return tuple([cotangent] * len(node.inputs if node else []))

        return _emit_shape_node(op_name, list(args), kwargs, getattr(val, "shape", ()), getattr(val, "dtype", None))

    return wrapper


def case(pred_fn_pairs: list[tuple[Tensor, Callable]], default: Callable = None) -> object:
    """Evaluate multiple predicates and run the corresponding function of the first true predicate.

    Args:
        pred_fn_pairs: List of (predicate, callable) pairs.
        default: Optional callable for default case.

    Returns:
        The result of the evaluated callable.
    """
    if not pred_fn_pairs:
        if default is not None:
            return default()
        raise ValueError("case requires at least one (pred, fn) pair or a default")

    def _build_case(idx: int) -> Callable:
        """Evaluate and process the build case operation.

        Args:
            idx (int): Required parameter for idx.

        Returns:
            Callable: The evaluated or processed output.
        """
        if idx == len(pred_fn_pairs):
            return default if default is not None else lambda: None
        pred, fn = pred_fn_pairs[idx]
        return lambda: cond(pred, fn, _build_case(idx + 1))

    return _build_case(0)()


def switch_case(branch_index: Tensor, branch_fns: dict[int, Callable], default: Callable = None) -> object:
    """Select a specific function based on a dynamic index mapping.

    Args:
        branch_index: The branch index tensor.
        branch_fns: Dictionary mapping indices to callables.
        default: Optional callable for default case.

    Returns:
        The result of the evaluated callable.
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
        """Determine the shape for the debugging infinite values operation.

        Args:
            *args (object): Positional arguments, typically the input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The resulting shape of the operation.
        """
        return args[0] if args else ()


@register_op("DebugNans")
class DebugNans(OpDef):
    """Operation definition for debugging NaN values in tensors."""

    op_name = "DebugNans"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Determine the shape for the debugging NaN values operation.

        Args:
            *args (object): Positional arguments, typically the input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The resulting shape of the operation.
        """
        return args[0] if args else ()


debug_infs = get_op("DebugInfs")()
debug_nans = get_op("DebugNans")()


@register_op("Switch")
class SwitchOp(OpDef):
    """Operation definition for a dynamic branching switch statement."""

    op_name = "Switch"

    def infer_shape(self, index: object, branches: object, *operands: object, **kwargs: object) -> object:
        """Determine the shape for the switch operation.

        Args:
            index (object): The index determining the branch.
            branches (object): The list of branch functions.
            *operands (object): The input operands.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The resulting shape of the operation.
        """
        return ()


@register_op("Scan")
class ScanOp(OpDef):
    """Operation definition for performing iterative scan computations."""

    op_name = "Scan"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Determine the shape for the scan operation.

        Args:
            *args (object): The input arguments including sequences to scan.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The resulting shape of the operation.
        """
        return ()


@register_op("AssociativeScan")
class AssociativeScan(OpDef):
    """Operation definition for performing associative scan computations."""

    op_name = "AssociativeScan"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Determine the shape for the associative scan operation.

        Args:
            *args (object): Positional arguments, typically the input sequences.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The resulting shape of the operation.
        """
        return args[0] if args else ()


def associative_scan(*args: object, **kwargs: object) -> object:
    """Evaluate and process the associative scan operation.

    Args:
        *args (object): Positional arguments for the scan.
        **kwargs (object): Keyword arguments for the scan.

    Returns:
        object: The resulting tensor.
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
