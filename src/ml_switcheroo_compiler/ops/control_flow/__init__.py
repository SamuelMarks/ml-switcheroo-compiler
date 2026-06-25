"""Control flow operators dispatcher."""

from __future__ import annotations

from typing import Callable, Any
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.vmap import vmap as vmap

from .eager import (
    cond_eager,
    while_loop_eager,
    scan_eager,
    map_fn_eager,
    pmap_eager,
    stop_gradient_eager,
    assert_value_eager,
)
from .tracing import (
    cond_tracing,
    while_loop_tracing,
    scan_tracing,
    map_fn_tracing,
    pmap_tracing,
    stop_gradient_tracing,
    assert_value_tracing,
)


def cond(pred: Tensor, true_fn: Callable[[], Any], false_fn: Callable[[], Any]) -> object:  # noqa: ANN401
    """Docstring."""
    if config.eager_mode:
        return cond_eager(pred, true_fn, false_fn)
    return cond_tracing(pred, true_fn, false_fn)


def while_loop(
    cond_fn: Callable[[Any], Tensor], body_fn: Callable[[Any], Any], init_val: object
) -> object:  # noqa: ANN401
    """Docstring."""
    if config.eager_mode:
        return while_loop_eager(cond_fn, body_fn, init_val)
    return while_loop_tracing(cond_fn, body_fn, init_val)


def scan(
    f: Callable[[Any, Any], tuple[Any, Any]], init: object, xs: object, length: int | None = None
) -> tuple[Any, Any]:  # noqa: ANN401
    """Docstring."""
    if config.eager_mode:
        return scan_eager(f, init, xs, length)
    return scan_tracing(f, init, xs, length)


def map_fn(fn: Callable[[Any], Any], elems: Tensor, dtype: DType | None = None) -> Tensor:  # noqa: ANN401
    """Docstring."""
    if config.eager_mode:
        return map_fn_eager(fn, elems, dtype)
    return map_fn_tracing(fn, elems, dtype)


def pmap(func: Callable, axis_name: str | None = None) -> Callable:
    """Docstring."""
    if config.eager_mode:
        return pmap_eager(func, axis_name)
    return pmap_tracing(func, axis_name)


def stop_gradient(x: object) -> object:
    """Docstring."""
    if config.eager_mode:
        return stop_gradient_eager(x)
    return stop_gradient_tracing(x)


def assert_value(condition: object, message: str = "") -> None:
    """Docstring."""
    if config.eager_mode:
        assert_value_eager(condition, message)
    else:
        assert_value_tracing(condition, message)


@register_op("Assert")
class AssertOp(OpDef):
    """Docstring."""

    def infer_shape(self, condition: object, **kwargs: object) -> object:
        """Docstring."""
        return ()


def fori_loop(
    lower: object, upper: object, body_fun: Callable[[object, object], object], init_val: object
) -> object:
    """fori_loop implementation."""
    from ml_switcheroo_compiler.ops.binary import less, add

    def cond_fn(val: object) -> object:
        """Function docstring.

        Args:
        val: Arg.
        """
        i, _ = val
        return less(i, upper)

    def body_wrapper(val: object) -> object:
        """Function docstring.

        Args:
        val: Arg.
        """
        i, x = val
        return add(i, 1), body_fun(i, x)

    _, res = while_loop(cond_fn, body_wrapper, (lower, init_val))
    return res


def map(fn: Callable[[Any], Any], elems: Tensor) -> Tensor:
    """Map over elems."""
    return map_fn(fn, elems)


def vectorized_map(fn: Callable[[Any], Any], elems: Tensor) -> Tensor:
    """Vectorized map."""
    return vmap(fn)(elems)


def switch(index: Tensor, branches: list[Callable], *operands: object) -> object:
    """Switch operator."""
    if not branches:  # pragma: no branch
        raise ValueError("branches cannot be empty")  # pragma: no cover

    def build_tree(start: int, end: int) -> object:
        """Function docstring.

        Args:
        start: Arg.
        end: Arg.
        """
        if end - start == 1:
            return branches[start](*operands)

        mid = (start + end) // 2

        def true_fn() -> object:
            """Function docstring."""
            return build_tree(start, mid)

        def false_fn() -> object:
            """Function docstring."""
            return build_tree(mid, end)

        from ml_switcheroo_compiler.ops.binary import less

        return cond(less(index, mid), true_fn, false_fn)

    return build_tree(0, len(branches))


def custom_gradient(func: Callable) -> Callable:
    """Custom gradient decorator."""

    def wrapper(*args: object, **kwargs: object) -> object:
        """Function docstring.

        Args:
        args: Arg.
        kwargs: Arg.
        """
        # A simple stub that just calls the function.
        # True custom gradients require backend-specific registration which we mock.
        val, grad_fn = func(*args, **kwargs)
        return val

    return wrapper


def case(pred_fn_pairs: list[tuple[Tensor, Callable]], default: Callable = None) -> object:
    """Execute case.

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
        if idx == len(pred_fn_pairs):
            return default if default is not None else lambda: None
        pred, fn = pred_fn_pairs[idx]
        return lambda: cond(pred, fn, _build_case(idx + 1))

    return _build_case(0)()


def switch_case(
    branch_index: Tensor, branch_fns: dict[int, Callable], default: Callable = None
) -> object:
    """Execute switch_case.

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

    # Build pred_fn_pairs from branch_fns
    from ml_switcheroo_compiler.ops.binary import equal

    pred_fn_pairs = []
    # Sort keys to ensure deterministic ordering (not strictly necessary but good practice)
    for key in sorted(branch_fns.keys()):
        key_tensor = Tensor(key, TensorConfig((), branch_index.dtype, branch_index.device))
        pred = equal(branch_index, key_tensor)
        pred_fn_pairs.append((pred, branch_fns[key]))

    return case(pred_fn_pairs, default)
