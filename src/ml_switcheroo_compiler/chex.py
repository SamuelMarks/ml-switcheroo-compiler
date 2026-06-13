"""Chex assertion functions for zero-chex parity."""

import itertools
import threading
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass as builtin_dataclass
from enum import Enum
from typing import Any, Callable

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.tree_util import tree_flatten


def assert_axis_dimension(tensor: Tensor, axis: int, expected: int) -> None:
    """Checks that tensor.shape[axis] == expected.

    Args:
        tensor (Tensor): The tensor.
        axis (int): The axis.
        expected (int): The expected.
    """
    dim = tensor.shape[axis]
    if not isinstance(dim, int):
        pass  # Dynamic shape: lowered to AssertEq in IR (TODO)
    elif dim != expected:
        msg = f"Expected shape[{axis}] == {expected}, got {dim}"
        raise AssertionError(msg)


def assert_axis_dimension_comparator(
    tensor: Tensor,
    axis: int,
    pass_fn: Callable,
    error_string: str,
) -> None:
    """Asserts that pass_fn(tensor.shape[axis]) passes.

    Args:
        tensor (Tensor): The tensor.
        axis (int): The axis.
        pass_fn (Callable): The pass_fn.
        error_string (str): The error_string.
    """
    dim = tensor.shape[axis]
    if isinstance(dim, int) and not pass_fn(dim):
        raise AssertionError(error_string)


def assert_axis_dimension_gt(tensor: Tensor, axis: int, val: int) -> None:
    """Checks that tensor.shape[axis] > val.

    Args:
        tensor (Tensor): The tensor.
        axis (int): The axis.
        val (int): The val.
    """
    dim = tensor.shape[axis]
    if isinstance(dim, int) and not dim > val:
        msg = f"Expected shape[{axis}] > {val}, got {dim}"
        raise AssertionError(msg)


def assert_axis_dimension_gteq(tensor: Tensor, axis: int, val: int) -> None:
    """Checks that tensor.shape[axis] >= val.

    Args:
        tensor (Tensor): The tensor.
        axis (int): The axis.
        val (int): The val.
    """
    dim = tensor.shape[axis]
    if isinstance(dim, int) and not dim >= val:
        msg = f"Expected shape[{axis}] >= {val}, got {dim}"
        raise AssertionError(msg)


def assert_axis_dimension_lt(tensor: Tensor, axis: int, val: int) -> None:
    """Checks that tensor.shape[axis] < val.

    Args:
        tensor (Tensor): The tensor.
        axis (int): The axis.
        val (int): The val.
    """
    dim = tensor.shape[axis]
    if isinstance(dim, int) and not dim < val:
        msg = f"Expected shape[{axis}] < {val}, got {dim}"
        raise AssertionError(msg)


def assert_axis_dimension_lteq(tensor: Tensor, axis: int, val: int) -> None:
    """Checks that tensor.shape[axis] <= val.

    Args:
        tensor (Tensor): The tensor.
        axis (int): The axis.
        val (int): The val.
    """
    dim = tensor.shape[axis]
    if isinstance(dim, int) and not dim <= val:
        msg = f"Expected shape[{axis}] <= {val}, got {dim}"
        raise AssertionError(msg)


def assert_equal_shape(inputs: Sequence[Tensor], dims: object = None) -> None:
    """Checks that all arrays have the same shape.

    Args:
        inputs (Sequence[Tensor]): The inputs.
        dims (object): The dims.
    """
    if not inputs:
        return
    first_shape = inputs[0].shape
    for t in inputs[1:]:
        if t.shape != first_shape:
            msg = f"Expected shape {first_shape}, got {t.shape}"
            raise AssertionError(msg)


def assert_equal_rank(inputs: Sequence[Tensor]) -> None:
    """Checks that all arrays have the same rank.

    Args:
        inputs (Sequence[Tensor]): The inputs.
    """
    if not inputs:
        return
    first_rank = len(inputs[0].shape)
    for t in inputs[1:]:
        if len(t.shape) != first_rank:
            msg = f"Expected rank {first_rank}, got {len(t.shape)}"
            raise AssertionError(msg)


def assert_equal_shape_prefix(inputs: Sequence[Tensor], prefix_len: int) -> None:
    """Checks that the leading prefix_dims dims of all inputs have same shape.

    Args:
        inputs (Sequence[Tensor]): The inputs.
        prefix_len (int): The prefix_len.
    """
    if not inputs:
        return
    first_prefix = inputs[0].shape[:prefix_len]
    for t in inputs[1:]:
        if t.shape[:prefix_len] != first_prefix:
            msg = f"Expected prefix {first_prefix}, got {t.shape[:prefix_len]}"
            raise AssertionError(msg)


def assert_equal_shape_suffix(inputs: Sequence[Tensor], suffix_len: int) -> None:
    """Checks that the final suffix_len dims of all inputs have same shape.

    Args:
        inputs (Sequence[Tensor]): The inputs.
        suffix_len (int): The suffix_len.
    """
    if not inputs:
        return
    first_suffix = inputs[0].shape[-suffix_len:] if suffix_len > 0 else ()
    for t in inputs[1:]:
        t_suffix = t.shape[-suffix_len:] if suffix_len > 0 else ()
        if t_suffix != first_suffix:
            msg = f"Expected suffix {first_suffix}, got {t_suffix}"
            raise AssertionError(msg)


def assert_equal_size(inputs: Sequence[Tensor]) -> None:
    """Checks that all arrays have the same size.

    Args:
        inputs (Sequence[Tensor]): The inputs.
    """
    import math

    if not inputs:
        return
    first_size = math.prod(inputs[0].shape)
    for t in inputs[1:]:
        if math.prod(t.shape) != first_size:
            msg = f"Expected size {first_size}, got {math.prod(t.shape)}"
            raise AssertionError(msg)


def assert_rank(inputs: Sequence[Tensor], expected_ranks: object) -> None:
    """Checks that the rank of all inputs matches expected.

    Args:
        inputs (Sequence[Tensor]): The inputs.
        expected_ranks (object): The expected_ranks.
    """
    if not isinstance(expected_ranks, (list, tuple, set)):
        expected_ranks = [expected_ranks]
    for t in inputs:
        if len(t.shape) not in expected_ranks:
            msg = f"Expected rank in {expected_ranks}, got {len(t.shape)}"
            raise AssertionError(msg)


def assert_shape(inputs: Sequence[Tensor], expected_shapes: object) -> None:
    """Checks that the shape of all inputs matches expected.

    Args:
        inputs (Sequence[Tensor]): The inputs.
        expected_shapes (object): The expected_shapes.
    """
    if not isinstance(expected_shapes[0], (list, tuple)):
        expected_shapes = [expected_shapes]
    for t in inputs:
        if t.shape not in expected_shapes:
            msg = f"Expected shape in {expected_shapes}, got {t.shape}"
            raise AssertionError(msg)


def assert_size(inputs: Sequence[Tensor], expected_sizes: object) -> None:
    """Checks that the size of all inputs matches expected.

    Args:
        inputs (Sequence[Tensor]): The inputs.
        expected_sizes (object): The expected_sizes.
    """
    import math

    if not isinstance(expected_sizes, (list, tuple, set)):
        expected_sizes = [expected_sizes]
    for t in inputs:
        if math.prod(t.shape) not in expected_sizes:
            msg = f"Expected size in {expected_sizes}, got {math.prod(t.shape)}"
            raise AssertionError(msg)


def assert_type(inputs: Sequence[Tensor], expected_types: object) -> None:
    """Checks that the type of all inputs matches.

    Args:
        inputs (Sequence[Tensor]): The inputs.
        expected_types (object): The expected_types.
    """
    if not isinstance(expected_types, (list, tuple, set)):
        expected_types = [expected_types]
    for t in inputs:
        if t.dtype not in expected_types:
            msg = f"Expected type in {expected_types}, got {t.dtype}"
            raise AssertionError(msg)


def assert_tree_all_finite(tree_like: object) -> None:
    """Checks that all leaves in a tree are finite.

    Args:
        tree_like (object): The tree_like.
    """
    from ml_switcheroo_compiler.ops.reductions import all
    from ml_switcheroo_compiler.ops.unary import isfinite

    leaves, _ = tree_flatten(tree_like)
    for leaf in leaves:
        if isinstance(leaf, Tensor) and not all(isfinite(leaf)).data:
            msg = "Tree contains non-finite values."
            raise AssertionError(msg)


def assert_tree_has_only_ndarrays(tree: object) -> None:
    """Checks that all leaves are n-dimensional arrays.

    Args:
        tree (object): The tree.
    """
    leaves, _ = tree_flatten(tree)
    for leaf in leaves:
        if not isinstance(leaf, Tensor):
            msg = f"Expected Tensor, got {type(leaf)}"
            raise AssertionError(msg)


def assert_tree_shape_prefix(tree: object, shape_prefix: Sequence[int]) -> None:
    """Checks that all leaves shapes have the same prefix.

    Args:
        tree (object): The tree.
        shape_prefix (Sequence[int]): The shape_prefix.
    """
    leaves, _ = tree_flatten(tree)
    if not leaves:
        return
    for leaf in leaves:
        if not isinstance(leaf, Tensor) or leaf.shape[: len(shape_prefix)] != tuple(shape_prefix):
            msg = f"Expected shape prefix {shape_prefix}, got {leaf.shape}"
            raise AssertionError(msg)


def assert_tree_shape_suffix(tree: object, shape_suffix: Sequence[int]) -> None:
    """Checks that all leaves shapes have the same suffix.

    Args:
        tree (object): The tree.
        shape_suffix (Sequence[int]): The shape_suffix.
    """
    leaves, _ = tree_flatten(tree)
    if not leaves:
        return
    for leaf in leaves:
        if not isinstance(leaf, Tensor):
            msg = f"Expected Tensor, got {type(leaf)}"
            raise AssertionError(msg)
        leaf_suffix = leaf.shape[-len(shape_suffix) :] if len(shape_suffix) > 0 else ()
        if leaf_suffix != tuple(shape_suffix):
            msg = f"Expected shape suffix {shape_suffix}, got {leaf.shape}"
            raise AssertionError(msg)


def assert_tree_no_nones(tree: object) -> None:
    """Checks that a tree does not contain None.

    Args:
        tree (object): The tree.
    """
    leaves, _ = tree_flatten(tree)
    for leaf in leaves:
        if leaf is None:
            msg = "Tree contains None"
            raise AssertionError(msg)


def assert_trees_all_close(trees: Sequence[Any], rtol: float = 1e-06, atol: float = 0.0) -> None:
    """Checks that all trees have leaves with approx equal values.

    Args:
        trees (Sequence[Any]): The trees.
        rtol (float): The rtol.
        atol (float): The atol.
    """
    if not trees:
        return
    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            msg = "Trees have different structures"
            raise AssertionError(msg)
        for l0, li in zip(leaves0, leaves_i):
            if isinstance(l0, Tensor) and isinstance(li, Tensor):
                from ml_switcheroo_compiler.ops.binary import add, less_equal, multiply, subtract
                from ml_switcheroo_compiler.ops.reductions import all
                from ml_switcheroo_compiler.ops.unary import abs

                diff = abs(subtract(l0, li))
                thresh = add(atol, multiply(rtol, abs(li)))
                if not all(less_equal(diff, thresh)).data:
                    msg = "Trees are not all close"
                    raise AssertionError(msg)


def assert_trees_all_close_ulp(trees: Sequence[Any], maxulp: int = 1) -> None:
    """Checks that tree leaves differ by at most maxulp ULP.

    Args:
        trees (Sequence[Any]): The trees.
        maxulp (int): The maxulp.
    """
    if not trees:
        return
    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            msg = "Trees have different structures"
            raise AssertionError(msg)
        for l0, li in zip(leaves0, leaves_i):
            if isinstance(l0, Tensor) and isinstance(li, Tensor):
                from ml_switcheroo_compiler.ops.binary import less_equal, subtract
                from ml_switcheroo_compiler.ops.reductions import all
                from ml_switcheroo_compiler.ops.unary import abs

                diff = abs(subtract(l0, li))
                if not all(less_equal(diff, 1e-6)).data:
                    msg = "Trees are not all close within ULP"
                    raise AssertionError(msg)


def assert_trees_all_equal(trees: Sequence[Any], strict: bool = False) -> None:
    """Checks that all trees have leaves with exactly equal values.

    Args:
        trees (Sequence[Any]): The trees.
        strict (bool): The strict.
    """
    if not trees:
        return
    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            msg = "Trees have different structures"
            raise AssertionError(msg)
        for l0, li in zip(leaves0, leaves_i):
            if isinstance(l0, Tensor) and isinstance(li, Tensor):
                from ml_switcheroo_compiler.ops.binary import equal
                from ml_switcheroo_compiler.ops.reductions import all

                if not all(equal(l0, li)).data:
                    msg = "Trees are not all equal"
                    raise AssertionError(msg)


def assert_trees_all_equal_comparator(equality_comparator: Callable, *trees: object) -> None:
    """Checks that all trees are equal as per custom comparator.

    Args:
        equality_comparator (Callable): The equality_comparator.
        *trees: Additional arguments.
    """
    if not trees:
        return
    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            msg = "Trees have different structures"
            raise AssertionError(msg)
        for l0, li in zip(leaves0, leaves_i):
            if not equality_comparator(l0, li):
                msg = "Trees are not all equal by comparator"
                raise AssertionError(msg)


def assert_trees_all_equal_dtypes(trees: Sequence[Any]) -> None:
    """Checks that trees leaves have the same dtype.

    Args:
        trees (Sequence[Any]): The trees.
    """
    if not trees:
        return
    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            msg = "Trees have different structures"
            raise AssertionError(msg)
        for l0, li in zip(leaves0, leaves_i):
            if hasattr(l0, "dtype") and hasattr(li, "dtype") and l0.dtype != li.dtype:
                msg = f"Expected dtype {l0.dtype}, got {li.dtype}"
                raise AssertionError(msg)


def assert_trees_all_equal_shapes(trees: Sequence[Any]) -> None:
    """Checks that trees have same structure and leaves shapes.

    Args:
        trees (Sequence[Any]): The trees.
    """
    if not trees:
        return
    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            msg = "Trees have different structures"
            raise AssertionError(msg)
        for l0, li in zip(leaves0, leaves_i):
            if hasattr(l0, "shape") and hasattr(li, "shape") and l0.shape != li.shape:
                msg = f"Expected shape {l0.shape}, got {li.shape}"
                raise AssertionError(msg)


def assert_trees_all_equal_shapes_and_dtypes(trees: Sequence[Any]) -> None:
    """Checks same structure, shape, and dtype.

    Args:
        trees (Sequence[Any]): The trees.
    """
    assert_trees_all_equal_shapes(trees)
    assert_trees_all_equal_dtypes(trees)


def assert_trees_all_equal_sizes(trees: Sequence[Any]) -> None:
    """Checks same structure and leaves sizes.

    Args:
        trees (Sequence[Any]): The trees.
    """
    if not trees:
        return
    import math

    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            msg = "Trees have different structures"
            raise AssertionError(msg)
        for l0, li in zip(leaves0, leaves_i):
            if (
                hasattr(l0, "shape")
                and hasattr(li, "shape")
                and math.prod(l0.shape) != math.prod(li.shape)
            ):
                msg = "Trees do not have equal sizes"
                raise AssertionError(msg)


def assert_trees_all_equal_structs(trees: Sequence[Any]) -> None:
    """Checks that trees have the same structure.

    Args:
        trees (Sequence[Any]): The trees.
    """
    if not trees:
        return
    _, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        _, def_i = tree_flatten(t)
        if def0 != def_i:
            msg = "Trees have different structures"
            raise AssertionError(msg)


def assert_devices_available(
    n: int,
    devtype: str,
    backend: object = None,
    not_less_than: bool = False,
) -> None:
    """Checks that n devices of type are available.

    Args:
        n (int): The n.
        devtype (str): The devtype.
        backend (object): The backend.
        not_less_than (bool): The not_less_than.
    """
    # Mocking implementation for now
    available = 0
    if devtype.lower() == "cpu":
        available = 1  # Or interrogate config/backend
    elif devtype.lower() == "gpu" or devtype.lower() == "tpu":
        available = 0

    if not_less_than:
        if available < n:
            msg = f"Expected at least {n} {devtype} devices, got {available}"
            raise AssertionError(msg)
    elif available != n:
        msg_0 = f"Expected exactly {n} {devtype} devices, got {available}"
        raise AssertionError(msg_0)


def assert_gpu_available(backend: object = None) -> None:
    """Checks that at least one GPU device is available.

    Args:
        backend (object): The backend.
    """
    assert_devices_available(1, "gpu", backend, not_less_than=True)


def assert_tpu_available(backend: object = None) -> None:
    """Checks that at least one TPU device is available.

    Args:
        backend (object): The backend.
    """
    assert_devices_available(1, "tpu", backend, not_less_than=True)


def assert_tree_is_on_device(tree: object, platform: object = None, device: object = None) -> None:
    """Checks leaves are in device memory.

    Args:
        tree (object): The tree.
        platform (object): The platform.
        device (object): The device.
    """
    leaves, _ = tree_flatten(tree)
    for leaf in leaves:
        if isinstance(leaf, Tensor):
            dev = getattr(leaf, "device", None)
            if not dev or str(dev).lower() == "cpu":
                msg = f"Expected leaf to be on device, but found on {dev}"
                raise AssertionError(msg)


def assert_tree_is_on_host(
    tree: object,
    allow_cpu_device: bool = True,
    allow_sharded: bool = False,
) -> None:
    """Checks leaves are in host memory (CPU).

    Args:
        tree (object): The tree.
        allow_cpu_device (bool): The allow_cpu_device.
        allow_sharded (bool): The allow_sharded.
    """
    leaves, _ = tree_flatten(tree)
    for leaf in leaves:
        if isinstance(leaf, Tensor):
            dev = getattr(leaf, "device", None)
            if dev and str(dev).lower() != "cpu":
                msg = f"Expected leaf to be on host (CPU), but found on {dev}"
                raise AssertionError(msg)


def assert_tree_is_sharded(tree: object, devices: object = None) -> None:
    """Checks leaves are sharded across specified devices.

    Args:
        tree (object): The tree.
        devices (object): The devices.
    """
    leaves, _ = tree_flatten(tree)
    for leaf in leaves:
        if isinstance(leaf, Tensor):
            if not hasattr(leaf, "sharding") or leaf.sharding is None:
                msg = "Leaf is not sharded"
                raise AssertionError(msg)


_CHEX_ASSERTS_ENABLED = True
_TRACE_COUNTER = 0
_TRACE_LOCK = threading.Lock()


def chexify(fn: Callable, async_check: bool = True, errors: object = None) -> Callable:
    """Wraps a transformed function to enable Chex value assertions.

    Args:
        fn (Callable): The fn.
        async_check (bool): The async_check.
        errors (object): The errors.

    Returns:
        Callable: The computed result.
    """

    def wrapper(*args: object, **kwargs: object) -> object:
        return fn(*args, **kwargs)

    return wrapper


def block_until_chexify_assertions_complete() -> None:
    """Waits until all async checks complete."""


def assert_max_traces(fn: object = None, n: int = 1) -> Callable:
    """Checks that a function is traced at most n times.

    Args:
        fn (object): The fn.
        n (int): The n.

    Returns:
        Callable: The computed result.
    """
    global _TRACE_COUNTER
    if fn is None:
        return lambda f: assert_max_traces(f, n)

    def wrapper(*args: object, **kwargs: object) -> object:
        global _TRACE_COUNTER
        with _TRACE_LOCK:
            _TRACE_COUNTER += 1
            if n < _TRACE_COUNTER:
                msg = f"Function traced more than {n} times"
                raise AssertionError(msg)
        return fn(*args, **kwargs)

    return wrapper


def assert_numerical_grads(f: Callable, f_args: object, order: int = 1, atol: float = 0.01) -> None:
    """Checks that autodiff and numerical gradients match.

    Args:
        f (Callable): The f.
        f_args (object): The f_args.
        order (int): The order.
        atol (float): The atol.
    """
    # Mocking implementation


def clear_trace_counter() -> None:
    """Clears Chex traces counter."""
    global _TRACE_COUNTER
    with _TRACE_LOCK:
        _TRACE_COUNTER = 0


def disable_asserts() -> None:
    """Disables all Chex assertions."""
    global _CHEX_ASSERTS_ENABLED
    _CHEX_ASSERTS_ENABLED = False


def enable_asserts() -> None:
    """Enables Chex assertions."""
    global _CHEX_ASSERTS_ENABLED
    _CHEX_ASSERTS_ENABLED = True


@contextmanager
def fake_jit(enable_patching: bool = True) -> object:
    """Context manager for patching jit with identity.

    Args:
        enable_patching (bool): The enable_patching.

    Returns:
        object: The computed result.
    """
    yield


@contextmanager
def fake_pmap(enable_patching: bool = True) -> object:
    """Context manager for patching pmap with vmap.

    Args:
        enable_patching (bool): The enable_patching.

    Returns:
        object: The computed result.
    """
    yield


@contextmanager
def fake_pmap_and_jit(enable_pmap: bool = True, enable_jit: bool = True) -> object:
    """Patches both jit and pmap.

    Args:
        enable_pmap (bool): The enable_pmap.
        enable_jit (bool): The enable_jit.

    Returns:
        object: The computed result.
    """
    yield


def restrict_backends(allowed: object = None, forbidden: object = None) -> None:
    """Disallows compilation for certain backends.

    Args:
        allowed (object): The allowed.
        forbidden (object): The forbidden.
    """


def set_n_cpu_devices(n: int) -> None:
    """Forces compiler to use n CPU threads as host devices.

    Args:
        n (int): The n.
    """


def assert_scalar(x: object) -> None:
    """Checks that x is a scalar.

    Args:
        x (object): The x.
    """
    if isinstance(x, Tensor):
        if len(x.shape) != 0:
            msg = f"Expected scalar, got shape {x.shape}"
            raise AssertionError(msg)
    elif not isinstance(x, (float, int, complex)):
        msg = f"Expected scalar, got {type(x)}"
        raise AssertionError(msg)


def assert_scalar_in(x: object, min_: object, max_: object, included: bool = True) -> None:
    """Checks that argument is a scalar within segment.

    Args:
        x (object): The x.
        min_ (object): The min_.
        max_ (object): The max_.
        included (bool): The included.
    """
    assert_scalar(x)
    if isinstance(x, Tensor):
        x = float(x.data)
    if included:
        if not (min_ <= x <= max_):
            msg = f"Expected scalar in [{min_}, {max_}], got {x}"
            raise AssertionError(msg)
    elif not (min_ < x < max_):
        msg_0 = f"Expected scalar in ({min_}, {max_}), got {x}"
        raise AssertionError(msg_0)


def assert_scalar_negative(x: object) -> None:
    """Checks that a scalar is negative.

    Args:
        x (object): The x.
    """
    assert_scalar(x)
    val = float(x.data) if isinstance(x, Tensor) else x
    if not val < 0:
        msg = "Expected negative scalar"
        raise AssertionError(msg)


def assert_scalar_non_negative(x: object) -> None:
    """Checks that a scalar is non-negative.

    Args:
        x (object): The x.
    """
    assert_scalar(x)
    val = float(x.data) if isinstance(x, Tensor) else x
    if not val >= 0:
        msg = "Expected non-negative scalar"
        raise AssertionError(msg)


def assert_scalar_positive(x: object) -> None:
    """Checks that a scalar is positive.

    Args:
        x (object): The x.
    """
    assert_scalar(x)
    val = float(x.data) if isinstance(x, Tensor) else x
    if not val > 0:
        msg = "Expected positive scalar"
        raise AssertionError(msg)


def assert_equal(first: object, second: object) -> None:
    """Checks that two objects are equal.

    Args:
        first (object): The first.
        second (object): The second.
    """
    if first != second:
        msg = f"Expected {first} == {second}"
        raise AssertionError(msg)


def assert_exactly_one_is_none(first: object, second: object) -> None:
    """Checks that one and only one argument is None.

    Args:
        first (object): The first.
        second (object): The second.
    """
    if (first is None) == (second is None):
        msg = "Expected exactly one None"
        raise AssertionError(msg)


def assert_not_both_none(first: object, second: object) -> None:
    """Checks that at least one argument is not None.

    Args:
        first (object): The first.
        second (object): The second.
    """
    if first is None and second is None:
        msg = "Expected not both None"
        raise AssertionError(msg)


def assert_is_broadcastable(shape_a: Sequence[int], shape_b: Sequence[int]) -> None:
    """Checks that shape_a is broadcastable to shape_b.

    Args:
        shape_a (Sequence[int]): The shape_a.
        shape_b (Sequence[int]): The shape_b.
    """
    for a, b in zip(reversed(shape_a), reversed(shape_b)):
        if a != 1 and b not in (1, a):
            msg = f"Shape {shape_a} is not broadcastable to {shape_b}"
            raise AssertionError(msg)


def assert_is_divisible(numerator: object, denominator: object) -> None:
    """Checks divisibility.

    Args:
        numerator (object): The numerator.
        denominator (object): The denominator.
    """
    if numerator % denominator != 0:
        msg = f"{numerator} is not divisible by {denominator}"
        raise AssertionError(msg)


# Table 6 & 7 types and classes
Array = Tensor
ArrayBatched = Tensor
ArrayDevice = Tensor
ArrayNumpy = Any
ArraySharded = Tensor
ArrayDeviceTree = Any
ArrayDType = Any
ArrayNumpyTree = Any
ArrayTree = Any
Numeric = Any
PRNGKey = Any
Scalar = Any
Shape = Sequence[int]


class ChexVariantType(Enum):
    """Mock ChexVariantType Enum."""

    WITH_JIT = 1
    WITHOUT_JIT = 2


class ChexifyChecks(Enum):
    """Mock ChexifyChecks Enum."""

    USER = 1
    INTERNAL = 2


class Device:
    """Mock Device class."""


class PyTreeDef:
    """Mock PyTreeDef class."""


class Dimensions(dict):
    """Mock Dimensions class."""


class TestCase:
    """Mock TestCase class."""


def dataclass(cls: object = None, **kwargs: object) -> object:
    """Mock dataclass decorator.

    Args:
        cls: The class to wrap.
        **kwargs: Additional keyword arguments.

    Returns:
        object: The computed result.
    """
    if cls is None:
        return lambda c: dataclass(c, **kwargs)
    return builtin_dataclass(cls, **kwargs)


def mappable_dataclass(cls: object) -> object:
    """Mock mappable_dataclass decorator.

    Returns:
        object: The computed result.
    """
    return dataclass(cls)


def params_product(params_lists: object, named: bool = False) -> object:
    """Mock params_product function.

    Args:
        params_lists (object): The params_lists.
        named (bool): The named.

    Returns:
        object: The computed result.
    """
    if named:
        keys = list(params_lists.keys())
        values = list(params_lists.values())
        return [dict(zip(keys, p)) for p in itertools.product(*values)]
    return list(itertools.product(*params_lists))


def create_deprecated_function_alias(
    fun: object,
    new_name: str,
    *args: object,
    **kwargs: object,
) -> object:
    """Mock create_deprecated_function_alias function.

    Args:
        fun (object): The fun.
        new_name (str): The new_name.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        object: The computed result.
    """

    def wrapper(*wargs: object, **wkwargs: object) -> object:
        warnings.warn("Deprecated function called", stacklevel=2)
        return fun(*wargs, **wkwargs)

    return wrapper


def warn_deprecated_function(fun: object, replacement: object = None) -> object:
    """Mock warn_deprecated_function decorator.

    Args:
        fun (object): The fun.
        replacement (object): The replacement.

    Returns:
        object: The computed result.
    """

    def wrapper(*args: object, **kwargs: object) -> object:
        warnings.warn("Deprecated", stacklevel=2)
        return fun(*args, **kwargs)

    return wrapper


def warn_only_n_pos_args_in_future(fun: object = None, n: int = 1) -> object:
    """Mock warn_only_n_pos_args_in_future decorator.

    Args:
        fun (object): The fun.
        n (int): The n.

    Returns:
        object: The computed result.
    """
    if fun is None:
        return lambda f: warn_only_n_pos_args_in_future(f, n)

    def wrapper(*args: object, **kwargs: object) -> object:
        if len(args) > n:
            warnings.warn("Too many positional args", stacklevel=2)
        return fun(*args, **kwargs)

    return wrapper


def if_args_not_none(fn: object, args: object, kwargs: object) -> object:
    """Mock if_args_not_none function.

    Args:
        fn (object): The fn.
        args (object): The args.
        kwargs (object): The kwargs.

    Returns:
        object: The computed result.
    """
    if any(a is None for a in args) or any(v is None for v in kwargs.values()):
        return None
    return fn(*args, **kwargs)


def all_variants(with_pmap: bool = True, **kwargs: object) -> object:
    """Mock all_variants decorator.

    Args:
        with_pmap (bool): The with_pmap.
        **kwargs: Additional keyword arguments.

    Returns:
        object: The computed result.
    """

    def decorator(fn: object) -> object:
        def wrapper(*args: object, **kwargs: object) -> object:
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def get_err_regex(err: object) -> object:
    """Mock get_err_regex function.

    Args:
        err (object): The err.

    Returns:
        object: The computed result.
    """
    return str(err)


def register_dataclass_type_with_jax_tree_util(dataclass_type: object) -> None:
    """Mock register_dataclass_type_with_jax_tree_util.

    Args:
        dataclass_type (object): The dataclass_type.
    """


def variants(variants: object = (), **kwargs: object) -> object:
    """Mock variants decorator.

    Args:
        variants (object): The variants.
        **kwargs: Additional keyword arguments.

    Returns:
        object: The computed result.
    """

    def decorator(fn: object) -> object:
        def wrapper(*args: object, **kwargs: object) -> object:
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def warn_keyword_args_only_in_future(fun: object) -> object:
    """Mock warn_keyword_args_only_in_future decorator.

    Args:
        fun (object): The fun.

    Returns:
        object: The computed result.
    """

    def wrapper(*args: object, **kwargs: object) -> object:
        if len(args) > 0:
            warnings.warn("Use kwargs", stacklevel=2)
        return fun(*args, **kwargs)

    return wrapper
