"""Eager evaluation dispatch for Awkward Array backend supporting native ragged structures."""

from __future__ import annotations

import importlib
import math
import operator
from collections.abc import Callable
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.distributed.host_collectives import _np_mod


def _unwrap_arg(arg: EagerValue) -> EagerValue:
    """Unwrap Tensor into underlying array or scalar.

    Args:
        arg (object): Input argument to unwrap.

    Returns:
        object: Unwrapped raw data.
    """
    if type(arg).__name__ == "Tensor" and hasattr(arg, "data"):
        return cast(EagerValue, getattr(arg, "data"))
    return arg


# Map of standard reduction and shape ops to awkward top-level functions
AK_OPS_MAP: dict[str, str] = {
    # Jagged reductions
    "sum": "sum",
    "mean": "mean",
    "min": "min",
    "max": "max",
    "prod": "prod",
    "all": "all",
    "any": "any",
    "std": "std",
    "var": "var",
    "count": "count",
    "count_nonzero": "count_nonzero",
    "ptp": "ptp",
    "argmin": "argmin",
    "argmax": "argmax",
    "moment": "moment",
    # Variable-length flattening & structure manipulation
    "flatten": "flatten",
    "unflatten": "unflatten",
    "num": "num",
    "pad_none": "pad_none",
    "fill_none": "fill_none",
    "drop_none": "drop_none",
    "is_none": "is_none",
    "nan_to_num": "nan_to_num",
    "firsts": "firsts",
    "singletons": "singletons",
    "mask": "mask",
    "local_index": "local_index",
    "run_lengths": "run_lengths",
    "ravel": "ravel",
    "argsort": "argsort",
    "sort": "sort",
    "cartesian": "cartesian",
    "combinations": "combinations",
    "concatenate": "concatenate",
    "where": "where",
    "broadcast_arrays": "broadcast_arrays",
    "broadcast_fields": "broadcast_fields",
    # Nested record transformations
    "zip": "zip",
    "unzip": "unzip",
    "with_field": "with_field",
    "without_field": "without_field",
    "fields": "fields",
    "to_regular": "to_regular",
    "from_regular": "from_regular",
    "to_numpy": "to_numpy",
    "from_numpy": "from_numpy",
    "to_list": "to_list",
    "from_iter": "from_iter",
}

UFUNC_NAMES: dict[str, str] = {
    "exp": "exp",
    "exp2": "exp2",
    "expm1": "expm1",
    "log": "log",
    "log2": "log2",
    "log10": "log10",
    "log1p": "log1p",
    "sqrt": "sqrt",
    "square": "square",
    "cbrt": "cbrt",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "arcsin",
    "arcsin": "arcsin",
    "acos": "arccos",
    "arccos": "arccos",
    "atan": "arctan",
    "arctan": "arctan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asinh": "arcsinh",
    "arcsinh": "arcsinh",
    "acosh": "arccosh",
    "arccosh": "arccosh",
    "atanh": "arctanh",
    "arctanh": "arctanh",
    "floor": "floor",
    "ceil": "ceil",
    "trunc": "trunc",
    "rint": "rint",
    "round": "round",
    "abs": "abs",
    "sign": "sign",
}

_BINARY_OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "subtract": operator.sub,
    "mul": operator.mul,
    "multiply": operator.mul,
    "div": operator.truediv,
    "divide": operator.truediv,
    "truedivide": operator.truediv,
    "floordivide": operator.floordiv,
    "floor_divide": operator.floordiv,
    "mod": operator.mod,
    "remainder": operator.mod,
    "pow": operator.pow,
    "power": operator.pow,
    "equal": operator.eq,
    "eq": operator.eq,
    "notequal": operator.ne,
    "not_equal": operator.ne,
    "ne": operator.ne,
    "greater": operator.gt,
    "gt": operator.gt,
    "greaterequal": operator.ge,
    "greater_equal": operator.ge,
    "ge": operator.ge,
    "less": operator.lt,
    "lt": operator.lt,
    "lessequal": operator.le,
    "less_equal": operator.le,
    "le": operator.le,
    "logicaland": operator.and_,
    "logical_and": operator.and_,
    "bitwiseand": operator.and_,
    "bitwise_and": operator.and_,
    "logicalor": operator.or_,
    "logical_or": operator.or_,
    "bitwiseor": operator.or_,
    "bitwise_or": operator.or_,
    "logicalxor": operator.xor,
    "logical_xor": operator.xor,
    "bitwisexor": operator.xor,
    "bitwise_xor": operator.xor,
}

_UNARY_OPS = {
    "neg": operator.neg,
    "negative": operator.neg,
    "negate": operator.neg,
    "abs": operator.abs,
    "absolute": operator.abs,
    "pos": operator.pos,
    "positive": operator.pos,
    "invert": operator.invert,
    "bitwisenot": operator.invert,
    "bitwise_not": operator.invert,
    "logicalnot": operator.not_,
    "logical_not": operator.not_,
}


def _eval_operator(fn_name: str, args: tuple[EagerValue, ...]) -> tuple[bool, EagerValue]:
    """Evaluate native overloaded operators on Awkward operands.

    Args:
        fn_name (str): Lowercase operation name.
        args (tuple[object, ...]): Operation arguments.

    Returns:
        tuple[bool, object]: Handled flag and result.
    """
    if fn_name in _BINARY_OPS and len(args) >= 2:
        return True, cast(Callable[..., EagerValue], _BINARY_OPS[fn_name])(args[0], args[1])
    if fn_name in _UNARY_OPS and len(args) >= 1:
        return True, cast(Callable[..., EagerValue], _UNARY_OPS[fn_name])(args[0])
    return False, None


def _flatten_ragged_items(data: list[EagerValue] | tuple[EagerValue, ...]) -> list[EagerValue]:
    """Flatten a 2-level ragged list or tuple into a 1D list of elements.

    Args:
        data (list[EagerValue] | tuple[EagerValue, ...]): Input ragged sequence.

    Returns:
        list[EagerValue]: Flattened elements.
    """
    res: list[EagerValue] = []
    for sub in data:
        if isinstance(sub, (list, tuple)):
            res.extend(sub)
        else:
            res.append(sub)
    return res


def _eval_ragged_axis1_reduction(
    fn_name: str,
    data: list[EagerValue] | tuple[EagerValue, ...],
) -> tuple[bool, EagerValue]:
    """Evaluate inner-axis (axis=1 or -1) ragged list reductions.

    Args:
        fn_name (str): Lowercase reduction operation name.
        data (list[EagerValue] | tuple[EagerValue, ...]): Ragged sequence data.

    Returns:
        tuple[bool, EagerValue]: Handled flag and list of per-row reduced values.
    """
    if fn_name == "count":
        res: EagerValue = [len([x for x in sub if x is not None]) if isinstance(sub, (list, tuple)) else 1 for sub in data]
    elif fn_name == "sum":
        res = [sum(sub) if isinstance(sub, (list, tuple)) else sub for sub in data]
    elif fn_name == "mean":
        res = [sum(sub) / len(sub) if isinstance(sub, (list, tuple)) and sub else 0.0 for sub in data]
    elif fn_name == "min":
        res = [min([float(v) for v in sub if isinstance(v, (int, float))]) if isinstance(sub, (list, tuple)) and sub else sub for sub in data]
    elif fn_name == "max":
        res = [max([float(v) for v in sub if isinstance(v, (int, float))]) if isinstance(sub, (list, tuple)) and sub else sub for sub in data]
    else:
        res = [math.prod(sub) if isinstance(sub, (list, tuple)) else sub for sub in data]
    return True, res


def _eval_ragged_global_reduction(
    fn_name: str,
    data: list[EagerValue] | tuple[EagerValue, ...],
) -> tuple[bool, EagerValue]:
    """Evaluate global (all-axis) ragged list reductions.

    Args:
        fn_name (str): Lowercase reduction operation name.
        data (list[EagerValue] | tuple[EagerValue, ...]): Ragged sequence data.

    Returns:
        tuple[bool, EagerValue]: Handled flag and scalar reduction result.
    """
    flat_items = _flatten_ragged_items(data)
    if fn_name == "count":
        res: EagerValue = len([x for x in flat_items if x is not None])
    elif fn_name == "sum":
        res = sum(getattr(flat_items, "__iter__")())
    elif fn_name == "mean":
        res = sum(getattr(flat_items, "__iter__")()) / len(flat_items) if flat_items else 0.0
    elif fn_name in ("min", "max"):
        flat_floats = [float(x) for x in flat_items if isinstance(x, (int, float))]
        res = (min(flat_floats) if fn_name == "min" else max(flat_floats)) if flat_floats else None
    else:
        res = getattr(math, "prod")(flat_items) if flat_items else 1
    return True, res


def _eval_ragged_fallback(
    fn_name: str,
    args: tuple[EagerValue, ...],
    kwargs: dict[str, EagerValue],
) -> tuple[bool, EagerValue]:
    """Evaluate reference ragged and nested operations on Python lists without padding.

    Args:
        fn_name (str): Lowercase operation name.
        args (tuple[object, ...]): Operation input arguments.
        kwargs (dict[str, object]): Keyword options such as axis.

    Returns:
        tuple[bool, object]: Handled flag and resulting structure.
    """
    if not args or not isinstance(args[0], (list, tuple)):
        return False, None

    data = args[0]
    if fn_name == "flatten":
        return True, _flatten_ragged_items(data)
    if fn_name == "num":
        return True, [len(sub) if isinstance(sub, (list, tuple)) else 1 for sub in data]

    axis = kwargs.get("axis", None)
    if axis in (-1, 1):
        return _eval_ragged_axis1_reduction(fn_name, data)
    return _eval_ragged_global_reduction(fn_name, data)


def _try_awkward_module_op(
    ak_mod: EagerValue,
    fn_name: str,
    actual_args: tuple[EagerValue, ...],
    kwargs: dict[str, EagerValue],
) -> tuple[bool, EagerValue]:
    """Attempt to execute operation directly on the awkward module.

    Args:
        ak_mod (EagerValue): Loaded awkward module or None.
        fn_name (str): Lowercase operation name.
        actual_args (tuple[EagerValue, ...]): Unwrapped positional arguments.
        kwargs (dict[str, EagerValue]): Keyword arguments.

    Returns:
        tuple[bool, EagerValue]: Handled flag and evaluated result.

    Raises:
        BackendNotSupportedError: When awkward function invocation fails.
    """
    if ak_mod is None:
        return False, None
    target_fn_name: str | None = fn_name if hasattr(ak_mod, fn_name) else AK_OPS_MAP.get(fn_name)
    if target_fn_name is not None and hasattr(ak_mod, target_fn_name):
        try:
            return True, getattr(ak_mod, target_fn_name)(*actual_args, **kwargs)
        except Exception as exc:
            msg = f"Failed executing awkward.{target_fn_name}: {exc}"
            raise BackendNotSupportedError(msg) from exc
    return False, None


def _try_numpy_ufunc(
    numpy_mod: EagerValue,
    fn_name: str,
    actual_args: tuple[EagerValue, ...],
    kwargs: dict[str, EagerValue],
) -> tuple[bool, EagerValue]:
    """Attempt to execute operation via NumPy ufunc dispatch.

    Args:
        numpy_mod (EagerValue): Loaded NumPy module.
        fn_name (str): Lowercase operation name.
        actual_args (tuple[EagerValue, ...]): Unwrapped positional arguments.
        kwargs (dict[str, EagerValue]): Keyword arguments.

    Returns:
        tuple[bool, EagerValue]: Handled flag and evaluated result.

    Raises:
        BackendNotSupportedError: When NumPy ufunc invocation fails.
    """
    np_name = UFUNC_NAMES.get(fn_name)
    if np_name and hasattr(numpy_mod, np_name):
        try:
            return True, getattr(numpy_mod, np_name)(*actual_args, **kwargs)
        except Exception as exc:
            msg = f"Failed executing NumPy ufunc {np_name} on Awkward array: {exc}"
            raise BackendNotSupportedError(msg) from exc
    return False, None


def execute_op(
    cls_or_op: EagerValue,
    op_type_or_first: EagerValue = None,
    *args: EagerValue,
    **kwargs: EagerValue,
) -> EagerValue:
    """Execute operation eagerly on Awkward Array backend supporting native ragged structures.

    Args:
        cls_or_op (object): Caller class context or operation name string.
        op_type_or_first (object): Operation name string or first argument.
        *args (object): Input tensor parameters.
        **kwargs (object): Additional keyword attributes.

    Returns:
        object: Evaluated Awkward array or scalar.

    Raises:
        BackendNotSupportedError: When the operation cannot be resolved or executed.
    """
    if isinstance(cls_or_op, type):
        op_type = str(op_type_or_first)
        raw_args = args
    else:
        op_type = str(cls_or_op)
        raw_args = (op_type_or_first,) + args if op_type_or_first is not None else args

    try:
        ak_mod = importlib.import_module("awkward")
    except Exception:
        ak_mod = None

    actual_args = tuple(_unwrap_arg(a) for a in raw_args)
    fn_name = op_type.lower()

    ak_handled, ak_res = _try_awkward_module_op(ak_mod, fn_name, actual_args, kwargs)
    if ak_handled:
        return ak_res

    handled, op_res = _eval_operator(fn_name, actual_args)
    if handled:
        return op_res

    numpy_mod = _np_mod()
    ufunc_handled, ufunc_res = _try_numpy_ufunc(numpy_mod, fn_name, actual_args, kwargs)
    if ufunc_handled:
        return ufunc_res

    ragged_handled, ragged_res = _eval_ragged_fallback(fn_name, actual_args, kwargs)
    if ragged_handled:
        return ragged_res

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    registered_func = global_eager_registry.get(op_type)
    if registered_func is not None:
        backend_context = ak_mod if ak_mod is not None else numpy_mod
        return registered_func(backend_context, *actual_args, **kwargs)

    msg = f"Operation '{op_type}' not supported in Awkward backend."
    raise BackendNotSupportedError(msg)
