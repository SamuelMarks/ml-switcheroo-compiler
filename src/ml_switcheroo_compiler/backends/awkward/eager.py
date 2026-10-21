"""Eager evaluation dispatch for Awkward Array backend supporting native ragged structures."""

from __future__ import annotations

import importlib
import math
import operator
from collections.abc import Callable

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable, as_float, as_float_list, as_tensor
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def _unwrap_arg(arg: EagerValue) -> EagerValue:
    """Unwrap Tensor into underlying array or scalar.

    Args:
        arg (EagerValue): Input argument to unwrap.

    Returns:
        EagerValue: Unwrapped raw data.
    """
    if type(arg).__name__ == "Tensor" and hasattr(arg, "data"):
        return as_tensor(getattr(arg, "data"))
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

_MATH_UFUNCS: dict[str, Callable[[float], float]] = {
    "exp": math.exp,
    "exp2": lambda x: 2.0**x,
    "expm1": math.expm1,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "log1p": math.log1p,
    "sqrt": math.sqrt,
    "square": lambda x: x * x,
    "cbrt": lambda x: math.copysign(abs(x) ** (1.0 / 3.0), x),
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "arcsin": math.asin,
    "acos": math.acos,
    "arccos": math.acos,
    "atan": math.atan,
    "arctan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "arcsinh": math.asinh,
    "acosh": math.acosh,
    "arccosh": math.acosh,
    "atanh": math.atanh,
    "arctanh": math.atanh,
    "floor": lambda x: float(math.floor(x)),
    "ceil": lambda x: float(math.ceil(x)),
    "trunc": lambda x: float(math.trunc(x)),
    "rint": lambda x: float(round(x)),
    "round": lambda x: float(round(x)),
    "abs": abs,
    "sign": lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0),
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
        args (tuple[EagerValue, ...]): Operation arguments.

    Returns:
        tuple[bool, EagerValue]: Handled flag and result.
    """
    if fn_name in _BINARY_OPS and len(args) >= 2:
        return True, as_callable(_BINARY_OPS[fn_name])(args[0], args[1])
    if fn_name in _UNARY_OPS and len(args) >= 1:
        return True, as_callable(_UNARY_OPS[fn_name])(args[0])
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
        res = [sum(as_float_list(sub)) if isinstance(sub, (list, tuple)) else sub for sub in data]
    elif fn_name == "mean":
        res = [sum(as_float_list(sub)) / len(sub) if isinstance(sub, (list, tuple)) and sub else 0.0 for sub in data]
    elif fn_name == "min":
        res = [min([float(v) for v in sub if isinstance(v, (int, float))]) if isinstance(sub, (list, tuple)) and sub else sub for sub in data]
    elif fn_name == "max":
        res = [max([float(v) for v in sub if isinstance(v, (int, float))]) if isinstance(sub, (list, tuple)) and sub else sub for sub in data]
    else:
        res = [math.prod(as_float_list(sub)) if isinstance(sub, (list, tuple)) else sub for sub in data]
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
        res = sum(as_float_list(flat_items))
    elif fn_name == "mean":
        res = sum(as_float_list(flat_items)) / len(flat_items) if flat_items else 0.0
    elif fn_name in ("min", "max"):
        flat_floats = [float(x) for x in flat_items if isinstance(x, (int, float))]
        res = (min(flat_floats) if fn_name == "min" else max(flat_floats)) if flat_floats else None
    else:
        res = math.prod(as_float_list(flat_items)) if flat_items else 1
    return True, res


def _eval_ragged_fallback(
    fn_name: str,
    args: tuple[EagerValue, ...],
    kwargs: dict[str, EagerValue],
) -> tuple[bool, EagerValue]:
    """Evaluate reference ragged and nested operations on Python lists without padding.

    Args:
        fn_name (str): Lowercase operation name.
        args (tuple[EagerValue, ...]): Operation input arguments.
        kwargs (dict[str, EagerValue]): Keyword options such as axis.

    Returns:
        tuple[bool, EagerValue]: Handled flag and resulting structure.
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


def _apply_math_recursive(fn: Callable[[float], float], val: EagerValue) -> EagerValue:
    """Recursively apply a scalar math function across ragged or nested structures.

    Args:
        fn (Callable[[float], float]): Scalar mathematical function.
        val (EagerValue): Input Awkward array, nested list, or scalar.

    Returns:
        EagerValue: Transformed structure preserving raggedness.
    """
    source = getattr(val, "to_list")() if hasattr(val, "to_list") else val
    if isinstance(source, (list, tuple)):
        import awkward as ak

        return as_tensor(ak.Array([_apply_math_recursive(fn, x) for x in source]))
    return fn(float(as_float(source)))


def _try_math_ufunc(
    fn_name: str,
    actual_args: tuple[EagerValue, ...],
) -> tuple[bool, EagerValue]:
    """Attempt to execute unary mathematical operation on Awkward or nested data.

    Args:
        fn_name (str): Lowercase operation name.
        actual_args (tuple[EagerValue, ...]): Unwrapped positional arguments.

    Returns:
        tuple[bool, EagerValue]: Handled flag and evaluated result.

    Raises:
        BackendNotSupportedError: When mathematical evaluation fails.
    """
    if fn_name in _MATH_UFUNCS and actual_args:
        try:
            fn = getattr(math, fn_name) if hasattr(math, fn_name) else _MATH_UFUNCS[fn_name]
            return True, _apply_math_recursive(fn, actual_args[0])
        except Exception as exc:
            msg = f"Failed executing math ufunc {fn_name} on Awkward array: {exc}"
            raise BackendNotSupportedError(msg) from exc
    return False, None


def execute_op(
    op_type: str,
    *args: EagerValue,
    **kwargs: EagerValue,
) -> EagerValue:
    """Execute operation eagerly on Awkward Array backend supporting native ragged structures.

    Args:
        op_type (str): Operation name string.
        *args (EagerValue): Input tensor parameters.
        **kwargs (EagerValue): Additional keyword attributes.

    Returns:
        EagerValue: Evaluated Awkward array or scalar.

    Raises:
        BackendNotSupportedError: When the operation cannot be resolved or executed.
    """
    try:
        ak_mod = importlib.import_module("awkward")
    except Exception:
        ak_mod = None

    actual_args = tuple(_unwrap_arg(a) for a in args)
    fn_name = op_type.lower()

    ak_handled, ak_res = _try_awkward_module_op(ak_mod, fn_name, actual_args, kwargs)
    if ak_handled:
        return ak_res

    handled, op_res = _eval_operator(fn_name, actual_args)
    if handled:
        return op_res

    ufunc_handled, ufunc_res = _try_math_ufunc(fn_name, actual_args)
    if ufunc_handled:
        return ufunc_res

    ragged_handled, ragged_res = _eval_ragged_fallback(fn_name, actual_args, kwargs)
    if ragged_handled:
        return ragged_res

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    registered_func = global_eager_registry.get(op_type)
    if registered_func is not None:
        return registered_func(ak_mod, *actual_args, **kwargs)

    msg = f"Operation '{op_type}' not supported in Awkward backend."
    raise BackendNotSupportedError(msg)
