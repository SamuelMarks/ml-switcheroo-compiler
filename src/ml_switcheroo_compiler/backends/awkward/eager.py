"""Eager evaluation dispatch for Awkward Array backend supporting native ragged structures."""

from __future__ import annotations

import importlib
import math
import operator
from typing import Any

from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def _unwrap_arg(arg: object) -> object:
    """Unwrap Tensor into underlying array or scalar.

    Args:
        arg (object): Input argument to unwrap.

    Returns:
        object: Unwrapped raw data.
    """
    if type(arg).__name__ == "Tensor" and hasattr(arg, "data"):
        return arg.data
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


def _eval_operator(fn_name: str, args: tuple[object, ...]) -> tuple[bool, object]:
    """Evaluate native overloaded operators on Awkward operands.

    Args:
        fn_name (str): Lowercase operation name.
        args (tuple[object, ...]): Operation arguments.

    Returns:
        tuple[bool, object]: Handled flag and result.
    """
    if fn_name in _BINARY_OPS and len(args) >= 2:
        return True, _BINARY_OPS[fn_name](args[0], args[1])
    if fn_name in _UNARY_OPS and len(args) >= 1:
        return True, _UNARY_OPS[fn_name](args[0])
    return False, None


def _eval_ragged_fallback(  # noqa: C901, PLR0911, PLR0912
    fn_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[bool, object]:
    """Evaluate reference ragged and nested operations on Python lists without padding.

    Args:
        fn_name (str): Lowercase operation name.
        args (tuple[object, ...]): Operation input arguments.
        kwargs (dict[str, object]): Keyword options such as axis.

    Returns:
        tuple[bool, object]: Handled flag and resulting structure.
    """
    if not args:
        return False, None

    data = args[0]
    axis = kwargs.get("axis", None)

    if fn_name == "flatten" and isinstance(data, (list, tuple)):
        res: list[object] = []
        for sub in data:
            if isinstance(sub, (list, tuple)):
                res.extend(sub)
            else:
                res.append(sub)
        return True, res

    if fn_name == "num" and isinstance(data, (list, tuple)):
        return True, [len(sub) if isinstance(sub, (list, tuple)) else 1 for sub in data]

    if fn_name == "count" and isinstance(data, (list, tuple)):
        if axis in (-1, 1):
            return True, [len([x for x in sub if x is not None]) if isinstance(sub, (list, tuple)) else 1 for sub in data]
        return True, sum(len([x for x in sub if x is not None]) if isinstance(sub, (list, tuple)) else 1 for sub in data)

    if fn_name == "sum" and isinstance(data, (list, tuple)):
        if axis in (-1, 1):
            return True, [sum(sub) if isinstance(sub, (list, tuple)) else sub for sub in data]
        flat_items: list[Any] = [x for sub in data for x in (sub if isinstance(sub, (list, tuple)) else [sub])]
        return True, sum(flat_items)

    if fn_name == "mean" and isinstance(data, (list, tuple)):
        if axis in (-1, 1):
            return True, [sum(sub) / len(sub) if isinstance(sub, (list, tuple)) and sub else 0.0 for sub in data]
        flat_items = [x for sub in data for x in (sub if isinstance(sub, (list, tuple)) else [sub])]
        return True, sum(flat_items) / len(flat_items) if flat_items else 0.0

    if fn_name == "min" and isinstance(data, (list, tuple)):
        if axis in (-1, 1):
            return True, [min(sub) if isinstance(sub, (list, tuple)) and sub else sub for sub in data]
        flat_items = [x for sub in data for x in (sub if isinstance(sub, (list, tuple)) else [sub])]
        return True, min(flat_items) if flat_items else None

    if fn_name == "max" and isinstance(data, (list, tuple)):
        if axis in (-1, 1):
            return True, [max(sub) if isinstance(sub, (list, tuple)) and sub else sub for sub in data]
        flat_items = [x for sub in data for x in (sub if isinstance(sub, (list, tuple)) else [sub])]
        return True, max(flat_items) if flat_items else None

    if fn_name == "prod" and isinstance(data, (list, tuple)):
        if axis in (-1, 1):
            return True, [math.prod(sub) if isinstance(sub, (list, tuple)) else sub for sub in data]
        flat_items = [x for sub in data for x in (sub if isinstance(sub, (list, tuple)) else [sub])]
        return True, math.prod(flat_items) if flat_items else 1

    return False, None


def execute_op(  # noqa: C901, PLR0912
    cls_or_op: object,
    op_type_or_first: object = None,
    *args: object,
    **kwargs: object,
) -> object:
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

    if ak_mod is not None:
        target_fn_name: str | None = fn_name if hasattr(ak_mod, fn_name) else AK_OPS_MAP.get(fn_name)
        if target_fn_name is not None and hasattr(ak_mod, target_fn_name):
            try:
                return getattr(ak_mod, target_fn_name)(*actual_args, **kwargs)
            except Exception as exc:
                msg = f"Failed executing awkward.{target_fn_name}: {exc}"
                raise BackendNotSupportedError(msg) from exc

    handled, op_res = _eval_operator(fn_name, actual_args)
    if handled:
        return op_res

    numpy_mod = importlib.import_module("numpy")
    np_name = UFUNC_NAMES.get(fn_name)
    if np_name and hasattr(numpy_mod, np_name):
        try:
            return getattr(numpy_mod, np_name)(*actual_args, **kwargs)
        except Exception as exc:
            msg = f"Failed executing NumPy ufunc {np_name} on Awkward array: {exc}"
            raise BackendNotSupportedError(msg) from exc

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
