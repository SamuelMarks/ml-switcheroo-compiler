"""Eager evaluation dispatch for Awkward Array backend supporting native ragged structures."""

from __future__ import annotations

import importlib
import operator

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
    "ptp": "ptp",
    "argmin": "argmin",
    "argmax": "argmax",
    # Variable-length flattening & structure manipulation
    "flatten": "flatten",
    "unflatten": "unflatten",
    "num": "num",
    "pad_none": "pad_none",
    "fill_none": "fill_none",
    "drop_none": "drop_none",
    "cartesian": "cartesian",
    "combinations": "combinations",
    "concatenate": "concatenate",
    "where": "where",
    # Nested record transformations
    "zip": "zip",
    "unzip": "unzip",
    "with_field": "with_field",
    "fields": "fields",
    "to_regular": "to_regular",
    "from_regular": "from_regular",
}

UFUNC_NAMES: dict[str, str] = {
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "floor": "floor",
    "ceil": "ceil",
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
    "pow": operator.pow,
    "power": operator.pow,
}

_UNARY_OPS = {
    "neg": operator.neg,
    "negative": operator.neg,
    "negate": operator.neg,
    "abs": operator.abs,
    "absolute": operator.abs,
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


def execute_op(
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

    msg = f"Operation '{op_type}' not supported in Awkward backend."
    raise BackendNotSupportedError(msg)
