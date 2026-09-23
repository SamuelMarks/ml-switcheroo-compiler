"""Eager evaluation dispatch for Apache Arrow Compute backend."""

from __future__ import annotations

import importlib

from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

# Comprehensive mapping of standard IR operations to pyarrow.compute functions
ARROW_COMPUTE_OP_MAP: dict[str, str] = {
    # Elementwise binary arithmetic
    "add": "add",
    "sub": "subtract",
    "subtract": "subtract",
    "mul": "multiply",
    "multiply": "multiply",
    "div": "divide",
    "divide": "divide",
    "truedivide": "divide",
    "pow": "power",
    "power": "power",
    "minimum": "min_element_wise",
    "min2": "min_element_wise",
    "maximum": "max_element_wise",
    "max2": "max_element_wise",
    # Elementwise unary arithmetic
    "neg": "negate",
    "negative": "negate",
    "negate": "negate",
    "abs": "abs",
    "absolute": "abs",
    "sign": "sign",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "ln",
    "ln": "ln",
    "log10": "log10",
    "log2": "log2",
    "log1p": "log1p",
    "expm1": "expm1",
    "floor": "floor",
    "ceil": "ceil",
    "ceiling": "ceil",
    "round": "round",
    "trunc": "trunc",
    "truncate": "trunc",
    # Trigonometry & Hyperbolic
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "arcsin": "asin",
    "acos": "acos",
    "arccos": "acos",
    "atan": "atan",
    "arctan": "atan",
    "atan2": "atan2",
    "arctan2": "atan2",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asinh": "asinh",
    "arcsinh": "asinh",
    "acosh": "acosh",
    "arccosh": "acosh",
    "atanh": "atanh",
    "arctanh": "atanh",
    # Comparisons
    "equal": "equal",
    "eq": "equal",
    "notequal": "not_equal",
    "ne": "not_equal",
    "greater": "greater",
    "gt": "greater",
    "greaterequal": "greater_equal",
    "ge": "greater_equal",
    "less": "less",
    "lt": "less",
    "lessequal": "less_equal",
    "le": "less_equal",
    # Logical & Bitwise
    "and": "and_",
    "logicaland": "and_",
    "bitwiseand": "bit_wise_and",
    "or": "or_",
    "logicalor": "or_",
    "bitwiseor": "bit_wise_or",
    "xor": "xor",
    "logicalxor": "xor",
    "bitwisexor": "bit_wise_xor",
    "not": "invert",
    "logicalnot": "invert",
    "invert": "invert",
    "bitwisenot": "bit_wise_not",
    # Reductions
    "sum": "sum",
    "mean": "mean",
    "min": "min",
    "max": "max",
    "all": "all",
    "any": "any",
    "std": "stddev",
    "stddev": "stddev",
    "var": "variance",
    "variance": "variance",
    "count": "count",
    # Cumulative
    "cumsum": "cumulative_sum",
    "cumulativesum": "cumulative_sum",
    "cumprod": "cumulative_prod",
    "cumulativeprod": "cumulative_prod",
    "cummax": "cumulative_max",
    "cumulativemax": "cumulative_max",
    "cummin": "cumulative_min",
    "cumulativemin": "cumulative_min",
    # Predicates & Selection
    "isnan": "is_nan",
    "isinf": "is_inf",
    "isfinite": "is_finite",
    "isnull": "is_null",
    "isvalid": "is_valid",
    "where": "if_else",
    "ifelse": "if_else",
    "select": "if_else",
    "cast": "cast",
}


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


def execute_op(
    cls_or_op: object,
    op_type_or_first: object = None,
    *args: object,
    **kwargs: object,
) -> object:
    """Execute operation eagerly on Apache Arrow Compute backend.

    Args:
        cls_or_op (object): Caller class context or operation name string.
        op_type_or_first (object): Operation name string or first argument.
        *args (object): Input tensor parameters.
        **kwargs (object): Additional keyword attributes.

    Returns:
        object: Evaluated Arrow array or scalar.

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
        pc_mod = importlib.import_module("pyarrow.compute")
    except Exception as exc:
        msg = f"pyarrow.compute is required for PyArrow Compute backend execution: {exc}"
        raise BackendNotSupportedError(msg) from exc

    actual_args = tuple(_unwrap_arg(a) for a in raw_args)
    fn_name = op_type.lower()

    # 1. Direct function lookup on pyarrow.compute
    if hasattr(pc_mod, fn_name):
        compute_fn = getattr(pc_mod, fn_name)
        try:
            return compute_fn(*actual_args, **kwargs)
        except Exception as exc:
            msg = f"Failed executing pyarrow.compute.{fn_name}: {exc}"
            raise BackendNotSupportedError(msg) from exc

    # 2. Lookup in standard operation mapping
    target_name = ARROW_COMPUTE_OP_MAP.get(fn_name)
    if target_name and hasattr(pc_mod, target_name):
        compute_fn = getattr(pc_mod, target_name)
        try:
            return compute_fn(*actual_args, **kwargs)
        except Exception as exc:
            msg = f"Failed executing pyarrow.compute.{target_name}: {exc}"
            raise BackendNotSupportedError(msg) from exc

    msg = f"Operation '{op_type}' not supported in PyArrow Compute backend."
    raise BackendNotSupportedError(msg)
