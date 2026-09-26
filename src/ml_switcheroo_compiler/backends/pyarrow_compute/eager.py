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
    "product": "product",
    "prod": "product",
    "mode": "mode",
    "quantile": "quantile",
    "median": "approximate_median",
    "approximatemedian": "approximate_median",
    "approximate_median": "approximate_median",
    # Cumulative
    "cumsum": "cumulative_sum",
    "cumulativesum": "cumulative_sum",
    "cumprod": "cumulative_prod",
    "cumulativeprod": "cumulative_prod",
    "cummax": "cumulative_max",
    "cumulativemax": "cumulative_max",
    "cummin": "cumulative_min",
    "cumulativemin": "cumulative_min",
    # Bitwise shift & extra math
    "shift_left": "shift_left",
    "left_shift": "shift_left",
    "shift_right": "shift_right",
    "right_shift": "shift_right",
    "cbrt": "cbrt",
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
    # Tabular & Columnar Array Kernels
    "filter": "filter",
    "take": "take",
    "gather": "take",
    "drop_null": "drop_null",
    "dropna": "drop_null",
    "fill_null": "fill_null",
    "fillna": "fill_null",
    "fill_null_forward": "fill_null_forward",
    "fill_null_backward": "fill_null_backward",
    "replace_with_mask": "replace_with_mask",
    "unique": "unique",
    "value_counts": "value_counts",
    "indices_nonzero": "indices_nonzero",
    "nonzero": "indices_nonzero",
    "flatnonzero": "indices_nonzero",
    "sort_indices": "sort_indices",
    "argsort": "sort_indices",
    "partition_nth_indices": "partition_nth_indices",
    "argpartition": "partition_nth_indices",
    "rank": "rank",
    "dictionary_encode": "dictionary_encode",
    # Nested & Structural Columnar Kernels
    "list_flatten": "list_flatten",
    "flatten": "list_flatten",
    "list_slice": "list_slice",
    "list_element": "list_element",
    "list_parent_indices": "list_parent_indices",
    "make_struct": "make_struct",
    "struct_field": "struct_field",
    # Columnar Text / String Kernels
    "lower": "utf8_lower",
    "ascii_lower": "ascii_lower",
    "utf8_lower": "utf8_lower",
    "upper": "utf8_upper",
    "ascii_upper": "ascii_upper",
    "utf8_upper": "utf8_upper",
    "length": "utf8_length",
    "string_length": "utf8_length",
    "utf8_length": "utf8_length",
    "replace_substring": "replace_substring",
    "replace_substring_regex": "replace_substring_regex",
    "match_substring": "match_substring",
    "match_substring_regex": "match_substring_regex",
    "split_pattern": "split_pattern",
    "split_pattern_regex": "split_pattern_regex",
    "capitalize": "ascii_capitalize",
    "ascii_capitalize": "ascii_capitalize",
    "title": "ascii_title",
    "ascii_title": "ascii_title",
    # GroupBy / Hash Aggregations
    "hash_count": "hash_count",
    "hash_count_distinct": "hash_count_distinct",
    "hash_sum": "hash_sum",
    "hash_mean": "hash_mean",
    "hash_min": "hash_min",
    "hash_max": "hash_max",
    "hash_stddev": "hash_stddev",
    "hash_variance": "hash_variance",
    "hash_any": "hash_any",
    "hash_all": "hash_all",
    "hash_product": "hash_product",
    "hash_first": "hash_first",
    "hash_last": "hash_last",
    "hash_list": "hash_list",
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


def _dispatch_groupby(
    fn_name: str,
    actual_args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[bool, object]:
    """Dispatch table group_by or hash aggregations if applicable.

    Args:
        fn_name (str): Lowercase operation name.
        actual_args (tuple[object, ...]): Operation positional arguments.
        kwargs (dict[str, object]): Keyword options.

    Returns:
        tuple[bool, object]: Handled flag and resulting object.

    Raises:
        BackendNotSupportedError: If group_by or hash aggregation fails.
    """
    if fn_name in ("group_by", "groupby") and actual_args:
        table_obj = actual_args[0]
        if hasattr(table_obj, "group_by"):
            keys = kwargs.get("keys", actual_args[1] if len(actual_args) > 1 else [])
            aggregations = kwargs.get("aggregations", actual_args[2] if len(actual_args) > 2 else [])
            try:
                return True, table_obj.group_by(keys).aggregate(aggregations)
            except Exception as exc:
                msg = f"Failed executing group_by: {exc}"
                raise BackendNotSupportedError(msg) from exc

    if fn_name.startswith("hash_") and len(actual_args) >= 3:
        table_obj = actual_args[0]
        keys = actual_args[1]
        agg_col = actual_args[2]
        agg_name = fn_name[5:]
        if hasattr(table_obj, "group_by"):
            try:
                return True, table_obj.group_by(keys).aggregate([(agg_col, agg_name)])
            except Exception as exc:
                msg = f"Failed executing {fn_name}: {exc}"
                raise BackendNotSupportedError(msg) from exc

    return False, None


def _dispatch_cbrt(
    fn_name: str,
    actual_args: tuple[object, ...],
    pc_mod: object,
) -> tuple[bool, object]:
    """Dispatch cube root via pc_mod power if applicable.

    Args:
        fn_name (str): Lowercase operation name.
        actual_args (tuple[object, ...]): Operation positional arguments.
        pc_mod (object): pyarrow.compute module.

    Returns:
        tuple[bool, object]: Handled flag and resulting object.

    Raises:
        BackendNotSupportedError: If cbrt execution fails.
    """
    if fn_name == "cbrt" and actual_args:
        try:
            return True, pc_mod.power(actual_args[0], 1.0 / 3.0)
        except Exception as exc:
            msg = f"Failed executing cbrt: {exc}"
            raise BackendNotSupportedError(msg) from exc
    return False, None


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

    # 1. GroupBy / Hash Aggregations on Table / RecordBatch
    handled_gb, res_gb = _dispatch_groupby(fn_name, actual_args, kwargs)
    if handled_gb:
        return res_gb

    # 2. Cube root
    handled_cbrt, res_cbrt = _dispatch_cbrt(fn_name, actual_args, pc_mod)
    if handled_cbrt:
        return res_cbrt

    # 3. Direct function lookup on pyarrow.compute
    if hasattr(pc_mod, fn_name):
        compute_fn = getattr(pc_mod, fn_name)
        try:
            return compute_fn(*actual_args, **kwargs)
        except Exception as exc:
            msg = f"Failed executing pyarrow.compute.{fn_name}: {exc}"
            raise BackendNotSupportedError(msg) from exc

    # 4. Lookup in standard operation mapping
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
