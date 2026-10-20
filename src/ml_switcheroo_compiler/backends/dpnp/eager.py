"""Eager evaluation dispatch for Data Parallel NumPy (dpnp)."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.dpnp.types import _get_dpnp_module
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable

_DPNP_OP_ALIASES: dict[str, str] = {
    # Unary arithmetic & elementary functions
    "abs": "abs",
    "neg": "negative",
    "negative": "negative",
    "sign": "sign",
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
    "reciprocal": "reciprocal",
    "floor": "floor",
    "ceil": "ceil",
    "trunc": "trunc",
    "rint": "rint",
    "round": "round",
    "logicalnot": "logical_not",
    "logical_not": "logical_not",
    "bitwisenot": "bitwise_not",
    "bitwise_not": "bitwise_not",
    "invert": "invert",
    "isnan": "isnan",
    "isinf": "isinf",
    "isfinite": "isfinite",
    # Binary math & arithmetic
    "add": "add",
    "sub": "subtract",
    "subtract": "subtract",
    "mul": "multiply",
    "multiply": "multiply",
    "div": "divide",
    "divide": "divide",
    "truedivide": "true_divide",
    "true_divide": "true_divide",
    "floordivide": "floor_divide",
    "floor_divide": "floor_divide",
    "power": "power",
    "pow": "power",
    "maximum": "maximum",
    "minimum": "minimum",
    "fmax": "fmax",
    "fmin": "fmin",
    "fmod": "fmod",
    "remainder": "remainder",
    "mod": "remainder",
    "hypot": "hypot",
    "atan2": "arctan2",
    "arctan2": "arctan2",
    "copysign": "copysign",
    "nextafter": "nextafter",
    # Comparisons & logical operations
    "equal": "equal",
    "notequal": "not_equal",
    "not_equal": "not_equal",
    "greater": "greater",
    "greaterequal": "greater_equal",
    "greater_equal": "greater_equal",
    "less": "less",
    "lessequal": "less_equal",
    "less_equal": "less_equal",
    "logicaland": "logical_and",
    "logical_and": "logical_and",
    "logicalor": "logical_or",
    "logical_or": "logical_or",
    "logicalxor": "logical_xor",
    "logical_xor": "logical_xor",
    "bitwiseand": "bitwise_and",
    "bitwise_and": "bitwise_and",
    "bitwiseor": "bitwise_or",
    "bitwise_or": "bitwise_or",
    "bitwisexor": "bitwise_xor",
    "bitwise_xor": "bitwise_xor",
    "leftshift": "left_shift",
    "left_shift": "left_shift",
    "rightshift": "right_shift",
    "right_shift": "right_shift",
    # Reductions
    "sum": "sum",
    "prod": "prod",
    "product": "prod",
    "mean": "mean",
    "std": "std",
    "var": "var",
    "max": "max",
    "amax": "amax",
    "reducemax": "amax",
    "min": "min",
    "amin": "amin",
    "reducemin": "amin",
    "argmax": "argmax",
    "argmin": "argmin",
    "all": "all",
    "any": "any",
    "cumsum": "cumsum",
    "cumprod": "cumprod",
    # Linear algebra & decompositions
    "dot": "dot",
    "matmul": "matmul",
    "tensordot": "tensordot",
    "outer": "outer",
    "inner": "inner",
    "kron": "kron",
    "trace": "trace",
    "transpose": "transpose",
    "cholesky": "linalg.cholesky",
    "det": "linalg.det",
    "inv": "linalg.inv",
    "matrix_power": "linalg.matrix_power",
    "qr": "linalg.qr",
    "svd": "linalg.svd",
    "eig": "linalg.eig",
    "eigh": "linalg.eigh",
    "norm": "linalg.norm",
    "solve": "linalg.solve",
    # Shape & manipulation operations
    "reshape": "reshape",
    "squeeze": "squeeze",
    "expanddims": "expand_dims",
    "expand_dims": "expand_dims",
    "flatten": "ravel",
    "concatenate": "concatenate",
    "concat": "concatenate",
    "stack": "stack",
    "split": "split",
    "tile": "tile",
    "repeat": "repeat",
    "clip": "clip",
    "where": "where",
    "broadcastto": "broadcast_to",
    "broadcast_to": "broadcast_to",
    "pad": "pad",
    "flip": "flip",
    "roll": "roll",
}


def _get_target_function(module: EagerValue, target_path: str) -> EagerValue | None:
    """Retrieve target function from module resolving dot-delimited submodules.

    Args:
        module (EagerValue): The root module (dpnp or numpy).
        target_path (str): Function attribute name or dotted path like 'linalg.cholesky'.

    Returns:
        object | None: Callable function if resolved, None otherwise.
    """
    curr = module
    for part in target_path.split("."):
        if not hasattr(curr, part):
            return None
        curr = getattr(curr, part)
    return curr if callable(curr) else None


def _unwrap_arg(arg: EagerValue) -> EagerValue:
    """Unwrap framework Tensor objects to underlying array buffers.

    Args:
        arg (EagerValue): Argument to inspect.

    Returns:
        EagerValue: Unwrapped buffer or original value.
    """
    if type(arg).__name__ == "Tensor" and hasattr(arg, "data"):
        return typing.cast(EagerValue, getattr(arg, "data"))
    return arg


def _resolve_dpnp_callable(dpnp_mod: EagerValue, fn_name: str) -> EagerValue | None:
    """Resolve callable function on the DPNP module via direct attribute, alias, or linalg submodule.

    Args:
        dpnp_mod (EagerValue): Loaded DPNP or fallback NumPy module.
        fn_name (str): Lowercase operation name.

    Returns:
        EagerValue | None: Resolved callable or None if not found.
    """
    if hasattr(dpnp_mod, fn_name):
        fn = getattr(dpnp_mod, fn_name)
        if callable(fn):
            return typing.cast(typing.Union[EagerValue, "None"], fn)

    target_path = _DPNP_OP_ALIASES.get(fn_name)
    if target_path is not None:
        fn = _get_target_function(dpnp_mod, target_path)
        if fn is not None:
            return fn

    linalg_mod = getattr(dpnp_mod, "linalg", None)
    if linalg_mod is not None and hasattr(linalg_mod, fn_name):
        fn = getattr(linalg_mod, fn_name)
        if callable(fn):
            return typing.cast(typing.Union[EagerValue, "None"], fn)

    return None


def execute_op(
    op_type: str,
    *args: EagerValue,
    **kwargs: EagerValue,
) -> EagerValue:
    """Execute operation eagerly on DPNP backend with Intel SYCL target.

    Args:
        op_type (str): Operation name string.
        *args (EagerValue): Input tensor parameters.
        **kwargs (EagerValue): Additional keyword attributes.

    Returns:
        EagerValue: Evaluated DPNP array.

    Raises:
        BackendNotSupportedError: When the operation cannot be resolved.
    """
    actual_args = tuple(_unwrap_arg(a) for a in args)
    dpnp_mod = _get_dpnp_module()

    fn = _resolve_dpnp_callable(dpnp_mod, op_type.lower())
    if fn is not None:
        return typing.cast(TypedVariadicCallable[EagerValue], fn)(*actual_args, **kwargs)

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    registered_func = global_eager_registry.get(op_type)
    if registered_func is not None:
        return registered_func(dpnp_mod, *actual_args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' not supported in DPNP backend.")
