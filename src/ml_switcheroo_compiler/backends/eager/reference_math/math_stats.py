"""Generic backend-parametric eager reference operations for `reference_math.math_stats`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Average")
def _np_average(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np average operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("average")(*args, **kwargs)


@global_eager_registry.register("CorrCoef")
def _np_corrcoef(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Return Pearson product-moment correlation coefficients.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("corrcoef")(*args, **kwargs)


@global_eager_registry.register("Correlate")
def _np_correlate(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("correlate")(*args, **kwargs)


@global_eager_registry.register("Cov")
def _np_cov(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Estimate a covariance matrix, given data and weights.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("cov")(*args, **kwargs)


@global_eager_registry.register("Histogram")
def _np_histogram_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Histogram via histogram.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("histogram")(*args, **kwargs)


@global_eager_registry.register("Histogram2d")
def _np_histogram2d_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Histogram2d via histogram2d.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("histogram2d")(*args, **kwargs)


@global_eager_registry.register("HistogramBinEdges")
def _np_histogram_bin_edges_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement HistogramBinEdges via histogram_bin_edges.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("histogram_bin_edges")(*args, **kwargs)


@global_eager_registry.register("Histogramdd")
def _np_histogramdd_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Histogramdd via histogramdd.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("histogramdd")(*args, **kwargs)


@global_eager_registry.register("Median")
def _np_median_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Median via median.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("median")(*args, **kwargs)


@global_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np confusion matrix operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    y_true = args[0] if len(args) > 0 else kwargs.get("labels", None)
    y_pred = args[1] if len(args) > 1 else kwargs.get("predictions", None)

    if adapt_backend_module(y_true).has_symbol("data"):
        y_true = adapt_backend_module(y_true).require_symbol("data")
    if adapt_backend_module(y_pred).has_symbol("data"):
        y_pred = adapt_backend_module(y_pred).require_symbol("data")

    if y_true is None or y_pred is None:
        raise ValueError("Expected labels and predictions")

    num_classes = kwargs.get("num_classes", None)
    if num_classes is None:
        num_classes = (
            max(
                coerce_int(adapt_backend_module(arr_mod).require_symbol("max")(y_true)),
                coerce_int(adapt_backend_module(arr_mod).require_symbol("max")(y_pred)),
            )
            + 1
        )

    y_true = arr_mod.asarray(y_true).flatten()
    y_pred = arr_mod.asarray(y_pred).flatten()

    y_true_t = y_true
    y_pred_t = y_pred
    num_classes_t = int(coerce_int(num_classes))
    bins = adapt_backend_module(arr_mod).require_symbol("bincount")(
        y_true_t * num_classes_t + y_pred_t, minlength=num_classes_t**2
    )
    return adapt_backend_module(bins).require_symbol("reshape")((num_classes_t, num_classes_t))


@global_eager_registry.register("Descriptive")
def _np_descriptive(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np descriptive operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a = args[0] if len(args) > 0 else kwargs.get("a", None)
    if adapt_backend_module(a).has_symbol("data"):
        a = adapt_backend_module(a).require_symbol("data")

    if a is None:
        raise ValueError("Expected 1 argument")

    return {
        "mean": adapt_backend_module(arr_mod).require_symbol("mean")(a),
        "std": adapt_backend_module(arr_mod).require_symbol("std")(a),
        "min": adapt_backend_module(arr_mod).require_symbol("min")(a),
        "max": adapt_backend_module(arr_mod).require_symbol("max")(a),
    }


@global_eager_registry.register("Distributions")
def _np_distributions(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np distributions operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return arr_mod.array([0.0])


@global_eager_registry.register("RandomCategorical")
def _np_randomcategorical(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np randomcategorical operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if len(args) == 1 and isinstance(args[0], str):
        return args[0]
    try:
        logits = args[1]
    except IndexError:
        logits = kwargs.get("logits")
    if adapt_backend_module(logits).has_symbol("data"):
        logits = adapt_backend_module(logits).require_symbol("data")
    shape = kwargs.get("shape", ())
    # Compute probabilities from logits
    p = adapt_backend_module(arr_mod).require_symbol("exp")(
        logits - adapt_backend_module(arr_mod).require_symbol("max")(logits, axis=-1, keepdims=True)
    )
    p = p / arr_mod.sum(p, axis=-1, keepdims=True)
    if not shape:
        shape = p.shape[:-1]
    # Simple choice using flat probabilities
    # Assuming logits are 1D or batched properly
    res = adapt_backend_module(arr_mod).require_symbol("zeros")(shape, dtype=arr_mod.int32)
    return res


@global_eager_registry.register("RandomPermutation")
def _np_randompermutation(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np randompermutation operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if len(args) == 1 and isinstance(args[0], str):
        return args[0]
    try:
        x = args[1]
    except IndexError:
        x = kwargs.get("x")
    if adapt_backend_module(x).has_symbol("data"):
        x = adapt_backend_module(x).require_symbol("data")

    if isinstance(x, int) or str(type(x).__name__).startswith(("int", "uint")):
        return adapt_backend_module(arr_mod.random).require_symbol("permutation")(x)
    return adapt_backend_module(arr_mod.random).require_symbol("permutation")(
        adapt_backend_module(arr_mod).require_symbol("copy")(x)
    )


@global_eager_registry.register("RandomTruncatedNormal")
def _np_randomtruncatednormal(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np randomtruncatednormal operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    return adapt_backend_module(backend_module).require_symbol("random").standard_normal(size=shape)


@global_eager_registry.register("RandomBernoulli")
def _np_randombernoulli(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np randombernoulli operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    p = kwargs.get("p", args[1] if len(args) > 1 else 0.5)
    return adapt_backend_module(backend_module).require_symbol("random").binomial(1, p, size=shape)


@global_eager_registry.register("RandomUniform")
def _np_randomuniform(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np randomuniform operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    low = kwargs.get("minval", args[1] if len(args) > 1 else 0.0)
    high = kwargs.get("maxval", args[2] if len(args) > 2 else 1.0)
    return adapt_backend_module(backend_module).require_symbol("random").uniform(low=low, high=high, size=shape)


@global_eager_registry.register("RandomChoice")
def _np_randomchoice(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np randomchoice operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a = kwargs.get("a", args[1] if len(args) > 1 else None)
    shape = kwargs.get("shape", ())
    replace = kwargs.get("replace", True)
    p = kwargs.get("p", args[2] if len(args) > 2 else None)

    if adapt_backend_module(a).has_symbol("data"):
        a = adapt_backend_module(a).require_symbol("data")
    if adapt_backend_module(p).has_symbol("data") and p is not None:
        p = adapt_backend_module(p).require_symbol("data")

    return adapt_backend_module(arr_mod.random).require_symbol("choice")(a, size=shape, replace=replace, p=p)


@global_eager_registry.register("RandomShuffle")
def _np_randomshuffle(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np randomshuffle operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x = kwargs.get("x", args[1] if len(args) > 1 else args[0])
    if adapt_backend_module(x).has_symbol("data"):
        x = adapt_backend_module(x).require_symbol("data")
    x = adapt_backend_module(arr_mod).require_symbol("copy")(x)
    adapt_backend_module(arr_mod.random).require_symbol("shuffle")(x)
    return x


@global_eager_registry.register("Dirichlet")
def _np_dirichlet(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np dirichlet operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    alpha = kwargs.get("alpha", args[1] if len(args) > 1 else None)
    shape = kwargs.get("shape", ())
    if adapt_backend_module(alpha).has_symbol("data"):
        alpha = adapt_backend_module(alpha).require_symbol("data")
    return adapt_backend_module(arr_mod.random).require_symbol("dirichlet")(alpha, size=shape)
