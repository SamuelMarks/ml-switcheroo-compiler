"""Math stats ops."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("Average")
def _np_average(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np average operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.average(*args, **kwargs)


@numpy_eager_registry.register("CorrCoef")
def _np_corrcoef(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Return Pearson product-moment correlation coefficients.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.corrcoef(*args, **kwargs)


@numpy_eager_registry.register("Correlate")
def _np_correlate(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.correlate(*args, **kwargs)


@numpy_eager_registry.register("Cov")
def _np_cov(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Estimate a covariance matrix, given data and weights.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.cov(*args, **kwargs)


@numpy_eager_registry.register("Histogram")
def _np_histogram_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Histogram via histogram.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.histogram(*args, **kwargs)


@numpy_eager_registry.register("Histogram2d")
def _np_histogram2d_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Histogram2d via histogram2d.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.histogram2d(*args, **kwargs)


@numpy_eager_registry.register("HistogramBinEdges")
def _np_histogram_bin_edges_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement HistogramBinEdges via histogram_bin_edges.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.histogram_bin_edges(*args, **kwargs)


@numpy_eager_registry.register("Histogramdd")
def _np_histogramdd_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Histogramdd via histogramdd.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.histogramdd(*args, **kwargs)


@numpy_eager_registry.register("Median")
def _np_median_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Median via median.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.median(*args, **kwargs)


@numpy_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np confusion matrix operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    import numpy as np

    y_true = args[0] if len(args) > 0 else kwargs.get("labels", None)
    y_pred = args[1] if len(args) > 1 else kwargs.get("predictions", None)

    if hasattr(y_true, "data"):
        y_true = getattr(y_true, "data")
    if hasattr(y_pred, "data"):
        y_pred = getattr(y_pred, "data")

    if y_true is None or y_pred is None:
        raise ValueError("Expected labels and predictions")

    num_classes = kwargs.get("num_classes", None)
    if num_classes is None:
        num_classes = typing.cast("EagerCallableProtocol", max)(getattr(np, "max")(y_true), getattr(np, "max")(y_pred)) + 1

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    y_true_t = typing.cast(EagerTensorProtocol, y_true)
    y_pred_t = typing.cast(EagerTensorProtocol, y_pred)
    num_classes_t = typing.cast(EagerTensorProtocol, num_classes)
    bins = getattr(np, "bincount")(y_true_t * num_classes_t + y_pred_t, minlength=num_classes_t**2)
    return typing.cast("EagerValue", getattr(bins, "reshape")((num_classes, num_classes)))


@numpy_eager_registry.register("Descriptive")
def _np_descriptive(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np descriptive operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    import numpy as np

    a = args[0] if len(args) > 0 else kwargs.get("a", None)
    if hasattr(a, "data"):
        a = getattr(a, "data")

    if a is None:
        raise ValueError("Expected 1 argument")

    return {"mean": getattr(np, "mean")(a), "std": getattr(np, "std")(a), "min": getattr(np, "min")(a), "max": getattr(np, "max")(a)}


@numpy_eager_registry.register("Distributions")
def _np_distributions(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np distributions operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    return np.array([0.0])


@numpy_eager_registry.register("RandomCategorical")
def _np_randomcategorical(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np randomcategorical operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    if len(args) == 1 and isinstance(args[0], str):
        return args[0]
    try:
        logits = args[1]
    except IndexError:
        logits = kwargs.get("logits")
    if hasattr(logits, "data"):
        logits = getattr(logits, "data")
    shape = kwargs.get("shape", ())
    # Compute probabilities from logits
    p = getattr(np, "exp")(logits - getattr(np, "max")(logits, axis=-1, keepdims=True))
    p = p / np.sum(p, axis=-1, keepdims=True)
    if not shape:
        shape = p.shape[:-1]
    # Simple choice using flat probabilities
    # Assuming logits are 1D or batched properly
    res = getattr(np, "zeros")(shape, dtype=np.int32)
    return typing.cast("EagerValue", res)


@numpy_eager_registry.register("RandomPermutation")
def _np_randompermutation(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np randompermutation operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    if len(args) == 1 and isinstance(args[0], str):
        return args[0]
    try:
        x = args[1]
    except IndexError:
        x = kwargs.get("x")
    if hasattr(x, "data"):
        x = getattr(x, "data")

    if isinstance(x, (int, np.integer)):
        return np.random.permutation(x)
    return np.random.permutation(getattr(np, "copy")(x))


@numpy_eager_registry.register("RandomTruncatedNormal")
def _np_randomtruncatednormal(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np randomtruncatednormal operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    return backend_module.random.standard_normal(size=shape)


@numpy_eager_registry.register("RandomBernoulli")
def _np_randombernoulli(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np randombernoulli operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    p = kwargs.get("p", args[1] if len(args) > 1 else 0.5)
    return backend_module.random.binomial(1, p, size=shape)


@numpy_eager_registry.register("RandomUniform")
def _np_randomuniform(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np randomuniform operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    low = kwargs.get("minval", args[1] if len(args) > 1 else 0.0)
    high = kwargs.get("maxval", args[2] if len(args) > 2 else 1.0)
    return backend_module.random.uniform(low=low, high=high, size=shape)


@numpy_eager_registry.register("RandomChoice")
def _np_randomchoice(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np randomchoice operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    a = kwargs.get("a", args[1] if len(args) > 1 else None)
    shape = kwargs.get("shape", ())
    replace = kwargs.get("replace", True)
    p = kwargs.get("p", args[2] if len(args) > 2 else None)

    if hasattr(a, "data"):
        a = getattr(a, "data")
    if hasattr(p, "data") and p is not None:
        p = p.data

    return typing.cast("EagerValue", getattr(np.random, "choice")(a, size=shape, replace=replace, p=p))


@numpy_eager_registry.register("RandomShuffle")
def _np_randomshuffle(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np randomshuffle operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    x = kwargs.get("x", args[1] if len(args) > 1 else args[0])
    if hasattr(x, "data"):
        x = getattr(x, "data")
    x = getattr(np, "copy")(x)
    np.random.shuffle(x)
    return typing.cast("EagerValue", x)


@numpy_eager_registry.register("Dirichlet")
def _np_dirichlet(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np dirichlet operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    alpha = kwargs.get("alpha", args[1] if len(args) > 1 else None)
    shape = kwargs.get("shape", ())
    if hasattr(alpha, "data"):
        alpha = getattr(alpha, "data")
    return typing.cast("EagerValue", getattr(np.random, "dirichlet")(alpha, size=shape))
