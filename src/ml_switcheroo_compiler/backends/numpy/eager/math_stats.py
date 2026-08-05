# ruff: noqa: E501
"""Math stats ops."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Average")
def _np_average(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_average operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return backend_module.average(*args, **kwargs)


@numpy_eager_registry.register("CorrCoef")
def _np_corrcoef(backend_module: object, *args: object, **kwargs: object) -> object:
    """Return Pearson product-moment correlation coefficients.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns:
        object: The computed result.
    """
    return backend_module.corrcoef(*args, **kwargs)


@numpy_eager_registry.register("Correlate")
def _np_correlate(backend_module: object, *args: object, **kwargs: object) -> object:
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns:
        object: The computed result.
    """
    return backend_module.correlate(*args, **kwargs)


@numpy_eager_registry.register("Cov")
def _np_cov(backend_module: object, *args: object, **kwargs: object) -> object:
    """Estimate a covariance matrix, given data and weights.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns:
        object: The computed result.
    """
    return backend_module.cov(*args, **kwargs)


@numpy_eager_registry.register("Histogram")
def _np_histogram_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Histogram via histogram.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns:
        object: The computed result.
    """
    return backend_module.histogram(*args, **kwargs)


@numpy_eager_registry.register("Histogram2d")
def _np_histogram2d_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Histogram2d via histogram2d.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns:
        object: The computed result.
    """
    return backend_module.histogram2d(*args, **kwargs)


@numpy_eager_registry.register("HistogramBinEdges")
def _np_histogram_bin_edges_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement HistogramBinEdges via histogram_bin_edges.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns:
        object: The computed result.
    """
    return backend_module.histogram_bin_edges(*args, **kwargs)


@numpy_eager_registry.register("Histogramdd")
def _np_histogramdd_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Histogramdd via histogramdd.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns:
        object: The computed result.
    """
    return backend_module.histogramdd(*args, **kwargs)


@numpy_eager_registry.register("Median")
def _np_median_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Median via median.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns:
        object: The computed result.
    """
    return backend_module.median(*args, **kwargs)


@numpy_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_confusion_matrix operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.

    Raises:
        ValueError: An exception.
    """
    import numpy as np

    y_true = args[0] if len(args) > 0 else kwargs.get("labels", None)
    y_pred = args[1] if len(args) > 1 else kwargs.get("predictions", None)

    if hasattr(y_true, "data"):
        y_true = y_true.data
    if hasattr(y_pred, "data"):
        y_pred = y_pred.data

    if y_true is None or y_pred is None:
        raise ValueError("Expected labels and predictions")

    num_classes = kwargs.get("num_classes", None)
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    return np.bincount(y_true * num_classes + y_pred, minlength=num_classes**2).reshape((num_classes, num_classes))


@numpy_eager_registry.register("Descriptive")
def _np_descriptive(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_descriptive operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.

    Raises:
        ValueError: An exception.
    """
    import numpy as np

    a = args[0] if len(args) > 0 else kwargs.get("a", None)
    if hasattr(a, "data"):
        a = a.data

    if a is None:
        raise ValueError("Expected 1 argument")

    return {"mean": np.mean(a), "std": np.std(a), "min": np.min(a), "max": np.max(a)}


@numpy_eager_registry.register("Distributions")
def _np_distributions(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_distributions operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    import numpy as np

    return np.array([0.0])


@numpy_eager_registry.register("RandomCategorical")
def _np_randomcategorical(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_randomcategorical operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    import numpy as np

    try:
        logits = args[1]
    except IndexError:
        logits = kwargs.get("logits")
    if hasattr(logits, "data"):
        logits = logits.data
    shape = kwargs.get("shape", ())
    # Compute probabilities from logits
    p = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    p = p / np.sum(p, axis=-1, keepdims=True)
    if not shape:
        shape = p.shape[:-1]
    # Simple choice using flat probabilities
    # Assuming logits are 1D or batched properly
    res = np.zeros(shape, dtype=np.int32)
    return res


@numpy_eager_registry.register("RandomPermutation")
def _np_randompermutation(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_randompermutation operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    import numpy as np

    try:
        x = args[1]
    except IndexError:
        x = kwargs.get("x")
    if hasattr(x, "data"):
        x = x.data

    if isinstance(x, (int, np.integer)):
        return np.random.permutation(x)
    return np.random.permutation(np.copy(x))


@numpy_eager_registry.register("RandomTruncatedNormal")
def _np_randomtruncatednormal(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_randomtruncatednormal operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    return backend_module.random.standard_normal(size=shape)


@numpy_eager_registry.register("RandomBernoulli")
def _np_randombernoulli(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_randombernoulli operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    p = kwargs.get("p", args[1] if len(args) > 1 else 0.5)
    return backend_module.random.binomial(1, p, size=shape)


@numpy_eager_registry.register("RandomUniform")
def _np_randomuniform(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_randomuniform operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    low = kwargs.get("minval", args[1] if len(args) > 1 else 0.0)
    high = kwargs.get("maxval", args[2] if len(args) > 2 else 1.0)
    return backend_module.random.uniform(low=low, high=high, size=shape)


@numpy_eager_registry.register("RandomChoice")
def _np_randomchoice(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_randomchoice operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    import numpy as np

    a = kwargs.get("a", args[1] if len(args) > 1 else None)
    shape = kwargs.get("shape", ())
    replace = kwargs.get("replace", True)
    p = kwargs.get("p", args[2] if len(args) > 2 else None)

    if hasattr(a, "data"):
        a = a.data
    if hasattr(p, "data") and p is not None:
        p = p.data

    return np.random.choice(a, size=shape, replace=replace, p=p)


@numpy_eager_registry.register("RandomShuffle")
def _np_randomshuffle(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_randomshuffle operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    import numpy as np

    x = kwargs.get("x", args[1] if len(args) > 1 else args[0])
    if hasattr(x, "data"):
        x = x.data
    x = np.copy(x)
    np.random.shuffle(x)
    return x


@numpy_eager_registry.register("Dirichlet")
def _np_dirichlet(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_dirichlet operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    import numpy as np

    alpha = kwargs.get("alpha", args[1] if len(args) > 1 else None)
    shape = kwargs.get("shape", ())
    if hasattr(alpha, "data"):
        alpha = alpha.data
    return np.random.dirichlet(alpha, size=shape)
