# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Misc operations."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Histogram")
class Histogram(OpDef):
    """Compute the histogram of a dataset."""

    op_name = "Histogram"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        bins = kwargs.get("bins", 10)
        if hasattr(bins, "shape") and len(bins.shape) > 0:
            return (bins.shape[0] - 1,)
        if isinstance(bins, int):
            return (bins,)
        return (10,)


@register_op("Histogram2d")
class Histogram2d(OpDef):
    """Compute the bi-dimensional histogram of two data samples."""

    op_name = "Histogram2d"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        bins = kwargs.get("bins", 10)
        if isinstance(bins, (list, tuple)):
            if len(bins) == 2:
                b1, b2 = bins
                b1_len = b1 if isinstance(b1, int) else (b1.shape[0] - 1 if hasattr(b1, "shape") else 10)
                b2_len = b2 if isinstance(b2, int) else (b2.shape[0] - 1 if hasattr(b2, "shape") else 10)
                return (b1_len, b2_len)
        return (10, 10)


@register_op("HistogramBinEdges")
class HistogramBinEdges(OpDef):
    """Provide function to calculate only the edges of the bins used by the histogram function."""

    op_name = "HistogramBinEdges"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        bins = kwargs.get("bins", 10)
        if hasattr(bins, "shape") and len(bins.shape) > 0:
            return bins.shape
        if isinstance(bins, int):
            return (bins + 1,)
        return (11,)


@register_op("Histogramdd")
class Histogramdd(OpDef):
    """Compute the multidimensional histogram of some data."""

    op_name = "Histogramdd"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if not args:
            return ()
        sample = args[0]
        n_dim = sample.shape[1] if (hasattr(sample, "shape") and len(sample.shape) == 2) else 1
        return tuple(10 for _ in range(n_dim))


def histogram(*args, **kwargs):
    """Evaluate histogram operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Histogram", *args, **kwargs)


def histogram2d(*args, **kwargs):
    """Evaluate histogram2d operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Histogram2d", *args, **kwargs)


def histogram_bin_edges(*args, **kwargs):
    """Provide function to calculate only the edges of the bins used by the histogram function.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("HistogramBinEdges", *args, **kwargs)


def histogramdd(*args, **kwargs):
    """Evaluate histogramdd operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Histogramdd", *args, **kwargs)
