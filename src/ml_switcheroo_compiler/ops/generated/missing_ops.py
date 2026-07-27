"""Missing ops."""

from ml_switcheroo_compiler.ops.dispatcher import dispatch_op


def flatnonzero(*args: object, **kwargs: object) -> object:
    """Flatnonzero frontend."""
    return dispatch_op("Flatnonzero", *args, **kwargs)


def from_dlpack(*args: object, **kwargs: object) -> object:
    """FromDlpack frontend."""
    return dispatch_op("FromDlpack", *args, **kwargs)


def fromfile(*args: object, **kwargs: object) -> object:
    """Fromfile frontend."""
    return dispatch_op("Fromfile", *args, **kwargs)


def fromfunction(*args: object, **kwargs: object) -> object:
    """Fromfunction frontend."""
    return dispatch_op("Fromfunction", *args, **kwargs)


def fromiter(*args: object, **kwargs: object) -> object:
    """Fromiter frontend."""
    return dispatch_op("Fromiter", *args, **kwargs)


def frompyfunc(*args: object, **kwargs: object) -> object:
    """Frompyfunc frontend."""
    return dispatch_op("Frompyfunc", *args, **kwargs)


def fromstring(*args: object, **kwargs: object) -> object:
    """Fromstring frontend."""
    return dispatch_op("Fromstring", *args, **kwargs)


def geometric(*args: object, **kwargs: object) -> object:
    """Geometric frontend."""
    return dispatch_op("Geometric", *args, **kwargs)


def geomspace(*args: object, **kwargs: object) -> object:
    """Geomspace frontend."""
    return dispatch_op("Geomspace", *args, **kwargs)


def get_printoptions(*args: object, **kwargs: object) -> object:
    """GetPrintoptions frontend."""
    return dispatch_op("GetPrintoptions", *args, **kwargs)


def gradient(*args: object, **kwargs: object) -> object:
    """Gradient frontend."""
    return dispatch_op("Gradient", *args, **kwargs)


def hard_silu(*args: object, **kwargs: object) -> object:
    """HardSilu frontend."""
    return dispatch_op("HardSilu", *args, **kwargs)


def hard_swish(*args: object, **kwargs: object) -> object:
    """HardSwish frontend."""
    return dispatch_op("HardSwish", *args, **kwargs)


def histogram(*args: object, **kwargs: object) -> object:
    """Histogram frontend."""
    return dispatch_op("Histogram", *args, **kwargs)


def histogram_bin_edges(*args: object, **kwargs: object) -> object:
    """HistogramBinEdges frontend."""
    return dispatch_op("HistogramBinEdges", *args, **kwargs)


def histogram2d(*args: object, **kwargs: object) -> object:
    """Histogram2d frontend."""
    return dispatch_op("Histogram2d", *args, **kwargs)


def histogramdd(*args: object, **kwargs: object) -> object:
    """Histogramdd frontend."""
    return dispatch_op("Histogramdd", *args, **kwargs)


def i0(*args: object, **kwargs: object) -> object:
    """I0 frontend."""
    return dispatch_op("I0", *args, **kwargs)


def igamma_grad_a(*args: object, **kwargs: object) -> object:
    """IgammaGradA frontend."""
    return dispatch_op("IgammaGradA", *args, **kwargs)


def iinfo(*args: object, **kwargs: object) -> object:
    """Iinfo frontend."""
    return dispatch_op("Iinfo", *args, **kwargs)


def index_in_dim(*args: object, **kwargs: object) -> object:
    """IndexInDim frontend."""
    return dispatch_op("IndexInDim", *args, **kwargs)


def indices(*args: object, **kwargs: object) -> object:
    """Indices frontend."""
    return dispatch_op("Indices", *args, **kwargs)


def infeed(*args: object, **kwargs: object) -> object:
    """Infeed frontend."""
    return dispatch_op("Infeed", *args, **kwargs)


def interp(*args: object, **kwargs: object) -> object:
    """Interp frontend."""
    return dispatch_op("Interp", *args, **kwargs)


def intersect1d(*args: object, **kwargs: object) -> object:
    """Intersect1d frontend."""
    return dispatch_op("Intersect1d", *args, **kwargs)


def iscomplex(*args: object, **kwargs: object) -> object:
    """Iscomplex frontend."""
    return dispatch_op("Iscomplex", *args, **kwargs)


def iscomplexobj(*args: object, **kwargs: object) -> object:
    """Iscomplexobj frontend."""
    return dispatch_op("Iscomplexobj", *args, **kwargs)


def isin(*args: object, **kwargs: object) -> object:
    """Isin frontend."""
    return dispatch_op("Isin", *args, **kwargs)


def isreal(*args: object, **kwargs: object) -> object:
    """Isreal frontend."""
    return dispatch_op("Isreal", *args, **kwargs)


def isrealobj(*args: object, **kwargs: object) -> object:
    """Isrealobj frontend."""
    return dispatch_op("Isrealobj", *args, **kwargs)


def isscalar(*args: object, **kwargs: object) -> object:
    """Isscalar frontend."""
    return dispatch_op("Isscalar", *args, **kwargs)


def issubdtype(*args: object, **kwargs: object) -> object:
    """Issubdtype frontend."""
    return dispatch_op("Issubdtype", *args, **kwargs)


def iterable(*args: object, **kwargs: object) -> object:
    """Iterable frontend."""
    return dispatch_op("Iterable", *args, **kwargs)


def ix_(*args: object, **kwargs: object) -> object:
    """Ix frontend."""
    return dispatch_op("Ix", *args, **kwargs)


def kron(*args: object, **kwargs: object) -> object:
    """Kron frontend."""
    return dispatch_op("Kron", *args, **kwargs)


def lexsort(*args: object, **kwargs: object) -> object:
    """Lexsort frontend."""
    return dispatch_op("Lexsort", *args, **kwargs)


def mask_indices(*args: object, **kwargs: object) -> object:
    """MaskIndices frontend."""
    return dispatch_op("MaskIndices", *args, **kwargs)


def median(*args: object, **kwargs: object) -> object:
    """Median frontend."""
    return dispatch_op("Median", *args, **kwargs)


def mgrid(*args: object, **kwargs: object) -> object:
    """Mgrid frontend."""
    return dispatch_op("Mgrid", *args, **kwargs)


def mish(*args: object, **kwargs: object) -> object:
    """Mish frontend."""
    return dispatch_op("Mish", *args, **kwargs)


def modf(*args: object, **kwargs: object) -> object:
    """Modf frontend."""
    return dispatch_op("Modf", *args, **kwargs)


def nonzero(*args: object, **kwargs: object) -> object:
    """Nonzero frontend."""
    return dispatch_op("Nonzero", *args, **kwargs)


def ogrid(*args: object, **kwargs: object) -> object:
    """Ogrid frontend."""
    return dispatch_op("Ogrid", *args, **kwargs)


def outfeed(*args: object, **kwargs: object) -> object:
    """Outfeed frontend."""
    return dispatch_op("Outfeed", *args, **kwargs)


def pdot(*args: object, **kwargs: object) -> object:
    """Pdot frontend."""
    return dispatch_op("Pdot", *args, **kwargs)


def percentile(*args: object, **kwargs: object) -> object:
    """Percentile frontend."""
    return dispatch_op("Percentile", *args, **kwargs)


def piecewise(*args: object, **kwargs: object) -> object:
    """Piecewise frontend."""
    return dispatch_op("Piecewise", *args, **kwargs)


def pmax(*args: object, **kwargs: object) -> object:
    """Pmax frontend."""
    return dispatch_op("Pmax", *args, **kwargs)


def pmin(*args: object, **kwargs: object) -> object:
    """Pmin frontend."""
    return dispatch_op("Pmin", *args, **kwargs)


def population_count(*args: object, **kwargs: object) -> object:
    """PopulationCount frontend."""
    return dispatch_op("PopulationCount", *args, **kwargs)


def ppermute(*args: object, **kwargs: object) -> object:
    """Ppermute frontend."""
    return dispatch_op("Ppermute", *args, **kwargs)


def promote_types(*args: object, **kwargs: object) -> object:
    """PromoteTypes frontend."""
    return dispatch_op("PromoteTypes", *args, **kwargs)


def pshuffle(*args: object, **kwargs: object) -> object:
    """Pshuffle frontend."""
    return dispatch_op("Pshuffle", *args, **kwargs)


def psum_scatter(*args: object, **kwargs: object) -> object:
    """PsumScatter frontend."""
    return dispatch_op("PsumScatter", *args, **kwargs)


def pswapaxes(*args: object, **kwargs: object) -> object:
    """Pswapaxes frontend."""
    return dispatch_op("Pswapaxes", *args, **kwargs)


def quantile(*args: object, **kwargs: object) -> object:
    """Quantile frontend."""
    return dispatch_op("Quantile", *args, **kwargs)


def r_(*args: object, **kwargs: object) -> object:
    """R frontend."""
    return dispatch_op("R", *args, **kwargs)


def rademacher(*args: object, **kwargs: object) -> object:
    """Rademacher frontend."""
    return dispatch_op("Rademacher", *args, **kwargs)


def random_gamma_grad(*args: object, **kwargs: object) -> object:
    """RandomGammaGrad frontend."""
    return dispatch_op("RandomGammaGrad", *args, **kwargs)


def ravel_multi_index(*args: object, **kwargs: object) -> object:
    """RavelMultiIndex frontend."""
    return dispatch_op("RavelMultiIndex", *args, **kwargs)


def reduce_precision(*args: object, **kwargs: object) -> object:
    """ReducePrecision frontend."""
    return dispatch_op("ReducePrecision", *args, **kwargs)


def repeat(*args: object, **kwargs: object) -> object:
    """Repeat frontend."""
    return dispatch_op("Repeat", *args, **kwargs)


def result_type(*args: object, **kwargs: object) -> object:
    """ResultType frontend."""
    return dispatch_op("ResultType", *args, **kwargs)


def rot90(*args: object, **kwargs: object) -> object:
    """Rot90 frontend."""
    return dispatch_op("Rot90", *args, **kwargs)


def searchsorted(*args: object, **kwargs: object) -> object:
    """Searchsorted frontend."""
    return dispatch_op("Searchsorted", *args, **kwargs)


def sort_complex(*args: object, **kwargs: object) -> object:
    """SortComplex frontend."""
    return dispatch_op("SortComplex", *args, **kwargs)


def sort_key_val(*args: object, **kwargs: object) -> object:
    """SortKeyVal frontend."""
    return dispatch_op("SortKeyVal", *args, **kwargs)


def squareplus(*args: object, **kwargs: object) -> object:
    """Squareplus frontend."""
    return dispatch_op("Squareplus", *args, **kwargs)


def tile(*args: object, **kwargs: object) -> object:
    """Tile frontend."""
    return dispatch_op("Tile", *args, **kwargs)


def trapezoid(*args: object, **kwargs: object) -> object:
    """Trapezoid frontend."""
    return dispatch_op("Trapezoid", *args, **kwargs)


def tri(*args: object, **kwargs: object) -> object:
    """Tri frontend."""
    return dispatch_op("Tri", *args, **kwargs)


def tril(*args: object, **kwargs: object) -> object:
    """Tril frontend."""
    return dispatch_op("Tril", *args, **kwargs)


def trim_zeros(*args: object, **kwargs: object) -> object:
    """TrimZeros frontend."""
    return dispatch_op("TrimZeros", *args, **kwargs)


def triu(*args: object, **kwargs: object) -> object:
    """Triu frontend."""
    return dispatch_op("Triu", *args, **kwargs)


def unique(*args: object, **kwargs: object) -> object:
    """Unique frontend."""
    return dispatch_op("Unique", *args, **kwargs)


def unwrap(*args: object, **kwargs: object) -> object:
    """Unwrap frontend."""
    return dispatch_op("Unwrap", *args, **kwargs)


def update_slice(*args: object, **kwargs: object) -> object:
    """UpdateSlice frontend."""
    return dispatch_op("UpdateSlice", *args, **kwargs)


def vander(*args: object, **kwargs: object) -> object:
    """Vander frontend."""
    return dispatch_op("Vander", *args, **kwargs)


def vectorize(*args: object, **kwargs: object) -> object:
    """Vectorize frontend."""
    return dispatch_op("Vectorize", *args, **kwargs)


__all__ = [
    "flatnonzero",
    "from_dlpack",
    "fromfile",
    "fromfunction",
    "fromiter",
    "frompyfunc",
    "fromstring",
    "geometric",
    "geomspace",
    "get_printoptions",
    "gradient",
    "hard_silu",
    "hard_swish",
    "histogram",
    "histogram2d",
    "histogram_bin_edges",
    "histogramdd",
    "i0",
    "igamma_grad_a",
    "iinfo",
    "index_in_dim",
    "indices",
    "infeed",
    "interp",
    "intersect1d",
    "iscomplex",
    "iscomplexobj",
    "isin",
    "isreal",
    "isrealobj",
    "isscalar",
    "issubdtype",
    "iterable",
    "ix_",
    "kron",
    "lexsort",
    "mask_indices",
    "median",
    "mgrid",
    "mish",
    "modf",
    "nonzero",
    "ogrid",
    "outfeed",
    "pdot",
    "percentile",
    "piecewise",
    "pmax",
    "pmin",
    "population_count",
    "ppermute",
    "promote_types",
    "pshuffle",
    "psum_scatter",
    "pswapaxes",
    "quantile",
    "r_",
    "rademacher",
    "random_gamma_grad",
    "ravel_multi_index",
    "reduce_precision",
    "repeat",
    "result_type",
    "rot90",
    "searchsorted",
    "sort_complex",
    "sort_key_val",
    "squareplus",
    "tile",
    "trapezoid",
    "tri",
    "tril",
    "trim_zeros",
    "triu",
    "unique",
    "unwrap",
    "update_slice",
    "vander",
    "vectorize",
]
