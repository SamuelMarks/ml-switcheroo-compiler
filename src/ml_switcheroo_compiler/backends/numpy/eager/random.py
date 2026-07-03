"""Module docstring."""

import functools
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2


def _unwrap(x: object) -> object:
    """Function docstring."""
    if type(x).__name__ == "Tensor" and hasattr(x, "data"):
        x = x.data
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        if x.ndim == 1 and x.size == 0:
            return ()
    if isinstance(x, (tuple, list)):
        return tuple(_unwrap(i) for i in x)
    return x


def unwrap_args(func: object) -> object:
    """Function docstring."""

    @functools.wraps(func)
    def wrapper(np_mod: object, *args: object, **kwargs: object) -> object:
        """Function docstring."""
        args = [_unwrap(a) for a in args]
        kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
        return np_mod.asarray(func(np_mod, *args, **kwargs))

    return wrapper


def pack_distribution_params(func: object) -> object:
    """Function docstring."""

    @functools.wraps(func)
    def wrapper(np_mod: object, key: object, *args: object, **kwargs: object) -> object:
        """Function docstring."""
        shape = kwargs.pop("shape", None)
        dtype = kwargs.pop("dtype", None)
        params = DistributionParams(key=key, shape=shape, dtype=dtype)
        return func(np_mod, params, *args, **kwargs)

    return wrapper


@dataclass
class DistributionParams:
    """Parameters for random distributions."""

    key: object
    shape: object = None
    dtype: object = None


@dataclass
class DropoutConfig:
    """Dropout config."""

    noise_shape: object = None
    training: object = True
    seed: object = None


def _randint(*args: object, **kwargs: object) -> object:
    r"""Execute _randint.\n\n    Args:\n        cls (Any): The class.\n        *args (Any): Argument *args.\n        **kwargs (Any): Argument **kwargs.\n\n    Returns:\n    Any: The result.\n."""
    size = kwargs.get("size")  # pragma: no cover
    if (size is None) and (len(args) > MAGIC_VAL_2):  # pragma: no cover
        size = args[2]  # pragma: no cover
    if size is None:  # pragma: no cover
        res = np.random.randint(*(args[:2] if (len(args) > 1) else args[:1]))  # pragma: no cover
    else:
        res = np.random.randint(*(args[:2] if (len(args) > 1) else args[:1]), size=size)  # pragma: no cover
    dt = getattr(kwargs.get("dtype", np.int64), "value", kwargs.get("dtype", np.int64))  # pragma: no cover
    if dt is None:  # pragma: no cover
        dt = np.int64  # pragma: no cover
    return np.asarray(res).astype(dt)  # pragma: no cover


@numpy_eager_registry.register("Rand")
@unwrap_args
def _np_rand(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    dtype = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype_str = str(dtype).split(".")[(-1)]
    dt = getattr(backend_module, dtype_str, dtype)
    return backend_module.random.rand(*args).astype(dt)


@numpy_eager_registry.register("Randn")
@unwrap_args
def _np_randn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    dtype = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype_str = str(dtype).split(".")[(-1)]
    dt = getattr(backend_module, dtype_str, dtype)
    return backend_module.random.randn(*args).astype(dt)


@numpy_eager_registry.register("Seed")
@unwrap_args
def _np_seed(backend_module: object, seed: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        seed: Arg.
    """
    backend_module.random.seed(seed)
    return seed


@numpy_eager_registry.register("ManualSeed")
@unwrap_args
def _np_manual_seed(backend_module: object, seed: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        seed: Arg.
    """
    backend_module.random.seed(seed)
    return seed


@numpy_eager_registry.register("Randint")
@unwrap_args
def _np_randint(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    dtype = kwargs.pop("dtype", None)
    res = backend_module.random.randint(*args, **kwargs)
    if dtype is not None:
        dtype_str = str(dtype).split(".")[(-1)]
        dt = getattr(backend_module, dtype_str, dtype)
        res = res.astype(dt)
    return res


@numpy_eager_registry.register("Dropout")
def dropout(
    np_mod: object,
    x: object,
    rate: object,
    config: Optional[DropoutConfig] = None,
) -> object:
    """Execute Dropout eager op."""
    conf = config if config is not None else DropoutConfig()
    noise_shape = conf.noise_shape
    training = conf.training
    seed = conf.seed
    """Execute Dropout eager op.

    Args:
        np_mod: Numpy module
        x: Input array
        rate: Dropout rate
        noise_shape: Noise shape
        training: Training boolean
        seed: Random seed

    Returns:
        Resulting array
    """
    if rate == 0.0 or not training:  # pragma: no branch
        return x  # pragma: no cover
    keep_prob = 1.0 - rate
    if seed is not None:  # pragma: no branch
        rng = np_mod.random.RandomState(seed)  # pragma: no cover
    else:
        rng = np_mod.random
    shape = noise_shape if noise_shape is not None else getattr(x, "shape", ())
    mask = rng.binomial(1, keep_prob, size=shape)
    if hasattr(x, "dtype"):  # pragma: no cover
        mask = mask.astype(x.dtype)  # pragma: no cover
    return (x * mask) / keep_prob  # pragma: no cover


@numpy_eager_registry.register("Key")
@unwrap_args
def _np_key(np_mod: object, seed: object) -> object:
    """Function docstring."""
    return np_mod.array([0, seed], dtype=np_mod.uint32)


@numpy_eager_registry.register("Ball")
def _np_ball(np_mod: object, key: object, d: int, p: float = 2.0, shape: object = ()) -> object:
    """Function docstring."""
    shape_tup = shape if isinstance(shape, tuple) else (shape,)
    return np_mod.random.randn(*(shape_tup + (d,)))


@numpy_eager_registry.register("Beta")
@pack_distribution_params
@unwrap_args
def _np_beta(np_mod: object, params: DistributionParams, a: object, b: object) -> object:
    """Function docstring."""
    return np_mod.random.beta(a, b, size=params.shape)


@numpy_eager_registry.register("Binomial")
@pack_distribution_params
@unwrap_args
def _np_binomial(np_mod: object, params: DistributionParams, n: object, p: object) -> object:
    """Function docstring."""
    return np_mod.random.binomial(n, p, size=params.shape)


@numpy_eager_registry.register("Bits")
@pack_distribution_params
@unwrap_args
def _np_bits(np_mod: object, params: DistributionParams) -> object:
    """Function docstring."""
    return np_mod.random.randint(0, 255, size=params.shape)


@numpy_eager_registry.register("Cauchy")
@pack_distribution_params
@unwrap_args
def _np_cauchy(np_mod: object, params: DistributionParams) -> object:
    """Function docstring."""
    return np_mod.random.standard_cauchy(size=params.shape)


@numpy_eager_registry.register("Chisquare")
@pack_distribution_params
@unwrap_args
def _np_chisquare(np_mod: object, params: DistributionParams, df: object) -> object:
    """Function docstring."""
    return np_mod.random.chisquare(df, size=params.shape)


@numpy_eager_registry.register("Clone")
@unwrap_args
def _np_clone(np_mod: object, key: object) -> object:
    """Function docstring."""
    return key


@numpy_eager_registry.register("Dirichlet")
@pack_distribution_params
@unwrap_args
def _np_dirichlet(np_mod: object, params: DistributionParams, alpha: object) -> object:
    """Function docstring."""
    if isinstance(params.shape, int):
        params.shape = (params.shape,)
    return np_mod.random.dirichlet(alpha, size=params.shape if params.shape is not None else ())


@numpy_eager_registry.register("DoubleSidedMaxwell")
@pack_distribution_params
@unwrap_args
def _np_ds_maxwell(
    np_mod: object,
    params: DistributionParams,
    loc: object,
    scale: object,
) -> object:
    """Function docstring."""
    return np_mod.random.randn(*(params.shape if params.shape is not None else ()))


@numpy_eager_registry.register("Exponential")
@pack_distribution_params
@unwrap_args
def _np_exponential(np_mod: object, params: DistributionParams) -> object:
    """Function docstring."""
    return np_mod.random.exponential(size=params.shape)


@numpy_eager_registry.register("F")
@pack_distribution_params
@unwrap_args
def _np_f(
    np_mod: object,
    params: DistributionParams,
    dfnum: object,
    dfden: object,
) -> object:
    """Function docstring."""
    return np_mod.random.f(dfnum, dfden, size=params.shape)


@numpy_eager_registry.register("Gamma")
@pack_distribution_params
@unwrap_args
def _np_gamma(np_mod: object, params: DistributionParams, a: object) -> object:
    """Function docstring."""
    return np_mod.random.gamma(a, size=params.shape)


@numpy_eager_registry.register("GeneralizedNormal")
@pack_distribution_params
@unwrap_args
def _np_gen_normal(np_mod: object, params: DistributionParams, p: object) -> object:
    """Function docstring."""
    return np_mod.random.randn(*(params.shape or ()))


@numpy_eager_registry.register("Geometric")
@pack_distribution_params
@unwrap_args
def _np_geometric(np_mod: object, params: DistributionParams, p: object) -> object:
    """Function docstring."""
    return np_mod.random.geometric(p, size=params.shape)


@numpy_eager_registry.register("Gumbel")
@pack_distribution_params
@unwrap_args
def _np_gumbel(np_mod: object, params: DistributionParams) -> object:
    """Function docstring."""
    return np_mod.random.gumbel(size=params.shape)


@numpy_eager_registry.register("KeyData")
@unwrap_args
def _np_key_data(np_mod: object, key: object) -> object:
    """Function docstring."""
    return key


@numpy_eager_registry.register("KeyImpl")
@unwrap_args
def _np_key_impl(np_mod: object, key: object) -> object:
    """Function docstring."""
    return key


@numpy_eager_registry.register("Laplace")
@pack_distribution_params
@unwrap_args
def _np_laplace(np_mod: object, params: DistributionParams) -> object:
    """Function docstring."""
    return np_mod.random.laplace(size=params.shape)


@numpy_eager_registry.register("Loggamma")
@pack_distribution_params
@unwrap_args
def _np_loggamma(np_mod: object, params: DistributionParams, a: object) -> object:
    """Function docstring."""
    return np_mod.log(np_mod.random.gamma(a, size=params.shape))


@numpy_eager_registry.register("Logistic")
@pack_distribution_params
@unwrap_args
def _np_logistic(np_mod: object, params: DistributionParams) -> object:
    """Function docstring."""
    return np_mod.random.logistic(size=params.shape)


@numpy_eager_registry.register("Lognormal")
@pack_distribution_params
@unwrap_args
def _np_lognormal(np_mod: object, params: DistributionParams) -> object:
    """Function docstring."""
    return np_mod.random.lognormal(size=params.shape)


@numpy_eager_registry.register("Maxwell")
@pack_distribution_params
@unwrap_args
def _np_maxwell(np_mod: object, params: DistributionParams) -> object:
    """Function docstring."""
    return np_mod.random.randn(*(params.shape or ()))


@numpy_eager_registry.register("MultivariateNormal")
@pack_distribution_params
@unwrap_args
def _np_multivariate_normal(
    np_mod: object,
    params: DistributionParams,
    mean: object,
    cov: object,
) -> object:
    """Function docstring."""
    return np_mod.random.multivariate_normal(mean, cov, size=params.shape)


@numpy_eager_registry.register("Orthogonal")
@pack_distribution_params
@unwrap_args
def _np_orthogonal(np_mod: object, params: DistributionParams, n: int) -> object:
    """Function docstring."""
    shape_tup = tuple(params.shape) if isinstance(params.shape, (list, tuple)) else ((params.shape,) if params.shape is not None else ())
    return np_mod.random.randn(*(shape_tup + (n, n)))


@numpy_eager_registry.register("Pareto")
@pack_distribution_params
@unwrap_args
def _np_pareto(np_mod: object, params: DistributionParams, b: object) -> object:
    """Function docstring."""
    return np_mod.random.pareto(b, size=params.shape)


@numpy_eager_registry.register("Poisson")
@pack_distribution_params
@unwrap_args
def _np_poisson(np_mod: object, params: DistributionParams, lam: object) -> object:
    """Function docstring."""
    return np_mod.random.poisson(lam, size=params.shape)


@numpy_eager_registry.register("Rademacher")
@pack_distribution_params
@unwrap_args
def _np_rademacher(np_mod: object, params: DistributionParams) -> object:
    """Function docstring."""
    return np_mod.random.randint(0, 2, size=params.shape) * 2 - 1


@numpy_eager_registry.register("Rayleigh")
@pack_distribution_params
@unwrap_args
def _np_rayleigh(np_mod: object, params: DistributionParams, scale: object) -> object:
    """Function docstring."""
    return np_mod.random.rayleigh(scale, size=params.shape)


@numpy_eager_registry.register("T")
@pack_distribution_params
@unwrap_args
def _np_t(np_mod: object, params: DistributionParams, df: object) -> object:
    """Function docstring."""
    return np_mod.random.standard_t(df, size=params.shape)


@numpy_eager_registry.register("Triangular")
@pack_distribution_params
@unwrap_args
def _np_triangular(
    np_mod: object,
    params: DistributionParams,
    left: object,
    mode: object,
    right: object,
) -> object:
    """Function docstring."""
    return np_mod.random.triangular(left, mode, right, size=params.shape)


@numpy_eager_registry.register("Wald")
@pack_distribution_params
@unwrap_args
def _np_wald(
    np_mod: object,
    params: DistributionParams,
    mean: object,
    scale: object,
) -> object:
    """Function docstring."""
    return np_mod.random.wald(mean, scale, size=params.shape)


@numpy_eager_registry.register("WeibullMin")
@pack_distribution_params
@unwrap_args
def _np_weibull(
    np_mod: object,
    params: DistributionParams,
    scale: object,
    concentration: object,
) -> object:
    """Function docstring."""
    return np_mod.random.weibull(concentration, size=params.shape) * scale


@numpy_eager_registry.register("WrapKeyData")
@unwrap_args
def _np_wrap_key(np_mod: object, key_data: object) -> object:
    """Function docstring."""
    return key_data


@numpy_eager_registry.register("RngBitGenerator")
@pack_distribution_params
@unwrap_args
def _np_rng_bit_generator(np_mod: object, params: DistributionParams) -> object:
    """Function docstring."""
    return np_mod.random.randint(0, 255, size=params.shape)


@numpy_eager_registry.register("RngUniform")
@unwrap_args
def _np_rng_uniform(np_mod: object, a: object, b: object, shape: object, dtype: object = None) -> object:
    """Function docstring."""
    return np_mod.random.uniform(a, b, size=shape)
