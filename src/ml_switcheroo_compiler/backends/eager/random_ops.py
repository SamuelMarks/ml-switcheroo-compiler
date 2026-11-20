"""Generate random operations for eager backends."""

from __future__ import annotations

import random as _py_random
import types
import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_int_list,
    coerce_operand_list,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

RandVal = BackendArg


@global_eager_registry.register("PRNGKey")
def prng_key(backend_module: BackendArg, seed: int) -> RandVal:
    """Create a PRNGKey.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        seed (int): Input `seed` argument (seed).

    Returns:
            RandVal: Computed `RandVal` output.
    """
    if adapt_backend_module(backend_module).has_symbol("array"):
        return adapt_backend_module(backend_module).require_symbol("array")([0, seed], dtype="uint32")
    return [0, seed]


@global_eager_registry.register("RandomSplit")
def random_split(backend_module: BackendArg, key: BackendArg, num: int = 2) -> RandVal:
    """Split a PRNGKey.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        key: Input `key` argument (key).
        num (int): Input `num` argument (num).

    Returns:
            RandVal: Computed `RandVal` output.
    """
    import random

    data_list = [[random.randint(0, 0xFFFFFFFF), random.randint(0, 0xFFFFFFFF)] for _ in range(num)]
    if adapt_backend_module(backend_module).has_symbol("array"):
        return adapt_backend_module(backend_module).require_symbol("array")(data_list, dtype="uint32")
    return data_list


@global_eager_registry.register("RandomFoldIn")
def random_fold_in(backend_module: BackendArg, key: BackendArg, data: int) -> RandVal:
    """Fold in data to a PRNGKey.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        key: Input `key` argument (key).
        data (int): Input `data` argument (data).

    Returns:
            RandVal: Computed `RandVal` output.
    """
    has_idx = isinstance(key, (list, tuple)) or adapt_backend_module(key).has_symbol("shape")
    val0 = int(coerce_int_list(key)[0]) if has_idx else 0
    val1 = int(coerce_int_list(key)[1]) if has_idx else 0
    data_list = [val0 + data, val1]
    if adapt_backend_module(backend_module).has_symbol("array"):
        return adapt_backend_module(backend_module).require_symbol("array")(data_list, dtype="uint32")
    return data_list


@global_eager_registry.register("Rand")
def rand(backend_module: BackendArg, *args: RandVal, **kwargs: RandVal) -> RandVal:
    """Generate uniform random values.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            RandVal: Computed `RandVal` output.
    """
    raw_shape = coerce_operand_list(
        kwargs.get("shape", args if args else ()) if adapt_backend_module(kwargs).has_symbol("get") else ()
    )
    shape: tuple[BackendArg, ...] = (
        tuple(raw_shape[0]) if len(raw_shape) == 1 and isinstance(raw_shape[0], (tuple, list)) else tuple(raw_shape)
    )
    if adapt_backend_module(backend_module).has_symbol("random"):
        rand_mod = adapt_backend_module(backend_module).require_symbol("random")
        if adapt_backend_module(rand_mod).has_symbol("uniform"):
            return rand_mod.uniform(size=shape)
        if adapt_backend_module(rand_mod).has_symbol("rand"):
            return rand_mod.rand(*shape)
    if adapt_backend_module(backend_module).has_symbol("rand"):
        return adapt_backend_module(backend_module).require_symbol("rand")(*shape)
    if adapt_backend_module(backend_module).has_symbol("uniform"):
        return adapt_backend_module(backend_module).require_symbol("uniform")(size=shape)

    return adapt_backend_module(backend_module).require_symbol("random").uniform(size=shape)


@global_eager_registry.register("Randn")
def randn(backend_module: BackendArg, *args: RandVal, **kwargs: RandVal) -> RandVal:
    """Generate normal random values.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            RandVal: Computed `RandVal` output.
    """
    raw_shape = coerce_operand_list(
        kwargs.get("shape", args if args else ()) if adapt_backend_module(kwargs).has_symbol("get") else ()
    )
    shape: tuple[BackendArg, ...] = (
        tuple(raw_shape[0]) if len(raw_shape) == 1 and isinstance(raw_shape[0], (tuple, list)) else tuple(raw_shape)
    )
    if adapt_backend_module(backend_module).has_symbol("random"):
        rand_mod = adapt_backend_module(backend_module).require_symbol("random")
        if adapt_backend_module(rand_mod).has_symbol("normal"):
            return rand_mod.normal(size=shape)
        if adapt_backend_module(rand_mod).has_symbol("randn"):
            return rand_mod.randn(*shape)
    if adapt_backend_module(backend_module).has_symbol("randn"):
        return adapt_backend_module(backend_module).require_symbol("randn")(*shape)
    if adapt_backend_module(backend_module).has_symbol("normal"):
        return adapt_backend_module(backend_module).require_symbol("normal")(size=shape)

    return adapt_backend_module(backend_module).require_symbol("random").normal(size=shape)


@global_eager_registry.register("Randint")
def randint(backend_module: BackendArg, *args: RandVal, **kwargs: RandVal) -> RandVal:
    """Generate integer random values.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            RandVal: Computed `RandVal` output.
    """
    low = kwargs.get("low", args[0] if len(args) > 0 else 0) if adapt_backend_module(kwargs).has_symbol("get") else 0
    high = (
        kwargs.get("high", args[1] if len(args) > 1 else 10) if adapt_backend_module(kwargs).has_symbol("get") else 10
    )
    shape = (
        kwargs.get("shape", args[2] if len(args) > 2 else ()) if adapt_backend_module(kwargs).has_symbol("get") else ()
    )
    if adapt_backend_module(backend_module).has_symbol("random") and adapt_backend_module(
        adapt_backend_module(backend_module).require_symbol("random")
    ).has_symbol("randint"):
        return adapt_backend_module(backend_module).require_symbol("random").randint(low, high, size=shape)

    return adapt_backend_module(backend_module).require_symbol("random").randint(low, high, size=shape)


@global_eager_registry.register("Seed")
@global_eager_registry.register("ManualSeed")
def seed_rng(backend_module: BackendArg, seed: int = 0) -> int:
    """Seed the backend random number generator state.

    Args:
        backend_module (BackendArg): Active backend runtime module.
        seed (int): Integer PRNG seed value.

    Returns:
        int: Applied integer seed value.
    """
    seed_int = int(seed)
    if adapt_backend_module(backend_module).has_symbol("random"):
        rand_mod = adapt_backend_module(backend_module).require_symbol("random")
        if adapt_backend_module(rand_mod).has_symbol("seed"):
            adapt_backend_module(rand_mod).require_symbol("seed")(seed_int)
            return seed_int
    if adapt_backend_module(backend_module).has_symbol("manual_seed"):
        adapt_backend_module(backend_module).require_symbol("manual_seed")(seed_int)
    return seed_int


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        BackendArg: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@typing.runtime_checkable
class RandomDistributionConfig(typing.Protocol):
    """Protocol describing configuration attributes for random distributions."""

    mean: float
    stddev: float
    minval: float
    maxval: float


@typing.runtime_checkable
class _MeanProtocol(typing.Protocol):
    """Protocol for config objects exposing `.mean`."""

    @property
    def mean(self) -> float | int:
        """Return distribution mean.

        Returns:
            float | int: Mean value.
        """
        ...


@typing.runtime_checkable
class _StddevProtocol(typing.Protocol):
    """Protocol for config objects exposing `.stddev`."""

    @property
    def stddev(self) -> float | int:
        """Return distribution standard deviation.

        Returns:
            float | int: Standard deviation value.
        """
        ...


@typing.runtime_checkable
class _MinvalProtocol(typing.Protocol):
    """Protocol for config objects exposing `.minval`."""

    @property
    def minval(self) -> float | int | None:
        """Return minimum uniform bound.

        Returns:
            float | int | None: Minimum bound.
        """
        ...


@typing.runtime_checkable
class _MaxvalProtocol(typing.Protocol):
    """Protocol for config objects exposing `.maxval`."""

    @property
    def maxval(self) -> float | int | None:
        """Return maximum uniform bound.

        Returns:
            float | int | None: Maximum bound.
        """
        ...


@typing.runtime_checkable
class LookupTableProtocol(typing.Protocol):
    """Protocol for table objects supporting lookup."""

    def lookup(self, keys: BackendArg) -> EagerTensorProtocol:
        """Lookup keys in table.

        Args:
            keys (BackendArg): Input keys to query.

        Returns:
            BackendArg: Looked up values.
        """
        ...


@global_eager_registry.register("Normal")
def _np_normal(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    shape: tuple[int, ...] | list[int] | int,
    **kwargs: BackendArg | RandomDistributionConfig,
) -> EagerTensorProtocol:
    """Evaluate normal distribution random sampling.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        shape (Union[tuple[int, ...], list[int], int]): Target output shape.
        **kwargs (BackendArg | RandomDistributionConfig): Keyword arguments.

    Returns:
        BackendArg: Random samples drawn from normal distribution.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    dtype = kwargs.get("dtype", "float32")
    config: BackendArg | RandomDistributionConfig | _MeanProtocol | _StddevProtocol = kwargs.get("config")
    mean = float(config.mean) if isinstance(config, _MeanProtocol) else 0.0
    stddev = float(config.stddev) if isinstance(config, _StddevProtocol) else 1.0
    return _ensure_arr(arr_mod.random.normal(loc=mean, scale=stddev, size=shape).astype(str(dtype)))


@global_eager_registry.register("Uniform")
def _np_uniform(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    shape: tuple[int, ...] | list[int] | int,
    **kwargs: BackendArg | RandomDistributionConfig,
) -> EagerTensorProtocol:
    """Evaluate uniform distribution random sampling.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        shape (Union[tuple[int, ...], list[int], int]): Target output shape.
        **kwargs (BackendArg | RandomDistributionConfig): Keyword arguments.

    Returns:
        BackendArg: Random samples drawn from uniform distribution.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    dtype = kwargs.get("dtype", "float32")
    config: BackendArg | RandomDistributionConfig | _MinvalProtocol | _MaxvalProtocol = kwargs.get("config")
    raw_min = config.minval if isinstance(config, _MinvalProtocol) else None
    minval: float = float(raw_min) if isinstance(raw_min, (int, float)) else 0.0
    raw_max = config.maxval if isinstance(config, _MaxvalProtocol) else None
    maxval: float = float(raw_max) if isinstance(raw_max, (int, float)) else 1.0
    return _ensure_arr(arr_mod.random.uniform(low=minval, high=maxval, size=shape).astype(str(dtype)))


@global_eager_registry.register("StatelessSplit")
def _np_stateless_split(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    seed: BackendArg | list[int | str] | int | str,
    **kwargs: int | None,
) -> EagerTensorProtocol:
    """Evaluate stateless PRNG key split.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        seed (BackendArg | list[int | str] | int | str): Seed value or array.
        **kwargs (Union[int, None]): Keyword arguments such as num.

    Returns:
        BackendArg: Split pseudo-random integer seeds.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    num_val = kwargs.get("num", 2)
    num = int(num_val) if num_val is not None else 2
    s = arr_mod.asarray(seed).flatten()
    if len(s) > 0:
        s0_item = s[0].item() if adapt_backend_module(s[0]).has_symbol("item") else s[0]
        if isinstance(s0_item, str):
            s_val = hash(s0_item) % (2**31 - 1)
        else:
            try:
                s_val = int(s[0])
            except (ValueError, TypeError):
                s_val = hash(str(s[0])) % (2**31 - 1)
    else:
        s_val = 0
    rng = _py_random.Random(s_val)
    pairs = [[rng.randint(0, 2**31 - 2), rng.randint(0, 2**31 - 2)] for _ in range(num)]
    return _ensure_arr(arr_mod.array(pairs, dtype="int64"))


@global_eager_registry.register("Lookup")
def _np_lookup(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    table: dict[int, int] | dict[int | str, int | float] | LookupTableProtocol | BackendArg,
    keys: BackendArg,
    **kwargs: int | float | None,
) -> EagerTensorProtocol:
    """Evaluate lookup table query on keys.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        table (dict[int, int] | dict[int | str, int | float] | LookupTableProtocol | BackendArg):
            Table dictionary or lookup object.
        keys (BackendArg): Keys to lookup.
        **kwargs (Union[int, float, None]): Keyword arguments such as default_value.

    Returns:
        BackendArg: Array of looked up values.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    arr = _ensure_arr(arr_mod.asarray(keys))
    default_value = kwargs.get("default_value", 0)
    if isinstance(table, dict):
        return _ensure_arr(arr_mod.vectorize(lambda k: table.get(k, default_value))(arr))
    if isinstance(table, LookupTableProtocol):
        return table.lookup(arr)
    return _ensure_arr(arr_mod.full_like(arr, default_value, dtype=arr_mod.int32))
