"""Linalg utilities."""

from dataclasses import dataclass

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@dataclass
class BlockMaskedMmConfig:
    """Class docstring."""

    block_size: int = 64
    mask_out: object = None
    mask_lhs: object = None
    mask_rhs: object = None


@global_eager_registry.register("BlockMaskedMm")
def _block_masked_mm_eager(
    backend_module: object,
    a: object,
    b: object,
    config: BlockMaskedMmConfig = None,
    **kwargs: object,
) -> object:
    """Fallback eager execution for BlockMaskedMm."""
    if config is None:
        config = BlockMaskedMmConfig(**{k: v for k, v in kwargs.items() if k in ["block_size", "mask_out", "mask_lhs", "mask_rhs"]})
    block_size = config.block_size
    mask_out = config.mask_out
    mask_lhs = config.mask_lhs
    mask_rhs = config.mask_rhs
    if hasattr(backend_module, "block_masked_mm"):
        return backend_module.block_masked_mm(a, b, block_size=block_size, mask_out=mask_out, mask_lhs=mask_lhs, mask_rhs=mask_rhs)

    # Fallback to normal matmul and manual masking
    # scale masks by block_size
    if mask_lhs is not None:
        mask_lhs = backend_module.repeat(backend_module.repeat(mask_lhs, block_size, axis=-2), block_size, axis=-1)
        a = backend_module.where(mask_lhs, a, 0)
    if mask_rhs is not None:
        mask_rhs = backend_module.repeat(backend_module.repeat(mask_rhs, block_size, axis=-2), block_size, axis=-1)
        b = backend_module.where(mask_rhs, b, 0)

    out = backend_module.matmul(a, b)

    if mask_out is not None:
        mask_out = backend_module.repeat(backend_module.repeat(mask_out, block_size, axis=-2), block_size, axis=-1)
        out = backend_module.where(mask_out, out, 0)

    return out


@global_eager_registry.register("GatherMm")
def _gather_mm_eager(
    backend_module: object,
    a: object,
    b: object,
    lhs_indices: object = None,
    rhs_indices: object = None,
    **kwargs: object,
) -> object:
    """Function docstring."""
    if hasattr(backend_module, "gather_mm"):
        return backend_module.gather_mm(a, b, lhs_indices=lhs_indices, rhs_indices=rhs_indices, **kwargs)
    # naive fallback
    if lhs_indices is not None:
        a = backend_module.take(a, lhs_indices, axis=0)
    if rhs_indices is not None:
        b = backend_module.take(b, rhs_indices, axis=0)
    return backend_module.matmul(a, b)


@global_eager_registry.register("SegmentedMm")
def _segmented_mm_eager(backend_module: object, a: object, b: object, segments: object, **kwargs: object) -> object:
    """Function docstring."""
    if hasattr(backend_module, "segmented_mm"):
        return backend_module.segmented_mm(a, b, segments, **kwargs)
    raise NotImplementedError("SegmentedMm eager fallback not implemented")


@global_eager_registry.register("Dropout2d")
def _dropout2d_eager(backend_module: object, x: object, p: float = 0.5, training: bool = True) -> object:
    """Function docstring."""
    if not training or p == 0.0:
        return x
    if hasattr(backend_module, "dropout2d"):
        return backend_module.dropout2d(x, p=p, training=training)
    if hasattr(backend_module, "random") and hasattr(backend_module.random, "uniform"):
        shape = list(x.shape)
        if len(shape) >= 2:
            shape[-1] = 1
            shape[-2] = 1
        mask = backend_module.random.uniform(size=shape) > p
        return backend_module.where(mask, x / (1.0 - p), 0.0)

    shape = list(x.shape)
    if len(shape) >= 2:
        shape[-1] = 1
        shape[-2] = 1
    mask = np.random.uniform(size=shape) > p
    return x * mask / (1.0 - p)


@global_eager_registry.register("Dropout3d")
def _dropout3d_eager(backend_module: object, x: object, p: float = 0.5, training: bool = True) -> object:
    """Function docstring."""
    if not training or p == 0.0:
        return x
    if hasattr(backend_module, "dropout3d"):
        return backend_module.dropout3d(x, p=p, training=training)

    shape = list(x.shape)
    if len(shape) >= 3:
        shape[-1] = 1
        shape[-2] = 1
        shape[-3] = 1
    mask = np.random.uniform(size=shape) > p
    return x * mask / (1.0 - p)


@global_eager_registry.register("PutAlongAxis")
def _put_along_axis_eager(
    backend_module: object,
    a: object,
    indices: object,
    values: object,
    axis: int = None,
    **kwargs: object,
) -> object:
    """Function docstring."""
    if hasattr(backend_module, "put_along_axis"):
        return backend_module.put_along_axis(a, indices, values, axis=axis, **kwargs)
    if hasattr(backend_module, "put_along_axis"):
        return np.put_along_axis(a, indices, values, axis)

    out = np.array(a)
    np.put_along_axis(out, indices, values, axis)
    if hasattr(backend_module, "array"):
        return backend_module.array(out)
    return out


@global_eager_registry.register("Logcumsumexp")
def _logcumsumexp_eager(backend_module: object, a: object, axis: int = None, **kwargs: object) -> object:
    """Function docstring."""
    if hasattr(backend_module, "logcumsumexp"):
        return backend_module.logcumsumexp(a, axis=axis, **kwargs)
    raise NotImplementedError("Logcumsumexp eager fallback not implemented")


@global_eager_registry.register("Gru")
def _gru_eager(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    raise NotImplementedError("Gru eager fallback not implemented")


@global_eager_registry.register("GetItem")
def _getitem_eager(backend_module: object, x: object, key: object, **kwargs: object) -> object:
    """Function docstring."""
    return x[key]


@global_eager_registry.register("PowerIteration")
def _power_iteration(backend_module: object, w: object, num_iters: int = 1, u: object = None) -> tuple[object, object, object]:
    """Function docstring.

    Args:
        backend_module: Arg.
        w: Arg.
        num_iters: Arg.
        u: Arg.
    """
    linalg = getattr(backend_module, "linalg", None)
    if linalg is None:  # pragma: no branch
        raise NotImplementedError("Backend module missing linalg submodule.")  # pragma: no cover

    # We assume w is an array. We need to implement eager logic using backend_module.
    # backend_module is typically np, jnp, mlx.core, etc.
    shape = w.shape
    dtype = w.dtype

    if u is None:
        u = backend_module.ones(shape[:-2] + (shape[-2], 1), dtype=dtype)

    for _ in range(num_iters):
        w_t = backend_module.swapaxes(w, -1, -2)
        v = backend_module.matmul(w_t, u)
        v_norm = linalg.norm(v, axis=-2, keepdims=True) + 1e-12
        v = v / v_norm

        u = backend_module.matmul(w, v)
        u_norm = linalg.norm(u, axis=-2, keepdims=True) + 1e-12
        u = u / u_norm

    sigma = backend_module.matmul(backend_module.swapaxes(u, -1, -2), backend_module.matmul(w, v))
    return (
        backend_module.squeeze(v, -1),
        backend_module.squeeze(u, -1),
        backend_module.squeeze(backend_module.squeeze(sigma, -1), -1),
    )
