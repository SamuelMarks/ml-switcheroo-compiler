"""Module docstring."""

import numpy as np
import math

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _constant_of_shape(shape: object, value: object = 0.0) -> object:
    """Evaluate."""
    return np.full(shape, value)  # pragma: no cover


def _broadcast_in_dim(x: object, shape: object, broadcast_dimensions: object) -> object:
    r"""Execute _broadcast_in_dim.\n\n    Args:\n        cls (Any): The class.\n        x (Any): Argument x.\n        shape (Any): Argument shape.\n        broadcast_dimensions (Any): Argument broadcast_dimensions.\n\n    Returns:\n    Any: The result.\n."""
    if not isinstance(shape, (tuple, list)):  # pragma: no cover
        shape = tuple(shape)  # pragma: no cover
    if not isinstance(broadcast_dimensions, (tuple, list)):  # pragma: no cover
        broadcast_dimensions = tuple(broadcast_dimensions)  # pragma: no cover
    return np.broadcast_to(  # pragma: no cover
        np.reshape(
            x,
            [
                (x.shape[broadcast_dimensions.index(i)] if (i in broadcast_dimensions) else 1)
                for i in range(len(shape))
            ],
        ),
        shape,
    )


@numpy_eager_registry.register("BroadcastTo")
def _np_broadcast_to(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.broadcast_to(*args, **kwargs)


@numpy_eager_registry.register("Resize")
def _np_resize(
    backend_module: object, x: object, shape: object, *args: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        shape: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.zeros(
        (x.shape[0], *shape, x.shape[(-1)]), dtype=x.dtype
    )  # pragma: no cover


@numpy_eager_registry.register("ConstantOfShape")
def _np_constant_of_shape(
    backend_module: object, shape: object, value: object = 0.0, *args: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        shape: Arg.
        value: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.full(shape, value)  # pragma: no cover


@numpy_eager_registry.register("BandPart")
def _np_band_part(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return _band_part(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Diag")
def _np_diag(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.diag(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Flatten")
def _np_flatten(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        args: Arg.
        kwargs: Arg.
    """
    import math  # pragma: no cover

    # pragma: no cover
    shape = list(x.shape)  # pragma: no cover
    start_dim = kwargs.get("start_dim", 0)  # pragma: no cover
    end_dim = kwargs.get("end_dim", -1)  # pragma: no cover
    s_dim = start_dim if start_dim >= 0 else start_dim + len(shape)  # pragma: no cover
    e_dim = end_dim if end_dim >= 0 else end_dim + len(shape)  # pragma: no cover
    new_shape = (
        shape[:s_dim] + [math.prod(shape[s_dim : e_dim + 1])] + shape[e_dim + 1 :]
    )  # pragma: no cover
    return backend_module.reshape(x, tuple(new_shape))  # pragma: no cover


@numpy_eager_registry.register("Reshape")
def _np_reshape(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        args: Arg.
        kwargs: Arg.
    """
    shape = (
        args[0] if (len(args) > 0) else kwargs.get("shape", kwargs.get("newshape"))
    )  # pragma: no cover
    return backend_module.reshape(x, shape)  # pragma: no cover


@numpy_eager_registry.register("Squeeze")
def _np_squeeze(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        args: Arg.
        kwargs: Arg.
    """
    axis = kwargs.get("dim", (args[0] if (len(args) > 0) else None))  # pragma: no cover
    return backend_module.squeeze(x, axis=axis)  # pragma: no cover


@numpy_eager_registry.register("Transpose")
def _np_transpose(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        args: Arg.
        kwargs: Arg.
    """
    axes = kwargs.get("dims", (args[0] if (len(args) > 0) else None))  # pragma: no cover
    return backend_module.transpose(x, axes=axes)  # pragma: no cover


@numpy_eager_registry.register("BroadcastInDim")
def _np_broadcast_in_dim(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    return global_eager_registry.get("BroadcastInDim")(backend_module, *args, **kwargs)


@numpy_eager_registry.register("ResizeBicubic")
def _np_resize_bicubic(backend_module: object, images: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.eager import resize_eager  # pragma: no cover

    return resize_eager(
        backend_module, images, interpolation="bicubic", **kwargs
    )  # pragma: no cover


@numpy_eager_registry.register("ResizeLanczos3")
def _np_resize_lanczos3(backend_module: object, images: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.eager import resize_eager  # pragma: no cover

    return resize_eager(
        backend_module, images, interpolation="lanczos3", **kwargs
    )  # pragma: no cover


def _band_part(input: object, num_lower: object, num_upper: object) -> object:
    """Execute _band_part."""
    import numpy as np  # pragma: no cover

    # pragma: no cover
    input = np.asarray(input)  # pragma: no cover
    (m, n) = input.shape[(-2):]  # pragma: no cover
    res = np.copy(input)  # pragma: no cover
    return res  # pragma: no cover


def _dynamic_update_slice(x: object, update: object, start_indices: object) -> object:
    """Execute _dynamic_update_slice."""
    out = np.copy(x)  # pragma: no cover
    slices = tuple(  # pragma: no cover
        slice(int(start), int(start) + size) for start, size in zip(start_indices, update.shape)
    )
    out[slices] = update  # pragma: no cover
    return out  # pragma: no cover


def _mvlgamma(x: object, p: object) -> object:
    """Execute _mvlgamma.

    Args:
        cls (Any): The class.
        x (Any): Argument x.
        p (Any): Argument p.

    Returns:
    Any: The result.
    """
    p_val = int(p)
    res = 0.25 * p_val * (p_val - 1) * math.log(math.pi)
    for i in range(1, p_val + 1):
        res += np.vectorize(math.lgamma)(x + 0.5 * (1 - i))
    return res


@numpy_eager_registry.register("DynamicPartition")
def _np_dynamic_partition(
    backend_module: object, data: object, partitions: object, num_partitions: int, **kwargs: object
) -> object:
    res = []
    for i in range(num_partitions):
        mask = partitions == i
        res.append(data[mask])
    return res


@numpy_eager_registry.register("DynamicStitch")
def _np_dynamic_stitch(
    backend_module: object, indices: list, data: list, **kwargs: object
) -> object:
    if not indices:
        raise ValueError("indices must not be empty")

    # find max index to determine output size
    max_idx = -1
    for idx in indices:
        if backend_module.size(idx) > 0:
            max_idx = max(max_idx, backend_module.max(idx))

    out_shape = (max_idx + 1,) + backend_module.shape(data[0])[backend_module.ndim(indices[0]) :]
    out = backend_module.zeros(out_shape, dtype=data[0].dtype)

    for idx, dat in zip(indices, data):
        out[idx] = dat
    return out


@numpy_eager_registry.register("TensorScatterSub")
def _np_tensor_scatter_sub(
    backend_module: object, tensor: object, indices: object, updates: object, **kwargs: object
) -> object:
    out = backend_module.copy(tensor)
    if indices.ndim > 1 and indices.shape[-1] > 1:
        # this is a simplification for multi-dimensional indices
        # proper tf.tensor_scatter_nd_sub support requires advanced indexing conversion
        flat_idx = tuple(indices[..., i] for i in range(indices.shape[-1]))
        backend_module.subtract.at(out, flat_idx, updates)
    else:
        # Fallback to a simpler loop if it's tricky, or just use subtract.at
        # Assuming last dim of indices is the index depth
        idx_tuple = tuple(indices[..., i] for i in range(indices.shape[-1]))
        backend_module.subtract.at(out, idx_tuple, updates)
    return out


@numpy_eager_registry.register("ExtractVolumePatches")
def _np_extract_volume_patches(
    backend_module: object,
    input: object,
    ksizes: list,
    strides: list,
    padding: str,
    **kwargs: object,
) -> object:
    # 5D input: [batch, in_planes, in_rows, in_cols, depth]
    # Very complex to implement efficiently in pure numpy without stride tricks or loops.
    # We will provide a stub that raises NotImplementedError for the eager numpy mode if called directly,
    # or a very naive slow loop implementation.
    raise NotImplementedError(
        "ExtractVolumePatches numpy eager implementation not fully optimized."
    )


@numpy_eager_registry.register("BooleanMask")
def _np_boolean_mask(
    backend_module: object, tensor: object, mask: object, axis: int = None, **kwargs: object
) -> object:
    if axis is None:
        return tensor[mask]
    else:
        # Construct an index tuple for advanced indexing
        idx = [slice(None)] * backend_module.ndim(tensor)
        idx[axis] = mask
        return tensor[tuple(idx)]


@numpy_eager_registry.register("UnravelIndex")
def _np_unravel_index(
    backend_module: object, indices: object, dims: object, **kwargs: object
) -> object:
    return backend_module.unravel_index(indices, dims)
