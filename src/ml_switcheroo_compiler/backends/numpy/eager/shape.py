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
