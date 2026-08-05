"""Nn ops module."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Dropout2d")
def _np_dropout2d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_dropout2d operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.

    Raises:
        ValueError: An exception.
    """
    x = args[0]
    p = kwargs.get("p", 0.5)
    training = kwargs.get("training", True)
    if not training or p == 0.0:
        return x

    # dropout2d zeros out entire channels (axis 1) for each sample in batch (axis 0)
    # Shape of x: (N, C, H, W)
    if x.ndim != 4:
        raise ValueError("Dropout2d requires a 4D tensor (N, C, H, W)")

    N, C, _, _ = x.shape
    mask = np.random.binomial(1, 1.0 - p, size=(N, C, 1, 1)).astype(x.dtype)
    return x * mask / (1.0 - p)


@numpy_eager_registry.register("BlockMaskedMm")
def _np_block_masked_mm(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_block_masked_mm operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    import numpy as np

    a, b = args[0], args[1]
    # BlockMaskedMm is just batched matmul where some blocks are skipped.
    # In eager mode without a provided mask, we just return a @ b
    return np.matmul(a, b)
