"""Extra audio ops for eager numpy execution."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
import numpy as np


@numpy_eager_registry.register("Dct")
def _np_dct(backend_module: object, *args: object, **kwargs: object) -> object:
    from scipy.fftpack import dct

    return dct(args[0], **kwargs)


@numpy_eager_registry.register("Idct")
def _np_idct(backend_module: object, *args: object, **kwargs: object) -> object:
    from scipy.fftpack import idct

    return idct(args[0], **kwargs)


@numpy_eager_registry.register("Mdct")
def _np_mdct(backend_module: object, *args: object, **kwargs: object) -> object:
    # Dummy mock execution
    return np.zeros_like(args[0])


@numpy_eager_registry.register("InverseMdct")
def _np_inverse_mdct(backend_module: object, *args: object, **kwargs: object) -> object:
    # Dummy mock execution
    return np.zeros_like(args[0])


@numpy_eager_registry.register("Frame")
def _np_frame(backend_module: object, *args: object, **kwargs: object) -> object:
    # Dummy mock execution
    return np.zeros_like(args[0])


@numpy_eager_registry.register("OverlapAndAdd")
def _np_overlap_and_add(backend_module: object, *args: object, **kwargs: object) -> object:
    # Dummy mock execution
    return np.zeros_like(args[0])
