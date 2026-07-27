"""Missing ops for random."""


def random_gamma_grad(*args: object, **kwargs: object) -> object:
    """RandomGammaGrad frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("RandomGammaGrad", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("RandomGammaGrad", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


__all__ = [
    "random_gamma_grad",
]
