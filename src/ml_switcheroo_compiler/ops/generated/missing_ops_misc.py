"""Missing ops for misc."""


def append(*args: object, **kwargs: object) -> object:
    """Append frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Append", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Append", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def array_equiv(*args: object, **kwargs: object) -> object:
    """ArrayEquiv frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ArrayEquiv", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ArrayEquiv", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def array_repr(*args: object, **kwargs: object) -> object:
    """ArrayRepr frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ArrayRepr", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ArrayRepr", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def array_str(*args: object, **kwargs: object) -> object:
    """ArrayStr frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ArrayStr", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ArrayStr", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def average(*args: object, **kwargs: object) -> object:
    """Average frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Average", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Average", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def ball(*args: object, **kwargs: object) -> object:
    """Ball frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Ball", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Ball", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def bartlett(*args: object, **kwargs: object) -> object:
    """Bartlett frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Bartlett", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Bartlett", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def beta(*args: object, **kwargs: object) -> object:
    """Beta frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Beta", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Beta", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def binomial(*args: object, **kwargs: object) -> object:
    """Binomial frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Binomial", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Binomial", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def bits(*args: object, **kwargs: object) -> object:
    """Bits frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Bits", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Bits", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def blackman(*args: object, **kwargs: object) -> object:
    """Blackman frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Blackman", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Blackman", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def cauchy(*args: object, **kwargs: object) -> object:
    """Cauchy frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Cauchy", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Cauchy", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def chisquare(*args: object, **kwargs: object) -> object:
    """Chisquare frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Chisquare", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Chisquare", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def choose(*args: object, **kwargs: object) -> object:
    """Choose frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Choose", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Choose", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def clone(*args: object, **kwargs: object) -> object:
    """Clone frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Clone", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Clone", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def compress(*args: object, **kwargs: object) -> object:
    """Compress frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Compress", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Compress", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def custom_linear_solve(*args: object, **kwargs: object) -> object:
    """CustomLinearSolve frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("CustomLinearSolve", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("CustomLinearSolve", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def custom_root(*args: object, **kwargs: object) -> object:
    """CustomRoot frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("CustomRoot", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("CustomRoot", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def diag_indices_from(*args: object, **kwargs: object) -> object:
    """DiagIndicesFrom frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DiagIndicesFrom", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DiagIndicesFrom", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def dirichlet(*args: object, **kwargs: object) -> object:
    """Dirichlet frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Dirichlet", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Dirichlet", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def ediff1d(*args: object, **kwargs: object) -> object:
    """Ediff1d frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Ediff1d", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Ediff1d", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def f(*args: object, **kwargs: object) -> object:
    """F frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("F", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("F", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def fft2(*args: object, **kwargs: object) -> object:
    """Fft2 frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Fft2", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Fft2", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def fftfreq(*args: object, **kwargs: object) -> object:
    """Fftfreq frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Fftfreq", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Fftfreq", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def finfo(*args: object, **kwargs: object) -> object:
    """Finfo frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Finfo", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Finfo", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def gamma(*args: object, **kwargs: object) -> object:
    """Gamma frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Gamma", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Gamma", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def gumbel(*args: object, **kwargs: object) -> object:
    """Gumbel frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Gumbel", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Gumbel", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def hanning(*args: object, **kwargs: object) -> object:
    """Hanning frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Hanning", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Hanning", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def hessenberg(*args: object, **kwargs: object) -> object:
    """Hessenberg frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Hessenberg", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Hessenberg", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def ihfft(*args: object, **kwargs: object) -> object:
    """Ihfft frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Ihfft", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Ihfft", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def irfft(*args: object, **kwargs: object) -> object:
    """Irfft frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Irfft", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Irfft", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def kaiser(*args: object, **kwargs: object) -> object:
    """Kaiser frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Kaiser", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Kaiser", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def key(*args: object, **kwargs: object) -> object:
    """Key frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Key", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Key", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def key_data(*args: object, **kwargs: object) -> object:
    """KeyData frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("KeyData", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("KeyData", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def key_impl(*args: object, **kwargs: object) -> object:
    """KeyImpl frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("KeyImpl", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("KeyImpl", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def laplace(*args: object, **kwargs: object) -> object:
    """Laplace frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Laplace", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Laplace", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def lu_pivots_to_permutation(*args: object, **kwargs: object) -> object:
    """LuPivotsToPermutation frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("LuPivotsToPermutation", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("LuPivotsToPermutation", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def orthogonal(*args: object, **kwargs: object) -> object:
    """Orthogonal frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Orthogonal", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Orthogonal", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def packbits(*args: object, **kwargs: object) -> object:
    """Packbits frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Packbits", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Packbits", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def pareto(*args: object, **kwargs: object) -> object:
    """Pareto frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Pareto", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Pareto", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def poisson(*args: object, **kwargs: object) -> object:
    """Poisson frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Poisson", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Poisson", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def poly(*args: object, **kwargs: object) -> object:
    """Poly frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Poly", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Poly", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def polyder(*args: object, **kwargs: object) -> object:
    """Polyder frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Polyder", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Polyder", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def polyfit(*args: object, **kwargs: object) -> object:
    """Polyfit frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Polyfit", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Polyfit", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def polyint(*args: object, **kwargs: object) -> object:
    """Polyint frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Polyint", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Polyint", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def polyval(*args: object, **kwargs: object) -> object:
    """Polyval frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Polyval", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Polyval", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def qdwh(*args: object, **kwargs: object) -> object:
    """Qdwh frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Qdwh", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Qdwh", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def qr(*args: object, **kwargs: object) -> object:
    """Qr frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Qr", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Qr", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def roll(*args: object, **kwargs: object) -> object:
    """Roll frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Roll", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Roll", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def roots(*args: object, **kwargs: object) -> object:
    """Roots frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Roots", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Roots", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def schur(*args: object, **kwargs: object) -> object:
    """Schur frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Schur", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Schur", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def setdiff1d(*args: object, **kwargs: object) -> object:
    """Setdiff1d frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Setdiff1d", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Setdiff1d", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def setxor1d(*args: object, **kwargs: object) -> object:
    """Setxor1d frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Setxor1d", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Setxor1d", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def solve(*args: object, **kwargs: object) -> object:
    """Solve frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Solve", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Solve", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def swapaxes(*args: object, **kwargs: object) -> object:
    """Swapaxes frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Swapaxes", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Swapaxes", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def t(*args: object, **kwargs: object) -> object:
    """T frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("T", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("T", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def take(*args: object, **kwargs: object) -> object:
    """Take frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Take", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Take", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def take_along_axis(*args: object, **kwargs: object) -> object:
    """TakeAlongAxis frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("TakeAlongAxis", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("TakeAlongAxis", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def top_k(*args: object, **kwargs: object) -> object:
    """TopK frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("TopK", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("TopK", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def triangular(*args: object, **kwargs: object) -> object:
    """Triangular frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Triangular", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Triangular", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def tridiagonal(*args: object, **kwargs: object) -> object:
    """Tridiagonal frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Tridiagonal", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Tridiagonal", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def tril_indices(*args: object, **kwargs: object) -> object:
    """TrilIndices frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("TrilIndices", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("TrilIndices", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def tril_indices_from(*args: object, **kwargs: object) -> object:
    """TrilIndicesFrom frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("TrilIndicesFrom", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("TrilIndicesFrom", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def triu_indices(*args: object, **kwargs: object) -> object:
    """TriuIndices frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("TriuIndices", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("TriuIndices", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def triu_indices_from(*args: object, **kwargs: object) -> object:
    """TriuIndicesFrom frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("TriuIndicesFrom", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("TriuIndicesFrom", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def union1d(*args: object, **kwargs: object) -> object:
    """Union1d frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Union1d", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Union1d", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def unpackbits(*args: object, **kwargs: object) -> object:
    """Unpackbits frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Unpackbits", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Unpackbits", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def wald(*args: object, **kwargs: object) -> object:
    """Wald frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Wald", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Wald", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def wrap_key_data(*args: object, **kwargs: object) -> object:
    """WrapKeyData frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("WrapKeyData", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("WrapKeyData", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


__all__ = [
    "append",
    "array_equiv",
    "array_repr",
    "array_str",
    "average",
    "ball",
    "bartlett",
    "beta",
    "binomial",
    "bits",
    "blackman",
    "cauchy",
    "chisquare",
    "choose",
    "clone",
    "compress",
    "custom_linear_solve",
    "custom_root",
    "diag_indices_from",
    "dirichlet",
    "ediff1d",
    "f",
    "fft2",
    "fftfreq",
    "finfo",
    "gamma",
    "gumbel",
    "hanning",
    "hessenberg",
    "ihfft",
    "irfft",
    "kaiser",
    "key",
    "key_data",
    "key_impl",
    "laplace",
    "lu_pivots_to_permutation",
    "orthogonal",
    "packbits",
    "pareto",
    "poisson",
    "poly",
    "polyder",
    "polyfit",
    "polyint",
    "polyval",
    "qdwh",
    "qr",
    "roll",
    "roots",
    "schur",
    "setdiff1d",
    "setxor1d",
    "solve",
    "swapaxes",
    "t",
    "take",
    "take_along_axis",
    "top_k",
    "triangular",
    "tridiagonal",
    "tril_indices",
    "tril_indices_from",
    "triu_indices",
    "triu_indices_from",
    "union1d",
    "unpackbits",
    "wald",
    "wrap_key_data",
]
