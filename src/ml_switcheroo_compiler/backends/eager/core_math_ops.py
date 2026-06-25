# ruff: noqa: F405, F403
"""Core utilities."""

import scipy.special

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("TrueDivide")
def _true_divide(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    func = getattr(
        backend_module, "divide", getattr(backend_module, "true_divide", None)
    )  # pragma: no cover
    return func(*args, **kwargs) if func else None  # pragma: no cover


@global_eager_registry.register("Fft")
def _fft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    fft_mod = getattr(backend_module, "fft", None)  # pragma: no cover
    return fft_mod.fft(*args, **kwargs) if fft_mod else None  # pragma: no cover


@global_eager_registry.register("Rfft")
def _rfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    fft_mod = getattr(backend_module, "fft", None)  # pragma: no cover
    return fft_mod.rfft(*args, **kwargs) if fft_mod else None  # pragma: no cover


@global_eager_registry.register("Fftn")
def _fftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    fft_mod = getattr(backend_module, "fft", None)  # pragma: no cover
    return fft_mod.fftn(*args, **kwargs) if fft_mod else None  # pragma: no cover


@global_eager_registry.register("Erfinv")
def _erfinv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    func = getattr(  # pragma: no cover
        backend_module,
        "erfinv",
        scipy.special.erfinv if getattr(backend_module, "__name__", "") == "numpy" else None,
    )
    return func(*args, **kwargs) if func else None  # pragma: no cover


@global_eager_registry.register("NanToNum")
def _nan_to_num(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    x = args[0]  # pragma: no cover
    nan = kwargs.get("nan", 0.0)  # pragma: no cover
    posinf = kwargs.get("posinf", None)  # pragma: no cover
    neginf = kwargs.get("neginf", None)  # pragma: no cover
    if hasattr(backend_module, "nan_to_num"):  # pragma: no cover
        return backend_module.nan_to_num(
            x, nan=nan, posinf=posinf, neginf=neginf
        )  # pragma: no cover
    return None  # pragma: no cover


@global_eager_registry.register("Einsum")
def _einsum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    eq = (  # pragma: no cover
        kwargs.pop("equation", "")
        if "equation" in kwargs
        else args[0]
        if len(args) > 0 and isinstance(args[0], str)
        else ""
    )
    op_args = args[1:] if len(args) > 0 and isinstance(args[0], str) else args  # pragma: no cover
    if hasattr(backend_module, "einsum"):  # pragma: no cover
        return backend_module.einsum(eq, *op_args, **kwargs)  # pragma: no cover
    return None  # pragma: no cover


@global_eager_registry.register("Allclose")
def _allclose(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    a = args[0]  # pragma: no cover
    b = args[1]  # pragma: no cover
    rtol = kwargs.get("rtol", 1e-5)  # pragma: no cover
    atol = kwargs.get("atol", 1e-8)  # pragma: no cover
    equal_nan = kwargs.get("equal_nan", False)  # pragma: no cover

    def _val(x: object) -> object:  # pragma: no cover
        """Function docstring.

        Args:
        x: Arg.
        """
        x_data = getattr(x, "data", x)  # pragma: no cover
        if hasattr(x_data, "item") and callable(x_data.item):  # pragma: no cover
            return x_data.item()  # pragma: no cover
        if hasattr(x_data, "tolist"):  # pragma: no cover
            return x_data.tolist()  # pragma: no cover
        return x_data  # pragma: no cover

    if hasattr(backend_module, "allclose"):  # pragma: no cover
        return backend_module.allclose(  # pragma: no cover
            a, b, rtol=float(_val(rtol)), atol=float(_val(atol)), equal_nan=bool(_val(equal_nan))
        )
    return None  # pragma: no cover


@global_eager_registry.register("Psum")
def _psum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return args[0]


@global_eager_registry.register("Pmean")
def _pmean(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return args[0]


@global_eager_registry.register("SegmentSum")
def _segment_sum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    if hasattr(backend_module, "zeros"):  # pragma: no branch
        return backend_module.zeros((1,))
    return None  # pragma: no cover


__all__ = [
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_allclose",
    "_einsum",
    "_erfinv",
    "_fft",
    "_fftn",
    "_nan_to_num",
    "_pmean",
    "_psum",
    "_rfft",
    "_segment_sum",
    "_true_divide",
    "global_eager_registry",
    "scipy",
]
