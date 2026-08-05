"""Missing ops for shape."""


def argpartition(*args: object, **kwargs: object) -> object:
    """Argpartition frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Argpartition", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Argpartition", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def argsort(*args: object, **kwargs: object) -> object:
    """Argsort frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Argsort", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Argsort", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def argwhere(*args: object, **kwargs: object) -> object:
    """Argwhere frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Argwhere", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Argwhere", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def column_stack(*args: object, **kwargs: object) -> object:
    """ColumnStack frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ColumnStack", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ColumnStack", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def dsplit(*args: object, **kwargs: object) -> object:
    """Dsplit frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Dsplit", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Dsplit", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def dstack(*args: object, **kwargs: object) -> object:
    """Dstack frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Dstack", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Dstack", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def dynamic_index_in_dim(*args: object, **kwargs: object) -> object:
    """DynamicIndexInDim frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DynamicIndexInDim", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DynamicIndexInDim", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def dynamic_slice(*args: object, **kwargs: object) -> object:
    """DynamicSlice frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DynamicSlice", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DynamicSlice", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def dynamic_slice_in_dim(*args: object, **kwargs: object) -> object:
    """DynamicSliceInDim frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DynamicSliceInDim", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DynamicSliceInDim", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def dynamic_update_index_in_dim(*args: object, **kwargs: object) -> object:
    """DynamicUpdateIndexInDim frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DynamicUpdateIndexInDim", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DynamicUpdateIndexInDim", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def dynamic_update_slice_in_dim(*args: object, **kwargs: object) -> object:
    """DynamicUpdateSliceInDim frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DynamicUpdateSliceInDim", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DynamicUpdateSliceInDim", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def gather(*args: object, **kwargs: object) -> object:
    """Gather frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Gather", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Gather", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def gather_nd(*args: object, **kwargs: object) -> object:
    """GatherNd frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("GatherNd", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("GatherNd", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def hsplit(*args: object, **kwargs: object) -> object:
    """Hsplit frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Hsplit", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Hsplit", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def hstack(*args: object, **kwargs: object) -> object:
    """Hstack frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Hstack", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Hstack", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def scatter_add(*args: object, **kwargs: object) -> object:
    """ScatterAdd frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ScatterAdd", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ScatterAdd", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def scatter_apply(*args: object, **kwargs: object) -> object:
    """ScatterApply frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ScatterApply", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ScatterApply", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def scatter_max(*args: object, **kwargs: object) -> object:
    """ScatterMax frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ScatterMax", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ScatterMax", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def scatter_min(*args: object, **kwargs: object) -> object:
    """ScatterMin frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ScatterMin", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ScatterMin", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def scatter_mul(*args: object, **kwargs: object) -> object:
    """ScatterMul frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ScatterMul", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ScatterMul", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def scatter_nd(*args: object, **kwargs: object) -> object:
    """ScatterNd frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ScatterNd", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ScatterNd", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def slice_in_dim(*args: object, **kwargs: object) -> object:
    """SliceInDim frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SliceInDim", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SliceInDim", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def sort(*args: object, **kwargs: object) -> object:
    """Sort frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Sort", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Sort", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def split(*args: object, **kwargs: object) -> object:
    """Split frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Split", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Split", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def squeeze(*args: object, **kwargs: object) -> object:
    """Squeeze frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Squeeze", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Squeeze", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def strided_slice(*args: object, **kwargs: object) -> object:
    """StridedSlice frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("StridedSlice", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("StridedSlice", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def transpose(*args: object, **kwargs: object) -> object:
    """Transpose frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Transpose", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Transpose", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def vsplit(*args: object, **kwargs: object) -> object:
    """Vsplit frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Vsplit", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Vsplit", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def vstack(*args: object, **kwargs: object) -> object:
    """Vstack frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Vstack", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Vstack", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def where(*args: object, **kwargs: object) -> object:
    """Where frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Where", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Where", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


__all__ = [
    "argpartition",
    "argsort",
    "argwhere",
    "column_stack",
    "dsplit",
    "dstack",
    "dynamic_index_in_dim",
    "dynamic_slice",
    "dynamic_slice_in_dim",
    "dynamic_update_index_in_dim",
    "dynamic_update_slice_in_dim",
    "gather",
    "gather_nd",
    "hsplit",
    "hstack",
    "scatter_add",
    "scatter_apply",
    "scatter_max",
    "scatter_min",
    "scatter_mul",
    "scatter_nd",
    "slice_in_dim",
    "sort",
    "split",
    "squeeze",
    "strided_slice",
    "transpose",
    "vsplit",
    "vstack",
    "where",
]
