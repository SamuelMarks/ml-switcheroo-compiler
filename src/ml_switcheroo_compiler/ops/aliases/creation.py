"""Aliases for creation."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_10_0

from ml_switcheroo_compiler.core.shape import broadcast_shapes as _bs
from ml_switcheroo_compiler.ops.binary import power
from ml_switcheroo_compiler.ops.configs import SpaceConfig
from ml_switcheroo_compiler.ops.creation.frontend import linspace
from ml_switcheroo_compiler.ops.shape.frontend import broadcast_to

from .common import create_eager_alias


def broadcast_shapes(*shapes: object) -> object:
    """Execute broadcast_shapes.

    Args:
        *shapes (Any): Argument *shapes.

    Returns:
    Any: The result.
    """
    return _bs(*shapes)


def logspace(
    start: object,
    stop: object,
    config: SpaceConfig = None,
) -> object:
    """Execute logspace.

    Args:
        start (Any): Argument start.
        stop (Any): Argument stop.
        config (SpaceConfig): Argument config.

    Returns:
    Any: The result.
    """
    if config is None:
        config = SpaceConfig()
    num = config.num

    base = config.base
    dtype = config.dtype

    y = linspace(start, stop, steps=num, dtype=dtype)
    if base == MAGIC_VAL_10_0:
        return power(10.0, y)
    return power(base, y)


def broadcast(x: object, sizes: object) -> object:
    """Execute broadcast.

    Args:
        x (Any): Argument x.
        sizes (Any): Argument sizes.

    Returns:
    Any: The result.
    """
    return broadcast_to(x, sizes)


fill_diagonal = create_eager_alias("fill_diagonal")


fromfunction = create_eager_alias("fromfunction")


fromiter = create_eager_alias("fromiter")


frompyfunc = create_eager_alias("frompyfunc")


fromstring = create_eager_alias("fromstring")


geomspace = create_eager_alias("geomspace")


indices = create_eager_alias("indices")


mask_indices = create_eager_alias("mask_indices")


class _MgridClass_mgrid:
    """Class docstring."""

    def __getitem__(self, key: object) -> object:
        """Function docstring.

        Args:
        key: Arg.
        """
        raise NotImplementedError("mgrid not fully supported")  # pragma: no cover


mgrid = _MgridClass_mgrid()


class _MgridClass_ogrid:
    """Class docstring."""

    def __getitem__(self, key: object) -> object:
        """Function docstring.

        Args:
        key: Arg.
        """
        raise NotImplementedError("ogrid not fully supported")  # pragma: no cover


ogrid = _MgridClass_ogrid()


ravel_multi_index = create_eager_alias("ravel_multi_index")


tri = create_eager_alias("tri")


tril_indices = create_eager_alias("tril_indices")


tril_indices_from = create_eager_alias("tril_indices_from")


triu_indices = create_eager_alias("triu_indices")


triu_indices_from = create_eager_alias("triu_indices_from")


vander = create_eager_alias("vander")
