"""Aliases for array_ops."""

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.reductions import cumsum
from ml_switcheroo_compiler.ops.shape import unsqueeze
from ml_switcheroo_compiler.ops.shape.frontend import concatenate
from ml_switcheroo_compiler.ops.unary import conj

from .common import create_eager_alias

expand_dims = unsqueeze

ndarray = Tensor

cumulative_sum = cumsum

concat = concatenate

conjugate = conj


def flip(m: object, axis: int = None) -> object:
    """Reverse the order of elements in an array along the given axis."""
    from ml_switcheroo_compiler.ops.creation.frontend import asarray
    from ml_switcheroo_compiler.ops.shape.frontend import reverse

    m = asarray(m)
    return reverse(m, dims=axis if axis is not None else tuple(range(len(m.shape))))


def fliplr(m: object) -> object:
    """Reverse the order of elements along axis 1 (left/right)."""
    from ml_switcheroo_compiler.ops.shape.frontend import reverse

    return reverse(m, dims=1)


def flipud(m: object) -> object:
    """Reverse the order of elements along axis 0 (up/down)."""
    from ml_switcheroo_compiler.ops.shape.frontend import reverse

    return reverse(m, dims=0)


def ediff1d(ary: object, to_end: object = None, to_begin: object = None) -> object:
    """The differences between consecutive elements of an array."""
    from ml_switcheroo_compiler.ops.creation.frontend import asarray
    from ml_switcheroo_compiler.ops.shape.frontend import diff
    from ml_switcheroo_compiler.ops.shape.manipulation import flatten

    ary = flatten(asarray(ary))
    return diff(ary, n=1, axis=-1, prepend=to_begin, append=to_end)


def extract(condition: object, arr: object) -> object:
    """Return the elements of an array that satisfy some condition."""
    from ml_switcheroo_compiler.ops.creation.frontend import asarray
    from ml_switcheroo_compiler.ops.shape.frontend import compress
    from ml_switcheroo_compiler.ops.shape.manipulation import flatten

    return compress(flatten(asarray(condition)), flatten(asarray(arr)), axis=0)


class _C_Class:
    def __getitem__(self, key: object) -> object:
        raise NotImplementedError("c_ is not fully supported yet.")


c_ = _C_Class()


class _R_Class:
    def __getitem__(self, key: object) -> object:
        raise NotImplementedError("r_ is not fully supported yet.")


r_ = _C_Class()


class _S_Class:
    def __getitem__(self, key: object) -> object:
        return key


s_ = _S_Class()


class _IndexExp_Class:
    def __getitem__(self, key: object) -> object:
        return key


index_exp = _IndexExp_Class()


gradient = create_eager_alias("gradient")


matrix_transpose = create_eager_alias("matrix_transpose")


rollaxis = create_eager_alias("rollaxis")
