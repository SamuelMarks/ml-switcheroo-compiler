"""Docstring."""

import ml_switcheroo.ops as ops
from ml_switcheroo.jnp.array import _to_tensor, _wrap


def einsum(subscripts: str, *operands: object) -> object:
    """Evaluates the Einstein summation convention on the operands.

    Args:
        subscripts (Any): Argument subscripts.
        *operands: Argument operands.

    Returns:
        Any: The result of the operation.
    """
    tensors = [_to_tensor(a) for a in operands]
    return _wrap(ops.einsum(subscripts, *tensors))


def dot(a: object, b: object) -> object:
    """Dot product of two arrays.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.dot(_to_tensor(a), _to_tensor(b)))


def matmul(a: object, b: object) -> object:
    """Matrix product of two arrays.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.matmul(_to_tensor(a), _to_tensor(b)))


def vdot(a: object, b: object) -> object:
    """Return the dot product of two vectors.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.vdot(_to_tensor(a), _to_tensor(b)))


def inner(a: object, b: object) -> object:
    """Inner product of two arrays.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.inner(_to_tensor(a), _to_tensor(b)))


def outer(a: object, b: object) -> object:
    """Compute the outer product of two vectors.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.outer(_to_tensor(a), _to_tensor(b)))


def tensordot(a: object, b: object, axes: object = 2) -> object:
    """Compute tensor dot product along specified axes.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.
        axes (Any): Argument axes.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.tensordot(_to_tensor(a), _to_tensor(b), axes=axes))
