"""Sequential scan control flow operator implementation."""

from __future__ import annotations

import typing
from collections.abc import Callable

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.tensor import Tensor

Carry = typing.TypeVar("Carry", bound=OpValue)
OutElem = typing.TypeVar("OutElem", bound=OpValue)


@typing.overload
def scan(
    f: Callable[[Carry, Tensor], tuple[Carry, OutElem]],
    init: Carry,
    xs: Tensor,
    length: int | None = None,
) -> tuple[OpValue, Tensor]: ...


@typing.overload
def scan(
    f: OpValue,
    init: Carry,
    xs: Tensor | OpValue,
    length: int | None = None,
) -> tuple[OpValue, Tensor]: ...


def scan(
    f: Callable[[Carry, Tensor], tuple[Carry, OutElem]] | OpValue,
    init: Carry,
    xs: Tensor | OpValue,
    length: int | None = None,
) -> tuple[OpValue, Tensor]:
    """Iterate over leading axis of ``xs``, carrying state and producing stacked outputs.

    Dispatches to eager loop accumulation when eager mode is enabled, or records a
    ``Scan`` control flow node into the active tracing IR graph.

    Args:
        f (Callable[[Carry, Tensor], tuple[Carry, OutElem]] | OpValue): Binary function taking
            ``(carry, x)`` and returning ``(carry, y)``.
        init (Carry): Initial carry value or tensor.
        xs (Tensor | OpValue): Tensor whose leading axis will be sliced and scanned.
        length (int | None): Optional maximum number of scan iterations.

    Returns:
        tuple[OpValue, Tensor]: Final carry state and stacked output ``Tensor``.
    """
    import ml_switcheroo_compiler.ops.control_flow as cf

    if config.eager_mode:
        return cf.scan_eager(f, init, xs, length)
    return cf.scan_tracing(f, init, xs, length)
