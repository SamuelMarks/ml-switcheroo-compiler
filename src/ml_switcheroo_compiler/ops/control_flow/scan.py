"""Sequential scan control flow operator implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor

Carry = TypeVar("Carry")
OutElem = TypeVar("OutElem")


def scan(
    f: Callable[[Carry, Tensor], tuple[Carry, OutElem]],
    init: Carry,
    xs: Tensor,
    length: int | None = None,
) -> tuple[Carry, Tensor]:
    """Iterate over leading axis of xs, carrying state and producing stacked outputs.

    Dispatches to eager loop accumulation when eager mode is enabled, or records a
    Scan control flow node into the active tracing IR graph.

    Args:
        f (Callable[[Carry, Tensor], tuple[Carry, OutElem]]): Binary function taking (carry, x) and returning (carry, y).
        init (Carry): Initial carry value or tensor.
        xs (Tensor): Tensor whose leading axis will be sliced and scanned.
        length (Optional[int]): Optional maximum number of scan iterations.

    Returns:
        tuple[Carry, Tensor]: Tuple containing the final carry state and the stacked output tensor.
    """
    import ml_switcheroo_compiler.ops.control_flow as cf

    if config.eager_mode:
        return cf.scan_eager(f, init, xs, length)
    return cf.scan_tracing(f, init, xs, length)
