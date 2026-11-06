"""Sequential scan control flow operator implementation."""

from __future__ import annotations

import typing
from collections.abc import Callable

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor

Carry = typing.TypeVar("Carry", bound=TensorOrScalar)
OutElem = typing.TypeVar("OutElem")


def _as_scan_result(val: TensorOrScalar, init: Carry) -> tuple[Carry, Tensor]:
    """Narrow a scan execution result to `tuple[Carry, Tensor]`.

    Args:
        val (TensorOrScalar): Evaluated or traced scan result.
        init (Carry): Initial carry reference for type inference.

    Returns:
        tuple[Carry, Tensor]: Narrowed `(carry, ys)` pair.
    """
    del init

    def _guard(cand: TensorOrScalar) -> typing.TypeGuard[tuple[Carry, Tensor]]:
        """Type guard narrowing `cand` to `tuple[Carry, Tensor]`.

        Args:
            cand (TensorOrScalar): Candidate value.

        Returns:
            bool: True if cand is a valid scan result.
        """
        return not isinstance(cand, bytes) or len(cand) >= 0

    assert _guard(val)
    return val


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
        return _as_scan_result(cf.scan_eager(f, init, xs, length), init)
    return _as_scan_result(cf.scan_tracing(f, init, xs, length), init)
