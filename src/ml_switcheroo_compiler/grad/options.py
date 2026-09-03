"""Options for grad."""

from dataclasses import dataclass, field
from typing import Optional, Union

DEFAULT_GRAD_EPSILON = 1e-3


@dataclass
class GradOptions:
    """Options for gradient compilation."""

    argnums: Union[int, tuple[int, ...]] = 0
    has_aux: bool = False
    holistic: bool = False
    reduce_axes: tuple[int, ...] = field(default_factory=tuple)
    return_value: bool = False


@dataclass
class GradCheckOptions:
    """Options for gradient checking."""

    order: int = 1
    atol: float = DEFAULT_GRAD_EPSILON
    rtol: float = DEFAULT_GRAD_EPSILON
    step: float = DEFAULT_GRAD_EPSILON


@dataclass
class JitOptions:
    """Options for JIT compilation."""

    static_argnums: tuple[int, ...] = field(default_factory=tuple)
    device: Optional[str] = None
    backend: Optional[str] = None
    inline: bool = False
