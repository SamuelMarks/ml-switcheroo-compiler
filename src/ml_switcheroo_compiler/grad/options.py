"""Options for grad."""

from __future__ import annotations

import pydantic

DEFAULT_GRAD_EPSILON = 1e-3


class GradOptions(pydantic.BaseModel):
    """Options for gradient compilation."""

    model_config = pydantic.ConfigDict(extra="forbid")

    argnums: int | tuple[int, ...] = 0
    has_aux: bool = False
    holistic: bool = False
    reduce_axes: tuple[int, ...] = pydantic.Field(default_factory=tuple)
    return_value: bool = False


class GradCheckOptions(pydantic.BaseModel):
    """Options for gradient checking."""

    model_config = pydantic.ConfigDict(extra="forbid")

    order: int = 1
    atol: float = DEFAULT_GRAD_EPSILON
    rtol: float = DEFAULT_GRAD_EPSILON
    step: float = DEFAULT_GRAD_EPSILON


class JitOptions(pydantic.BaseModel):
    """Options for JIT compilation."""

    model_config = pydantic.ConfigDict(extra="forbid")

    static_argnums: tuple[int, ...] = pydantic.Field(default_factory=tuple)
    device: str | None = None
    backend: str | None = None
    inline: bool = False
