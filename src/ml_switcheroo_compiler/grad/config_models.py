"""Pydantic models for finite difference config."""

from pydantic import BaseModel


class FiniteDifferenceDtypeConfig(BaseModel):
    """Configuration for finite difference dtype."""

    epsilon: float


class FiniteDifferenceConfig(BaseModel):
    """Configuration for finite difference."""

    float32: FiniteDifferenceDtypeConfig
    float64: FiniteDifferenceDtypeConfig
