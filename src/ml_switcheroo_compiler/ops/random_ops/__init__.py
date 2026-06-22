"""Random ops module."""

from .frontend import sobol_sample
from .sobol import SobolSample

__all__ = ["sobol_sample", "SobolSample"]
