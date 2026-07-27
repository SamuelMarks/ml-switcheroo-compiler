"""Shared utilities for tests."""


class DummyDimSpecs:
    """Dummy dimension specifications for convolution tests."""

    def __init__(self) -> None:
        """Initialize the dummy dimension specs."""
        self.lhs_spec = (0, 1, 2)
        self.rhs_spec = (0, 1, 2)
        self.out_spec = (0, 1, 2)
