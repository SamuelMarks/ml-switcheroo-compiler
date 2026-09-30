"""Tests for test_facades_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.export as export_pkg
import ml_switcheroo_compiler.interpreter as interpreter_pkg
import ml_switcheroo_compiler.nn as nn_pkg
import ml_switcheroo_compiler.nn.activations as nn_activations
import ml_switcheroo_compiler.transforms as transforms_pkg
import ml_switcheroo_compiler.transforms.autodiff_rules as autodiff_rules_pkg


class _DummyBadShape:
    """Shape object whose iterator raises TypeError."""

    def __iter__(self) -> _DummyBadShape:
        """Return self as iterator.

        Returns:
            _DummyBadShape: Self.
        """
        return self

    def __next__(self) -> int:
        """Raise TypeError on iteration.

        Raises:
            TypeError: Simulated invalid dimension.
        """
        raise TypeError("Not iterable dimension")


def test_package_facades_exhaustive() -> None:
    """Verify exports and facades for export, interpreter, nn, transforms, and autodiff packages."""
    assert hasattr(export_pkg, "ExportArchive")

    assert hasattr(interpreter_pkg, "Environment")
    assert hasattr(interpreter_pkg, "evaluate_graph")

    assert nn_pkg.__all__ == []
    assert nn_activations.__all__ == []

    assert hasattr(transforms_pkg, "PassManager")
    assert hasattr(transforms_pkg, "constant_folding_pass")
    assert hasattr(transforms_pkg, "grad")

    assert hasattr(autodiff_rules_pkg, "custom_rules")
