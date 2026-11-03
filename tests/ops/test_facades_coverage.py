"""Tests for test_facades_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.export as export_pkg
import ml_switcheroo_compiler.interpreter as interpreter_pkg
import ml_switcheroo_compiler.nn as nn_pkg
import ml_switcheroo_compiler.transforms as transforms_pkg
import ml_switcheroo_compiler.transforms.autodiff_rules as autodiff_rules_pkg
from ml_switcheroo_compiler.core.shape import _has_attr


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
    assert _has_attr(export_pkg, "ExportArchive")

    assert _has_attr(interpreter_pkg, "Environment")
    assert _has_attr(interpreter_pkg, "evaluate_graph")

    assert "glorot_uniform" in nn_pkg.__all__
    assert _has_attr(nn_pkg, "glorot_uniform")

    assert _has_attr(transforms_pkg, "PassManager")
    assert _has_attr(transforms_pkg, "constant_folding_pass")
    assert _has_attr(transforms_pkg, "grad")

    assert _has_attr(autodiff_rules_pkg, "custom_rules")
