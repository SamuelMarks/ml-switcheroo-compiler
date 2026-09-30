"""Tests for test_debugging_coverage."""

from __future__ import annotations

import os
import shutil
import tempfile

import ml_switcheroo_compiler.diagnostics.debugging as debugging_mod


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


def test_debugging_exhaustive() -> None:
    """Verify dump debug info directory creation."""
    temp_dir = tempfile.mkdtemp()
    try:
        dump_target = os.path.join(temp_dir, "subdir", "debug_dump")
        assert not os.path.exists(dump_target)
        debugging_mod.enable_dump_debug_info(dump_target)
        assert os.path.exists(dump_target)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
