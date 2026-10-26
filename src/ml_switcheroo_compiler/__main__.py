"""CLI entrypoint module for python -m ml_switcheroo_compiler."""

from __future__ import annotations

import sys

from ml_switcheroo_compiler.cli import main

__all__ = ["main"]

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
