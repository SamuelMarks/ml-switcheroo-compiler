"""CLI verifier for Haskell-style backend capability typeclasses and curated YAML mappings."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.typeclass_verifier import verify_all_typeclasses


def main(argv: Sequence[str] | None = None) -> int:
    """Run static AST, typeclass dictionary, and curated YAML verification across backends.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments (`--backends-dir`).

    Returns:
        int: Exit code `0` when all capability obligations hold, or `1` on violation.
    """
    parser = argparse.ArgumentParser(
        description="Verify Haskell-style backend typeclass capabilities and curated YAML mappings."
    )
    parser.add_argument(
        "--backends-dir",
        default="src/ml_switcheroo_compiler/backends",
        help="Path to backends package directory.",
    )
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    backends_dir = str(args.backends_dir)
    violations = verify_all_typeclasses(backends_dir=backends_dir)
    if violations:
        for msg in violations:
            sys.stderr.write(msg + os.linesep)
        return 1
    sys.stdout.write("Haskell-style backend typeclass and YAML verification passed." + os.linesep)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
