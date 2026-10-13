"""Lint Any."""

import argparse
import subprocess
import sys


def main():
    """Main."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-count", type=int, default=0)
    args = parser.parse_args()

    result = subprocess.run(["rg", "-n", r"typing\.Any", "src/ml_switcheroo_compiler/"], capture_output=True, text=True)
    count = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0

    if count > args.max_count:
        print(f"Found {count} usages of typing.Any. Maximum allowed is {args.max_count}.")
        sys.exit(1)

    print("typing.Any check passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
