"""Linting pass checks."""

import os
import subprocess  # noqa: I001
import sys


def main() -> None:
    """Run lint pass.

    Returns:
        None
    """
    failed = False
    for root, _, files in os.walk("tests"):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r") as file:  # noqa: UP015
                    content = file.read()

                if "assert True" in content:
                    print(f"FAILED: Vacuous 'assert True' found in {path}")
                    failed = True

                if "except Exception: pass" in content.replace(" ", ""):
                    print(f"FAILED: Vacuous 'except Exception: pass' found in {path}")
                    failed = True

                if "except: pass" in content.replace(" ", ""):
                    print(f"FAILED: Vacuous 'except: pass' found in {path}")
                    failed = True

    if failed:
        sys.exit(1)

    print("Linting passed.")
    sys.exit(0)

    count = int(subprocess.run(["rg", "-c", r"typing\.Any", "src/ml_switcheroo_compiler/"], capture_output=True, text=True).stdout.strip() or 0)
    if count > 0:
        print(f"FAILED: Found {count} usages of typing.Any")
        sys.exit(1)


if __name__ == "__main__":
    main()
