"""Remove numpy eval from ops."""

import os
import re


def remove_numpy_eval(content: str) -> str:
    """Remove numpy eval block.

    Args:
        content (str): Content string.

    Returns:
        str: Resulting content string.
    """
    content = re.sub(r"^[ \t]*import numpy as np\n", "", content, flags=re.MULTILINE)

    lines = content.split("\n")
    new_lines = []
    skip = False
    indent_level = -1

    for line in lines:
        if re.match(r"^(\s*)def numpy_eval\(", line):
            skip = True
            indent_level = len(re.match(r"^(\s*)", line).group(1))
            continue

        if skip:
            if re.match(r"^(\s+)def ", line) or re.match(r"^(\s+)@", line):
                curr_indent = len(re.match(r"^(\s+)", line).group(1))
                if curr_indent <= indent_level:
                    skip = False
            elif re.match(r"^\S", line):  # Dedent back to global
                skip = False

        if not skip:
            new_lines.append(line)

    # clear trailing newlines
    result = "\n".join(new_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


for root, _, files in os.walk("src/ml_switcheroo_compiler/ops/"):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath) as f:
                content = f.read()
            new_content = remove_numpy_eval(content)
            with open(filepath, "w") as f:
                f.write(new_content)
