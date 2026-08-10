import os


def fix_file(filepath):
    with open(filepath) as f:
        content = f.read()

    # We want to replace parameter names dim with axis for consistency across ops.
    if "shape/" in filepath or "nn/" in filepath or "linalg/" in filepath:
        content = content.replace("axisension", "axis")
        content = content.replace("valid_dim", "valid_axis")

    with open(filepath, "w") as f:
        f.write(content)


for root, _, files in os.walk("src/ml_switcheroo_compiler/ops"):
    for file in files:
        if file.endswith(".py"):
            fix_file(os.path.join(root, file))
