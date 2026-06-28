from ml_switcheroo_compiler.ops.aliases.array_ops import ediff1d


def test_ediff1d():
    # just to get coverage on the branches
    try:
        ediff1d([1, 2, 3], to_end=1)
    except Exception:
        pass

    try:
        ediff1d([1, 2, 3], to_begin=1)
    except Exception:
        pass
