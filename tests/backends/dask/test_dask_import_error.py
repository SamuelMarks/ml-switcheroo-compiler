import sys
from unittest.mock import patch


def test_dask_import_error():
    # Remove from sys.modules to force a fresh execution
    for mod in list(sys.modules.keys()):
        if "ml_switcheroo_compiler.backends.dask" in mod:
            del sys.modules[mod]

    with patch.dict("sys.modules", {"dask.array": None}):
        import ml_switcheroo_compiler.backends.dask.eager as eager

        assert eager.da is None

        import ml_switcheroo_compiler.backends.dask.types as types

        assert types.da is None

        import ml_switcheroo_compiler.backends.dask.generator as generator

        assert generator.da is None

    # restore to hit the if da is not None branch
    for mod in list(sys.modules.keys()):
        if "ml_switcheroo_compiler.backends.dask" in mod:
            del sys.modules[mod]
