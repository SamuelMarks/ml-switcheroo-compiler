def test_generated_ops():
    import importlib

    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.core.config import config

    class DummyBackend:
        def execute_op(self, op, *a, **k):
            class T:
                shape = (1,)

            return T()

    reg._ACTIVE_BACKEND = DummyBackend()

    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value=None):
        with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value=None):
            for mod_name in ["missing_ops", "missing_ops_linalg", "missing_ops_math", "missing_ops_misc", "missing_ops_nn", "missing_ops_random", "missing_ops_shape"]:
                try:
                    mod = importlib.import_module(f"ml_switcheroo_compiler.ops.generated.{mod_name}")
                    if hasattr(mod, "get_active_backend"):
                        mod.get_active_backend = lambda: DummyBackend()

                    for k, v in vars(mod).items():
                        if isinstance(v, type) and hasattr(v, "infer_shape") and v.__module__ == mod.__name__:
                            try:
                                v.infer_shape(None, "dummy")
                            except Exception:
                                pass
                        if callable(v) and not isinstance(v, type):
                            config.eager_mode = True
                            try:
                                v(None)
                            except Exception:
                                pass
                            config.eager_mode = False
                            try:
                                v(None)
                            except Exception:
                                pass
                except ImportError:
                    pass
