from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.io import _eager_base64, decode_bmp, decode_gif, decode_jpeg, decode_png, set_default_stream, set_memory_limit, set_wired_limit


def test_io_missing_branches(monkeypatch):
    class MockBackend:
        pass

    class MockBackendWithOps:
        @staticmethod
        def set_default_stream(s):
            return "stream"

        @staticmethod
        def set_memory_limit(l):
            return "mem"

        @staticmethod
        def set_wired_limit(l):
            return "wired"

        @staticmethod
        def decode_jpeg(*args, **kwargs):
            return "jpeg"

        @staticmethod
        def decode_png(*args, **kwargs):
            return "png"

        @staticmethod
        def decode_gif(*args, **kwargs):
            return "gif"

        @staticmethod
        def decode_bmp(*args, **kwargs):
            return "bmp"

    register_backend("mlx_mock")(MockBackend)
    register_backend("mlx_mock_ops")(MockBackendWithOps)

    import builtins

    original_import = builtins.__import__

    config.backend = "not_mlx"
    set_default_stream(None)
    set_memory_limit(None)
    set_wired_limit(None)

    def mocked_import(name, *args, **kwargs):
        if "registry" in name:
            raise ImportError("no registry")
        return original_import(name, *args, **kwargs)

    config.backend = "mlx"

    builtins.__import__ = mocked_import
    try:
        set_default_stream(None)
        set_memory_limit(None)
        set_wired_limit(None)
    finally:
        builtins.__import__ = original_import

    config.backend = "mlx_mock"
    assert decode_jpeg(None).data is None
    assert decode_png(None).data is None
    assert decode_gif(None).data is None
    assert decode_bmp(None).data is None

    config.backend = "mlx_mock_ops"
    assert decode_jpeg(None) == "jpeg"
    assert decode_png(None) == "png"
    assert decode_gif(None) == "gif"
    assert decode_bmp(None) == "bmp"

    register_backend("mlx")(MockBackend)
    config.backend = "mlx"
    set_default_stream(None)
    set_memory_limit(None)
    set_wired_limit(None)

    assert _eager_base64("encode", [b"hello", b"world"], pad=False) == [b"aGVsbG8", b"d29ybGQ"]
    assert _eager_base64("encode", (b"hello",), pad=False) == [b"aGVsbG8"]
    assert _eager_base64("encode", [None]) == [b""]
