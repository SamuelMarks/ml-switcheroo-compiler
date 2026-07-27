from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import io
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_fallback_load_branches():
    assert io._fallback_load(123) is None
    assert io._fallback_load("unsupported.txt") is None


def test_tracing_mode_emits():
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        t = Tensor([1], TensorConfig((1,), "float32", "cpu"))

        io.load(t)
        io.save("path.npy", t)
        io.save_gguf("path.gguf", t)
        io.savez("path.npz", t)
        io.savez_compressed("path.npz", t)
        io.read_file("path")
        io.write_file("path", t)
        io.decode_image(t)
        io.decode_csv(t, [1.0])
        io.parse_example(t, {"f": 1})
        io.serialize_tensor(t)
        io.parse_tensor(t, "float32")
        io.encode_base64(t)
        io.decode_base64(t)
        io.parse_sequence_example(t)
    finally:
        global_tracing_state.stop_tracing()


def test_eager_save_gguf():
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.side_effect = NotImplementedError()
        with pytest.raises(RuntimeError):
            io.save_gguf("path")
