# ruff: noqa: E501
"""Core abstractions and logic definitions for generator.py."""

from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors

from .mlx_mixins import MLXAudioVisitor, MLXNNOpsVisitor, MLXShapeOpsVisitor, MLXVisionVisitor

_MLX_RESIZE_TMPL = "def mlx_resize(images, size, interpolation, align_corners):\n    import mlx.core as mx\n    from ml_switcheroo_compiler.backends.eager.vision_geometric import resize_eager\n    from ml_switcheroo_compiler.ops.configs import ResizeOptions\n    # Fallback to eager numpy for bicubic/lanczos3 on mlx\n\n    imgs_np = np.array(images)\n    config = ResizeOptions(interpolation=interpolation, align_corners=align_corners)\n    out = resize_eager(np, imgs_np, size, config)\n    return mx.array(out)"  # noqa: E501
_MLX_ISTFT_TMPL = "def mlx_istft(matrix, config: STFTConfig):\n    import mlx.core as mx\n\n    from ml_switcheroo_compiler.backends.eager.audio import istft_eager\n    from ml_switcheroo_compiler.ops.configs import STFTConfig\n    config = STFTConfig(frame_length=config.frame_length, frame_step=config.frame_step, fft_length=config.fft_length, window=config.window, center=config.center)\n    out = istft_eager(np, np.array(matrix), config)\n    return mx.array(out)"  # noqa: E501
_MLX_MEL_FILTERBANK_TMPL = "def mlx_mel_filterbank(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz):\n    import mlx.core as mx\n\n    from ml_switcheroo_compiler.backends.eager.audio import mel_filterbank_eager\n    from ml_switcheroo_compiler.ops.configs import MelConfig\n    config = MelConfig(num_mel_bins=config.num_mel_bins, num_spectrogram_bins=num_spectrogram_bins, sample_rate=config.sample_rate, lower_edge_hertz=config.lower_edge_hertz, upper_edge_hertz=config.upper_edge_hertz)\n    out = mel_filterbank_eager(np, config)\n    return mx.array(out)"  # noqa: E501
_MLX_MFCC_TMPL = "def mlx_mfcc(spectrogram, config: MFCCConfig):\n    import mlx.core as mx\n\n    from ml_switcheroo_compiler.backends.eager.audio import mfcc_eager\n    from ml_switcheroo_compiler.ops.configs import MelConfig\n    config = MelConfig(num_mel_bins=config.num_mel_bins, num_spectrogram_bins=spectrogram.shape[-1], sample_rate=config.sample_rate, lower_edge_hertz=config.lower_edge_hertz, upper_edge_hertz=config.upper_edge_hertz, num_mfccs=config.num_mfccs)\n    out = mfcc_eager(np, np.array(spectrogram), config)\n    return mx.array(out)"  # noqa: E501
_MLX_POWER_ITERATION_TMPL = "def mlx_power_iteration(w, num_iters, u=None):\n    import mlx.core as mx\n    if u is None:\n        u = mx.ones(w.shape[:-2] + [w.shape[-2], 1], dtype=w.dtype)\n    def body_fn(val):\n        i, u_curr, _ = val\n        w_t = mx.swapaxes(w, -1, -2)\n        v_next = mx.matmul(w_t, u_curr)\n        v_next = v_next / (mx.linalg.norm(v_next, axis=-2, keepdims=True) + 1e-12)\n        u_next = mx.matmul(w, v_next)\n        u_next = u_next / (mx.linalg.norm(u_next, axis=-2, keepdims=True) + 1e-12)\n        return i + 1, u_next, v_next\n    def cond_fn(val):\n        return val[0] < num_iters\n    init_v = mx.zeros(w.shape[:-2] + [w.shape[-1], 1], dtype=w.dtype)\n    _, u_final, v_final = mx.while_loop(cond_fn, body_fn)( (mx.array(0), u, init_v) )\n    sigma = mx.matmul(mx.swapaxes(u_final, -1, -2), mx.matmul(w, v_final))\n    return mx.squeeze(v_final, -1), mx.squeeze(u_final, -1), mx.squeeze(mx.squeeze(sigma, -1), -1)"  # noqa: E501


class MLXCodeGenerator(ClassBasedGenerator):
    """Emit MLX-compatible code from IR."""

    def __init__(self, graph: object) -> None:
        """Init."""
        super().__init__(graph)
        self.visitors.extend(
            [
                *get_shared_ast_visitors(generator=self),
                MLXNNOpsVisitor(),
                MLXVisionVisitor(),
                MLXAudioVisitor(),
                MLXShapeOpsVisitor(),
            ]
        )

    def _get_backend_prefix(self) -> str:
        """Retrieve the backend prefix property or mapping.

        Returns:
            str: The evaluated or processed output.
        """
        return "mlx"

    @classmethod
    def save_gguf(cls, file: str, arrays: dict[str, object]) -> None:
        """Save a dictionary of arrays to GGUF format."""
        import mlx.core as mx

        if hasattr(mx, "save_gguf"):
            mx.save_gguf(file, arrays)

    @classmethod
    def set_default_stream(cls, stream: object) -> None:
        """Set the default stream."""
        import mlx.core as mx

        if hasattr(mx, "set_default_stream"):
            mx.set_default_stream(stream)

    @classmethod
    def set_memory_limit(cls, limit: int) -> None:
        """Set the memory limit."""
        import mlx.core as mx

        if hasattr(mx, "metal") and hasattr(mx.metal, "set_memory_limit"):
            mx.metal.set_memory_limit(limit)

    @classmethod
    def set_wired_limit(cls, limit: int) -> None:
        """Set the wired limit."""
        import mlx.core as mx

        if hasattr(mx, "metal") and hasattr(mx.metal, "set_wired_limit"):
            mx.metal.set_wired_limit(limit)

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations."""
        return "mx"

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = mx.array({val_repr})")

    _forward_method_name = "__call__"

    def _get_prefix_code(self) -> list[str]:
        """Retrieve the prefix code property or mapping.

        Returns:
            list: The evaluated or processed output.
        """
        res = [
            "import mlx.core as mx",
            "import mlx.nn as nn\n",
            *_MLX_RESIZE_TMPL.split("\n"),
            *_MLX_ISTFT_TMPL.split("\n"),
            *_MLX_MEL_FILTERBANK_TMPL.split("\n"),
            *_MLX_MFCC_TMPL.split("\n"),
            *_MLX_POWER_ITERATION_TMPL.split("\n"),
        ]

        return res

    @classmethod
    def load(cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII") -> object:
        """Load."""
        import numpy as np

        return np.load(filepath, allow_pickle=allow_pickle, fix_imports=fix_imports, encoding=encoding)

    @classmethod
    def save(cls: type, file: str, arr: object, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save."""
        import numpy as np

        np.save(file, arr, allow_pickle=allow_pickle, fix_imports=fix_imports)

    @classmethod
    def savez(cls: type, file: str, *args: object, **kwds: object) -> None:
        """Savez."""
        import numpy as np

        np.savez(file, *args, **kwds)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: object, **kwds: object) -> None:
        """Savez compressed."""
        import numpy as np

        np.savez_compressed(file, *args, **kwds)
