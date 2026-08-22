# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for generator.py."""

from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors

from .mlx_mixins import MLXAudioVisitor, MLXNNOpsVisitor, MLXShapeOpsVisitor, MLXVisionVisitor

_MLX_RESIZE_TMPL = """def mlx_resize(images, size, interpolation, align_corners):
    import mlx.core as mx
    N, H, W, C = images.shape
    new_H, new_W = size
    if align_corners:
        h_idx = mx.round(mx.arange(new_H) * ((H - 1) / max(1, new_H - 1)))
        w_idx = mx.round(mx.arange(new_W) * ((W - 1) / max(1, new_W - 1)))
    else:
        h_idx = mx.floor(mx.arange(new_H) * (H / new_H))
        w_idx = mx.floor(mx.arange(new_W) * (W / new_W))
    h_idx = mx.clip(h_idx.astype(mx.int32), 0, H - 1)
    w_idx = mx.clip(w_idx.astype(mx.int32), 0, W - 1)
    return images[:, h_idx[:, None], w_idx[None, :], :]
"""

_MLX_ISTFT_TMPL = """def mlx_istft(matrix, config):
    import mlx.core as mx
    frames = matrix.shape[-2] if matrix.ndim == 3 else matrix.shape[-3]
    window = mx.array(config.window) if config.window is not None else mx.ones((config.fft_length,))
    if matrix.dtype != mx.complex64:
        comp_matrix = matrix[..., 0] + 1j * matrix[..., 1]
    else:
        comp_matrix = matrix
    time_frames = mx.fft.irfft(comp_matrix, n=config.fft_length, axis=-1)
    time_frames = time_frames * window
    expected_length = (frames - 1) * config.frame_step + config.fft_length
    batch_shape = time_frames.shape[:-2]
    out = mx.zeros((*batch_shape, expected_length))
    for i in range(frames):
        start = i * config.frame_step
        out[..., start:start + config.fft_length] = out[..., start:start + config.fft_length] + time_frames[..., i, :]
    return out
"""

_MLX_MEL_FILTERBANK_TMPL = """def mlx_mel_filterbank(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz):
    import mlx.core as mx
    def hz_to_mel(hz): return 2595.0 * mx.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_low = hz_to_mel(mx.array(lower_edge_hertz))
    mel_high = hz_to_mel(mx.array(upper_edge_hertz))
    mel_points = mx.linspace(mel_low, mel_high, num_mel_bins + 2)
    hz_points = mel_to_hz(mel_points)
    bin_freqs = mx.linspace(0.0, sample_rate / 2.0, num_spectrogram_bins)
    fbank = mx.zeros((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        low, center, high = hz_points[i], hz_points[i+1], hz_points[i+2]
        up_slope = (bin_freqs - low) / (center - low)
        down_slope = (high - bin_freqs) / (high - center)
        weights = mx.maximum(mx.zeros_like(bin_freqs), mx.minimum(up_slope, down_slope))
        fbank[:, i] = weights
    return fbank
"""

_MLX_MFCC_TMPL = """def mlx_mfcc(spectrogram, config):
    import mlx.core as mx
    mel_weights = mlx_mel_filterbank(config.num_mel_bins, spectrogram.shape[-1], config.sample_rate, config.lower_edge_hertz, config.upper_edge_hertz)
    mel_spectrogram = mx.matmul(spectrogram, mel_weights)
    log_mel = mx.log(mel_spectrogram + 1e-6)
    N = config.num_mel_bins
    n_mfcc = config.num_mfccs
    n = mx.arange(N)
    k = mx.arange(n_mfcc)[:, None]
    dct_mat = mx.cos(3.141592653589793 / N * (n + 0.5) * k)
    dct_mat = dct_mat * mx.sqrt(2.0 / N)
    dct_mat[0, :] = dct_mat[0, :] * 0.7071067811865476
    return mx.matmul(log_mel, dct_mat.T)
"""
from typing import Any

_MLX_POWER_ITERATION_TMPL = """def mlx_power_iteration(w, num_iters, u=None):
    import mlx.core as mx
    if u is None:
        u = mx.ones(w.shape[:-2] + [w.shape[-2], 1], dtype=w.dtype)
    def body_fn(val):
        i, u_curr, _ = val
        w_t = mx.swapaxes(w, -1, -2)
        v_next = mx.matmul(w_t, u_curr)
        v_next = v_next / (mx.linalg.norm(v_next, axis=-2, keepdims=True) + 1e-12)
        u_next = mx.matmul(w, v_next)
        u_next = u_next / (mx.linalg.norm(u_next, axis=-2, keepdims=True) + 1e-12)
        return i + 1, u_next, v_next
    def cond_fn(val):
        return val[0] < num_iters
    init_v = mx.zeros(w.shape[:-2] + [w.shape[-1], 1], dtype=w.dtype)
    _, u_final, v_final = mx.while_loop(cond_fn, body_fn)( (mx.array(0), u, init_v) )
    sigma = mx.matmul(mx.swapaxes(u_final, -1, -2), mx.matmul(w, v_final))
    return mx.squeeze(v_final, -1), mx.squeeze(u_final, -1), mx.squeeze(mx.squeeze(sigma, -1), -1)
"""


class MLXCodeGenerator(ClassBasedGenerator):
    """Emit MLX-compatible code from IR."""

    _base_class_name: str = "nn.Module"

    def __init__(self, graph: Any) -> None:
        """Init.

        Args:
            graph (object): The graph parameter.
        """
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

    @classmethod
    def save_gguf(cls, file: str, arrays: dict[str, Any]) -> None:
        """Save a dictionary of arrays to GGUF format.

        Args:
            file (str): The file parameter.
            arrays (dict): The arrays parameter.
        """
        import mlx.core as mx

        if hasattr(mx, "save_gguf"):
            mx.save_gguf(file, arrays)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    @classmethod
    def set_default_stream(cls, stream: Any) -> None:
        """Set the default stream.

        Args:
            stream (object): The stream parameter.
        """
        import mlx.core as mx

        if hasattr(mx, "set_default_stream"):
            mx.set_default_stream(stream)

    @classmethod
    def set_memory_limit(cls, limit: int) -> None:
        """Set the memory limit.

        Args:
            limit (int): The limit parameter.
        """
        import mlx.core as mx

        if hasattr(mx, "metal") and hasattr(mx.metal, "set_memory_limit"):
            mx.metal.set_memory_limit(limit)

    @classmethod
    def set_wired_limit(cls, limit: int) -> None:
        """Set the wired limit.

        Args:
            limit (int): The limit parameter.
        """
        import mlx.core as mx

        if hasattr(mx, "metal") and hasattr(mx.metal, "set_wired_limit"):
            mx.metal.set_wired_limit(limit)

    def generate(self) -> str:
        """Generate code using strict AST construction (CST) from a base NumPy string."""
        from ml_switcheroo_compiler.backends.cst_transpiler import transpile_source
        from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator

        gen = NumpyGenerator(self.graph)
        base_code = gen.generate()
        return transpile_source(base_code, target_framework="mlx")

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations.

        Returns:
        str: Result.
        """
        return "mx"

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate _emit_constant_assignment operation.

        Args:
            var_name (str): The var_name parameter.
            val_repr (str): The val_repr parameter.
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
    def load(cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII") -> Any:
        """Load.

        Args:
        filepath (str): The filepath parameter.
        allow_pickle (bool): The allow_pickle parameter.
        fix_imports (bool): The fix_imports parameter.
        encoding (str): The encoding parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        import mlx.core as mx

        # Note: MLX load does not use allow_pickle, fix_imports, or encoding
        return mx.load(filepath)

    @classmethod
    def save(cls: type, file: str, arr: Any, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save.

        Args:
            file (str): The file parameter.
            arr (object): The arr parameter.
            allow_pickle (bool): The allow_pickle parameter.
            fix_imports (bool): The fix_imports parameter.
        """
        import mlx.core as mx

        mx.save(file, arr)

    @classmethod
    def savez(cls: type, file: str, *args: Any, **kwds: Any) -> None:
        """Savez.

        Args:
            file (str): The file parameter.
            *args (object): Positional args.
            **kwds (object): Keyword args.
        """
        import mlx.core as mx

        data = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        mx.save_safetensors(file, data)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: Any, **kwds: Any) -> None:
        """Savez compressed.

        Args:
            file (str): The file parameter.
            *args (object): Positional args.
            **kwds (object): Keyword args.
        """
        import mlx.core as mx

        data = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        # MLX doesn't have a specific compressed method, safetensors is efficient
        mx.save_safetensors(file, data)
