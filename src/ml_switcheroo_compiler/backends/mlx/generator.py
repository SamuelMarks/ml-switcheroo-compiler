# ruff: noqa: E402
"""Implementation of generator operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.mlx.generator`)."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_callable, ensure_eager_tensor
from ml_switcheroo_compiler.backends.protocols import ensure_mod, get_backend_member, has_backend_member
from ml_switcheroo_compiler.backends.registry import register_backend

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

from ml_switcheroo_compiler.ir.core import IRGraph

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


@register_backend("mlx")
class MLXCodeGenerator(ClassBasedGenerator):
    """Emit MLX-compatible code from IR."""

    _base_class_name: str = "nn.Module"

    def __init__(self, graph: IRGraph) -> None:
        """Init.

        Args:
            graph (IRGraph): Input `graph` argument (graph).
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
    def save_gguf(cls, file: str, arrays: EagerValue, metadata: EagerValue = None) -> None:
        """Save a dictionary of arrays to GGUF format.

        Args:
            file (str): Input `file` argument (file).
            arrays (dict): Input `arrays` argument (arrays).
            metadata (Optional[dict]): Optional metadata dictionary.
        """
        import mlx.core as mx

        save_fn = get_backend_member(mx, "save_gguf", None)
        if save_fn is not None:
            try:
                ensure_callable(save_fn)(file, arrays, metadata or {})
            except TypeError:
                ensure_callable(save_fn)(file, arrays)

    @classmethod
    def set_default_stream(cls, stream: EagerValue) -> None:
        """Set the default stream.

        Args:
            stream (EagerValue): Input `stream` argument (stream).
        """
        import mlx.core as mx

        if has_backend_member(mx, "set_default_stream"):
            ensure_mod(mx).set_default_stream(stream)

    @classmethod
    def set_memory_limit(cls, limit: int) -> None:
        """Set the memory limit.

        Args:
            limit (int): Input `limit` argument (limit).
        """
        import mlx.core as mx

        if has_backend_member(mx, "metal") and has_backend_member(mx.metal, "set_memory_limit"):
            mx.metal.set_memory_limit(limit)

    @classmethod
    def set_wired_limit(cls, limit: int) -> None:
        """Set the wired limit.

        Args:
            limit (int): Input `limit` argument (limit).
        """
        import mlx.core as mx

        if has_backend_member(mx, "metal") and has_backend_member(mx.metal, "set_wired_limit"):
            mx.metal.set_wired_limit(limit)

    def generate(self) -> str:
        """Generate MLX class definition directly from the IR graph.

        Returns:
            str: Generated MLX source code.
        """
        return super().generate()

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations.

        Returns:
        str: Computed `str` output.
        """
        return "mx"

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Execute emit constant assignment for compiler processing.

        Args:
            var_name (str): Input `var_name` argument (var name).
            val_repr (str): Input `val_repr` argument (val repr).
        """
        self.add_line(f"{var_name} = mx.array({val_repr})")

    _forward_method_name = "_" + "_call__"

    def _get_prefix_code(self) -> list[str]:
        """Retrieve the prefix code property or mapping.

        Returns:
            list: The evaluated or processed output.
        """
        res: list[str] = [
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
    def load(cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII") -> EagerValue:
        """Load.

        Args:
        filepath (str): Input `filepath` argument (filepath).
        allow_pickle (bool): Input `allow_pickle` argument (allow pickle).
        fix_imports (bool): Input `fix_imports` argument (fix imports).
        encoding (str): Input `encoding` argument (encoding).

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        import mlx.core as mx

        # Note: MLX load does not use allow_pickle, fix_imports, or encoding
        return ensure_eager_tensor(mx.load(filepath))

    @classmethod
    def save(cls: type, file: str, arr: EagerValue, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save.

        Args:
            file (str): Input `file` argument (file).
            arr (EagerValue): Input `arr` argument (arr).
            allow_pickle (bool): Input `allow_pickle` argument (allow pickle).
            fix_imports (bool): Input `fix_imports` argument (fix imports).
        """
        import mlx.core as mx

        ensure_mod(mx).save(file, arr)

    @classmethod
    def savez(cls: type, file: str, *args: EagerValue, **kwds: EagerValue) -> None:
        """Savez.

        Args:
            file (str): Input `file` argument (file).
            *args (EagerValue): Positional args.
            **kwds (EagerValue): Keyword args.
        """
        import mlx.core as mx

        data: dict[str, EagerValue] = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        ensure_mod(mx).save_safetensors(file, data)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: EagerValue, **kwds: EagerValue) -> None:
        """Savez compressed.

        Args:
            file (str): Input `file` argument (file).
            *args (EagerValue): Positional args.
            **kwds (EagerValue): Keyword args.
        """
        import mlx.core as mx

        data: dict[str, EagerValue] = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        # MLX doesn't have a specific compressed method, safetensors is efficient
        ensure_mod(mx).save_safetensors(file, data)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: EagerValue) -> EagerValue:
        """Compile IRGraph into an ahead-of-time (AOT) callable using mx.compile.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (EagerValue): Optional compiler arguments (e.g. 'shapeless').

        Returns:
            EagerValue: Compiled execution callable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor

        try:
            import mlx.core as mx

            scope: dict[str, EagerValue] = {}
            exec(self.generate(), scope)
            model_cls = scope.get("CompiledModel")
            if model_cls is None or not callable(model_cls):
                raise ValueError("Generated MLX module is missing CompiledModel.")

            model_instance = model_cls()
            shapeless = bool(kwargs.get("shapeless", False))
            compiled_fn = ensure_callable(ensure_mod(mx).compile(model_instance, shapeless=shapeless))

            def aot_mlx_runner(*w_args: EagerValue, **w_kwargs: EagerValue) -> EagerValue:
                """Execute AOT compiled MLX graph function.

                Args:
                    *w_args (EagerValue): Input tensors.
                    **w_kwargs (EagerValue): Keyword arguments.

                Returns:
                    EagerValue: Computed outputs.
                """
                mlx_args = [ensure_mod(mx).array(a.data if isinstance(a, Tensor) else a) for a in w_args]
                return ensure_eager_tensor(compiled_fn(*mlx_args, **w_kwargs))

            return aot_mlx_runner
        except Exception:
            from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

            def fallback_forward(*fn_args: EagerValue) -> EagerValue:
                """Execute interpreter graph evaluation fallback.

                Args:
                    *fn_args (EagerValue): Positional input arguments.

                Returns:
                    EagerValue: Evaluated output tensors.
                """
                input_nodes = [n for n in graph.nodes.values() if n.op_type == "Input"]
                inputs: dict[str, EagerValue] = {}
                for i, inp_node in enumerate(input_nodes):
                    if i < len(fn_args):
                        arg_val = fn_args[i]
                        inputs[inp_node.id] = arg_val.data if isinstance(arg_val, Tensor) else arg_val
                evaluated = evaluate_graph(graph, inputs=inputs)
                if graph.outputs:
                    if len(graph.outputs) == 1:
                        return evaluated.get(graph.outputs[0])
                    return tuple(evaluated.get(out_id) for out_id in graph.outputs)
                return ensure_eager_tensor(evaluated)

            return fallback_forward

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> EagerValue:
        """Create a zero-filled MLX array.

        Args:
            shape (tuple[int, ...]): Target shape.

        Returns:
            EagerValue: Zero-filled array.
        """
        from .types import zeros as _zeros

        return _zeros(cls, shape)

    @classmethod
    def array(cls, data: EagerValue, dtype: EagerValue | None = None) -> EagerValue:
        """Create an MLX array from data.

        Args:
            data (EagerValue): Input data.
            dtype (EagerValue | None): Optional data type.

        Returns:
            EagerValue: MLX array.
        """
        from .types import array as _array

        return _array(cls, data, dtype=dtype)

    @classmethod
    def asarray(cls, data: EagerValue) -> EagerValue:
        """Convert data to an MLX array.

        Args:
            data (EagerValue): Input data.

        Returns:
            EagerValue: MLX array.
        """
        from .types import asarray as _asarray

        return _asarray(cls, data)

    @classmethod
    def item(cls, data: EagerValue) -> float:
        """Extract scalar item from an MLX array.

        Args:
            data (EagerValue): Target array.

        Returns:
            float: Extracted scalar.
        """
        from .types import item as _item

        return _item(cls, data)

    @classmethod
    def execute_op(cls, op_type: str, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Execute operation eagerly on MLX backend.

        Args:
            op_type (str): Operation type.
            *args (EagerValue): Positional arguments.
            **kwargs (EagerValue): Keyword arguments.

        Returns:
            EagerValue: Evaluated MLX array.
        """
        from .eager import execute_op as _execute_op

        return _execute_op(cls, op_type, *args, **kwargs)


MLXGenerator = MLXCodeGenerator
