# ruff: noqa: C901, PLR0911
"""Ahead-of-Time compilation hooks for frontend integrations."""

import importlib
from collections.abc import Callable

from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from ml_switcheroo_compiler.transforms.pass_manager import PassManager

# Cache for compiled artifacts
_COMPILATION_CACHE: dict[str, Callable] = {}


def _fallback_eager(fn: Callable, args: tuple, kw: dict) -> object:
    """Execute function eagerly as a fallback.

    Args:
        fn: The original function.
        args: Positional arguments.
        kw: Keyword arguments.

    Returns:
        object: Result of eager evaluation.
    """
    was_eager = config.eager_mode
    config.eager_mode = True
    try:
        return fn(*args, **kw)
    finally:
        config.eager_mode = was_eager


def _build_signature_key(fn: Callable, backend: str, args: tuple) -> str:
    """Build a cache key for the compilation signature."""
    sig_parts = []
    for a in args:
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            sig_parts.append(f"T({getattr(a, 'shape', ())},{getattr(a, 'dtype', '')})")
        else:
            sig_parts.append(f"S({type(a).__name__})")
    return f"{id(fn)}_{backend}_" + "_".join(sig_parts)


def _prepare_proxy_args(args: tuple) -> list[object]:
    """Prepare proxy arguments for tracing."""
    proxy_args = []
    for i, a in enumerate(args):
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            arg_id = f"arg_{i}"
            shape = getattr(a, "shape", ())
            proxy = ProxyTensor(id=arg_id, shape=shape, dtype=str(getattr(a, "dtype", "")))
            dtype = getattr(a, "dtype", DType.Float32)
            device = getattr(a, "device", "cpu")
            proxy_args.append(Tensor(proxy, TensorConfig(proxy.shape, dtype, device)))
            TracingNodeBuilder.create_tracing_logical_node("Input", [], {}, shape)

            # Overwrite id to match
            last_node = list(global_tracing_state.active_graph.nodes.values())[-1]
            old_id = last_node.id
            del global_tracing_state.active_graph.nodes[old_id]
            last_node.id = arg_id
            global_tracing_state.active_graph.nodes[arg_id] = last_node
        else:
            proxy_args.append(a)
    return proxy_args


def _capture_outputs(out: object) -> None:
    """Capture outputs in the active tracing graph."""
    if isinstance(out, Tensor):
        out_id, _ = TracingNodeBuilder.extract_from_tensor(out)
        out_node_id = TracingNodeBuilder.create_tracing_logical_node("Output", [out_id], {}, getattr(out, "shape", ()))
        global_tracing_state.active_graph.outputs.append(out_node_id)
    elif isinstance(out, (list, tuple)):
        out_ids = []
        for x in out:
            if isinstance(x, Tensor):
                oid, _ = TracingNodeBuilder.extract_from_tensor(x)
                out_ids.append(oid)
        if out_ids:
            out_node_id = TracingNodeBuilder.create_tracing_logical_node("Output", out_ids, {}, ())
            global_tracing_state.active_graph.outputs.append(out_node_id)


def _get_namespace(backend: str, generator_cls: type) -> dict[str, object]:
    """Get the namespace for executing generated code."""
    namespace = {}
    if hasattr(generator_cls, "get_module"):
        namespace[backend] = generator_cls.get_module()
    else:
        try:
            namespace[backend] = importlib.import_module(backend)
        except ImportError:
            pass
    return namespace


def compile_function(fn: Callable[..., object], backend: str = "numpy", **kwargs: object) -> Callable[..., object]:
    """Compiles a function functionally, intended for torch.compile integrations.

    Args:
        fn: The function to compile.
        backend: The target execution backend.
        kwargs: Additional compilation options.

    Returns:
        Callable: The compiled function.
    """

    def compiled_wrapper(*args: object, **kw: object) -> object:
        key = _build_signature_key(fn, backend, args)
        if key in _COMPILATION_CACHE:
            return _COMPILATION_CACHE[key](*args, **kw)

        tape = TracerTape()
        tape.start_tracing()
        proxy_args = _prepare_proxy_args(args)

        try:
            out = fn(*proxy_args, **kw)
            _capture_outputs(out)
        except Exception:
            tape.stop_tracing()
            return _fallback_eager(fn, args, kw)

        graph = tape.stop_tracing()
        if not graph or not graph.outputs:
            return _fallback_eager(fn, args, kw)

        pm = PassManager()
        graph = pm.run(graph)

        try:
            generator_cls = BackendRegistry.get(backend)
        except ValueError:
            return _fallback_eager(fn, args, kw)

        if hasattr(generator_cls, "compile_aot"):
            compiled_fn = generator_cls.compile_aot(graph, **kwargs)
            _COMPILATION_CACHE[key] = compiled_fn
            return compiled_fn(*args, **kw)

        generator = generator_cls(graph)
        code = generator.generate()
        namespace = _get_namespace(backend, generator_cls)

        try:
            exec(code, namespace)  # noqa: S102
            if "apply_model" in namespace:

                def apply_wrapper(*w_args: object, **w_kw: object) -> object:
                    return namespace["apply_model"]({}, *w_args, **w_kw)

                compiled_fn = apply_wrapper
            elif "evaluate" in namespace:

                def eval_wrapper(*w_args: object, **w_kw: object) -> object:
                    numpy_args = [a.data if isinstance(a, Tensor) else a for a in w_args]
                    return namespace["evaluate"](numpy_args)

                compiled_fn = eval_wrapper
            else:
                return _fallback_eager(fn, args, kw)

            res = compiled_fn(*args, **kw)
            _COMPILATION_CACHE[key] = compiled_fn
            return res
        except Exception:
            return _fallback_eager(fn, args, kw)

    return compiled_wrapper
